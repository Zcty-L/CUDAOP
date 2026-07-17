#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cuda_utils.cuh"
#include "ptx_utils.cuh"

// =============================================================================
// Conv2D Kernel: 128x128 output tile, K_chunk=8
// Supports ANN (FP32/FP16) and SNN (Packed Spikes uint8)
// =============================================================================

struct Conv2DK128Param
{
    uint32_t M;              // total output pixels per image: Oh * Ow
    uint32_t K;              // total inner product dim: C * R * S
    uint32_t N;              // output channels: K_out
    uint32_t N_padded;       // K_out padded to 128
    
    uint32_t C, H, W;
    uint32_t Oh, Ow, R, S;
    uint32_t U, V, P, Q;
    
    uint32_t inBatchNumel;   // C * H * W
    uint32_t outBatchNumel;  // N * Oh * Ow
};

// -----------------------------------------------------------------------------
// ANN FP32 Optimized Kernel (Double Buffer)
// -----------------------------------------------------------------------------
__global__ void conv2d_128x128x8_kernel(
    const float *inputs,
    const float *weights, // [N_padded, K_padded]
    const float *bias,
    float *outputs,
    Conv2DK128Param param)
{
    __shared__ __align__(16 * 1024) char smem[16 * 1024];
    float *A_smem_base = reinterpret_cast<float *>(smem);
    float *B_smem_base = reinterpret_cast<float *>(smem + 8 * 1024);
    float *smembias = reinterpret_cast<float *>(smem);

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;
    const int mma_tid_x = (lane_id / 2) % 8;
    const int mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);
    const int thread_m_base = warp_m * 32 + mma_tid_y * 8;
    const int thread_n_base = warp_n * 64 + mma_tid_x * 8;

    if (tid < 128) {
        const int n_global = blockIdx.x * 128 + tid;
        smembias[tid] = (n_global < param.N) ? bias[n_global] : 0.0f;
    }
    __syncthreads();

    float C_frag[8][8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const int m_global = blockIdx.y * 128 + thread_m_base + i;
        float mask = (m_global < param.M) ? 1.0f : 0.0f;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            C_frag[i][j] = smembias[thread_n_base + j] * mask;
        }
    }
    __syncthreads();

    const int load_idx = tid / 2;    // 0..127
    const int load_k   = (tid % 2) * 4; // 0, 4
    const int b_n_global = blockIdx.x * 128 + load_idx;
    const int K_padded = (param.K + 7) / 8 * 8;
    const float *b_src_row = weights + (size_t)b_n_global * K_padded;

    auto get_input_val = [&](int m_idx, int k_idx) {
        if (m_idx >= param.M || k_idx >= param.K) return 0.0f;
        int oh = m_idx / param.Ow;
        int ow = m_idx % param.Ow;
        int c = k_idx / (param.R * param.S);
        int rs = k_idx % (param.R * param.S);
        int r = rs / param.S;
        int s = rs % param.S;
        int ih = oh * param.U - param.P + r;
        int iw = ow * param.V - param.Q + s;
        if (ih >= 0 && ih < param.H && iw >= 0 && iw < param.W) {
            return inputs[(size_t)blockIdx.z * param.inBatchNumel + (size_t)c * param.H * param.W + ih * param.W + iw];
        }
        return 0.0f;
    };

    const int k_iters = (param.K + 7) / 8;

    auto load_chunk = [&](int k_tile, int buf) {
        int k_curr = k_tile * 8 + load_k;
        uint32_t a_reg[4];
        #pragma unroll
        for (int i=0; i<4; i++) {
            float val = get_input_val(blockIdx.y * 128 + load_idx, k_curr + i);
            a_reg[i] = reinterpret_cast<uint32_t&>(val);
        }
        uint32_t a_dst = smem_u32addr(&A_smem_base[buf * 1024 + load_idx * 8 + load_k]);
        ptx::sts128(a_reg[0], a_reg[1], a_reg[2], a_reg[3], a_dst);

        int b_src_bytes = (b_n_global < param.N_padded) ? 16 : 0;
        uint32_t b_dst = smem_u32addr(&B_smem_base[buf * 1024 + load_idx * 8 + load_k]);
        asm volatile ("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" :: "r"(b_dst), "l"(b_src_row + k_curr), "r"(b_src_bytes));
    };

    if (k_iters > 0) { load_chunk(0, 0); asm volatile ("cp.async.commit_group;\n" :::); }
    for (int kt = 0; kt < k_iters; kt++) {
        if (kt + 1 < k_iters) { load_chunk(kt + 1, (kt + 1) % 2); asm volatile ("cp.async.commit_group;\n" :::); }
        if (kt + 1 < k_iters) { asm volatile ("cp.async.wait_group 1;\n" :::); }
        else { asm volatile ("cp.async.wait_group 0;\n" :::); }
        __syncthreads();
        float *A_curr = &A_smem_base[(kt % 2) * 1024];
        float *B_curr = &B_smem_base[(kt % 2) * 1024];
#pragma unroll
        for (int k = 0; k < 8; k++) {
            float a_vals[8], b_vals[8];
#pragma unroll
            for (int i = 0; i < 8; i++) a_vals[i] = A_curr[(thread_m_base + i) * 8 + k];
#pragma unroll
            for (int j = 0; j < 8; j++) b_vals[j] = B_curr[(thread_n_base + j) * 8 + k];
#pragma unroll
            for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++) C_frag[i][j] += a_vals[i] * b_vals[j];
        }
        __syncthreads();
    }

    // --- Vectorized Output Write-back (Direct STG128) ---
#pragma unroll
    for (int i = 0; i < 8; i += 4) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int n_global = blockIdx.x * 128 + thread_n_base + j;
            const int m_global = blockIdx.y * 128 + thread_m_base + i;
            if (n_global < param.N) {
                if (m_global + 3 < param.M) {
                    ptx::stg128(C_frag[i][j], C_frag[i+1][j], C_frag[i+2][j], C_frag[i+3][j], 
                                &outputs[(size_t)blockIdx.z * param.outBatchNumel + (size_t)n_global * param.M + m_global], true);
                } else {
#pragma unroll
                    for (int ii = 0; ii < 4; ii++) {
                        if (m_global + ii < param.M)
                            outputs[(size_t)blockIdx.z * param.outBatchNumel + (size_t)n_global * param.M + m_global + ii] = C_frag[i+ii][j];
                    }
                }
            }
        }
    }
}

__global__ void conv2d_128x128x8_kernel_v1(
    const float *inputs,
    const float *weights, // [K_padded, N_padded]
    const float *bias,
    float *outputs,
    Conv2DK128Param param)
{
    __shared__ __align__(16 * 1024) char smem[16 * 1024];
    float *A_smem_base = reinterpret_cast<float *>(smem);
    float *B_smem_base = reinterpret_cast<float *>(smem + 8 * 1024);
    float *smembias = reinterpret_cast<float *>(smem);

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;
    const int mma_tid_x = (lane_id / 2) % 8;
    const int mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);
    const int thread_m_base = warp_m * 32 + mma_tid_y * 8;
    const int thread_n_base = warp_n * 64 + mma_tid_x * 8;

    if (tid < 128) {
        const int n_global = blockIdx.x * 128 + tid;
        smembias[tid] = (n_global < param.N) ? bias[n_global] : 0.0f;
    }
    __syncthreads();

    float C_frag[8][8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const int m_global = blockIdx.y * 128 + thread_m_base + i;
        float mask = (m_global < param.M) ? 1.0f : 0.0f;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            C_frag[i][j] = smembias[thread_n_base + j] * mask;
        }
    }
    __syncthreads();

    const int load_idx = tid / 2;
    const int load_k = (tid % 2) * 4;
    const int K_padded = (param.K + 7) / 8 * 8;

    const int b_load_k = tid / 32;
    const int b_load_n = (tid % 32) * 4;
    const int b_n_global = blockIdx.x * 128 + b_load_n;

    auto get_input_val = [&](int m_idx, int k_idx) {
        if (m_idx >= param.M || k_idx >= param.K) return 0.0f;
        int oh = m_idx / param.Ow;
        int ow = m_idx % param.Ow;
        int c = k_idx / (param.R * param.S);
        int rs = k_idx % (param.R * param.S);
        int r = rs / param.S;
        int s = rs % param.S;
        int ih = oh * param.U - param.P + r;
        int iw = ow * param.V - param.Q + s;
        if (ih >= 0 && ih < param.H && iw >= 0 && iw < param.W) {
            return inputs[(size_t)blockIdx.z * param.inBatchNumel + (size_t)c * param.H * param.W + ih * param.W + iw];
        }
        return 0.0f;
    };

    const int k_iters = (param.K + 7) / 8;

    auto load_chunk = [&](int k_tile, int buf) {
        int k_curr = k_tile * 8 + load_k;
        uint32_t a_reg[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            float val = get_input_val(blockIdx.y * 128 + load_idx, k_curr + i);
            a_reg[i] = reinterpret_cast<uint32_t&>(val);
        }
        uint32_t a_dst = smem_u32addr(&A_smem_base[buf * 1024 + load_idx * 8 + load_k]);
        ptx::sts128(a_reg[0], a_reg[1], a_reg[2], a_reg[3], a_dst);

        int b_k_global = k_tile * 8 + b_load_k;
        int b_src_bytes = (b_k_global < K_padded && b_n_global < param.N_padded) ? 16 : 0;
        uint32_t b_dst = smem_u32addr(&B_smem_base[buf * 1024 + b_load_k * 128 + b_load_n]);
        asm volatile ("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::
            "r"(b_dst),
            "l"(weights + (size_t)b_k_global * param.N_padded + b_n_global),
            "r"(b_src_bytes));
    };

    if (k_iters > 0) { load_chunk(0, 0); asm volatile ("cp.async.commit_group;\n" :::); }
    for (int kt = 0; kt < k_iters; kt++) {
        if (kt + 1 < k_iters) { load_chunk(kt + 1, (kt + 1) % 2); asm volatile ("cp.async.commit_group;\n" :::); }
        if (kt + 1 < k_iters) { asm volatile ("cp.async.wait_group 1;\n" :::); }
        else { asm volatile ("cp.async.wait_group 0;\n" :::); }
        __syncthreads();
        float *A_curr = &A_smem_base[(kt % 2) * 1024];
        float *B_curr = &B_smem_base[(kt % 2) * 1024];
#pragma unroll
        for (int k = 0; k < 8; k++) {
            float a_vals[8], b_vals[8];
#pragma unroll
            for (int i = 0; i < 8; i++) a_vals[i] = A_curr[(thread_m_base + i) * 8 + k];
#pragma unroll
            for (int j = 0; j < 8; j++) b_vals[j] = B_curr[k * 128 + thread_n_base + j];
#pragma unroll
            for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++) C_frag[i][j] += a_vals[i] * b_vals[j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 8; i += 4) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int n_global = blockIdx.x * 128 + thread_n_base + j;
            const int m_global = blockIdx.y * 128 + thread_m_base + i;
            if (n_global < param.N) {
                if (m_global + 3 < param.M) {
                    ptx::stg128(C_frag[i][j], C_frag[i + 1][j], C_frag[i + 2][j], C_frag[i + 3][j],
                                &outputs[(size_t)blockIdx.z * param.outBatchNumel + (size_t)n_global * param.M + m_global], true);
                } else {
#pragma unroll
                    for (int ii = 0; ii < 4; ii++) {
                        if (m_global + ii < param.M)
                            outputs[(size_t)blockIdx.z * param.outBatchNumel + (size_t)n_global * param.M + m_global + ii] = C_frag[i + ii][j];
                    }
                }
            }
        }
    }
}

__global__ void conv2d_128x128x8_kernel_v2(
    const float *inputs,
    const float *weights, // [K_padded, N_padded]
    const float *bias,
    float *outputs,
    Conv2DK128Param param)
{
    __shared__ __align__(16 * 1024) char smem[16 * 1024];
    float *smemweight = reinterpret_cast<float *>(smem);
    float *smeminput = reinterpret_cast<float *>(smem + 8 * 1024);

    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int mma_tid_x = (lane_id / 2) % 8;
    const int mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);
    const int K_padded = (param.K + 7) / 8 * 8;

    uint32_t input_sts_addr = ptx::smem_u32addr(
        smeminput + warp_id * 128 + lane_id);
    uint32_t weight_sts_addr = ptx::smem_u32addr(
        smemweight + warp_id * 128 + lane_id * 4);

    uint32_t input_lds_addr = ptx::smem_u32addr(
        smeminput + (warp_id % 2) * 64 + mma_tid_x * 4);
    uint32_t weight_lds_addr = ptx::smem_u32addr(
        smemweight + (warp_id / 2) * 32 + mma_tid_y * 4);

    float input_frag[2][8];
    float weight_frag[2][8];
    float output_frag[8][8];

#pragma unroll
    for (int i = 0; i < 8; i++)
    {
        const int n_offset = (i < 4) ? i : i + 12;
        const int n_global = blockIdx.y * 128
            + (warp_id / 2) * 32 + mma_tid_y * 4 + n_offset;
        const float bias_val = (n_global < param.N) ? bias[n_global] : 0.0f;
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            output_frag[i][j] = bias_val;
        }
    }

    int posh_ori[4];
    int posw_ori[4];
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        int m_idx = blockIdx.x * 128 + lane_id + i * 32;
        posh_ori[i] = (m_idx / param.Ow) * param.U - param.P;
        posw_ori[i] = (m_idx % param.Ow) * param.V - param.Q;
    }

    const int k_iters = (param.K + 7) / 8;

    auto load_chunk = [&](int k_tile)
    {
        const int k_base = k_tile * 8;
        const char *input_base = reinterpret_cast<const char *>(
            inputs + (size_t)blockIdx.z * param.inBatchNumel);
        const int cur_k = k_base + warp_id;
        const int curC = cur_k / (param.R * param.S);
        const int rs = cur_k % (param.R * param.S);
        const int curR = rs / param.S;
        const int curS = rs % param.S;

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            float input_reg = 0.0f;
            int curH = posh_ori[i] + curR;
            int curW = posw_ori[i] + curS;
            int inOffset = curC * param.H * param.W + curH * param.W + curW;
            bool guard = cur_k < param.K
                && curH >= 0 && curW >= 0
                && curH < param.H && curW < param.W;
            ptx::ldg32_nc_0(input_reg, input_base + inOffset * sizeof(float), guard);
            ptx::sts32(input_reg, input_sts_addr + i * 32 * sizeof(float));
        }

        const int weight_k = k_base + warp_id;
        const int weight_n = blockIdx.y * 128 + lane_id * 4;
        const float *weight_src = weights
            + (size_t)weight_k * param.N_padded + weight_n;
        int weight_bytes = (weight_k < K_padded) ? 16 : 0;
        asm volatile ("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::
            "r"(weight_sts_addr),
            "l"(weight_src),
            "r"(weight_bytes));
    };

    for (int kt = 0; kt < k_iters; kt++)
    {
        load_chunk(kt);
        asm volatile ("cp.async.commit_group;\n" :::);
        asm volatile ("cp.async.wait_group 0;\n" :::);
        __syncthreads();

#pragma unroll
        for (int k_frag = 0; k_frag < 8; k_frag++)
        {
            ptx::lds128(
                weight_frag[k_frag % 2][0],
                weight_frag[k_frag % 2][1],
                weight_frag[k_frag % 2][2],
                weight_frag[k_frag % 2][3],
                weight_lds_addr + k_frag * 128 * sizeof(float));
            ptx::lds128(
                weight_frag[k_frag % 2][4],
                weight_frag[k_frag % 2][5],
                weight_frag[k_frag % 2][6],
                weight_frag[k_frag % 2][7],
                weight_lds_addr + (k_frag * 128 + 16) * sizeof(float));
            ptx::lds128(
                input_frag[k_frag % 2][0],
                input_frag[k_frag % 2][1],
                input_frag[k_frag % 2][2],
                input_frag[k_frag % 2][3],
                input_lds_addr + k_frag * 128 * sizeof(float));
            ptx::lds128(
                input_frag[k_frag % 2][4],
                input_frag[k_frag % 2][5],
                input_frag[k_frag % 2][6],
                input_frag[k_frag % 2][7],
                input_lds_addr + (k_frag * 128 + 32) * sizeof(float));

#pragma unroll
            for (int i = 0; i < 8; i++)
            {
#pragma unroll
                for (int j = 0; j < 8; j++)
                {
                    output_frag[i][j] += weight_frag[k_frag % 2][i]
                        * input_frag[k_frag % 2][j];
                }
            }
        }
        __syncthreads();
    }

    uint32_t output_sts_addr = ptx::smem_u32addr(
        reinterpret_cast<float4 *>(smem + warp_id * 2048)
            + mma_tid_y * 4 * 8 + mma_tid_x);
    const float *output_lds_ptr = reinterpret_cast<float *>(
        smem + warp_id * 2048) + lane_id;

    const int n_idx = blockIdx.y * 128 + (warp_id / 2) * 32;
    const int m_idx = blockIdx.x * 128 + (warp_id % 2) * 64 + lane_id;
    const int safe_n_idx = (n_idx < param.N) ? n_idx : 0;
    float *output_stg_ptr = outputs
        + (size_t)blockIdx.z * param.outBatchNumel
        + (size_t)safe_n_idx * param.M + m_idx;

    const uint32_t m_guard = (m_idx < param.M) ? param.M - m_idx : 0;
    const uint32_t n_guard = (n_idx < param.N) ? param.N - n_idx : 0;

#pragma unroll
    for (int i = 0; i < 2; i++)
    {
#pragma unroll
        for (int j = 0; j < 2; j++)
        {
            __syncthreads();

#pragma unroll
            for (int p = 0; p < 4; p++)
            {
                ptx::sts128(
                    output_frag[i * 4 + p][j * 4 + 0],
                    output_frag[i * 4 + p][j * 4 + 1],
                    output_frag[i * 4 + p][j * 4 + 2],
                    output_frag[i * 4 + p][j * 4 + 3],
                    output_sts_addr + p * 8 * sizeof(float4));
            }
            __syncthreads();

#pragma unroll
            for (int p = 0; p < 16; p++)
            {
                ptx::stg32(
                    output_lds_ptr[p * 32],
                    output_stg_ptr + (i * 16 + p) * param.M + j * 32,
                    i * 16 + p < n_guard && j * 32 < m_guard);
            }
        }
    }
}

__global__ void conv2d_128x128x8_kernel_v3(
    const float *inputs,
    const float *weights, // [K_padded, N_padded]
    const float *bias,
    float *outputs,
    Conv2DK128Param param)
{
    __shared__ __align__(16 * 1024) char smem[16 * 1024];
    float *smemweight = reinterpret_cast<float *>(smem);
    float *smeminput = reinterpret_cast<float *>(smem + 8 * 1024);

    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int mma_tid_x = (lane_id / 2) % 8;
    const int mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);
    const int K_padded = (param.K + 7) / 8 * 8;

    float input_frag[2][8];
    float weight_frag[2][8];
    float output_frag[8][8];

#pragma unroll
    for (int i = 0; i < 8; i++)
    {
        const int n_offset = (i < 4) ? i : i + 12;
        const int n_global = blockIdx.y * 128
            + (warp_id / 2) * 32 + mma_tid_y * 4 + n_offset;
        const float bias_val = (n_global < param.N) ? bias[n_global] : 0.0f;
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            output_frag[i][j] = bias_val;
        }
    }

    int posh_ori[4];
    int posw_ori[4];
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        int m_idx = blockIdx.x * 128 + lane_id + i * 32;
        posh_ori[i] = (m_idx / param.Ow) * param.U - param.P;
        posw_ori[i] = (m_idx % param.Ow) * param.V - param.Q;
    }

    const int k_iters = (param.K + 7) / 8;

    auto load_chunk = [&](int k_tile, int buf)
    {
        const int k_base = k_tile * 8;
        const char *input_base = reinterpret_cast<const char *>(
            inputs + (size_t)blockIdx.z * param.inBatchNumel);
        const int cur_k = k_base + warp_id;
        const int curC = cur_k / (param.R * param.S);
        const int rs = cur_k % (param.R * param.S);
        const int curR = rs / param.S;
        const int curS = rs % param.S;

        uint32_t input_sts_addr = ptx::smem_u32addr(
            smeminput + buf * 1024 + warp_id * 128 + lane_id);
        uint32_t weight_sts_addr = ptx::smem_u32addr(
            smemweight + buf * 1024 + warp_id * 128 + lane_id * 4);

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            float input_reg = 0.0f;
            int curH = posh_ori[i] + curR;
            int curW = posw_ori[i] + curS;
            int inOffset = curC * param.H * param.W + curH * param.W + curW;
            bool guard = cur_k < param.K
                && curH >= 0 && curW >= 0
                && curH < param.H && curW < param.W;
            ptx::ldg32_nc_0(input_reg, input_base + inOffset * sizeof(float), guard);
            ptx::sts32(input_reg, input_sts_addr + i * 32 * sizeof(float));
        }

        const int weight_k = k_base + warp_id;
        const int weight_n = blockIdx.y * 128 + lane_id * 4;
        const float *weight_src = weights
            + (size_t)weight_k * param.N_padded + weight_n;
        int weight_bytes = (weight_k < K_padded) ? 16 : 0;
        asm volatile ("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::
            "r"(weight_sts_addr),
            "l"(weight_src),
            "r"(weight_bytes));
    };

    auto compute_chunk = [&](int buf)
    {
        uint32_t input_lds_addr = ptx::smem_u32addr(
            smeminput + buf * 1024 + (warp_id % 2) * 64 + mma_tid_x * 4);
        uint32_t weight_lds_addr = ptx::smem_u32addr(
            smemweight + buf * 1024 + (warp_id / 2) * 32 + mma_tid_y * 4);

#pragma unroll
        for (int k_frag = 0; k_frag < 8; k_frag++)
        {
            ptx::lds128(
                weight_frag[k_frag % 2][0],
                weight_frag[k_frag % 2][1],
                weight_frag[k_frag % 2][2],
                weight_frag[k_frag % 2][3],
                weight_lds_addr + k_frag * 128 * sizeof(float));
            ptx::lds128(
                weight_frag[k_frag % 2][4],
                weight_frag[k_frag % 2][5],
                weight_frag[k_frag % 2][6],
                weight_frag[k_frag % 2][7],
                weight_lds_addr + (k_frag * 128 + 16) * sizeof(float));
            ptx::lds128(
                input_frag[k_frag % 2][0],
                input_frag[k_frag % 2][1],
                input_frag[k_frag % 2][2],
                input_frag[k_frag % 2][3],
                input_lds_addr + k_frag * 128 * sizeof(float));
            ptx::lds128(
                input_frag[k_frag % 2][4],
                input_frag[k_frag % 2][5],
                input_frag[k_frag % 2][6],
                input_frag[k_frag % 2][7],
                input_lds_addr + (k_frag * 128 + 32) * sizeof(float));

#pragma unroll
            for (int i = 0; i < 8; i++)
            {
#pragma unroll
                for (int j = 0; j < 8; j++)
                {
                    output_frag[i][j] += weight_frag[k_frag % 2][i]
                        * input_frag[k_frag % 2][j];
                }
            }
        }
    };

    if (k_iters > 0)
    {
        load_chunk(0, 0);
        asm volatile ("cp.async.commit_group;\n" :::);
        asm volatile ("cp.async.wait_group 0;\n" :::);
        __syncthreads();
    }

    for (int kt = 0; kt < k_iters; kt++)
    {
        const int cur_buf = kt % 2;
        const int next_buf = (kt + 1) % 2;
        if (kt + 1 < k_iters)
        {
            load_chunk(kt + 1, next_buf);
            asm volatile ("cp.async.commit_group;\n" :::);
        }

        compute_chunk(cur_buf);

        if (kt + 1 < k_iters)
        {
            asm volatile ("cp.async.wait_group 0;\n" :::);
        }
        __syncthreads();
    }

    uint32_t output_sts_addr = ptx::smem_u32addr(
        reinterpret_cast<float4 *>(smem + warp_id * 2048)
            + mma_tid_y * 4 * 8 + mma_tid_x);
    const float *output_lds_ptr = reinterpret_cast<float *>(
        smem + warp_id * 2048) + lane_id;

    const int n_idx = blockIdx.y * 128 + (warp_id / 2) * 32;
    const int m_idx = blockIdx.x * 128 + (warp_id % 2) * 64 + lane_id;
    const int safe_n_idx = (n_idx < param.N) ? n_idx : 0;
    float *output_stg_ptr = outputs
        + (size_t)blockIdx.z * param.outBatchNumel
        + (size_t)safe_n_idx * param.M + m_idx;

    const uint32_t m_guard = (m_idx < param.M) ? param.M - m_idx : 0;
    const uint32_t n_guard = (n_idx < param.N) ? param.N - n_idx : 0;

#pragma unroll
    for (int i = 0; i < 2; i++)
    {
#pragma unroll
        for (int j = 0; j < 2; j++)
        {
            __syncthreads();

#pragma unroll
            for (int p = 0; p < 4; p++)
            {
                ptx::sts128(
                    output_frag[i * 4 + p][j * 4 + 0],
                    output_frag[i * 4 + p][j * 4 + 1],
                    output_frag[i * 4 + p][j * 4 + 2],
                    output_frag[i * 4 + p][j * 4 + 3],
                    output_sts_addr + p * 8 * sizeof(float4));
            }
            __syncthreads();

#pragma unroll
            for (int p = 0; p < 16; p++)
            {
                ptx::stg32(
                    output_lds_ptr[p * 32],
                    output_stg_ptr + (i * 16 + p) * param.M + j * 32,
                    i * 16 + p < n_guard && j * 32 < m_guard);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// SNN FP32 Optimized Kernel (uint32 0/1 spikes)
// -----------------------------------------------------------------------------
__global__ void conv2d_128x128x8_S_kernel(
    const uint32_t *inputs, // [T, C, H, W], values are 0 or 1
    const float *weights,   // [K_padded, N_padded]
    const float *bias,
    float *outputs,
    Conv2DK128Param param)
{
    __shared__ __align__(16 * 1024) char smem[16 * 1024];
    float *smemweight = reinterpret_cast<float *>(smem);
    uint32_t *smeminput = reinterpret_cast<uint32_t *>(smem + 8 * 1024);

    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int mma_tid_x = (lane_id / 2) % 8;
    const int mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);
    const int K_padded = (param.K + 7) / 8 * 8;

    uint32_t input_frag[2][8];
    float weight_frag[2][8];
    float output_frag[8][8];

#pragma unroll
    for (int i = 0; i < 8; i++)
    {
        const int n_offset = (i < 4) ? i : i + 12;
        const int n_global = blockIdx.y * 128
            + (warp_id / 2) * 32 + mma_tid_y * 4 + n_offset;
        const float bias_val = (n_global < param.N) ? bias[n_global] : 0.0f;
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            output_frag[i][j] = bias_val;
        }
    }

    int posh_ori[4];
    int posw_ori[4];
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        int m_idx = blockIdx.x * 128 + lane_id + i * 32;
        posh_ori[i] = (m_idx / param.Ow) * param.U - param.P;
        posw_ori[i] = (m_idx % param.Ow) * param.V - param.Q;
    }

    const int k_iters = (param.K + 7) / 8;

    auto load_chunk = [&](int k_tile, int buf)
    {
        const int k_base = k_tile * 8;
        const char *input_base = reinterpret_cast<const char *>(
            inputs + (size_t)blockIdx.z * param.inBatchNumel);
        const int cur_k = k_base + warp_id;
        const int curC = cur_k / (param.R * param.S);
        const int rs = cur_k % (param.R * param.S);
        const int curR = rs / param.S;
        const int curS = rs % param.S;

        uint32_t input_sts_addr = ptx::smem_u32addr(
            smeminput + buf * 1024 + warp_id * 128 + lane_id);
        uint32_t weight_sts_addr = ptx::smem_u32addr(
            smemweight + buf * 1024 + warp_id * 128 + lane_id * 4);

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            uint32_t input_reg = 0;
            int curH = posh_ori[i] + curR;
            int curW = posw_ori[i] + curS;
            int inOffset = curC * param.H * param.W + curH * param.W + curW;
            bool guard = cur_k < param.K
                && curH >= 0 && curW >= 0
                && curH < param.H && curW < param.W;
            ptx::ldg32_nc_0(input_reg, input_base + inOffset * sizeof(uint32_t), guard);
            ptx::sts32(input_reg, input_sts_addr + i * 32 * sizeof(uint32_t));
        }

        const int weight_k = k_base + warp_id;
        const int weight_n = blockIdx.y * 128 + lane_id * 4;
        const float *weight_src = weights
            + (size_t)weight_k * param.N_padded + weight_n;
        int weight_bytes = (weight_k < K_padded) ? 16 : 0;
        asm volatile ("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::
            "r"(weight_sts_addr),
            "l"(weight_src),
            "r"(weight_bytes));
    };

    auto compute_chunk = [&](int buf)
    {
        uint32_t input_lds_addr = ptx::smem_u32addr(
            smeminput + buf * 1024 + (warp_id % 2) * 64 + mma_tid_x * 4);
        uint32_t weight_lds_addr = ptx::smem_u32addr(
            smemweight + buf * 1024 + (warp_id / 2) * 32 + mma_tid_y * 4);

#pragma unroll
        for (int k_frag = 0; k_frag < 8; k_frag++)
        {
            ptx::lds128(
                weight_frag[k_frag % 2][0],
                weight_frag[k_frag % 2][1],
                weight_frag[k_frag % 2][2],
                weight_frag[k_frag % 2][3],
                weight_lds_addr + k_frag * 128 * sizeof(float));
            ptx::lds128(
                weight_frag[k_frag % 2][4],
                weight_frag[k_frag % 2][5],
                weight_frag[k_frag % 2][6],
                weight_frag[k_frag % 2][7],
                weight_lds_addr + (k_frag * 128 + 16) * sizeof(float));
            ptx::lds128(
                input_frag[k_frag % 2][0],
                input_frag[k_frag % 2][1],
                input_frag[k_frag % 2][2],
                input_frag[k_frag % 2][3],
                input_lds_addr + k_frag * 128 * sizeof(uint32_t));
            ptx::lds128(
                input_frag[k_frag % 2][4],
                input_frag[k_frag % 2][5],
                input_frag[k_frag % 2][6],
                input_frag[k_frag % 2][7],
                input_lds_addr + (k_frag * 128 + 32) * sizeof(uint32_t));

#pragma unroll
            for (int j = 0; j < 8; j++)
            {
                if (input_frag[k_frag % 2][j])
                {
#pragma unroll
                    for (int i = 0; i < 8; i++)
                    {
                        output_frag[i][j] += weight_frag[k_frag % 2][i];
                    }
                }
            }
        }
    };

    if (k_iters > 0)
    {
        load_chunk(0, 0);
        asm volatile ("cp.async.commit_group;\n" :::);
        asm volatile ("cp.async.wait_group 0;\n" :::);
        __syncthreads();
    }

    for (int kt = 0; kt < k_iters; kt++)
    {
        const int cur_buf = kt % 2;
        const int next_buf = (kt + 1) % 2;
        if (kt + 1 < k_iters)
        {
            load_chunk(kt + 1, next_buf);
            asm volatile ("cp.async.commit_group;\n" :::);
        }

        compute_chunk(cur_buf);

        if (kt + 1 < k_iters)
        {
            asm volatile ("cp.async.wait_group 0;\n" :::);
        }
        __syncthreads();
    }

    uint32_t output_sts_addr = ptx::smem_u32addr(
        reinterpret_cast<float4 *>(smem + warp_id * 2048)
            + mma_tid_y * 4 * 8 + mma_tid_x);
    const float *output_lds_ptr = reinterpret_cast<float *>(
        smem + warp_id * 2048) + lane_id;

    const int n_idx = blockIdx.y * 128 + (warp_id / 2) * 32;
    const int m_idx = blockIdx.x * 128 + (warp_id % 2) * 64 + lane_id;
    const int safe_n_idx = (n_idx < param.N) ? n_idx : 0;
    float *output_stg_ptr = outputs
        + (size_t)blockIdx.z * param.outBatchNumel
        + (size_t)safe_n_idx * param.M + m_idx;

    const uint32_t m_guard = (m_idx < param.M) ? param.M - m_idx : 0;
    const uint32_t n_guard = (n_idx < param.N) ? param.N - n_idx : 0;

#pragma unroll
    for (int i = 0; i < 2; i++)
    {
#pragma unroll
        for (int j = 0; j < 2; j++)
        {
            __syncthreads();

#pragma unroll
            for (int p = 0; p < 4; p++)
            {
                ptx::sts128(
                    output_frag[i * 4 + p][j * 4 + 0],
                    output_frag[i * 4 + p][j * 4 + 1],
                    output_frag[i * 4 + p][j * 4 + 2],
                    output_frag[i * 4 + p][j * 4 + 3],
                    output_sts_addr + p * 8 * sizeof(float4));
            }
            __syncthreads();

#pragma unroll
            for (int p = 0; p < 16; p++)
            {
                ptx::stg32(
                    output_lds_ptr[p * 32],
                    output_stg_ptr + (i * 16 + p) * param.M + j * 32,
                    i * 16 + p < n_guard && j * 32 < m_guard);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// ANN FP16 Optimized Kernel
// -----------------------------------------------------------------------------
__global__ void conv2d_128x128x8_FP16_kernel(
    const half *inputs,
    const half2 *weights,
    const half *bias,
    half *outputs,
    Conv2DK128Param param)
{
    __shared__ __align__(16 * 1024) char smem[16 * 1024];
    half2 *smemweight = reinterpret_cast<half2 *>(smem);
    half2 *smeminput = reinterpret_cast<half2 *>(smem + 8 * 1024);

    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int mma_tid_x = (lane_id / 2) % 8;
    const int mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);
    const int K_pair = (param.K + 1) / 2;
    const int K_pair_padded = (K_pair + 7) / 8 * 8;

    half2 input_frag[2][8];
    half2 weight_frag[2][8];
    half2 output_frag[8][8];

#pragma unroll
    for (int i = 0; i < 8; i++)
    {
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            output_frag[i][j] = __float2half2_rn(0.0f);
        }
    }

    int posh_ori[4];
    int posw_ori[4];
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        int m_idx = blockIdx.x * 128 + lane_id + i * 32;
        posh_ori[i] = (m_idx / param.Ow) * param.U - param.P;
        posw_ori[i] = (m_idx % param.Ow) * param.V - param.Q;
    }

    const int k_iters = (K_pair + 7) / 8;

    auto load_chunk = [&](int k_tile, int buf)
    {
        const int k_base = k_tile * 8;
        const char *input_base = reinterpret_cast<const char *>(
            inputs + (size_t)blockIdx.z * param.inBatchNumel);
        const int cur_pair = k_base + warp_id;
        const int k0 = cur_pair * 2;
        const int k1 = k0 + 1;
        const int curC0 = k0 / (param.R * param.S);
        const int rs0 = k0 % (param.R * param.S);
        const int curR0 = rs0 / param.S;
        const int curS0 = rs0 % param.S;
        const int curC1 = k1 / (param.R * param.S);
        const int rs1 = k1 % (param.R * param.S);
        const int curR1 = rs1 / param.S;
        const int curS1 = rs1 % param.S;

        uint32_t input_sts_addr = ptx::smem_u32addr(
            smeminput + buf * 1024 + warp_id * 128 + lane_id);
        uint32_t weight_sts_addr = ptx::smem_u32addr(
            smemweight + buf * 1024 + warp_id * 128 + lane_id * 4);

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            half2 input_reg;
            input_reg.x = __float2half(0.0f);
            input_reg.y = __float2half(0.0f);

            int curH0 = posh_ori[i] + curR0;
            int curW0 = posw_ori[i] + curS0;
            int inOffset0 = curC0 * param.H * param.W + curH0 * param.W + curW0;
            bool guard0 = k0 < param.K
                && curH0 >= 0 && curW0 >= 0
                && curH0 < param.H && curW0 < param.W;
            ptx::ldg16_nc_0(input_reg.x,
                input_base + inOffset0 * sizeof(half), guard0);

            int curH1 = posh_ori[i] + curR1;
            int curW1 = posw_ori[i] + curS1;
            int inOffset1 = curC1 * param.H * param.W + curH1 * param.W + curW1;
            bool guard1 = k1 < param.K
                && curH1 >= 0 && curW1 >= 0
                && curH1 < param.H && curW1 < param.W;
            ptx::ldg16_nc_0(input_reg.y,
                input_base + inOffset1 * sizeof(half), guard1);
            ptx::sts32(input_reg, input_sts_addr + i * 32 * sizeof(half2));
        }

        const int weight_n = blockIdx.y * 128 + lane_id * 4;
        const half2 *weight_src = weights
            + (size_t)cur_pair * param.N_padded + weight_n;
        int weight_bytes = (cur_pair < K_pair_padded) ? 16 : 0;
        asm volatile ("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::
            "r"(weight_sts_addr),
            "l"(weight_src),
            "r"(weight_bytes));
    };

    auto compute_chunk = [&](int buf)
    {
        uint32_t input_lds_addr = ptx::smem_u32addr(
            smeminput + buf * 1024 + (warp_id % 2) * 64 + mma_tid_x * 4);
        uint32_t weight_lds_addr = ptx::smem_u32addr(
            smemweight + buf * 1024 + (warp_id / 2) * 32 + mma_tid_y * 4);

#pragma unroll
        for (int k_frag = 0; k_frag < 8; k_frag++)
        {
            ptx::lds128(
                weight_frag[k_frag % 2][0],
                weight_frag[k_frag % 2][1],
                weight_frag[k_frag % 2][2],
                weight_frag[k_frag % 2][3],
                weight_lds_addr + k_frag * 128 * sizeof(half2));
            ptx::lds128(
                weight_frag[k_frag % 2][4],
                weight_frag[k_frag % 2][5],
                weight_frag[k_frag % 2][6],
                weight_frag[k_frag % 2][7],
                weight_lds_addr + (k_frag * 128 + 16) * sizeof(half2));
            ptx::lds128(
                input_frag[k_frag % 2][0],
                input_frag[k_frag % 2][1],
                input_frag[k_frag % 2][2],
                input_frag[k_frag % 2][3],
                input_lds_addr + k_frag * 128 * sizeof(half2));
            ptx::lds128(
                input_frag[k_frag % 2][4],
                input_frag[k_frag % 2][5],
                input_frag[k_frag % 2][6],
                input_frag[k_frag % 2][7],
                input_lds_addr + (k_frag * 128 + 32) * sizeof(half2));

#pragma unroll
            for (int i = 0; i < 8; i++)
            {
#pragma unroll
                for (int j = 0; j < 8; j++)
                {
                    output_frag[i][j] = __hadd2(
                        output_frag[i][j],
                        __hmul2(weight_frag[k_frag % 2][i],
                            input_frag[k_frag % 2][j]));
                }
            }
        }
    };

    if (k_iters > 0)
    {
        load_chunk(0, 0);
        asm volatile ("cp.async.commit_group;\n" :::);
        asm volatile ("cp.async.wait_group 0;\n" :::);
        __syncthreads();
    }

    for (int kt = 0; kt < k_iters; kt++)
    {
        const int cur_buf = kt % 2;
        const int next_buf = (kt + 1) % 2;
        if (kt + 1 < k_iters)
        {
            load_chunk(kt + 1, next_buf);
            asm volatile ("cp.async.commit_group;\n" :::);
        }

        compute_chunk(cur_buf);

        if (kt + 1 < k_iters)
        {
            asm volatile ("cp.async.wait_group 0;\n" :::);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 8; i++)
    {
        const int n_offset = (i < 4) ? i : i + 12;
        const int n_global = blockIdx.y * 128
            + (warp_id / 2) * 32 + mma_tid_y * 4 + n_offset;
        if (n_global < param.N)
        {
#pragma unroll
            for (int j = 0; j < 8; j++)
            {
                const int m_offset = (j < 4) ? j : j + 28;
                const int m_global = blockIdx.x * 128
                    + (warp_id % 2) * 64 + mma_tid_x * 4 + m_offset;
                if (m_global < param.M)
                {
                    float value = __half2float(output_frag[i][j].x)
                        + __half2float(output_frag[i][j].y)
                        + __half2float(bias[n_global]);
                    outputs[(size_t)blockIdx.z * param.outBatchNumel
                        + (size_t)n_global * param.M + m_global] = __float2half(value);
                }
            }
        }
    }
}
__global__ void conv2d_128x128x8_FP16_kernel_v1(
    const half *inputs,
    const half2 *weights,
    const half *bias,
    half *outputs,
    Conv2DK128Param param)
{
    __shared__ __align__(16 * 1024) char smem[16 * 1024];
    half2 *smemweight = reinterpret_cast<half2 *>(smem);
    half2 *smeminput = reinterpret_cast<half2 *>(smem + 8 * 1024);

    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int mma_tid_x = (lane_id / 2) % 8;
    const int mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);
    const int K_pair = (param.K + 1) / 2;
    const int K_pair_padded = (K_pair + 7) / 8 * 8;

    half2 input_frag[2][8];
    half2 weight_frag[2][8];
    half2 output_frag[8][8];

#pragma unroll
    for (int i = 0; i < 8; i++)
    {
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            output_frag[i][j] = __float2half2_rn(0.0f);
        }
    }

    int posh_ori[4];
    int posw_ori[4];
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        int m_idx = blockIdx.x * 128 + lane_id + i * 32;
        posh_ori[i] = (m_idx / param.Ow) * param.U - param.P;
        posw_ori[i] = (m_idx % param.Ow) * param.V - param.Q;
    }

    const int k_iters = (K_pair + 7) / 8;

    auto load_chunk = [&](int k_tile, int buf)
    {
        const int k_base = k_tile * 8;
        const char *input_base = reinterpret_cast<const char *>(
            inputs + (size_t)blockIdx.z * param.inBatchNumel);
        const int cur_pair = k_base + warp_id;
        const int k0 = cur_pair * 2;
        const int k1 = k0 + 1;
        const int curC0 = k0 / (param.R * param.S);
        const int rs0 = k0 % (param.R * param.S);
        const int curR0 = rs0 / param.S;
        const int curS0 = rs0 % param.S;
        const int curC1 = k1 / (param.R * param.S);
        const int rs1 = k1 % (param.R * param.S);
        const int curR1 = rs1 / param.S;
        const int curS1 = rs1 % param.S;

        uint32_t input_sts_addr = ptx::smem_u32addr(
            smeminput + buf * 1024 + warp_id * 128 + lane_id);
        uint32_t weight_sts_addr = ptx::smem_u32addr(
            smemweight + buf * 1024 + warp_id * 128 + lane_id * 4);

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            half2 input_reg;
            input_reg.x = __float2half(0.0f);
            input_reg.y = __float2half(0.0f);

            int curH0 = posh_ori[i] + curR0;
            int curW0 = posw_ori[i] + curS0;
            int inOffset0 = curC0 * param.H * param.W + curH0 * param.W + curW0;
            bool guard0 = k0 < param.K
                && curH0 >= 0 && curW0 >= 0
                && curH0 < param.H && curW0 < param.W;
            ptx::ldg16_nc_0(input_reg.x,
                input_base + inOffset0 * sizeof(half), guard0);

            int curH1 = posh_ori[i] + curR1;
            int curW1 = posw_ori[i] + curS1;
            int inOffset1 = curC1 * param.H * param.W + curH1 * param.W + curW1;
            bool guard1 = k1 < param.K
                && curH1 >= 0 && curW1 >= 0
                && curH1 < param.H && curW1 < param.W;
            ptx::ldg16_nc_0(input_reg.y,
                input_base + inOffset1 * sizeof(half), guard1);
            ptx::sts32(input_reg, input_sts_addr + i * 32 * sizeof(half2));
        }

        const int weight_n = blockIdx.y * 128 + lane_id * 4;
        const half2 *weight_src = weights
            + (size_t)cur_pair * param.N_padded + weight_n;
        int weight_bytes = (cur_pair < K_pair_padded) ? 16 : 0;
        asm volatile ("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::
            "r"(weight_sts_addr),
            "l"(weight_src),
            "r"(weight_bytes));
    };

    auto compute_chunk = [&](int buf)
    {
        uint32_t input_lds_addr = ptx::smem_u32addr(
            smeminput + buf * 1024 + (warp_id % 2) * 64 + mma_tid_x * 4);
        uint32_t weight_lds_addr = ptx::smem_u32addr(
            smemweight + buf * 1024 + (warp_id / 2) * 32 + mma_tid_y * 4);

#pragma unroll
        for (int k_frag = 0; k_frag < 8; k_frag++)
        {
            ptx::lds128(
                weight_frag[k_frag % 2][0],
                weight_frag[k_frag % 2][1],
                weight_frag[k_frag % 2][2],
                weight_frag[k_frag % 2][3],
                weight_lds_addr + k_frag * 128 * sizeof(half2));
            ptx::lds128(
                weight_frag[k_frag % 2][4],
                weight_frag[k_frag % 2][5],
                weight_frag[k_frag % 2][6],
                weight_frag[k_frag % 2][7],
                weight_lds_addr + (k_frag * 128 + 16) * sizeof(half2));
            ptx::lds128(
                input_frag[k_frag % 2][0],
                input_frag[k_frag % 2][1],
                input_frag[k_frag % 2][2],
                input_frag[k_frag % 2][3],
                input_lds_addr + k_frag * 128 * sizeof(half2));
            ptx::lds128(
                input_frag[k_frag % 2][4],
                input_frag[k_frag % 2][5],
                input_frag[k_frag % 2][6],
                input_frag[k_frag % 2][7],
                input_lds_addr + (k_frag * 128 + 32) * sizeof(half2));

#pragma unroll
            for (int i = 0; i < 8; i++)
            {
#pragma unroll
                for (int j = 0; j < 8; j++)
                {
                    output_frag[i][j] = __hadd2(
                        output_frag[i][j],
                        __hmul2(weight_frag[k_frag % 2][i],
                            input_frag[k_frag % 2][j]));
                }
            }
        }
    };

    if (k_iters > 0)
    {
        load_chunk(0, 0);
        asm volatile ("cp.async.commit_group;\n" :::);
        asm volatile ("cp.async.wait_group 0;\n" :::);
        __syncthreads();
    }

    for (int kt = 0; kt < k_iters; kt++)
    {
        const int cur_buf = kt % 2;
        const int next_buf = (kt + 1) % 2;
        if (kt + 1 < k_iters)
        {
            load_chunk(kt + 1, next_buf);
            asm volatile ("cp.async.commit_group;\n" :::);
        }

        compute_chunk(cur_buf);

        if (kt + 1 < k_iters)
        {
            asm volatile ("cp.async.wait_group 0;\n" :::);
        }
        __syncthreads();
    }

    half2 output[4][4];

#pragma unroll
    for (int group = 0; group < 2; group++)
    {
#pragma unroll
        for (int p = 0; p < 4; p++)
        {
            const int n_global = blockIdx.y * 128
                + (warp_id / 2) * 32
                + mma_tid_y * 4 + group * 16 + p;
            const float bias_value = (n_global < param.N)
                ? __half2float(bias[n_global]) : 0.0f;
#pragma unroll
            for (int q = 0; q < 4; q++)
            {
                const int n_frag = group * 4 + p;
                const int m_frag = q * 2;
                output[p][q].x = __float2half(
                    __half2float(output_frag[n_frag][m_frag].x)
                    + __half2float(output_frag[n_frag][m_frag].y)
                    + bias_value);
                output[p][q].y = __float2half(
                    __half2float(output_frag[n_frag][m_frag + 1].x)
                    + __half2float(output_frag[n_frag][m_frag + 1].y)
                    + bias_value);
            }
        }

        uint32_t output_sts_addr = ptx::smem_u32addr(
            reinterpret_cast<half2 *>(smem + warp_id * 512 * sizeof(half2))
                + mma_tid_y * 128 + mma_tid_x * 2);
        const half2 *output_lds_ptr = reinterpret_cast<half2 *>(
            smem + warp_id * 512 * sizeof(half2)) + lane_id;

        const int n_base = blockIdx.y * 128 + (warp_id / 2) * 32;
        const int m_base = blockIdx.x * 128 + (warp_id % 2) * 64 + lane_id * 2;
        const int safe_n_base = (n_base < param.N) ? n_base : 0;
        half *output_stg_ptr = outputs
            + (size_t)blockIdx.z * param.outBatchNumel
            + (size_t)safe_n_base * param.M + m_base;

        const uint32_t n_guard = (n_base < param.N) ? param.N - n_base : 0;
        const uint32_t m_guard = (m_base < param.M) ? param.M - m_base : 0;

        __syncthreads();
#pragma unroll
        for (int p = 0; p < 4; p++)
        {
            ptx::sts64(
                output[p][0],
                output[p][1],
                output_sts_addr + p * 32 * sizeof(half2));
            ptx::sts64(
                output[p][2],
                output[p][3],
                output_sts_addr + (p * 32 + 16) * sizeof(half2));
        }
        __syncthreads();

#pragma unroll
        for (int p = 0; p < 16; p++)
        {
            half2 x_out = output_lds_ptr[p * 32];
            ptx::stg16(
                x_out.x,
                output_stg_ptr + (group * 16 + p) * param.M,
                group * 16 + p < n_guard && 0 < m_guard);
            ptx::stg16(
                x_out.y,
                output_stg_ptr + (group * 16 + p) * param.M + 1,
                group * 16 + p < n_guard && 1 < m_guard);
        }
    }

}
__global__ void conv2d_128x128x8_S_FP16_kernel(
    const uint16_t *inputs, // [T, C, H, W], 0x0000 or non-zero such as 0xFFFF
    const half2 *weights,
    const half *bias,
    half *outputs,
    Conv2DK128Param param)
{
    __shared__ __align__(16 * 1024) char smem[16 * 1024];
    half2 *smemweight = reinterpret_cast<half2 *>(smem);
    uint16_t *smeminput = reinterpret_cast<uint16_t *>(smem + 8 * 1024);

    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int mma_tid_x = (lane_id / 2) % 8;
    const int mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);
    const int K_pair = (param.K + 1) / 2;
    const int K_pair_padded = (K_pair + 7) / 8 * 8;

    uint16_t input_frag[2][8][2];
    half2 weight_frag[2][8];
    half2 output_frag[8][8];

#pragma unroll
    for (int i = 0; i < 8; i++)
    {
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            output_frag[i][j] = __float2half2_rn(0.0f);
        }
    }

    int posh_ori[4];
    int posw_ori[4];
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        int m_idx = blockIdx.x * 128 + lane_id + i * 32;
        posh_ori[i] = (m_idx / param.Ow) * param.U - param.P;
        posw_ori[i] = (m_idx % param.Ow) * param.V - param.Q;
    }

    const int k_iters = (K_pair + 7) / 8;

    auto load_chunk = [&](int k_tile, int buf)
    {
        const int k_base = k_tile * 8;
        const char *input_base = reinterpret_cast<const char *>(
            inputs + (size_t)blockIdx.z * param.inBatchNumel);
        const int cur_pair = k_base + warp_id;
        const int k0 = cur_pair * 2;
        const int k1 = k0 + 1;
        const int curC0 = k0 / (param.R * param.S);
        const int rs0 = k0 % (param.R * param.S);
        const int curR0 = rs0 / param.S;
        const int curS0 = rs0 % param.S;
        const int curC1 = k1 / (param.R * param.S);
        const int rs1 = k1 % (param.R * param.S);
        const int curR1 = rs1 / param.S;
        const int curS1 = rs1 % param.S;

        uint32_t input_sts_addr = ptx::smem_u32addr(
            smeminput + buf * 2048 + warp_id * 256 + lane_id * 2);
        uint32_t weight_sts_addr = ptx::smem_u32addr(
            smemweight + buf * 1024 + warp_id * 128 + lane_id * 4);

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            uint16_t input0 = 0;
            uint16_t input1 = 0;

            int curH0 = posh_ori[i] + curR0;
            int curW0 = posw_ori[i] + curS0;
            int inOffset0 = curC0 * param.H * param.W + curH0 * param.W + curW0;
            bool guard0 = k0 < param.K
                && curH0 >= 0 && curW0 >= 0
                && curH0 < param.H && curW0 < param.W;
            ptx::ldg16_nc_0(input0,
                input_base + inOffset0 * sizeof(uint16_t), guard0);

            int curH1 = posh_ori[i] + curR1;
            int curW1 = posw_ori[i] + curS1;
            int inOffset1 = curC1 * param.H * param.W + curH1 * param.W + curW1;
            bool guard1 = k1 < param.K
                && curH1 >= 0 && curW1 >= 0
                && curH1 < param.H && curW1 < param.W;
            ptx::ldg16_nc_0(input1,
                input_base + inOffset1 * sizeof(uint16_t), guard1);

            uint32_t packed = static_cast<uint32_t>(input0)
                | (static_cast<uint32_t>(input1) << 16);
            ptx::sts32(packed, input_sts_addr + i * 32 * sizeof(uint32_t));
        }

        const int weight_n = blockIdx.y * 128 + lane_id * 4;
        const half2 *weight_src = weights
            + (size_t)cur_pair * param.N_padded + weight_n;
        int weight_bytes = (cur_pair < K_pair_padded) ? 16 : 0;
        asm volatile ("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::
            "r"(weight_sts_addr),
            "l"(weight_src),
            "r"(weight_bytes));
    };

    auto compute_chunk = [&](int buf)
    {
        uint32_t input_lds_addr = ptx::smem_u32addr(
            smeminput + buf * 2048 + (warp_id % 2) * 128 + mma_tid_x * 8);
        uint32_t weight_lds_addr = ptx::smem_u32addr(
            smemweight + buf * 1024 + (warp_id / 2) * 32 + mma_tid_y * 4);

#pragma unroll
        for (int k_frag = 0; k_frag < 8; k_frag++)
        {
            ptx::lds128(
                weight_frag[k_frag % 2][0],
                weight_frag[k_frag % 2][1],
                weight_frag[k_frag % 2][2],
                weight_frag[k_frag % 2][3],
                weight_lds_addr + k_frag * 128 * sizeof(half2));
            ptx::lds128(
                weight_frag[k_frag % 2][4],
                weight_frag[k_frag % 2][5],
                weight_frag[k_frag % 2][6],
                weight_frag[k_frag % 2][7],
                weight_lds_addr + (k_frag * 128 + 16) * sizeof(half2));

#pragma unroll
            for (int j = 0; j < 4; j++)
            {
                uint32_t packed;
                ptx::lds32(
                    packed,
                    input_lds_addr + (k_frag * 256 + j * 2) * sizeof(uint16_t));
                input_frag[k_frag % 2][j][0] = static_cast<uint16_t>(packed & 0xFFFF);
                input_frag[k_frag % 2][j][1] = static_cast<uint16_t>(packed >> 16);
            }
#pragma unroll
            for (int j = 4; j < 8; j++)
            {
                uint32_t packed;
                ptx::lds32(
                    packed,
                    input_lds_addr + (k_frag * 256 + 64 + (j - 4) * 2) * sizeof(uint16_t));
                input_frag[k_frag % 2][j][0] = static_cast<uint16_t>(packed & 0xFFFF);
                input_frag[k_frag % 2][j][1] = static_cast<uint16_t>(packed >> 16);
            }

#pragma unroll
            for (int j = 0; j < 8; j++)
            {
                const bool input0 = input_frag[k_frag % 2][j][0] != 0;
                const bool input1 = input_frag[k_frag % 2][j][1] != 0;
#pragma unroll
                for (int i = 0; i < 8; i++)
                {
                    half2 add_value = __float2half2_rn(0.0f);
                    if (input0)
                    {
                        add_value.x = weight_frag[k_frag % 2][i].x;
                    }
                    if (input1)
                    {
                        add_value.y = weight_frag[k_frag % 2][i].y;
                    }
                    output_frag[i][j] = __hadd2(output_frag[i][j], add_value);
                }
            }
        }
    };

    if (k_iters > 0)
    {
        load_chunk(0, 0);
        asm volatile ("cp.async.commit_group;\n" :::);
        asm volatile ("cp.async.wait_group 0;\n" :::);
        __syncthreads();
    }

    for (int kt = 0; kt < k_iters; kt++)
    {
        const int cur_buf = kt % 2;
        const int next_buf = (kt + 1) % 2;
        if (kt + 1 < k_iters)
        {
            load_chunk(kt + 1, next_buf);
            asm volatile ("cp.async.commit_group;\n" :::);
        }

        compute_chunk(cur_buf);

        if (kt + 1 < k_iters)
        {
            asm volatile ("cp.async.wait_group 0;\n" :::);
        }
        __syncthreads();
    }

    half2 output[4][4];

#pragma unroll
    for (int group = 0; group < 2; group++)
    {
#pragma unroll
        for (int p = 0; p < 4; p++)
        {
            const int n_global = blockIdx.y * 128
                + (warp_id / 2) * 32
                + mma_tid_y * 4 + group * 16 + p;
            const float bias_value = (n_global < param.N)
                ? __half2float(bias[n_global]) : 0.0f;
#pragma unroll
            for (int q = 0; q < 4; q++)
            {
                const int n_frag = group * 4 + p;
                const int m_frag = q * 2;
                output[p][q].x = __float2half(
                    __half2float(output_frag[n_frag][m_frag].x)
                    + __half2float(output_frag[n_frag][m_frag].y)
                    + bias_value);
                output[p][q].y = __float2half(
                    __half2float(output_frag[n_frag][m_frag + 1].x)
                    + __half2float(output_frag[n_frag][m_frag + 1].y)
                    + bias_value);
            }
        }

        uint32_t output_sts_addr = ptx::smem_u32addr(
            reinterpret_cast<half2 *>(smem + warp_id * 512 * sizeof(half2))
                + mma_tid_y * 128 + mma_tid_x * 2);
        const half2 *output_lds_ptr = reinterpret_cast<half2 *>(
            smem + warp_id * 512 * sizeof(half2)) + lane_id;

        const int n_base = blockIdx.y * 128 + (warp_id / 2) * 32;
        const int m_base = blockIdx.x * 128 + (warp_id % 2) * 64 + lane_id * 2;
        const int safe_n_base = (n_base < param.N) ? n_base : 0;
        half *output_stg_ptr = outputs
            + (size_t)blockIdx.z * param.outBatchNumel
            + (size_t)safe_n_base * param.M + m_base;

        const uint32_t n_guard = (n_base < param.N) ? param.N - n_base : 0;
        const uint32_t m_guard = (m_base < param.M) ? param.M - m_base : 0;

        __syncthreads();
#pragma unroll
        for (int p = 0; p < 4; p++)
        {
            ptx::sts64(
                output[p][0],
                output[p][1],
                output_sts_addr + p * 32 * sizeof(half2));
            ptx::sts64(
                output[p][2],
                output[p][3],
                output_sts_addr + (p * 32 + 16) * sizeof(half2));
        }
        __syncthreads();

#pragma unroll
        for (int p = 0; p < 16; p++)
        {
            half2 x_out = output_lds_ptr[p * 32];
            ptx::stg16(
                x_out.x,
                output_stg_ptr + (group * 16 + p) * param.M,
                group * 16 + p < n_guard && 0 < m_guard);
            ptx::stg16(
                x_out.y,
                output_stg_ptr + (group * 16 + p) * param.M + 1,
                group * 16 + p < n_guard && 1 < m_guard);
        }
    }
}

// -----------------------------------------------------------------------------
// CPU Reference & Test
// -----------------------------------------------------------------------------
void direct_conv2d_cpu(
    const float *input, const float *filter, const float *bias, float *output,
    int N, int C, int H, int W, int K, int R, int S, int U, int V, int P, int Q)
{
    int Oh = (H + 2 * P - R) / U + 1;
    int Ow = (W + 2 * Q - S) / V + 1;
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            for (int oh = 0; oh < Oh; oh++) {
                for (int ow = 0; ow < Ow; ow++) {
                    float sum = bias[k];
                    for (int c = 0; c < C; c++) {
                        for (int r = 0; r < R; r++) {
                            for (int s = 0; s < S; s++) {
                                int ih = oh * U - P + r;
                                int iw = ow * V - Q + s;
                                if (iw >= 0 && ih >= 0 && iw < W && ih < H) {
                                    sum += input[(size_t)n * C * H * W + (size_t)c * H * W + ih * W + iw] *
                                           filter[(size_t)k * C * R * S + (size_t)c * R * S + r * S + s];
                                }
                            }
                        }
                    }
                    output[(size_t)n * K * Oh * Ow + (size_t)k * Oh * Ow + oh * Ow + ow] = sum;
                }
            }
        }
    }
}

void test_conv2d_fp32(int N, int C, int H, int W, int K, int R, int S, int U, int V, int P, int Q)
{
    int Oh = (H + 2 * P - R) / U + 1;
    int Ow = (W + 2 * Q - S) / V + 1;
    int M = Oh * Ow;
    int InK = C * R * S;
    int N_padded = (K + 127) / 128 * 128;
    int K_padded = (InK + 7) / 8 * 8;
    std::cout << "Testing Conv2D FP32: N=" << N << " C=" << C << " H=" << H << " W=" << W << " K=" << K << std::endl;
    std::vector<float> h_in(N * C * H * W), h_wt_orig(K * InK);
    std::vector<float> h_wt_nk(N_padded * K_padded, 0.0f);
    std::vector<float> h_wt_kn(N_padded * K_padded, 0.0f);
    std::vector<float> h_bias(K), h_out(N * K * M, 0.0f);
    std::vector<float> h_out_v1(N * K * M, 0.0f);
    std::vector<float> h_out_v2(N * K * M, 0.0f);
    std::vector<float> h_out_v3(N * K * M, 0.0f), h_ref(N * K * M);
    srand(42);
    for (auto &x : h_in) x = (float)rand() / RAND_MAX;
    for (auto &x : h_wt_orig) x = (float)rand() / RAND_MAX;
    for (auto &x : h_bias) x = (float)rand() / RAND_MAX;
    for (int k = 0; k < K; k++) for (int i = 0; i < InK; i++) h_wt_nk[(size_t)k * K_padded + i] = h_wt_orig[(size_t)k * InK + i];
    for (int i = 0; i < InK; i++) for (int k = 0; k < K; k++) h_wt_kn[(size_t)i * N_padded + k] = h_wt_orig[(size_t)k * InK + i];
    float *d_in, *d_wt_nk, *d_wt_kn, *d_bias, *d_out, *d_out_v1, *d_out_v2, *d_out_v3;
    cudaMalloc(&d_in, h_in.size() * sizeof(float)); cudaMalloc(&d_wt_nk, h_wt_nk.size() * sizeof(float));
    cudaMalloc(&d_wt_kn, h_wt_kn.size() * sizeof(float));
    cudaMalloc(&d_bias, h_bias.size() * sizeof(float)); cudaMalloc(&d_out, h_out.size() * sizeof(float));
    cudaMalloc(&d_out_v1, h_out_v1.size() * sizeof(float));
    cudaMalloc(&d_out_v2, h_out_v2.size() * sizeof(float));
    cudaMalloc(&d_out_v3, h_out_v3.size() * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wt_nk, h_wt_nk.data(), h_wt_nk.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wt_kn, h_wt_kn.data(), h_wt_kn.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(float), cudaMemcpyHostToDevice);
    Conv2DK128Param param = {(uint32_t)M, (uint32_t)InK, (uint32_t)K, (uint32_t)N_padded, (uint32_t)C, (uint32_t)H, (uint32_t)W, (uint32_t)Oh, (uint32_t)Ow, (uint32_t)R, (uint32_t)S, (uint32_t)U, (uint32_t)V, (uint32_t)P, (uint32_t)Q, (uint32_t)(C * H * W), (uint32_t)(K * M)};
    dim3 grid((K + 127) / 128, (M + 127) / 128, N);
    dim3 grid_v2((M + 127) / 128, (K + 127) / 128, N);
    dim3 block(256);
    conv2d_128x128x8_kernel<<<grid, block>>>(d_in, d_wt_nk, d_bias, d_out, param);
    conv2d_128x128x8_kernel_v1<<<grid, block>>>(d_in, d_wt_kn, d_bias, d_out_v1, param);
    conv2d_128x128x8_kernel_v2<<<grid_v2, block>>>(d_in, d_wt_kn, d_bias, d_out_v2, param);
    conv2d_128x128x8_kernel_v3<<<grid_v2, block>>>(d_in, d_wt_kn, d_bias, d_out_v3, param);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_v1.data(), d_out_v1, h_out_v1.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_v2.data(), d_out_v2, h_out_v2.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_v3.data(), d_out_v3, h_out_v3.size() * sizeof(float), cudaMemcpyDeviceToHost);
    direct_conv2d_cpu(h_in.data(), h_wt_orig.data(), h_bias.data(), h_ref.data(), N, C, H, W, K, R, S, U, V, P, Q);
    int errs = 0;
    int errs_v1 = 0;
    int errs_v2 = 0;
    int errs_v3 = 0;
    for (int n = 0; n < N; n++) for (int k = 0; k < K; k++) for (int m = 0; m < M; m++) {
        if (std::abs(h_out[n * K * M + k * M + m] - h_ref[n * K * M + k * M + m]) > 1e-3) errs++;
        if (std::abs(h_out_v1[n * K * M + k * M + m] - h_ref[n * K * M + k * M + m]) > 1e-3) errs_v1++;
        if (std::abs(h_out_v2[n * K * M + k * M + m] - h_ref[n * K * M + k * M + m]) > 1e-3) errs_v2++;
        if (std::abs(h_out_v3[n * K * M + k * M + m] - h_ref[n * K * M + k * M + m]) > 1e-3) errs_v3++;
    }
    std::cout << "  base " << (errs == 0 ? "PASSED" : "FAILED") << " (" << errs << " errors)" << std::endl;
    std::cout << "  v1   " << (errs_v1 == 0 ? "PASSED" : "FAILED") << " (" << errs_v1 << " errors)" << std::endl;
    std::cout << "  v2   " << (errs_v2 == 0 ? "PASSED" : "FAILED") << " (" << errs_v2 << " errors)" << std::endl;
    std::cout << "  v3   " << (errs_v3 == 0 ? "PASSED" : "FAILED") << " (" << errs_v3 << " errors)" << std::endl;

    const int warmup_iters = 20;
    const int benchmark_iters = 200;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warmup_iters; i++) {
        conv2d_128x128x8_kernel<<<grid, block>>>(d_in, d_wt_nk, d_bias, d_out, param);
    }
    cudaEventRecord(start);
    for (int i = 0; i < benchmark_iters; i++) {
        conv2d_128x128x8_kernel<<<grid, block>>>(d_in, d_wt_nk, d_bias, d_out, param);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float base_ms = 0.0f;
    cudaEventElapsedTime(&base_ms, start, stop);
    base_ms /= benchmark_iters;

    for (int i = 0; i < warmup_iters; i++) {
        conv2d_128x128x8_kernel_v1<<<grid, block>>>(d_in, d_wt_kn, d_bias, d_out_v1, param);
    }
    cudaEventRecord(start);
    for (int i = 0; i < benchmark_iters; i++) {
        conv2d_128x128x8_kernel_v1<<<grid, block>>>(d_in, d_wt_kn, d_bias, d_out_v1, param);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float v1_ms = 0.0f;
    cudaEventElapsedTime(&v1_ms, start, stop);
    v1_ms /= benchmark_iters;

    for (int i = 0; i < warmup_iters; i++) {
        conv2d_128x128x8_kernel_v2<<<grid_v2, block>>>(d_in, d_wt_kn, d_bias, d_out_v2, param);
    }
    cudaEventRecord(start);
    for (int i = 0; i < benchmark_iters; i++) {
        conv2d_128x128x8_kernel_v2<<<grid_v2, block>>>(d_in, d_wt_kn, d_bias, d_out_v2, param);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float v2_ms = 0.0f;
    cudaEventElapsedTime(&v2_ms, start, stop);
    v2_ms /= benchmark_iters;

    for (int i = 0; i < warmup_iters; i++) {
        conv2d_128x128x8_kernel_v3<<<grid_v2, block>>>(d_in, d_wt_kn, d_bias, d_out_v3, param);
    }
    cudaEventRecord(start);
    for (int i = 0; i < benchmark_iters; i++) {
        conv2d_128x128x8_kernel_v3<<<grid_v2, block>>>(d_in, d_wt_kn, d_bias, d_out_v3, param);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float v3_ms = 0.0f;
    cudaEventElapsedTime(&v3_ms, start, stop);
    v3_ms /= benchmark_iters;

    std::cout << "  base avg: " << base_ms << " ms" << std::endl;
    std::cout << "  v1   avg: " << v1_ms << " ms";
    std::cout << "  speedup: " << (base_ms / v1_ms) << "x" << std::endl;
    std::cout << "  v2   avg: " << v2_ms << " ms";
    std::cout << "  speedup: " << (base_ms / v2_ms) << "x" << std::endl;
    std::cout << "  v3   avg: " << v3_ms << " ms";
    std::cout << "  speedup: " << (base_ms / v3_ms) << "x" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in); cudaFree(d_wt_nk); cudaFree(d_wt_kn); cudaFree(d_bias); cudaFree(d_out); cudaFree(d_out_v1); cudaFree(d_out_v2); cudaFree(d_out_v3);
}

void test_conv2d_snn_fp32(int T, int C, int H, int W, int K, int R, int S, int U, int V, int P, int Q)
{
    int Oh = (H + 2 * P - R) / U + 1;
    int Ow = (W + 2 * Q - S) / V + 1;
    int M = Oh * Ow;
    int InK = C * R * S;
    int N_padded = (K + 127) / 128 * 128;
    int K_padded = (InK + 7) / 8 * 8;
    std::cout << "Testing SNN Conv2D FP32: T=" << T << " C=" << C << " H=" << H << " W=" << W << " K=" << K << std::endl;
    std::vector<uint32_t> h_in((size_t)T * C * H * W);
    std::vector<float> h_wt_orig(K * InK), h_wt_padded(N_padded * K_padded, 0.0f), h_bias(K), h_out(T * K * M, 0.0f), h_ref(T * K * M);
    srand(42);
    for (auto &x : h_in) x = (rand() & 1) ? 1u : 0u;
    for (auto &x : h_wt_orig) x = (float)rand() / RAND_MAX;
    for (auto &x : h_bias) x = (float)rand() / RAND_MAX;
    for (int i = 0; i < InK; i++) for (int k = 0; k < K; k++) h_wt_padded[(size_t)i * N_padded + k] = h_wt_orig[(size_t)k * InK + i];
    uint32_t *d_in; float *d_wt, *d_bias, *d_out;
    cudaMalloc(&d_in, h_in.size() * sizeof(uint32_t)); cudaMalloc(&d_wt, h_wt_padded.size() * sizeof(float));
    cudaMalloc(&d_bias, h_bias.size() * sizeof(float)); cudaMalloc(&d_out, h_out.size() * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wt, h_wt_padded.data(), h_wt_padded.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(float), cudaMemcpyHostToDevice);
    Conv2DK128Param param = {(uint32_t)M, (uint32_t)InK, (uint32_t)K, (uint32_t)N_padded, (uint32_t)C, (uint32_t)H, (uint32_t)W, (uint32_t)Oh, (uint32_t)Ow, (uint32_t)R, (uint32_t)S, (uint32_t)U, (uint32_t)V, (uint32_t)P, (uint32_t)Q, (uint32_t)(C * H * W), (uint32_t)(K * M)};
    conv2d_128x128x8_S_kernel<<<dim3((M + 127) / 128, (K + 127) / 128, T), 256>>>(d_in, d_wt, d_bias, d_out, param);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost);
    for (int t = 0; t < T; t++) {
        for (int k = 0; k < K; k++) {
            for (int oh = 0; oh < Oh; oh++) {
                for (int ow = 0; ow < Ow; ow++) {
                    float sum = h_bias[k];
                    for (int c = 0; c < C; c++) {
                        for (int r = 0; r < R; r++) {
                            for (int s = 0; s < S; s++) {
                                int ih = oh * U - P + r;
                                int iw = ow * V - Q + s;
                                if (iw >= 0 && ih >= 0 && iw < W && ih < H) {
                                    if (h_in[(size_t)t * C * H * W + c * H * W + ih * W + iw])
                                        sum += h_wt_orig[k * InK + c * R * S + r * S + s];
                                }
                            }
                        }
                    }
                    h_ref[t * K * M + k * M + oh * Ow + ow] = sum;
                }
            }
        }
    }
    int errs = 0;
    for (size_t i = 0; i < h_out.size(); i++) if (std::abs(h_out[i] - h_ref[i]) > 1e-3) errs++;
    std::cout << (errs == 0 ? "  PASSED!" : "  FAILED") << " (" << errs << " errors)" << std::endl;
    cudaFree(d_in); cudaFree(d_wt); cudaFree(d_bias); cudaFree(d_out);
}

void test_conv2d_fp16(int N, int C, int H, int W, int K, int R, int S, int U, int V, int P, int Q)
{
    int Oh = (H + 2 * P - R) / U + 1;
    int Ow = (W + 2 * Q - S) / V + 1;
    int M = Oh * Ow;
    int InK = C * R * S;
    int N_padded = (K + 127) / 128 * 128;
    int K_pair = (InK + 1) / 2;
    int K_pair_padded = (K_pair + 7) / 8 * 8;
    std::cout << "Testing Conv2D FP16: N=" << N << " C=" << C << " H=" << H << " W=" << W << " K=" << K << std::endl;
    std::vector<half> h_in(N * C * H * W), h_wt_orig(K * InK), h_bias(K), h_out(N * K * M), h_ref(N * K * M);
    std::vector<half> h_out_v1(N * K * M);
    std::vector<half2> h_wt_padded(N_padded * K_pair_padded, __float2half2_rn(0.0f));
    srand(42);
    for (auto &x : h_in) x = __float2half((float)rand() / RAND_MAX);
    for (auto &x : h_wt_orig) x = __float2half((float)rand() / RAND_MAX);
    for (auto &x : h_bias) x = __float2half((float)rand() / RAND_MAX);
    for (int pair = 0; pair < K_pair; pair++) for (int k = 0; k < K; k++) {
        half2 v = __float2half2_rn(0.0f);
        int k0 = pair * 2;
        int k1 = k0 + 1;
        v.x = h_wt_orig[(size_t)k * InK + k0];
        if (k1 < InK) v.y = h_wt_orig[(size_t)k * InK + k1];
        h_wt_padded[(size_t)pair * N_padded + k] = v;
    }
    half *d_in, *d_bias, *d_out, *d_out_v1;
    half2 *d_wt;
    cudaMalloc(&d_in, h_in.size() * sizeof(half)); cudaMalloc(&d_wt, h_wt_padded.size() * sizeof(half2));
    cudaMalloc(&d_bias, h_bias.size() * sizeof(half)); cudaMalloc(&d_out, h_out.size() * sizeof(half));
    cudaMalloc(&d_out_v1, h_out_v1.size() * sizeof(half));
    cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wt, h_wt_padded.data(), h_wt_padded.size() * sizeof(half2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(half), cudaMemcpyHostToDevice);
    Conv2DK128Param param = {(uint32_t)M, (uint32_t)InK, (uint32_t)K, (uint32_t)N_padded, (uint32_t)C, (uint32_t)H, (uint32_t)W, (uint32_t)Oh, (uint32_t)Ow, (uint32_t)R, (uint32_t)S, (uint32_t)U, (uint32_t)V, (uint32_t)P, (uint32_t)Q, (uint32_t)(C * H * W), (uint32_t)(K * M)};
    dim3 grid((M + 127) / 128, (K + 127) / 128, N);
    dim3 block(256);
    conv2d_128x128x8_FP16_kernel<<<grid, block>>>(d_in, d_wt, d_bias, d_out, param);
    conv2d_128x128x8_FP16_kernel_v1<<<grid, block>>>(d_in, d_wt, d_bias, d_out_v1, param);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_v1.data(), d_out_v1, h_out_v1.size() * sizeof(half), cudaMemcpyDeviceToHost);
    for (int n = 0; n < N; n++) for (int k = 0; k < K; k++) for (int oh = 0; oh < Oh; oh++) for (int ow = 0; ow < Ow; ow++) {
        half even_acc = __float2half(0.0f);
        half odd_acc = __float2half(0.0f);
        for (int pair = 0; pair < K_pair; pair++) {
            int i0 = pair * 2;
            int i1 = i0 + 1;
            int c0 = i0 / (R * S);
            int rs0 = i0 % (R * S);
            int r0 = rs0 / S;
            int s0 = rs0 % S;
            int ih0 = oh * U - P + r0;
            int iw0 = ow * V - Q + s0;
            if (i0 < InK && iw0 >= 0 && ih0 >= 0 && iw0 < W && ih0 < H) {
                float prod = __half2float(h_in[n * C * H * W + c0 * H * W + ih0 * W + iw0])
                    * __half2float(h_wt_orig[k * InK + i0]);
                even_acc = __float2half(__half2float(even_acc) + prod);
            }

            int c1 = i1 / (R * S);
            int rs1 = i1 % (R * S);
            int r1 = rs1 / S;
            int s1 = rs1 % S;
            int ih1 = oh * U - P + r1;
            int iw1 = ow * V - Q + s1;
            if (i1 < InK && iw1 >= 0 && ih1 >= 0 && iw1 < W && ih1 < H) {
                float prod = __half2float(h_in[n * C * H * W + c1 * H * W + ih1 * W + iw1])
                    * __half2float(h_wt_orig[k * InK + i1]);
                odd_acc = __float2half(__half2float(odd_acc) + prod);
            }
        }
        float sum = __half2float(even_acc) + __half2float(odd_acc) + __half2float(h_bias[k]);
        h_ref[n * K * M + k * M + oh * Ow + ow] = __float2half(sum);
    }
    int errs = 0;
    int errs_v1 = 0;
    float max_err = 0.0f;
    float max_err_v1 = 0.0f;
    const float fp16_tol = 1.5e-1f;
    for (size_t i = 0; i < h_out.size(); i++) {
        float err = std::abs(__half2float(h_out[i]) - __half2float(h_ref[i]));
        if (err > max_err) max_err = err;
        if (err > fp16_tol) errs++;
        float err_v1 = std::abs(__half2float(h_out_v1[i]) - __half2float(h_ref[i]));
        if (err_v1 > max_err_v1) max_err_v1 = err_v1;
        if (err_v1 > fp16_tol) errs_v1++;
    }
    std::cout << "  base " << (errs == 0 ? "PASSED" : "FAILED") << " (" << errs << " errors, max_err=" << max_err << ")" << std::endl;
    std::cout << "  v1   " << (errs_v1 == 0 ? "PASSED" : "FAILED") << " (" << errs_v1 << " errors, max_err=" << max_err_v1 << ")" << std::endl;

    const int warmup_iters = 20;
    const int benchmark_iters = 200;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warmup_iters; i++) {
        conv2d_128x128x8_FP16_kernel<<<grid, block>>>(d_in, d_wt, d_bias, d_out, param);
    }
    cudaEventRecord(start);
    for (int i = 0; i < benchmark_iters; i++) {
        conv2d_128x128x8_FP16_kernel<<<grid, block>>>(d_in, d_wt, d_bias, d_out, param);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float base_ms = 0.0f;
    cudaEventElapsedTime(&base_ms, start, stop);
    base_ms /= benchmark_iters;

    for (int i = 0; i < warmup_iters; i++) {
        conv2d_128x128x8_FP16_kernel_v1<<<grid, block>>>(d_in, d_wt, d_bias, d_out_v1, param);
    }
    cudaEventRecord(start);
    for (int i = 0; i < benchmark_iters; i++) {
        conv2d_128x128x8_FP16_kernel_v1<<<grid, block>>>(d_in, d_wt, d_bias, d_out_v1, param);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float v1_ms = 0.0f;
    cudaEventElapsedTime(&v1_ms, start, stop);
    v1_ms /= benchmark_iters;

    std::cout << "  base avg: " << base_ms << " ms" << std::endl;
    std::cout << "  v1   avg: " << v1_ms << " ms";
    std::cout << "  speedup: " << (base_ms / v1_ms) << "x" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in); cudaFree(d_wt); cudaFree(d_bias); cudaFree(d_out); cudaFree(d_out_v1);
}

void test_conv2d_snn_fp16(int T, int C, int H, int W, int K, int R, int S, int U, int V, int P, int Q)
{
    int Oh = (H + 2 * P - R) / U + 1;
    int Ow = (W + 2 * Q - S) / V + 1;
    int M = Oh * Ow;
    int InK = C * R * S;
    int N_padded = (K + 127) / 128 * 128;
    int K_pair = (InK + 1) / 2;
    int K_pair_padded = (K_pair + 7) / 8 * 8;
    std::cout << "Testing SNN Conv2D FP16: T=" << T << " C=" << C << " H=" << H << " W=" << W << " K=" << K << std::endl;
    std::vector<uint16_t> h_in((size_t)T * C * H * W);
    std::vector<half> h_wt_orig(K * InK), h_bias(K), h_out(T * K * M), h_ref(T * K * M);
    std::vector<half2> h_wt_padded(N_padded * K_pair_padded, __float2half2_rn(0.0f));
    srand(42);
    for (auto &x : h_in) x = (rand() & 1) ? 0xFFFFu : 0u;
    for (auto &x : h_wt_orig) x = __float2half((float)rand() / RAND_MAX);
    for (auto &x : h_bias) x = __float2half((float)rand() / RAND_MAX);
    for (int pair = 0; pair < K_pair; pair++) for (int k = 0; k < K; k++) {
        half2 v = __float2half2_rn(0.0f);
        int k0 = pair * 2;
        int k1 = k0 + 1;
        v.x = h_wt_orig[(size_t)k * InK + k0];
        if (k1 < InK) v.y = h_wt_orig[(size_t)k * InK + k1];
        h_wt_padded[(size_t)pair * N_padded + k] = v;
    }
    uint16_t *d_in;
    half2 *d_wt;
    half *d_bias, *d_out;
    cudaMalloc(&d_in, h_in.size() * sizeof(uint16_t));
    cudaMalloc(&d_wt, h_wt_padded.size() * sizeof(half2));
    cudaMalloc(&d_bias, h_bias.size() * sizeof(half));
    cudaMalloc(&d_out, h_out.size() * sizeof(half));
    cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wt, h_wt_padded.data(), h_wt_padded.size() * sizeof(half2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(half), cudaMemcpyHostToDevice);
    Conv2DK128Param param = {(uint32_t)M, (uint32_t)InK, (uint32_t)K, (uint32_t)N_padded, (uint32_t)C, (uint32_t)H, (uint32_t)W, (uint32_t)Oh, (uint32_t)Ow, (uint32_t)R, (uint32_t)S, (uint32_t)U, (uint32_t)V, (uint32_t)P, (uint32_t)Q, (uint32_t)(C * H * W), (uint32_t)(K * M)};
    conv2d_128x128x8_S_FP16_kernel<<<dim3((M + 127) / 128, (K + 127) / 128, T), 256>>>(d_in, d_wt, d_bias, d_out, param);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(half), cudaMemcpyDeviceToHost);
    for (int t = 0; t < T; t++) for (int k = 0; k < K; k++) for (int oh = 0; oh < Oh; oh++) for (int ow = 0; ow < Ow; ow++) {
        half even_acc = __float2half(0.0f);
        half odd_acc = __float2half(0.0f);
        for (int pair = 0; pair < K_pair; pair++) {
            int i0 = pair * 2;
            int i1 = i0 + 1;
            int c0 = i0 / (R * S);
            int rs0 = i0 % (R * S);
            int r0 = rs0 / S;
            int s0 = rs0 % S;
            int ih0 = oh * U - P + r0;
            int iw0 = ow * V - Q + s0;
            if (i0 < InK && iw0 >= 0 && ih0 >= 0 && iw0 < W && ih0 < H) {
                if (h_in[(size_t)t * C * H * W + c0 * H * W + ih0 * W + iw0]) {
                    even_acc = __float2half(__half2float(even_acc)
                        + __half2float(h_wt_orig[k * InK + i0]));
                }
            }

            int c1 = i1 / (R * S);
            int rs1 = i1 % (R * S);
            int r1 = rs1 / S;
            int s1 = rs1 % S;
            int ih1 = oh * U - P + r1;
            int iw1 = ow * V - Q + s1;
            if (i1 < InK && iw1 >= 0 && ih1 >= 0 && iw1 < W && ih1 < H) {
                if (h_in[(size_t)t * C * H * W + c1 * H * W + ih1 * W + iw1]) {
                    odd_acc = __float2half(__half2float(odd_acc)
                        + __half2float(h_wt_orig[k * InK + i1]));
                }
            }
        }
        float sum = __half2float(even_acc) + __half2float(odd_acc) + __half2float(h_bias[k]);
        h_ref[t * K * M + k * M + oh * Ow + ow] = __float2half(sum);
    }
    int errs = 0;
    float max_err = 0.0f;
    const float fp16_tol = 1.5e-1f;
    for (size_t i = 0; i < h_out.size(); i++) {
        float err = std::abs(__half2float(h_out[i]) - __half2float(h_ref[i]));
        if (err > max_err) max_err = err;
        if (err > fp16_tol) errs++;
    }
    std::cout << (errs == 0 ? "  PASSED!" : "  FAILED") << " (" << errs << " errors, max_err=" << max_err << ")" << std::endl;
    cudaFree(d_in); cudaFree(d_wt); cudaFree(d_bias); cudaFree(d_out);
}

int main()
{
    test_conv2d_fp32(2, 64, 32, 32, 64, 3, 3, 1, 1, 1, 1);
    test_conv2d_fp32(1, 128, 40, 40, 128, 1, 1, 1, 1, 0, 0);
    test_conv2d_snn_fp32(1, 64, 32, 32, 64, 3, 3, 1, 1, 1, 1);
    test_conv2d_snn_fp32(8, 64, 32, 32, 64, 1, 1, 1, 1, 0, 0);
    test_conv2d_fp16(2, 64, 32, 32, 64, 3, 3, 1, 1, 1, 1);
    test_conv2d_fp16(1, 128, 40, 40, 128, 1, 1, 1, 1, 0, 0);
    test_conv2d_snn_fp16(1, 64, 32, 32, 64, 3, 3, 1, 1, 1, 1);
    test_conv2d_snn_fp16(8, 64, 32, 32, 64, 1, 1, 1, 1, 0, 0);
    return 0;
}
