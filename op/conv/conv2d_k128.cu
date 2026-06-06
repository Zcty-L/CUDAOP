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
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const int m_global = blockIdx.y * 128 + thread_m_base + i;
        if (m_global < param.M) {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int n_global = blockIdx.x * 128 + thread_n_base + j;
                if (n_global < param.N)
                    outputs[(size_t)blockIdx.z * param.outBatchNumel + (size_t)n_global * param.M + m_global] = C_frag[i][j];
            }
        }
    }
}

// -----------------------------------------------------------------------------
// SNN FP32 Optimized Kernel (Double Buffer, Packed uint8 spikes)
// -----------------------------------------------------------------------------
__global__ void conv2d_128x128x8_S_kernel(
    const uint8_t *inputs, // [Batch, C, H, W] packed spikes
    const float *weights,
    const float *bias,
    float *outputs,
    Conv2DK128Param param)
{
    __shared__ __align__(16 * 1024) char smem[16 * 1024];
    uint8_t *A_smem_base = reinterpret_cast<uint8_t *>(smem);
    float   *B_smem_base = reinterpret_cast<float *>(smem + 4 * 1024);
    float   *smembias = reinterpret_cast<float *>(smem);

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
    const int load_k   = (tid % 2) * 4;
    const int b_n_global = blockIdx.x * 128 + load_idx;
    const int K_padded = (param.K + 7) / 8 * 8;
    const float *b_src_row = weights + (size_t)b_n_global * K_padded;

    auto get_input_val_S = [&](int m_idx, int k_idx) -> uint8_t {
        if (m_idx >= param.M || k_idx >= param.K) return 0;
        int oh = m_idx / param.Ow;
        int ow = m_idx % param.Ow;
        int c = k_idx / (param.R * param.S);
        int rs = k_idx % (param.R * param.S);
        int r = rs / param.S;
        int s = rs % param.S;
        int ih = oh * param.U - param.P + r;
        int iw = ow * param.V - param.Q + s;
        if (ih >= 0 && ih < param.H && iw >= 0 && iw < param.W) {
            return inputs[(size_t)c * param.H * param.W + ih * param.W + iw];
        }
        return 0;
    };

    const int k_iters = (param.K + 7) / 8;

    auto load_chunk = [&](int k_tile, int buf) {
        int k_curr = k_tile * 8;
        if (tid < 128) {
            uint32_t a_dst = smem_u32addr(&A_smem_base[buf * 2048 + (tid % 128) * 16]);
            uint32_t a_reg[2] = {0, 0};
            int m_curr = blockIdx.y * 128 + (tid % 128);
            if (m_curr < param.M) {
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    uint8_t val = get_input_val_S(m_curr, k_curr + i);
                    if (i < 4) a_reg[0] |= ((uint32_t)val << (i * 8));
                    else a_reg[1] |= ((uint32_t)val << ((i - 4) * 8));
                }
            }
            ptx::sts64(a_reg[0], a_reg[1], a_dst);
        }
        {
            int k_off = k_curr + load_k;
            int b_src_bytes = (b_n_global < param.N_padded) ? 16 : 0;
            uint32_t b_dst = smem_u32addr(&B_smem_base[buf * 1024 + load_idx * 8 + load_k]);
            asm volatile ("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" :: "r"(b_dst), "l"(b_src_row + k_off), "r"(b_src_bytes));
        }
    };

    if (k_iters > 0) { load_chunk(0, 0); asm volatile ("cp.async.commit_group;\n" :::); }
    for (int kt = 0; kt < k_iters; kt++) {
        if (kt + 1 < k_iters) { load_chunk(kt + 1, (kt + 1) % 2); asm volatile ("cp.async.commit_group;\n" :::); }
        if (kt + 1 < k_iters) { asm volatile ("cp.async.wait_group 1;\n" :::); }
        else { asm volatile ("cp.async.wait_group 0;\n" :::); }
        __syncthreads();
        uint8_t *A_curr = &A_smem_base[(kt % 2) * 2048];
        float   *B_curr = &B_smem_base[(kt % 2) * 1024];
#pragma unroll
        for (int k = 0; k < 8; k++) {
            float b_vals[8];
#pragma unroll
            for (int j = 0; j < 8; j++) b_vals[j] = B_curr[(thread_n_base + j) * 8 + k];
#pragma unroll
            for (int i = 0; i < 8; i++) {
                uint8_t packed = A_curr[(thread_m_base + i) * 16 + k];
                if ((packed >> blockIdx.z) & 1) { 
#pragma unroll
                    for (int j = 0; j < 8; j++) C_frag[i][j] += b_vals[j];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const int m_global = blockIdx.y * 128 + thread_m_base + i;
        if (m_global < param.M) {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int n_global = blockIdx.x * 128 + thread_n_base + j;
                if (n_global < param.N)
                    outputs[(size_t)blockIdx.z * param.outBatchNumel + (size_t)n_global * param.M + m_global] = C_frag[i][j];
            }
        }
    }
}

// -----------------------------------------------------------------------------
// ANN FP16 Optimized Kernel
// -----------------------------------------------------------------------------
__global__ void conv2d_128x128x8_FP16_kernel(
    const half *inputs,
    const half *weights,
    const half *bias,
    half *outputs,
    Conv2DK128Param param)
{
    __shared__ __align__(16 * 1024) char smem[8 * 1024];
    half *A_smem_base = reinterpret_cast<half *>(smem);
    half *B_smem_base = reinterpret_cast<half *>(smem + 4 * 1024);
    half *smembias = reinterpret_cast<half *>(smem);

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
        smembias[tid] = (n_global < param.N) ? bias[n_global] : __float2half(0.0f);
    }
    __syncthreads();

    float C_frag[8][8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const int m_global = blockIdx.y * 128 + thread_m_base + i;
        float mask = (m_global < param.M) ? 1.0f : 0.0f;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            C_frag[i][j] = __half2float(smembias[thread_n_base + j]) * mask;
        }
    }
    __syncthreads();

    const int load_idx = tid % 128;
    const int is_load_B = tid / 128;
    const int b_n_global = blockIdx.x * 128 + load_idx;
    const int K_padded = (param.K + 7) / 8 * 8;
    const half *b_src_row = weights + (size_t)b_n_global * K_padded;

    auto get_input_val_H = [&](int m_idx, int k_idx) {
        if (m_idx >= param.M || k_idx >= param.K) return __float2half(0.0f);
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
        return __float2half(0.0f);
    };

    const int k_iters = (param.K + 7) / 8;
    auto load_chunk = [&](int k_tile, int buf) {
        int k_curr = k_tile * 8;
        if (is_load_B == 0) {
            uint32_t a_reg[4];
            #pragma unroll
            for (int i=0; i<4; i++) {
                half2 vals;
                vals.x = get_input_val_H(blockIdx.y * 128 + load_idx, k_curr + i*2);
                vals.y = get_input_val_H(blockIdx.y * 128 + load_idx, k_curr + i*2 + 1);
                a_reg[i] = reinterpret_cast<uint32_t&>(vals);
            }
            uint32_t a_dst = smem_u32addr(&A_smem_base[buf * 1024 + load_idx * 8]);
            ptx::sts128(a_reg[0], a_reg[1], a_reg[2], a_reg[3], a_dst);
        } else {
            int b_src_bytes = (b_n_global < param.N_padded) ? 16 : 0;
            uint32_t b_dst = smem_u32addr(&B_smem_base[buf * 1024 + load_idx * 8]);
            asm volatile ("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" :: "r"(b_dst), "l"(b_src_row + k_curr), "r"(b_src_bytes));
        }
    };

    if (k_iters > 0) { load_chunk(0, 0); asm volatile ("cp.async.commit_group;\n" :::); }
    for (int kt = 0; kt < k_iters; kt++) {
        if (kt + 1 < k_iters) { load_chunk(kt + 1, (kt + 1) % 2); asm volatile ("cp.async.commit_group;\n" :::); }
        if (kt + 1 < k_iters) { asm volatile ("cp.async.wait_group 1;\n" :::); }
        else { asm volatile ("cp.async.wait_group 0;\n" :::); }
        __syncthreads();
        half *A_curr = &A_smem_base[(kt % 2) * 1024];
        half *B_curr = &B_smem_base[(kt % 2) * 1024];
#pragma unroll
        for (int k = 0; k < 8; k++) {
            float a_vals[8], b_vals[8];
#pragma unroll
            for (int i = 0; i < 8; i++) a_vals[i] = __half2float(A_curr[(thread_m_base + i) * 8 + k]);
#pragma unroll
            for (int j = 0; j < 8; j++) b_vals[j] = __half2float(B_curr[(thread_n_base + j) * 8 + k]);
#pragma unroll
            for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++) C_frag[i][j] += a_vals[i] * b_vals[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const int m_global = blockIdx.y * 128 + thread_m_base + i;
        if (m_global < param.M) {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int n_global = blockIdx.x * 128 + thread_n_base + j;
                if (n_global < param.N)
                    outputs[(size_t)blockIdx.z * param.outBatchNumel + (size_t)n_global * param.M + m_global] = __float2half(C_frag[i][j]);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// SNN FP16 Optimized Kernel
// -----------------------------------------------------------------------------
__global__ void conv2d_128x128x8_S_FP16_kernel(
    const uint8_t *inputs,
    const half *weights,
    const half *bias,
    half *outputs,
    Conv2DK128Param param)
{
    __shared__ __align__(16 * 1024) char smem[16 * 1024];
    uint8_t *A_smem_base = reinterpret_cast<uint8_t *>(smem);
    half     *B_smem_base = reinterpret_cast<half *>(smem + 4 * 1024);
    half     *smembias = reinterpret_cast<half *>(smem);

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
        smembias[tid] = (n_global < param.N) ? bias[n_global] : __float2half(0.0f);
    }
    __syncthreads();

    float C_frag[8][8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const int m_global = blockIdx.y * 128 + thread_m_base + i;
        float mask = (m_global < param.M) ? 1.0f : 0.0f;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            C_frag[i][j] = __half2float(smembias[thread_n_base + j]) * mask;
        }
    }
    __syncthreads();

    const int load_idx = tid % 128;
    const int is_load_B = tid / 128;
    const int b_n_global = blockIdx.x * 128 + load_idx;
    const int K_padded = (param.K + 7) / 8 * 8;
    const half *b_src_row = weights + (size_t)b_n_global * K_padded;

    auto get_input_val_S = [&](int m_idx, int k_idx) -> uint8_t {
        if (m_idx >= param.M || k_idx >= param.K) return 0;
        int oh = m_idx / param.Ow;
        int ow = m_idx % param.Ow;
        int c = k_idx / (param.R * param.S);
        int rs = k_idx % (param.R * param.S);
        int r = rs / param.S;
        int s = rs % param.S;
        int ih = oh * param.U - param.P + r;
        int iw = ow * param.V - param.Q + s;
        if (ih >= 0 && ih < param.H && iw >= 0 && iw < param.W) {
            return inputs[(size_t)c * param.H * param.W + ih * param.W + iw];
        }
        return 0;
    };

    const int k_iters = (param.K + 7) / 8;
    auto load_chunk = [&](int k_tile, int buf) {
        int k_curr = k_tile * 8;
        if (is_load_B == 0) {
            uint32_t a_dst = smem_u32addr(&A_smem_base[buf * 2048 + load_idx * 16]);
            uint32_t a_reg[2] = {0, 0};
            int m_curr = blockIdx.y * 128 + load_idx;
            if (m_curr < param.M) {
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    uint8_t val = get_input_val_S(m_curr, k_curr + i);
                    if (i < 4) a_reg[0] |= ((uint32_t)val << (i * 8));
                    else a_reg[1] |= ((uint32_t)val << ((i - 4) * 8));
                }
            }
            ptx::sts64(a_reg[0], a_reg[1], a_dst);
        } else {
            int b_src_bytes = (b_n_global < param.N_padded) ? 16 : 0;
            uint32_t b_dst = smem_u32addr(&B_smem_base[buf * 1024 + load_idx * 8]);
            asm volatile ("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" :: "r"(b_dst), "l"(b_src_row + k_curr), "r"(b_src_bytes));
        }
    };

    if (k_iters > 0) { load_chunk(0, 0); asm volatile ("cp.async.commit_group;\n" :::); }
    for (int kt = 0; kt < k_iters; kt++) {
        if (kt + 1 < k_iters) { load_chunk(kt + 1, (kt + 1) % 2); asm volatile ("cp.async.commit_group;\n" :::); }
        if (kt + 1 < k_iters) { asm volatile ("cp.async.wait_group 1;\n" :::); }
        else { asm volatile ("cp.async.wait_group 0;\n" :::); }
        __syncthreads();
        uint8_t *A_curr = &A_smem_base[(kt % 2) * 2048];
        half     *B_curr = &B_smem_base[(kt % 2) * 1024];
#pragma unroll
        for (int k = 0; k < 8; k++) {
            float b_vals[8];
#pragma unroll
            for (int j = 0; j < 8; j++) b_vals[j] = __half2float(B_curr[(thread_n_base + j) * 8 + k]);
#pragma unroll
            for (int i = 0; i < 8; i++) {
                uint8_t packed = A_curr[(thread_m_base + i) * 16 + k];
                if ((packed >> blockIdx.z) & 1) { 
#pragma unroll
                    for (int j = 0; j < 8; j++) C_frag[i][j] += b_vals[j];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const int m_global = blockIdx.y * 128 + thread_m_base + i;
        if (m_global < param.M) {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int n_global = blockIdx.x * 128 + thread_n_base + j;
                if (n_global < param.N)
                    outputs[(size_t)blockIdx.z * param.outBatchNumel + (size_t)n_global * param.M + m_global] = __float2half(C_frag[i][j]);
            }
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
    std::vector<float> h_in(N * C * H * W), h_wt_orig(K * InK), h_wt_padded(N_padded * K_padded, 0.0f), h_bias(K), h_out(N * K * M, 0.0f), h_ref(N * K * M);
    srand(42);
    for (auto &x : h_in) x = (float)rand() / RAND_MAX;
    for (auto &x : h_wt_orig) x = (float)rand() / RAND_MAX;
    for (auto &x : h_bias) x = (float)rand() / RAND_MAX;
    for (int k = 0; k < K; k++) for (int i = 0; i < InK; i++) h_wt_padded[(size_t)k * K_padded + i] = h_wt_orig[(size_t)k * InK + i];
    float *d_in, *d_wt, *d_bias, *d_out;
    cudaMalloc(&d_in, h_in.size() * sizeof(float)); cudaMalloc(&d_wt, h_wt_padded.size() * sizeof(float));
    cudaMalloc(&d_bias, h_bias.size() * sizeof(float)); cudaMalloc(&d_out, h_out.size() * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wt, h_wt_padded.data(), h_wt_padded.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(float), cudaMemcpyHostToDevice);
    Conv2DK128Param param = {(uint32_t)M, (uint32_t)InK, (uint32_t)K, (uint32_t)N_padded, (uint32_t)C, (uint32_t)H, (uint32_t)W, (uint32_t)Oh, (uint32_t)Ow, (uint32_t)R, (uint32_t)S, (uint32_t)U, (uint32_t)V, (uint32_t)P, (uint32_t)Q, (uint32_t)(C * H * W), (uint32_t)(K * M)};
    conv2d_128x128x8_kernel<<<dim3((K + 127) / 128, (M + 127) / 128, N), 256>>>(d_in, d_wt, d_bias, d_out, param);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost);
    direct_conv2d_cpu(h_in.data(), h_wt_orig.data(), h_bias.data(), h_ref.data(), N, C, H, W, K, R, S, U, V, P, Q);
    int errs = 0;
    for (int n = 0; n < N; n++) for (int k = 0; k < K; k++) for (int m = 0; m < M; m++) {
        if (std::abs(h_out[n * K * M + k * M + m] - h_ref[n * K * M + k * M + m]) > 1e-3) errs++;
    }
    std::cout << (errs == 0 ? "  PASSED!" : "  FAILED") << " (" << errs << " errors)" << std::endl;
    cudaFree(d_in); cudaFree(d_wt); cudaFree(d_bias); cudaFree(d_out);
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
    std::vector<uint8_t> h_in(C * H * W);
    std::vector<float> h_wt_orig(K * InK), h_wt_padded(N_padded * K_padded, 0.0f), h_bias(K), h_out(T * K * M, 0.0f), h_ref(T * K * M);
    srand(42);
    for (auto &x : h_in) {
        uint8_t packed = 0;
        for (int t = 0; t < T; t++) if (rand() & 1) packed |= (1 << t);
        x = packed;
    }
    for (auto &x : h_wt_orig) x = (float)rand() / RAND_MAX;
    for (auto &x : h_bias) x = (float)rand() / RAND_MAX;
    for (int k = 0; k < K; k++) for (int i = 0; i < InK; i++) h_wt_padded[(size_t)k * K_padded + i] = h_wt_orig[(size_t)k * InK + i];
    uint8_t *d_in; float *d_wt, *d_bias, *d_out;
    cudaMalloc(&d_in, h_in.size()); cudaMalloc(&d_wt, h_wt_padded.size() * sizeof(float));
    cudaMalloc(&d_bias, h_bias.size() * sizeof(float)); cudaMalloc(&d_out, h_out.size() * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), h_in.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wt, h_wt_padded.data(), h_wt_padded.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(float), cudaMemcpyHostToDevice);
    Conv2DK128Param param = {(uint32_t)M, (uint32_t)InK, (uint32_t)K, (uint32_t)N_padded, (uint32_t)C, (uint32_t)H, (uint32_t)W, (uint32_t)Oh, (uint32_t)Ow, (uint32_t)R, (uint32_t)S, (uint32_t)U, (uint32_t)V, (uint32_t)P, (uint32_t)Q, (uint32_t)(C * H * W), (uint32_t)(K * M)};
    conv2d_128x128x8_S_kernel<<<dim3((K + 127) / 128, (M + 127) / 128, T), 256>>>(d_in, d_wt, d_bias, d_out, param);
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
                                    if ((h_in[c * H * W + ih * W + iw] >> t) & 1)
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
    int K_padded = (InK + 7) / 8 * 8;
    std::cout << "Testing Conv2D FP16: N=" << N << " C=" << C << " H=" << H << " W=" << W << " K=" << K << std::endl;
    std::vector<half> h_in(N * C * H * W), h_wt_orig(K * InK), h_wt_padded(N_padded * K_padded, __float2half(0.0f)), h_bias(K), h_out(N * K * M), h_ref(N * K * M);
    srand(42);
    for (auto &x : h_in) x = __float2half((float)rand() / RAND_MAX);
    for (auto &x : h_wt_orig) x = __float2half((float)rand() / RAND_MAX);
    for (auto &x : h_bias) x = __float2half((float)rand() / RAND_MAX);
    for (int k = 0; k < K; k++) for (int i = 0; i < InK; i++) h_wt_padded[(size_t)k * K_padded + i] = h_wt_orig[(size_t)k * InK + i];
    half *d_in, *d_wt, *d_bias, *d_out;
    cudaMalloc(&d_in, h_in.size() * sizeof(half)); cudaMalloc(&d_wt, h_wt_padded.size() * sizeof(half));
    cudaMalloc(&d_bias, h_bias.size() * sizeof(half)); cudaMalloc(&d_out, h_out.size() * sizeof(half));
    cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wt, h_wt_padded.data(), h_wt_padded.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(half), cudaMemcpyHostToDevice);
    Conv2DK128Param param = {(uint32_t)M, (uint32_t)InK, (uint32_t)K, (uint32_t)N_padded, (uint32_t)C, (uint32_t)H, (uint32_t)W, (uint32_t)Oh, (uint32_t)Ow, (uint32_t)R, (uint32_t)S, (uint32_t)U, (uint32_t)V, (uint32_t)P, (uint32_t)Q, (uint32_t)(C * H * W), (uint32_t)(K * M)};
    conv2d_128x128x8_FP16_kernel<<<dim3((K + 127) / 128, (M + 127) / 128, N), 256>>>(d_in, d_wt, d_bias, d_out, param);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(half), cudaMemcpyDeviceToHost);
    for (int n = 0; n < N; n++) for (int k = 0; k < K; k++) for (int oh = 0; oh < Oh; oh++) for (int ow = 0; ow < Ow; ow++) {
        float sum = __half2float(h_bias[k]);
        for (int c = 0; c < C; c++) for (int r = 0; r < R; r++) for (int s = 0; s < S; s++) {
            int ih = oh * U - P + r;
            int iw = ow * V - Q + s;
            if (iw >= 0 && ih >= 0 && iw < W && ih < H)
                sum += __half2float(h_in[n * C * H * W + c * H * W + ih * W + iw]) * __half2float(h_wt_orig[k * InK + c * R * S + r * S + s]);
        }
        h_ref[n * K * M + k * M + oh * Ow + ow] = __float2half(sum);
    }
    int errs = 0;
    for (size_t i = 0; i < h_out.size(); i++) if (std::abs(__half2float(h_out[i]) - __half2float(h_ref[i])) > 5e-2) errs++;
    std::cout << (errs == 0 ? "  PASSED!" : "  FAILED") << " (" << errs << " errors)" << std::endl;
    cudaFree(d_in); cudaFree(d_wt); cudaFree(d_bias); cudaFree(d_out);
}

void test_conv2d_snn_fp16(int T, int C, int H, int W, int K, int R, int S, int U, int V, int P, int Q)
{
    int Oh = (H + 2 * P - R) / U + 1;
    int Ow = (W + 2 * Q - S) / V + 1;
    int M = Oh * Ow;
    int InK = C * R * S;
    int N_padded = (K + 127) / 128 * 128;
    int K_padded = (InK + 7) / 8 * 8;
    std::cout << "Testing SNN Conv2D FP16: T=" << T << " C=" << C << " H=" << H << " W=" << W << " K=" << K << std::endl;
    std::vector<uint8_t> h_in(C * H * W);
    std::vector<half> h_wt_orig(K * InK), h_wt_padded(N_padded * K_padded, __float2half(0.0f)), h_bias(K), h_out(T * K * M), h_ref(T * K * M);
    srand(42);
    for (auto &x : h_in) {
        uint8_t packed = 0;
        for (int t = 0; t < T; t++) if (rand() & 1) packed |= (1 << t);
        x = packed;
    }
    for (auto &x : h_wt_orig) x = __float2half((float)rand() / RAND_MAX);
    for (auto &x : h_bias) x = __float2half((float)rand() / RAND_MAX);
    for (int k = 0; k < K; k++) for (int i = 0; i < InK; i++) h_wt_padded[(size_t)k * K_padded + i] = h_wt_orig[(size_t)k * InK + i];
    uint8_t *d_in; half *d_wt, *d_bias, *d_out;
    cudaMalloc(&d_in, h_in.size()); cudaMalloc(&d_wt, h_wt_padded.size() * sizeof(half));
    cudaMalloc(&d_bias, h_bias.size() * sizeof(half)); cudaMalloc(&d_out, h_out.size() * sizeof(half));
    cudaMemcpy(d_in, h_in.data(), h_in.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wt, h_wt_padded.data(), h_wt_padded.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(half), cudaMemcpyHostToDevice);
    Conv2DK128Param param = {(uint32_t)M, (uint32_t)InK, (uint32_t)K, (uint32_t)N_padded, (uint32_t)C, (uint32_t)H, (uint32_t)W, (uint32_t)Oh, (uint32_t)Ow, (uint32_t)R, (uint32_t)S, (uint32_t)U, (uint32_t)V, (uint32_t)P, (uint32_t)Q, (uint32_t)(C * H * W), (uint32_t)(K * M)};
    conv2d_128x128x8_S_FP16_kernel<<<dim3((K + 127) / 128, (M + 127) / 128, T), 256>>>(d_in, d_wt, d_bias, d_out, param);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(half), cudaMemcpyDeviceToHost);
    for (int t = 0; t < T; t++) for (int k = 0; k < K; k++) for (int oh = 0; oh < Oh; oh++) for (int ow = 0; ow < Ow; ow++) {
        float sum = __half2float(h_bias[k]);
        for (int c = 0; c < C; c++) for (int r = 0; r < R; r++) for (int s = 0; s < S; s++) {
            int ih = oh * U - P + r;
            int iw = ow * V - Q + s;
            if (iw >= 0 && ih >= 0 && iw < W && ih < H)
                if ((h_in[c * H * W + ih * W + iw] >> t) & 1)
                    sum += __half2float(h_wt_orig[k * InK + c * R * S + r * S + s]);
        }
        h_ref[t * K * M + k * M + oh * Ow + ow] = __float2half(sum);
    }
    int errs = 0;
    for (size_t i = 0; i < h_out.size(); i++) if (std::abs(__half2float(h_out[i]) - __half2float(h_ref[i])) > 5e-2) errs++;
    std::cout << (errs == 0 ? "  PASSED!" : "  FAILED") << " (" << errs << " errors)" << std::endl;
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
