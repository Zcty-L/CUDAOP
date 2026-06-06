#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cuda_utils.cuh"
#include "ptx_utils.cuh"

// =============================================================================
// Linear Kernel: 128x128 output tile, K_chunk=8
// Supports ANN (FP32/FP16) and SNN (Packed Spikes uint8)
// =============================================================================

struct LinearParam
{
    uint32_t in_ch;          // M (real)
    uint32_t in_dim;         // K (real)
    uint32_t out_dim;        // N (real)
    uint32_t out_dim_padded; // N padded to multiple of 128
    uint32_t inBatchNumel;   // T stride (M * K)
    uint32_t outBatchNumel;  // T stride (M * N)
};

// -----------------------------------------------------------------------------
// ANN FP32 Optimized Kernel (Double Buffer)
// -----------------------------------------------------------------------------
__global__ void linear_128x128x8_kernel(
    const float *inputs,
    const float *weights, // [N_padded, K_padded]
    const float *bias,
    float *outputs,
    LinearParam param)
{
    // SMEM layout: [2][128][8] for A and [2][128][8] for B
    __shared__ __align__(16 * 1024) char smem[16 * 1024];
    float *A_smem_base = reinterpret_cast<float *>(smem);
    float *B_smem_base = reinterpret_cast<float *>(smem + 8 * 1024);
    
    // Alias smem[0..511] for bias staging (128 floats)
    float *smembias = reinterpret_cast<float *>(smem);

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int warp_m = warp_id / 2; // 0..3
    const int warp_n = warp_id % 2; // 0..1

    const int mma_tid_x = (lane_id / 2) % 8;
    const int mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);

    const int thread_m_base = warp_m * 32 + mma_tid_y * 8;
    const int thread_n_base = warp_n * 64 + mma_tid_x * 8;

    // --- 1. Bias Staging & Accumulators Initialization ---
    if (tid < 128) {
        const int n_global = blockIdx.x * 128 + tid;
        smembias[tid] = (n_global < param.out_dim) ? bias[n_global] : 0.0f;
    }
    __syncthreads();

    float C_frag[8][8];
#pragma unroll
    for (int i = 0; i < 8; i++)
    {
        const int m_global = blockIdx.y * 128 + thread_m_base + i;
        float mask = (m_global < param.in_ch) ? 1.0f : 0.0f;
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            C_frag[i][j] = smembias[thread_n_base + j] * mask;
        }
    }
    __syncthreads(); // Ensure bias is used before SMEM is overwritten by A/B

    // --- 2. Setup Loading indices ---
    const int load_idx = tid / 2;    // 0..127
    const int load_k   = (tid % 2) * 4; // 0 or 4

    const int a_m_global = blockIdx.y * 128 + load_idx;
    const float *a_src_row = inputs + (size_t)blockIdx.z * param.inBatchNumel + (size_t)a_m_global * param.in_dim;
    const int b_n_global = blockIdx.x * 128 + load_idx;
    const int K_padded = (param.in_dim + 7) / 8 * 8;
    const float *b_src_row = weights + (size_t)b_n_global * K_padded;

    const int k_iters = (param.in_dim + 7) / 8;

    auto load_chunk = [&](int k_tile, int buf) {
        int k_curr = k_tile * 8 + load_k;
        int a_src_bytes = 0;
        if (a_m_global < (int)param.in_ch) {
            int rem = (int)param.in_dim - k_curr;
            a_src_bytes = max(0, min(16, rem * 4));
        }
        uint32_t a_dst = smem_u32addr(&A_smem_base[buf * 1024 + load_idx * 8 + load_k]);
        asm volatile ("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(a_dst), "l"(a_src_row + k_curr), "r"(a_src_bytes));

        int b_src_bytes = (b_n_global < (int)param.out_dim_padded) ? 16 : 0;
        uint32_t b_dst = smem_u32addr(&B_smem_base[buf * 1024 + load_idx * 8 + load_k]);
        asm volatile ("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" :: "r"(b_dst), "l"(b_src_row + k_curr), "r"(b_src_bytes));
    };

    if (k_iters > 0) {
        load_chunk(0, 0);
        asm volatile ("cp.async.commit_group;\n" :::);
    }

    for (int kt = 0; kt < k_iters; kt++)
    {
        if (kt + 1 < k_iters) {
            load_chunk(kt + 1, (kt + 1) % 2);
            asm volatile ("cp.async.commit_group;\n" :::);
        }
        if (kt + 1 < k_iters) { asm volatile ("cp.async.wait_group 1;\n" :::); }
        else { asm volatile ("cp.async.wait_group 0;\n" :::); }
        __syncthreads();

        float *A_curr = &A_smem_base[(kt % 2) * 1024];
        float *B_curr = &B_smem_base[(kt % 2) * 1024];

#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            float a_vals[8], b_vals[8];
#pragma unroll
            for (int i = 0; i < 8; i++) a_vals[i] = A_curr[(thread_m_base + i) * 8 + k];
#pragma unroll
            for (int j = 0; j < 8; j++) b_vals[j] = B_curr[(thread_n_base + j) * 8 + k];
#pragma unroll
            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 8; j++)
                    C_frag[i][j] += a_vals[i] * b_vals[j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 8; i++)
    {
        const int m_global = blockIdx.y * 128 + thread_m_base + i;
        if (m_global < param.in_ch)
        {
#pragma unroll
            for (int j = 0; j < 8; j++)
            {
                const int n_global = blockIdx.x * 128 + thread_n_base + j;
                if (n_global < param.out_dim)
                    outputs[(size_t)blockIdx.z * param.outBatchNumel + (size_t)m_global * param.out_dim + n_global] = C_frag[i][j];
            }
        }
    }
}

// -----------------------------------------------------------------------------
// SNN FP32 Optimized Kernel (Double Buffer, Packed uint8 spikes)
// -----------------------------------------------------------------------------
__global__ void linear_128x128x8_S_kernel(
    const uint8_t *inputs, // [M, K] packed spikes
    const float *weights,
    const float *bias,
    float *outputs,
    LinearParam param)
{
    __shared__ __align__(16 * 1024) char smem[16 * 1024];
    uint8_t *A_smem_base = reinterpret_cast<uint8_t *>(smem);
    float   *B_smem_base = reinterpret_cast<float *>(smem + 4 * 1024);
    
    // Alias smem for bias (128 floats)
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
        smembias[tid] = (n_global < param.out_dim) ? bias[n_global] : 0.0f;
    }
    __syncthreads();

    float C_frag[8][8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const int m_global = blockIdx.y * 128 + thread_m_base + i;
        float mask = (m_global < param.in_ch) ? 1.0f : 0.0f;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            C_frag[i][j] = smembias[thread_n_base + j] * mask;
        }
    }
    __syncthreads();

    const int a_m_global = blockIdx.y * 128 + (tid % 128); 
    const uint8_t *a_src_row = inputs + (size_t)a_m_global * param.in_dim;
    const int b_n_idx = tid / 2; 
    const int b_k_off = (tid % 2) * 4;
    const int b_n_global = blockIdx.x * 128 + b_n_idx;
    const int K_padded = (param.in_dim + 7) / 8 * 8;
    const float *b_src_row = weights + (size_t)b_n_global * K_padded;

    const int k_iters = (param.in_dim + 7) / 8;

    auto load_chunk = [&](int k_tile, int buf) {
        int k_curr = k_tile * 8;
        if (tid < 128) {
            uint32_t a_dst = smem_u32addr(&A_smem_base[buf * 2048 + (tid % 128) * 16]);
            uint32_t a_reg[2] = {0, 0};
            if (a_m_global < param.in_ch) {
                int rem = (int)param.in_dim - k_curr;
                if (rem >= 8) {
                    ptx::ldg32_nc_0(a_reg[0], a_src_row + k_curr, true);
                    ptx::ldg32_nc_0(a_reg[1], a_src_row + k_curr + 4, true);
                } else {
                    for (int i = 0; i < rem; i++) {
                        uint8_t val = a_src_row[k_curr + i];
                        if (i < 4) a_reg[0] |= ((uint32_t)val << (i * 8));
                        else a_reg[1] |= ((uint32_t)val << ((i - 4) * 8));
                    }
                }
            }
            ptx::sts64(a_reg[0], a_reg[1], a_dst);
        }
        
        {
            int k_off = k_curr + b_k_off;
            int b_src_bytes = (b_n_global < param.out_dim_padded) ? 16 : 0;
            uint32_t b_dst = smem_u32addr(&B_smem_base[buf * 1024 + b_n_idx * 8 + b_k_off]);
            asm volatile ("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" :: "r"(b_dst), "l"(b_src_row + k_off), "r"(b_src_bytes));
        }
    };

    if (k_iters > 0) {
        load_chunk(0, 0);
        asm volatile ("cp.async.commit_group;\n" :::);
    }

    for (int kt = 0; kt < k_iters; kt++) {
        if (kt + 1 < k_iters) {
            load_chunk(kt + 1, (kt + 1) % 2);
            asm volatile ("cp.async.commit_group;\n" :::);
        }
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
        if (m_global < param.in_ch) {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int n_global = blockIdx.x * 128 + thread_n_base + j;
                if (n_global < param.out_dim)
                    outputs[(size_t)blockIdx.z * param.outBatchNumel + (size_t)m_global * param.out_dim + n_global] = C_frag[i][j];
            }
        }
    }
}

// -----------------------------------------------------------------------------
// ANN FP16 Optimized Kernel (Double Buffer)
// -----------------------------------------------------------------------------
__global__ void linear_128x128x8_FP16_kernel(
    const half *inputs,
    const half *weights, // [N_padded, K_padded]
    const half *bias,
    half *outputs,
    LinearParam param)
{
    __shared__ __align__(16 * 1024) char smem[8 * 1024];
    half *A_smem_base = reinterpret_cast<half *>(smem);
    half *B_smem_base = reinterpret_cast<half *>(smem + 4 * 1024);
    
    // Alias smem for bias (128 halfs -> 256 bytes)
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
        smembias[tid] = (n_global < param.out_dim) ? bias[n_global] : __float2half(0.0f);
    }
    __syncthreads();

    float C_frag[8][8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const int m_global = blockIdx.y * 128 + thread_m_base + i;
        float mask = (m_global < param.in_ch) ? 1.0f : 0.0f;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            C_frag[i][j] = __half2float(smembias[thread_n_base + j]) * mask;
        }
    }
    __syncthreads();

    const int load_idx = tid % 128;
    const int is_load_B = tid / 128;
    const int a_m_global = blockIdx.y * 128 + load_idx;
    const half *a_src_row = inputs + (size_t)blockIdx.z * param.inBatchNumel + (size_t)a_m_global * param.in_dim;
    const int b_n_global = blockIdx.x * 128 + load_idx;
    const int K_padded = (param.in_dim + 7) / 8 * 8;
    const half *b_src_row = weights + (size_t)b_n_global * K_padded;
    const int k_iters = (param.in_dim + 7) / 8;

    auto load_chunk = [&](int k_tile, int buf) {
        int k_curr = k_tile * 8;
        if (is_load_B == 0) {
            int a_src_bytes = 0;
            if (a_m_global < (int)param.in_ch) {
                int rem = (int)param.in_dim - k_curr;
                a_src_bytes = max(0, min(16, rem * 2));
            }
            uint32_t a_dst = smem_u32addr(&A_smem_base[buf * 1024 + load_idx * 8]);
            asm volatile ("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(a_dst), "l"(a_src_row + k_curr), "r"(a_src_bytes));
        } else {
            int b_src_bytes = (b_n_global < (int)param.out_dim_padded) ? 16 : 0;
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
        if (m_global < param.in_ch) {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int n_global = blockIdx.x * 128 + thread_n_base + j;
                if (n_global < param.out_dim)
                    outputs[(size_t)blockIdx.z * param.outBatchNumel + (size_t)m_global * param.out_dim + n_global] = __float2half(C_frag[i][j]);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// SNN FP16 Optimized Kernel (Double Buffer, Packed uint8 spikes)
// -----------------------------------------------------------------------------
__global__ void linear_128x128x8_S_FP16_kernel(
    const uint8_t *inputs, // [M, K] packed spikes
    const half *weights,
    const half *bias,
    half *outputs,
    LinearParam param)
{
    __shared__ __align__(16 * 1024) char smem[16 * 1024];
    uint8_t *A_smem_base = reinterpret_cast<uint8_t *>(smem);
    half     *B_smem_base = reinterpret_cast<half *>(smem + 4 * 1024);
    
    // Alias bias
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
        smembias[tid] = (n_global < param.out_dim) ? bias[n_global] : __float2half(0.0f);
    }
    __syncthreads();

    float C_frag[8][8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const int m_global = blockIdx.y * 128 + thread_m_base + i;
        float mask = (m_global < param.in_ch) ? 1.0f : 0.0f;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            C_frag[i][j] = __half2float(smembias[thread_n_base + j]) * mask;
        }
    }
    __syncthreads();

    const int load_idx = tid % 128;
    const int is_load_B = tid / 128;
    const int a_m_global = blockIdx.y * 128 + load_idx; 
    const uint8_t *a_src_row = inputs + (size_t)a_m_global * param.in_dim;
    const int b_n_global = blockIdx.x * 128 + load_idx;
    const int K_padded = (param.in_dim + 7) / 8 * 8;
    const half *b_src_row = weights + (size_t)b_n_global * K_padded;
    const int k_iters = (param.in_dim + 7) / 8;

    auto load_chunk = [&](int k_tile, int buf) {
        int k_curr = k_tile * 8;
        if (is_load_B == 0) {
            uint32_t a_dst = smem_u32addr(&A_smem_base[buf * 2048 + load_idx * 16]);
            uint32_t a_reg[2] = {0, 0};
            if (a_m_global < param.in_ch) {
                int rem = (int)param.in_dim - k_curr;
                if (rem >= 8) {
                    ptx::ldg32_nc_0(a_reg[0], a_src_row + k_curr, true);
                    ptx::ldg32_nc_0(a_reg[1], a_src_row + k_curr + 4, true);
                } else {
                    for (int i = 0; i < rem; i++) {
                        uint8_t val = a_src_row[k_curr + i];
                        if (i < 4) a_reg[0] |= ((uint32_t)val << (i * 8));
                        else a_reg[1] |= ((uint32_t)val << ((i - 4) * 8));
                    }
                }
            }
            ptx::sts64(a_reg[0], a_reg[1], a_dst);
        } else {
            int b_src_bytes = (b_n_global < param.out_dim_padded) ? 16 : 0;
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
        if (m_global < param.in_ch) {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int n_global = blockIdx.x * 128 + thread_n_base + j;
                if (n_global < param.out_dim)
                    outputs[(size_t)blockIdx.z * param.outBatchNumel + (size_t)m_global * param.out_dim + n_global] = __float2half(C_frag[i][j]);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Test Functions
// -----------------------------------------------------------------------------
void test_linear_fp32(int T, int M, int K, int N)
{
    if (K % 4 != 0) return;
    int K_padded = (K + 7) / 8 * 8;
    int N_padded = (N + 127) / 128 * 128;
    std::cout << "Testing Double-Buffered ANN FP32: T=" << T << " M=" << M << " K=" << K << " N=" << N << std::endl;
    std::vector<float> h_in(T * M * K), h_wt_orig(N * K), h_wt_padded(N_padded * K_padded, 0.0f), h_bias(N), h_out(T * M * N, 0.0f);
    srand(42);
    for (auto &x : h_in) x = (float)rand() / RAND_MAX;
    for (auto &x : h_wt_orig) x = (float)rand() / RAND_MAX;
    for (auto &x : h_bias) x = (float)rand() / RAND_MAX;
    for (int n = 0; n < N; n++) for (int k = 0; k < K; k++) h_wt_padded[(size_t)n * K_padded + k] = h_wt_orig[(size_t)n * K + k];
    float *d_in, *d_wt, *d_bias, *d_out;
    cudaMalloc(&d_in, h_in.size() * sizeof(float)); cudaMalloc(&d_wt, h_wt_padded.size() * sizeof(float));
    cudaMalloc(&d_bias, h_bias.size() * sizeof(float)); cudaMalloc(&d_out, h_out.size() * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wt, h_wt_padded.data(), h_wt_padded.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(float), cudaMemcpyHostToDevice);
    LinearParam param = {(uint32_t)M, (uint32_t)K, (uint32_t)N, (uint32_t)N_padded, (uint32_t)(M * K), (uint32_t)(M * N)};
    linear_128x128x8_kernel<<<dim3((N + 127) / 128, (M + 127) / 128, T), 256>>>(d_in, d_wt, d_bias, d_out, param);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost);
    int errors = 0;
    for (int t = 0; t < T; t++) for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
        float sum = h_bias[n];
        for (int k = 0; k < K; k++) sum += h_in[(size_t)t * (M * K) + m * K + k] * h_wt_orig[(size_t)n * K + k];
        if (std::abs(h_out[(size_t)t * (M * N) + m * N + n] - sum) > 1e-3) errors++;
    }
    std::cout << (errors == 0 ? "  PASSED!" : "  FAILED") << " (" << errors << " errors)" << std::endl;
    cudaFree(d_in); cudaFree(d_wt); cudaFree(d_bias); cudaFree(d_out);
}

void test_linear_snn_fp32(int T, int M, int K, int N)
{
    if (K % 8 != 0) return;
    int K_padded = (K + 7) / 8 * 8;
    int N_padded = (N + 127) / 128 * 128;
    std::cout << "Testing Optimized SNN FP32: T=" << T << " M=" << M << " K=" << K << " N=" << N << std::endl;
    std::vector<uint8_t> h_in(M * K);
    std::vector<float> h_wt_orig(N * K), h_wt_padded(N_padded * K_padded, 0.0f), h_bias(N), h_out(T * M * N, 0.0f);
    srand(42);
    for (size_t i = 0; i < h_in.size(); i++) {
        uint8_t packed = 0;
        for (int t = 0; t < T; t++) if ((rand() & 1)) packed |= (1u << t);
        h_in[i] = packed;
    }
    for (auto &x : h_wt_orig) x = (float)rand() / RAND_MAX;
    for (auto &x : h_bias) x = (float)rand() / RAND_MAX;
    for (int n = 0; n < N; n++) for (int k = 0; k < K; k++) h_wt_padded[(size_t)n * K_padded + k] = h_wt_orig[(size_t)n * K + k];
    uint8_t *d_in; float *d_wt, *d_bias, *d_out;
    cudaMalloc(&d_in, h_in.size() * sizeof(uint8_t)); cudaMalloc(&d_wt, h_wt_padded.size() * sizeof(float));
    cudaMalloc(&d_bias, h_bias.size() * sizeof(float)); cudaMalloc(&d_out, h_out.size() * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wt, h_wt_padded.data(), h_wt_padded.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, h_out.size() * sizeof(float));
    LinearParam param = {(uint32_t)M, (uint32_t)K, (uint32_t)N, (uint32_t)N_padded, 0, (uint32_t)(M * N)};
    linear_128x128x8_S_kernel<<<dim3((N + 127) / 128, (M + 127) / 128, T), 256>>>(d_in, d_wt, d_bias, d_out, param);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost);
    int errors = 0;
    for (int t = 0; t < T; t++) for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
        float sum = h_bias[n];
        for (int k = 0; k < K; k++) if ((h_in[m * K + k] >> t) & 1) sum += h_wt_orig[n * K + k];
        if (std::abs(h_out[(size_t)t * (M * N) + m * N + n] - sum) > 1e-3) errors++;
    }
    std::cout << (errors == 0 ? "  PASSED!" : "  FAILED") << " (" << errors << " errors)" << std::endl;
    cudaFree(d_in); cudaFree(d_wt); cudaFree(d_bias); cudaFree(d_out);
}

void test_linear_fp16(int T, int M, int K, int N)
{
    if (K % 8 != 0) return;
    int K_padded = (K + 7) / 8 * 8;
    int N_padded = (N + 127) / 128 * 128;
    std::cout << "Testing Double-Buffered ANN FP16: T=" << T << " M=" << M << " K=" << K << " N=" << N << std::endl;
    std::vector<half> h_in(T * M * K), h_wt_orig(N * K), h_wt_padded(N_padded * K_padded, __float2half(0.0f)), h_bias(N), h_out(T * M * N);
    srand(42);
    for (auto &x : h_in) x = __float2half((float)rand() / RAND_MAX);
    for (auto &x : h_wt_orig) x = __float2half((float)rand() / RAND_MAX);
    for (auto &x : h_bias) x = __float2half((float)rand() / RAND_MAX);
    for (int n = 0; n < N; n++) for (int k = 0; k < K; k++) h_wt_padded[(size_t)n * K_padded + k] = h_wt_orig[(size_t)n * K + k];
    half *d_in, *d_wt, *d_bias, *d_out;
    cudaMalloc(&d_in, h_in.size() * sizeof(half)); cudaMalloc(&d_wt, h_wt_padded.size() * sizeof(half));
    cudaMalloc(&d_bias, h_bias.size() * sizeof(half)); cudaMalloc(&d_out, h_out.size() * sizeof(half));
    cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wt, h_wt_padded.data(), h_wt_padded.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(half), cudaMemcpyHostToDevice);
    LinearParam param = {(uint32_t)M, (uint32_t)K, (uint32_t)N, (uint32_t)N_padded, (uint32_t)(M * K), (uint32_t)(M * N)};
    linear_128x128x8_FP16_kernel<<<dim3((N + 127) / 128, (M + 127) / 128, T), 256>>>(d_in, d_wt, d_bias, d_out, param);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(half), cudaMemcpyDeviceToHost);
    int errors = 0;
    for (int t = 0; t < T; t++) for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
        float sum = __half2float(h_bias[n]);
        for (int k = 0; k < K; k++) sum += __half2float(h_in[(size_t)t * (M * K) + m * K + k]) * __half2float(h_wt_orig[(size_t)n * K + k]);
        if (std::abs(__half2float(h_out[(size_t)t * (M * N) + m * N + n]) - sum) > 5e-2) errors++;
    }
    std::cout << (errors == 0 ? "  PASSED!" : "  FAILED") << " (" << errors << " errors)" << std::endl;
    cudaFree(d_in); cudaFree(d_wt); cudaFree(d_bias); cudaFree(d_out);
}

void test_linear_snn_fp16(int T, int M, int K, int N)
{
    if (K % 8 != 0) return;
    int K_padded = (K + 7) / 8 * 8;
    int N_padded = (N + 127) / 128 * 128;
    std::cout << "Testing Optimized SNN FP16: T=" << T << " M=" << M << " K=" << K << " N=" << N << std::endl;
    size_t in_sz = (size_t)M * K;
    size_t wt_sz = (size_t)N * K;
    size_t out_sz = (size_t)T * M * N;
    std::vector<uint8_t> h_in(in_sz);
    std::vector<half> h_wt_orig(wt_sz), h_bias(N), h_out(out_sz), h_ref(out_sz);
    std::vector<half> h_wt_padded(N_padded * K_padded, __float2half(0.0f));
    srand(42);
    for (size_t i = 0; i < h_in.size(); i++) {
        uint8_t packed = 0;
        for (int t = 0; t < T; t++) if ((rand() & 1)) packed |= (1u << t);
        h_in[i] = packed;
    }
    for (auto &x : h_wt_orig) x = __float2half((float)rand() / RAND_MAX);
    for (auto &x : h_bias) x = __float2half((float)rand() / RAND_MAX);
    for (int n = 0; n < N; n++) for (int k = 0; k < K; k++) h_wt_padded[(size_t)n * K_padded + k] = h_wt_orig[(size_t)n * K + k];
    uint8_t *d_in; half *d_wt, *d_bias, *d_out;
    cudaMalloc(&d_in, in_sz * sizeof(uint8_t)); cudaMalloc(&d_wt, h_wt_padded.size() * sizeof(half));
    cudaMalloc(&d_bias, N * sizeof(half)); cudaMalloc(&d_out, out_sz * sizeof(half));
    cudaMemcpy(d_in, h_in.data(), in_sz * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wt, h_wt_padded.data(), h_wt_padded.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias.data(), N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, out_sz * sizeof(half));
    LinearParam param = {(uint32_t)M, (uint32_t)K, (uint32_t)N, (uint32_t)N_padded, 0, (uint32_t)(M * N)};
    linear_128x128x8_S_FP16_kernel<<<dim3((N + 127) / 128, (M + 127) / 128, T), 256>>>(d_in, d_wt, d_bias, d_out, param);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out, out_sz * sizeof(half), cudaMemcpyDeviceToHost);
    for (int t = 0; t < T; t++)
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                float sum = __half2float(h_bias[n]);
                for (int k = 0; k < K; k++) if ((h_in[m * K + k] >> t) & 1) sum += __half2float(h_wt_orig[n * K + k]);
                h_ref[t * M * N + m * N + n] = __float2half(sum);
            }
    int errs = 0;
    for (size_t i = 0; i < out_sz; i++) if (std::abs(__half2float(h_out[i]) - __half2float(h_ref[i])) > 5e-2) errs++;
    std::cout << (errs == 0 ? "  PASSED!" : "  FAILED") << " (" << errs << " errors)" << std::endl;
    cudaFree(d_in); cudaFree(d_wt); cudaFree(d_bias); cudaFree(d_out);
}

int main()
{
    test_linear_fp32(1, 128, 128, 128);
    test_linear_fp32(4, 130, 72, 140);
    test_linear_snn_fp32(1, 128, 128, 128);
    test_linear_snn_fp32(8, 64, 64, 64);
    test_linear_snn_fp32(8, 130, 72, 140);
    test_linear_fp16(1, 128, 128, 128);
    test_linear_fp16(4, 130, 72, 140);
    test_linear_snn_fp16(1, 128, 128, 128);
    test_linear_snn_fp16(8, 64, 64, 64);
    test_linear_snn_fp16(8, 130, 72, 140);
    return 0;
}
