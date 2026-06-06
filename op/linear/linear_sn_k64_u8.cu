#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include "cuda_utils.cuh"
#include "ptx_utils.cuh"

// =============================================================================
// linear_sn_64x64_k16_u8  —  Fused SNN Linear + IF HardReset Neuron
//
// Inputs : uint8_t inputs [M][K]           — T spike bits/feature (bit t = step t)
// Weights: float   weights [N_pad][K_pad]  — fp32, K padded to 16, N padded to 64
// Bias   : float   bias [N_pad]
// Outputs: uint8_t outputs [M][N]          — T spike bits/feature
//
// Architecture:
//   64×64 output tile (M=64, N=64), K_CHUNK=16 inner-product loop.
//   Double-buffered cp.async pipeline for weights (padded, fully asynchronous).
//   Scalar/vector loads for inputs (unpadded, handle boundary safely).
//   After accumulation over K, IF HardReset is applied entirely in registers
//   across T time steps. The packed uint8 result is written via a single smem 
//   transpose epilogue.
// =============================================================================

struct LinearParam
{
    uint32_t in_ch;          // M (Batch)
    uint32_t in_dim;         // K (In Features)
    uint32_t out_dim;        // N (Out Features)
    uint32_t out_dim_padded; // N padded to 64
    uint32_t inBatchNumel;   // T stride (0 for packed inputs)
    uint32_t outBatchNumel;  // T stride (0 for packed outputs)
};

static void pad_weights_sn(
    const float *src, float *dst,
    int K, int N,
    int K_padded, int N_padded)
{
    for (int n = 0; n < N_padded; n++)
        for (int k = 0; k < K_padded; k++)
            dst[n * K_padded + k] =
                (n < N && k < K) ? src[n * K + k] : 0.f;
}

// =============================================================================
// Kernel
// =============================================================================

template <int T_STEPS>
__global__
void linear_sn_64x64_k16_u8(
    const uint8_t * __restrict__ inputs,
    const float   * __restrict__ weights,
    const float   * __restrict__ bias,
    uint8_t       * __restrict__ outputs,
    LinearParam param,
    float v_th,
    float v_reset)
{
    static_assert(T_STEPS >= 1 && T_STEPS <= 8, "T_STEPS must be in [1,8] to fit in uint8 output");

    constexpr int K_CHUNK = 16;
    constexpr int M_TILE  = 64;
    constexpr int N_TILE  = 64;

    // 10 KB smem: 4KB×2 weight double-buffer, 1KB×2 input double-buffer.
    __shared__ __align__(128) char smem[10 * 1024];

    float   *smemweight[2];
    smemweight[0] = reinterpret_cast<float *>(smem);
    smemweight[1] = reinterpret_cast<float *>(smem + 4 * 1024);

    uint8_t *smeminput[2];
    smeminput[0] = reinterpret_cast<uint8_t *>(smem + 8 * 1024);
    smeminput[1] = reinterpret_cast<uint8_t *>(smem + 9 * 1024);

    float *smembias = reinterpret_cast<float *>(smem);

    const int tid     = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;

    const int mma_tid_x = lane_id / 16 * 2 + lane_id % 2;   // 0..3
    const int mma_tid_y = lane_id % 16 / 2;                 // 0..7

    const int warp_m = warp_id / 2;   // 0..3
    const int warp_n = warp_id % 2;   // 0..1

    const int thread_m_base = warp_m * 16 + mma_tid_y * 2;   // M offset in tile
    const int thread_n_base = warp_n * 32 + mma_tid_x * 8;   // N offset in tile

    const int m_tile_base = blockIdx.y * M_TILE;
    const int n_tile_base = blockIdx.x * N_TILE;

    // Preload bias
    if (tid < N_TILE) {
        const int n_global = n_tile_base + tid;
        smembias[tid] = (n_global < param.out_dim) ? bias[n_global] : 0.f;
    }
    __syncthreads();

    // Accumulation registers: T × 2 M-rows × 8 N-cols per thread
    float output_frag[T_STEPS][2][8];
#pragma unroll
    for (int t = 0; t < T_STEPS; t++)
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 8; j++)
                output_frag[t][i][j] = smembias[thread_n_base + j]; // Initialize with Bias for column N

    __syncthreads();

    const int k_iters = (param.in_dim + K_CHUNK - 1) / K_CHUNK;

    // ---- Weight loader (cp.async into smemweight) ----
    // weights layout: [N_padded][K_padded]
    const int K_padded = (param.in_dim + K_CHUNK - 1) / K_CHUNK * K_CHUNK;

    auto load_weight = [&](int k_iter, int buf)
    {
        const int k_base = k_iter * K_CHUNK;
        const int n_idx = tid / 4;          // 0..63
        const int k_idx = (tid % 4) * 4;    // 0,4,8,12
        const int n_global = n_tile_base + n_idx;
        const float *src = weights + (size_t)n_global * K_padded + k_base + k_idx;
        uint32_t smem_ptr = ptx::smem_u32addr(&smemweight[buf][n_idx * K_CHUNK + k_idx]);
        
        int src_bytes = (n_global < param.out_dim_padded) ? 16 : 0;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" :: "r"(smem_ptr), "l"(src), "r"(src_bytes));
    };

    auto load_input = [&](int k_iter, int buf)
    {
        const int k_base = k_iter * K_CHUNK;
        // Input A layout: [M, K] packed spikes (uint8_t)
        // Need to load M_TILE * K_CHUNK = 64 * 16 bytes = 1024 bytes.
        // We have 256 threads. Each thread loads 4 bytes (uint32_t).
        const int m_idx = tid / 4;       // 0..63
        const int k_idx = (tid % 4) * 4; // 0,4,8,12
        const int m_global = m_tile_base + m_idx;
        const uint8_t *src = inputs + (size_t)m_global * param.in_dim + k_base + k_idx;
        uint32_t smem_ptr = ptx::smem_u32addr(&smeminput[buf][m_idx * K_CHUNK + k_idx]);

        uint32_t packed = 0;
        if (m_global < param.in_ch)
        {
            int rem = (int)param.in_dim - (k_base + k_idx);
            if (rem >= 4) {
                ptx::ldg32_nc_0(packed, src, true);
            } else {
                for (int i = 0; i < rem; i++) {
                    packed |= ((uint32_t)src[i] << (i * 8));
                }
            }
        }
        ptx::sts32(packed, smem_ptr);
    };

    if (k_iters > 0)
    {
        load_weight(0, 0);
        load_input(0, 0);
        asm volatile("cp.async.commit_group;\n" :::);
    }

    for (int k_iter = 0; k_iter < k_iters; k_iter++)
    {
        const int cur = k_iter & 1;
        const int next = cur ^ 1;

        if (k_iter + 1 < k_iters)
        {
            load_weight(k_iter + 1, next);
            load_input(k_iter + 1, next);
            asm volatile("cp.async.commit_group;\n" :::);
        }

        asm volatile("cp.async.wait_group 1;\n" :::);
        __syncthreads();

#pragma unroll
        for (int k = 0; k < K_CHUNK; k++)
        {
            float weight_frag[8];
            // thread_m_base covers 2 rows? thread_m_base = warp_m * 16 + mma_tid_y * 2. So M offsets are +0 and +1.
            uint8_t in_val[2];
            in_val[0] = smeminput[cur][(thread_m_base + 0) * K_CHUNK + k];
            in_val[1] = smeminput[cur][(thread_m_base + 1) * K_CHUNK + k];

#pragma unroll
            for (int j = 0; j < 8; j++)
            {
                weight_frag[j] = smemweight[cur][(thread_n_base + j) * K_CHUNK + k];
            }

#pragma unroll
            for (int t = 0; t < T_STEPS; t++)
            {
#pragma unroll
                for (int i = 0; i < 2; i++)
                {
                    int spike = (in_val[i] >> t) & 1;
                    if (spike)
                    {
#pragma unroll
                        for (int j = 0; j < 8; j++)
                        {
                            output_frag[t][i][j] += weight_frag[j];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    asm volatile("cp.async.wait_all;\n" :::);

    // IF HardReset
    float v_state[2][8];
    uint8_t packed_out[2][8];

#pragma unroll
    for (int i = 0; i < 2; i++)
    {
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            v_state[i][j] = 0.f;
            packed_out[i][j] = 0;
        }
    }

#pragma unroll
    for (int t = 0; t < T_STEPS; t++)
    {
#pragma unroll
        for (int i = 0; i < 2; i++)
        {
#pragma unroll
            for (int j = 0; j < 8; j++)
            {
                v_state[i][j] += output_frag[t][i][j];
                int spike = (v_state[i][j] >= v_th) ? 1 : 0;
                packed_out[i][j] |= (uint8_t)(spike << t);
                v_state[i][j] -= (float)spike * (v_state[i][j] - v_reset);
            }
        }
    }

    // Epilogue Transpose
    uint8_t *warp_smem8 = reinterpret_cast<uint8_t *>(smem) + warp_id * 512;
    const int warp_m_global = m_tile_base + warp_m * 16;
    const int warp_n_global = n_tile_base + warp_n * 32;
    const int n_global = warp_n_global + lane_id;
    const bool n_valid = (n_global < param.out_dim);

#pragma unroll
    for (int i = 0; i < 2; i++)
    {
        const int smem_m = mma_tid_y * 2 + i;
        uint32_t lo = (uint32_t)packed_out[i][0] | ((uint32_t)packed_out[i][1] << 8) | ((uint32_t)packed_out[i][2] << 16) | ((uint32_t)packed_out[i][3] << 24);
        uint32_t hi = (uint32_t)packed_out[i][4] | ((uint32_t)packed_out[i][5] << 8) | ((uint32_t)packed_out[i][6] << 16) | ((uint32_t)packed_out[i][7] << 24);
        uint32_t addr_lo = ptx::smem_u32addr(&warp_smem8[smem_m * 32 + mma_tid_x * 8]);
        ptx::sts32(lo, addr_lo);
        ptx::sts32(hi, addr_lo + 4);
    }

    __syncthreads();

#pragma unroll
    for (int m_row = 0; m_row < 16; m_row++)
    {
        const int m_global = warp_m_global + m_row;
        const bool m_valid = (m_global < param.in_ch);
        const uint8_t val = warp_smem8[lane_id + m_row * 32];
        if (m_valid && n_valid) {
            outputs[(size_t)m_global * param.out_dim + n_global] = val;
        }
    }
}

// =============================================================================
// CPU Reference & Test
// =============================================================================

template <int T>
static void linear_sn_cpu_ref(
    const uint8_t *inputs,    // [M][K] packed spikes
    const float   *weights,   // [N][K] (unpadded)
    const float   *bias,      // [N]
    uint8_t       *outputs,   // [M][N] packed spikes
    int M, int K, int N,
    float v_th, float v_reset)
{
    memset(outputs, 0, (size_t)M * N);

    for (int m = 0; m < M; m++)
    {
        for (int n = 0; n < N; n++)
        {
            float v = 0.f;
            for (int t = 0; t < T; t++)
            {
                float sum = bias[n];
                for (int k = 0; k < K; k++)
                {
                    uint8_t pk = inputs[m * K + k];
                    if ((pk >> t) & 1)
                        sum += weights[n * K + k];
                }
                v += sum;
                int spike = (v >= v_th) ? 1 : 0;
                if (spike)
                {
                    outputs[m * N + n] |= (uint8_t)(1 << t);
                    v = v_reset;
                }
            }
        }
    }
}

template <int T>
void linear_sn_test(int M, int K, int N, float v_th, float v_reset, const char *label)
{
    constexpr int K_CHUNK = 16;
    int K_padded = (K + K_CHUNK - 1) / K_CHUNK * K_CHUNK;
    int N_padded = (N + 63) / 64 * 64;

    printf("  [%s] T=%d M=%d K=%d->%d N=%d->%d v_th=%.1f v_reset=%.1f  ",
           label, T, M, K, K_padded, N, N_padded, (double)v_th, (double)v_reset);

    size_t input_sz   = (size_t)M * K;
    size_t weight_sz  = (size_t)N * K;
    size_t weightP_sz = (size_t)N_padded * K_padded;
    size_t bias_sz    = (size_t)N;
    size_t output_sz  = (size_t)M * N;

    uint8_t *h_inputs   = new uint8_t[input_sz];
    float   *h_weights  = new float[weight_sz];
    float   *h_weightsP = new float[weightP_sz];
    float   *h_bias     = new float[bias_sz];
    uint8_t *h_outputs  = new uint8_t[output_sz];
    uint8_t *h_ref      = new uint8_t[output_sz];

    srand(42);
    for (size_t i = 0; i < input_sz; i++) {
        uint8_t packed = 0;
        for (int t = 0; t < T; t++)
            if (rand() & 1) packed |= (1u << t);
        h_inputs[i] = packed;
    }
    for (size_t i = 0; i < weight_sz; i++) h_weights[i] = (float)(rand() & 255) / 128.f - 1.f;
    for (size_t i = 0; i < bias_sz; i++) h_bias[i] = (float)((int)(i % 17) - 8) / 16.f;

    pad_weights_sn(h_weights, h_weightsP, K, N, K_padded, N_padded);

    uint8_t *d_inputs;
    float   *d_weightsP, *d_bias;
    uint8_t *d_outputs;
    cudaMalloc(&d_inputs,   input_sz   * sizeof(uint8_t));
    cudaMalloc(&d_weightsP, weightP_sz * sizeof(float));
    cudaMalloc(&d_bias,     bias_sz    * sizeof(float));
    cudaMalloc(&d_outputs,  output_sz  * sizeof(uint8_t));
    
    cudaMemcpy(d_inputs,   h_inputs,   input_sz   * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weightsP, h_weightsP, weightP_sz * sizeof(float),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias,     h_bias,     bias_sz    * sizeof(float),   cudaMemcpyHostToDevice);
    cudaMemset(d_outputs, 0, output_sz * sizeof(uint8_t));

    LinearParam param;
    param.in_ch = M; param.in_dim = K; param.out_dim = N;
    param.out_dim_padded = N_padded;
    param.inBatchNumel = 0; param.outBatchNumel = 0;

    dim3 block(256);
    dim3 grid((N + 63) / 64, (M + 63) / 64, 1);
    
    linear_sn_64x64_k16_u8<T><<<grid, block>>>(d_inputs, d_weightsP, d_bias, d_outputs, param, v_th, v_reset);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  CUDA error: %s\n", cudaGetErrorString(err));
    } else {
        cudaDeviceSynchronize();
        cudaMemcpy(h_outputs, d_outputs, output_sz * sizeof(uint8_t), cudaMemcpyDeviceToHost);

        linear_sn_cpu_ref<T>(h_inputs, h_weights, h_bias, h_ref, M, K, N, v_th, v_reset);

        int errors = 0;
        for (size_t i = 0; i < output_sz; i++) {
            if (h_outputs[i] != h_ref[i]) {
                if (errors < 5) printf("  Err[%zu]: gpu=0x%02x cpu=0x%02x\n", i, h_outputs[i], h_ref[i]);
                errors++;
            }
        }
        printf("  %s (%d errors)\n", errors == 0 ? "PASSED!" : "FAILED", errors);
    }

    cudaFree(d_inputs); cudaFree(d_weightsP); cudaFree(d_bias); cudaFree(d_outputs);
    delete[] h_inputs; delete[] h_weights; delete[] h_weightsP;
    delete[] h_bias; delete[] h_outputs; delete[] h_ref;
}

int main()
{
    printf("\n=== linear_sn_64x64_k16_u8 tests ===\n");
    const float V_TH = 1.0f;
    const float V_RESET = 0.0f;

    linear_sn_test<4>(64,  128, 64,  V_TH, V_RESET, "base aligned");
    linear_sn_test<4>(130, 72,  140, V_TH, V_RESET, "unaligned");
    linear_sn_test<8>(64,  64,  64,  V_TH, V_RESET, "T=8 aligned");
    linear_sn_test<1>(32,  256, 32,  V_TH, V_RESET, "T=1 aligned");

    return 0;
}