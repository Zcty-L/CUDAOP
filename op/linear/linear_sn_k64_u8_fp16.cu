#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cuda_utils.cuh"
#include "ptx_utils.cuh"

// =============================================================================
// linear_sn_if_64x64_k16_fp16 — Fused SNN Linear + IF HardReset (fp16)
//
// Based on conv2d_sn_k64_u8_fp16.cu architecture.
//
// Architecture:
//   64×64 output tile: 64 Batch (M) × 64 Output Features (N)
//   Reduction over K (In Features) in chunks of 16.
//   Uses __half2 for processing 2 N-channels per thread.
//   Weight Matrix: [K_padded, N_padded] FP16.
//   Input Matrix:  [M, K] packed uint8 spikes.
//   Output Matrix: [M, N] packed uint8 spikes.
// =============================================================================

struct LinearK64Param
{
    uint32_t M;              // Batch
    uint32_t K;              // In Features
    uint32_t N;              // Out Features
    uint32_t N_padded;       // N padded to multiple of 64
};

template <int T_STEPS>
__global__ __launch_bounds__(256, 3)
void linear_sn_if_64x64_k16_fp16(
    const uint8_t * __restrict__ inputs,   // [M, K]
    const __half  * __restrict__ weights,  // [K_pad, N_pad]
    const __half2 * __restrict__ bias,     // [N_pad / 2]
    uint8_t       * __restrict__ outputs,  // [M, N]
    LinearK64Param param,
    float v_th_f,
    float v_reset_f)
{
    static_assert(T_STEPS >= 1 && T_STEPS <= 8);

    constexpr int K_CHUNK = 16;
    constexpr int M_TILE  = 64; // Logical N in kernel logic (Out Channels)
    constexpr int N_TILE  = 64; // Logical M in kernel logic (Batch)

    // 6 KB smem: 2KB×2 weight double-buffer, 1KB×2 input double-buffer
    __shared__ __align__(128) char smem[6 * 1024];

    __half  *smemweight[2];
    smemweight[0] = reinterpret_cast<__half *>(smem);
    smemweight[1] = reinterpret_cast<__half *>(smem + 2 * 1024);

    uint8_t *smeminput[2];
    smeminput[0] = reinterpret_cast<uint8_t *>(smem + 4 * 1024);
    smeminput[1] = reinterpret_cast<uint8_t *>(smem + 5 * 1024);

    const int tid     = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;

    const int mma_tid_x = lane_id / 16 * 2 + lane_id % 2;
    const int mma_tid_y = lane_id % 16 / 2;

    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;

    const int thread_m_base = warp_m * 16 + mma_tid_y * 2; // Offset in M_TILE (N channels)
    const int thread_n_base = warp_n * 32 + mma_tid_x * 8; // Offset in N_TILE (M batch)

    const int m_tile_base = blockIdx.y * M_TILE; // Global N base
    const int n_tile_base = blockIdx.x * N_TILE; // Global M base

    // --- Bias: single half2 load ---
    int m_global = m_tile_base + thread_m_base;
    __half2 bias_h2 = __float2half2_rn(0.f);
    if (m_global < (int)param.N)
        bias_h2 = bias[m_global >> 1];

    // __half2 accumulators: .x = Channel 0, .y = Channel 1
    __half2 output_frag[T_STEPS][8];
#pragma unroll
    for (int t = 0; t < T_STEPS; t++)
#pragma unroll
        for (int j = 0; j < 8; j++)
            output_frag[t][j] = bias_h2;

    const int k_iters = (param.K + K_CHUNK - 1) / K_CHUNK;

    auto load_weight = [&](int k_iter, int buf)
    {
        const int k_base = k_iter * K_CHUNK;
        const int w_row  = tid / 16;      // 0..15 (K)
        const int w_col  = (tid % 16) * 4; // 0..60 (N)
        uint32_t smem_ptr = ptx::smem_u32addr(&smemweight[buf][w_row * M_TILE + w_col]);
        const __half *src = &weights[(size_t)(k_base + w_row) * param.N_padded + m_tile_base + w_col];
        
        bool guard = (k_base + w_row < param.K);
        // Load 8 bytes (4 halves)
        asm volatile (
            "{.reg .pred p;\n"
            " setp.ne.b32 p, %2, 0;\n"
            " @p cp.async.ca.shared.global [%0], [%1], 8;}\n"
            :: "r"(smem_ptr), "l"(src), "r"((int)guard));
    };

    auto load_input = [&](int k_iter, int buf)
    {
        const int k_base = k_iter * K_CHUNK;
        const int i_k    = tid / 16;      // 0..15 (K)
        const int i_n4   = tid % 16;      // 0..15 (M / 4)
        const int global_k = k_base + i_k;

        uint32_t packed = 0;
        if (global_k < param.K)
        {
            #pragma unroll
            for (int b = 0; b < 4; b++) 
            {
                int global_m = n_tile_base + i_n4 * 4 + b;
                if (global_m < (int)param.M)
                {
                    packed |= (uint32_t)inputs[(size_t)global_m * param.K + global_k] << (b * 8);
                }
            }
        }
        ptx::sts32(packed, ptx::smem_u32addr(&reinterpret_cast<uint32_t *>(smeminput[buf])[i_k * (N_TILE/4) + i_n4]));
    };

    if (k_iters > 0) 
    {
        load_weight(0, 0);
        load_input(0, 0);
        asm volatile("cp.async.commit_group;\n" :::);
    }

    for (int k_iter = 0; k_iter < k_iters; k_iter++)
    {
        const int cur = k_iter & 1, next = cur ^ 1;
        if (k_iter + 1 < k_iters) 
        {
            load_weight(k_iter+1, next);
            load_input(k_iter+1, next);
            asm volatile("cp.async.commit_group;\n" :::);
        }
        asm volatile("cp.async.wait_group 1;\n" :::);
        __syncthreads();

#pragma unroll
        for (int k = 0; k < K_CHUNK; k++)
        {
            uint32_t w_raw;
            ptx::lds32(w_raw, ptx::smem_u32addr(&smemweight[cur][k * M_TILE + thread_m_base]));
            __half2 weight2 = *reinterpret_cast<__half2 *>(&w_raw);

            uint32_t input_lo, input_hi;
            ptx::lds64(input_lo, input_hi, ptx::smem_u32addr(&smeminput[cur][k * N_TILE + thread_n_base]));

#pragma unroll
            for (int t = 0; t < T_STEPS; t++)
#pragma unroll
                for (int j = 0; j < 8; j++) 
                {
                    uint32_t word  = (j < 4) ? input_lo : input_hi;
                    int      spike = (word >> ((j%4)*8 + t)) & 1;
                    ptx::add_f16x2(output_frag[t][j], weight2, spike);
                }
        }
        __syncthreads();
    }
    asm volatile("cp.async.wait_all;\n" :::);

    // --- IF HardReset ---
    const __half v_th_h    = __float2half(v_th_f);
    const __half v_reset_h = __float2half(v_reset_f);

    __half2  v_state[8];
    uint8_t  packed_row0[8], packed_row1[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        v_state[j]    = __float2half2_rn(0.f);
        packed_row0[j] = 0;
        packed_row1[j] = 0;
    }

#pragma unroll
    for (int t = 0; t < T_STEPS; t++)
    {
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            v_state[j] = __hadd2(v_state[j], output_frag[t][j]);

            int s0 = __hge(v_state[j].x, v_th_h) ? 1 : 0;
            int s1 = __hge(v_state[j].y, v_th_h) ? 1 : 0;

            packed_row0[j] |= (uint8_t)(s0 << t);
            packed_row1[j] |= (uint8_t)(s1 << t);

            v_state[j].x = s0 ? v_reset_h : v_state[j].x;
            v_state[j].y = s1 ? v_reset_h : v_state[j].y;
        }
    }

    // --- Epilogue Transpose ---
    uint8_t *warp_smem8   = reinterpret_cast<uint8_t *>(smem) + warp_id * 512;
    const int warp_m_global = n_tile_base + warp_n * 32; // Batch dimension
    const int warp_n_global = m_tile_base + warp_m * 16; // Feature dimension
    const int m_global_out  = warp_m_global + lane_id;
    const bool m_valid      = (m_global_out < (int)param.M);

    const int smem_n0 = mma_tid_y * 2;
    const int smem_n1 = mma_tid_y * 2 + 1;
    const int smem_m  = mma_tid_x * 8;

    uint32_t lo0 = (uint32_t)packed_row0[0] | ((uint32_t)packed_row0[1] << 8) | ((uint32_t)packed_row0[2] << 16) | ((uint32_t)packed_row0[3] << 24);
    uint32_t hi0 = (uint32_t)packed_row0[4] | ((uint32_t)packed_row0[5] << 8) | ((uint32_t)packed_row0[6] << 16) | ((uint32_t)packed_row0[7] << 24);
    uint32_t lo1 = (uint32_t)packed_row1[0] | ((uint32_t)packed_row1[1] << 8) | ((uint32_t)packed_row1[2] << 16) | ((uint32_t)packed_row1[3] << 24);
    uint32_t hi1 = (uint32_t)packed_row1[4] | ((uint32_t)packed_row1[5] << 8) | ((uint32_t)packed_row1[6] << 16) | ((uint32_t)packed_row1[7] << 24);

    uint32_t a0 = ptx::smem_u32addr(&warp_smem8[smem_m * 32 + smem_n0]);
    uint32_t a1 = ptx::smem_u32addr(&warp_smem8[smem_m * 32 + smem_n1]);
    
    // Note: Re-indexing for Batch-Major output [M, N]
    // smem region [32 rows (Batch) x 16 cols (N features)]?
    // Wait, the smem transpose region is 16 rows (N features) x 32 cols (Batch) for one warp.
    // thread (mma_tid_y, mma_tid_x) handles 2 N-channels and 8 Batch-samples.
    // So writing to smem: row index = channel index, col index = sample index.
    ptx::sts32(lo0, ptx::smem_u32addr(&warp_smem8[smem_n0 * 32 + smem_m]));
    ptx::sts32(hi0, ptx::smem_u32addr(&warp_smem8[smem_n0 * 32 + smem_m + 4]));
    ptx::sts32(lo1, ptx::smem_u32addr(&warp_smem8[smem_n1 * 32 + smem_m]));
    ptx::sts32(hi1, ptx::smem_u32addr(&warp_smem8[smem_n1 * 32 + smem_m + 4]));

    __syncthreads();

#pragma unroll
    for (int n_row = 0; n_row < 16; n_row++) {
        const int  n_global = warp_n_global + n_row;
        const bool n_valid  = (n_global < (int)param.N);
        const uint8_t val   = warp_smem8[n_row * 32 + lane_id];
        if (m_valid && n_valid) {
            outputs[(size_t)m_global_out * param.N + n_global] = val;
        }
    }
}

// =============================================================================
// CPU Reference & Test
// =============================================================================

static void linear_sn_if_fp16_cpu_ref(
    const uint8_t *inputs,    // [M, K] packed
    const __half  *h_weightsP, // [K_pad, N_pad]
    uint8_t       *outputs,   // [M, N] packed
    int T, int M, int K, int N,
    int N_padded,
    const float *bias,
    float v_th, float v_reset)
{
    memset(outputs, 0, (size_t)M * N);
    for (int m = 0; m < M; m++)
    {
        for (int n = 0; n < N; n++)
        {
            float conv_t[8] = {};
            for (int t = 0; t < T; t++)
            {
                __half acc_h = __float2half(0.f);
                for (int k = 0; k < K; k++)
                {
                    if ((inputs[m * K + k] >> t) & 1)
                    {
                        float new_acc = __half2float(acc_h) + __half2float(h_weightsP[k * N_padded + n]);
                        acc_h = __float2half(new_acc);
                    }
                }
                conv_t[t] = __half2float(acc_h);
            }

            __half v_h = __float2half(0.f);
            for (int t = 0; t < T; t++)
            {
                // Bias added every step to match GPU logic
                float new_v = __half2float(v_h) + conv_t[t] + bias[n];
                v_h = __float2half(new_v);
                int spike = (__half2float(v_h) >= v_th) ? 1 : 0;
                if (spike) {
                    outputs[m * N + n] |= (uint8_t)(1 << t);
                    v_h = __float2half(v_reset);
                }
            }
        }
    }
}

void linear_sn_fp16_test(int M, int K, int N, int T, float v_th, float v_reset, const char *label)
{
    int N_padded = (N + 63) / 64 * 64;
    int K_padded = (K + 15) / 16 * 16;

    printf("  [%s] T=%d M=%d K=%d N=%d  ", label, T, M, K, N);

    size_t in_sz  = (size_t)M * K;
    size_t wt_sz  = (size_t)K_padded * N_padded;
    size_t out_sz = (size_t)M * N;

    uint8_t *h_in   = new uint8_t[in_sz];
    __half  *h_wt   = new __half[wt_sz];
    float   *h_bias = new float[N];
    __half2 *h_bias2 = new __half2[N_padded / 2];
    uint8_t *h_out  = new uint8_t[out_sz];
    uint8_t *h_ref  = new uint8_t[out_sz];

    srand(42);
    for (size_t i = 0; i < in_sz; i++) {
        uint8_t p = 0;
        for (int t = 0; t < T; t++) if (rand() & 1) p |= (1 << t);
        h_in[i] = p;
    }
    for (size_t i = 0; i < wt_sz; i++) h_wt[i] = __float2half((float)(rand() & 255) / 256.f - 0.5f);
    for (int i = 0; i < N; i++) h_bias[i] = (float)(rand() & 63) / 64.f;
    for (int i = 0; i < N_padded; i += 2)
        h_bias2[i >> 1] = __half2{__float2half(h_bias[i]), __float2half(i + 1 < N ? h_bias[i+1] : 0.f)};

    uint8_t *d_in; __half *d_wt; __half2 *d_bias; uint8_t *d_out;
    cudaMalloc(&d_in, in_sz); cudaMalloc(&d_wt, wt_sz * sizeof(__half));
    cudaMalloc(&d_bias, (N_padded/2) * sizeof(__half2)); cudaMalloc(&d_out, out_sz);
    
    cudaMemcpy(d_in, h_in, in_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_wt, h_wt, wt_sz * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias2, (N_padded/2) * sizeof(__half2), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, out_sz);

    LinearK64Param param = {(uint32_t)M, (uint32_t)K, (uint32_t)N, (uint32_t)N_padded};
    
    dim3 block(256);
    dim3 grid((M + 63) / 64, (N + 63) / 64, 1);
    
    switch(T) {
        case 1: linear_sn_if_64x64_k16_fp16<1><< <grid, block>>>(d_in, d_wt, d_bias, d_out, param, v_th, v_reset); break;
        case 2: linear_sn_if_64x64_k16_fp16<2><< <grid, block>>>(d_in, d_wt, d_bias, d_out, param, v_th, v_reset); break;
        case 4: linear_sn_if_64x64_k16_fp16<4><< <grid, block>>>(d_in, d_wt, d_bias, d_out, param, v_th, v_reset); break;
        case 8: linear_sn_if_64x64_k16_fp16<8><< <grid, block>>>(d_in, d_wt, d_bias, d_out, param, v_th, v_reset); break;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, out_sz, cudaMemcpyDeviceToHost);

    linear_sn_if_fp16_cpu_ref(h_in, h_wt, h_ref, T, M, K, N, N_padded, h_bias, v_th, v_reset);

    int errors = 0;
    for (size_t i = 0; i < out_sz; i++) {
        if (h_out[i] != h_ref[i]) {
            if (errors < 5) printf("\n    Err[%zu] gpu=0x%02x ref=0x%02x", i, h_out[i], h_ref[i]);
            errors++;
        }
    }
    printf("  %s (%d errors)\n", errors == 0 ? "PASSED!" : "FAILED", errors);

    cudaFree(d_in); cudaFree(d_wt); cudaFree(d_bias); cudaFree(d_out);
    delete[] h_in; delete[] h_wt; delete[] h_bias; delete[] h_bias2; delete[] h_out; delete[] h_ref;
}

int main()
{
    const float V_TH = 1.0f;
    const float V_RESET = 0.0f;

    printf("\n=== linear_sn_if_64x64_k16_fp16 tests ===\n");
    linear_sn_fp16_test(64, 128, 64, 4, V_TH, V_RESET, "aligned");
    linear_sn_fp16_test(130, 72, 140, 4, V_TH, V_RESET, "unaligned");
    linear_sn_fp16_test(64, 64, 64, 8, V_TH, V_RESET, "T=8");
    linear_sn_fp16_test(32, 256, 32, 1, V_TH, V_RESET, "T=1");

    return 0;
}
