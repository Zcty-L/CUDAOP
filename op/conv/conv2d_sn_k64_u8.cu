#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>

#include "config.h"
#include "ptx_utils.cuh"

// =============================================================================
// snn_conv2d_lif_64x64_k16  —  Fused SNN Conv2D + LIF HardReset Neuron
//
// Inputs : uint8_t inputs [C_in][H][W]         — T spike bits/pixel (bit t = step t)
// Weights: float   weights [K_feat][C_out_pad]  — fp32, K_feat padded to 16-multiple,
//                                                  C_out padded to 64-multiple
// Outputs: uint8_t outputs [C_out][H_out*W_out] — T spike bits/pixel
//
// Architecture:
//   64×64 output tile (M=C_out, N=H_out*W_out), K_CHUNK=16 inner-product loop.
//   Double-buffered cp.async pipeline for weights; synchronous sts32 for inputs.
//   After accumulation, LIF HardReset is applied entirely in registers across T
//   time steps. The packed uint8 result is written via a single smem transpose
//   epilogue (one __syncthreads vs 2T in v1 float output).
//
// Output memory savings vs v1 float output:
//   T=4: 4× C_out × H_out × W_out bytes  vs  16× (float × T)  → 16× smaller
//   No intermediate float[T][C_out][H*W] buffer needed.
// =============================================================================

static void pad_weights_sn(
    const float *src, float *dst,
    int in_features, int C_out,
    int in_features_padded, int C_out_padded)
{
    for (int k = 0; k < in_features_padded; k++)
        for (int m = 0; m < C_out_padded; m++)
            dst[k * C_out_padded + m] =
                (k < in_features && m < C_out) ? src[k * C_out + m] : 0.f;
}


// =============================================================================
// Kernel
// =============================================================================

template <int T_STEPS, int Kh, int Kw, int Sh, int Sw, int Ph, int Pw>
__global__
void snn_conv2d_lif_64x64_k16(
    const uint8_t * __restrict__ inputs,
    const float   * __restrict__ weights,
    const float   * __restrict__ bias,
    uint8_t       * __restrict__ outputs,
    Conv2DParam param,
    int   out_ch_padded,
    float v_th,
    float v_reset,
    float tau)
{
    static_assert(T_STEPS >= 1 && T_STEPS <= 8,
                  "T_STEPS must be in [1,8] to fit in uint8 output");

    constexpr int K_CHUNK  = 16;
    constexpr int M_TILE   = 64;
    constexpr int N_TILE   = 64;
    constexpr int KhKw     = Kh * Kw;

    // 10 KB smem: 4KB×2 weight double-buffer, 1KB×2 input double-buffer.
    // The weight region (0..4KB) is reused by the uint8 epilogue transpose
    // (8 warps × 512 B = 4 KB) once the main loop finishes.
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

    // MMA thread coordinates (same layout as v1)
    const int mma_tid_x = lane_id / 16 * 2 + lane_id % 2;   // 0..3
    const int mma_tid_y = lane_id % 16 / 2;                   // 0..7

    const int warp_m = warp_id / 2;   // 0..3  (4 warps in M direction)
    const int warp_n = warp_id % 2;   // 0..1  (2 warps in N direction)

    const int thread_m_base = warp_m * 16 + mma_tid_y * 2;   // M offset in tile
    const int thread_n_base = warp_n * 32 + mma_tid_x * 8;   // N offset in tile

    const int m_tile_base = blockIdx.y * M_TILE;
    const int n_tile_base = blockIdx.x * N_TILE;

    if (tid < M_TILE)
    {
        const int m_global = m_tile_base + tid;
        smembias[tid] = (m_global < (int)param.out_ch) ? bias[m_global] : 0.f;
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
                output_frag[t][i][j] = smembias[thread_m_base + i];

    __syncthreads();

    const int in_features = param.inChKhKw;
    const int k_iters     = in_features / K_CHUNK;

    // ---- Weight loader (cp.async into smemweight) ----
    auto load_weight = [&](int k_iter, int buf)
    {
        const int k_base = k_iter * K_CHUNK;
        const int w_row  = tid / 16;        // 0..15
        const int w_col  = (tid % 16) * 4;  // 0,4,8,...,60
        uint32_t smem_ptr = ptx::smem_u32addr(
            &smemweight[buf][w_row * M_TILE + w_col]);
        const float *src = &weights[(k_base + w_row) * out_ch_padded
                                    + m_tile_base + w_col];
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 16;\n"
            :: "r"(smem_ptr), "l"(src)
        );
    };

    // ---- Input loader (synchronous sts32 into smeminput) ----
    auto load_input = [&](int k_iter, int buf)
    {
        const int k_base   = k_iter * K_CHUNK;
        const int i_k      = tid / 16;    // which of 16 K-rows this thread loads
        const int i_n4     = tid % 16;    // which group of 4 N positions
        const int global_k = k_base + i_k;

        uint32_t packed = 0;
        if (global_k < in_features)
        {
            if constexpr (KhKw == 1)
            {
                int c_idx = global_k;
#pragma unroll
                for (int b = 0; b < 4; b++)
                {
                    int n_idx = n_tile_base + i_n4 * 4 + b;
                    if (n_idx < (int)param.outHW)
                    {
                        int oh = n_idx / param.out_w;
                        int ow = n_idx % param.out_w;
                        int ih = oh * Sh - Ph;
                        int iw = ow * Sw - Pw;
                        packed |= (uint32_t)inputs[c_idx * param.inHW
                                                   + ih * param.in_w + iw]
                                  << (b * 8);
                    }
                }
            }
            else
            {
                int c_idx = global_k / KhKw;
                int ky    = global_k % KhKw / Kw;
                int kx    = global_k % KhKw % Kw;
#pragma unroll
                for (int b = 0; b < 4; b++)
                {
                    int n_idx = n_tile_base + i_n4 * 4 + b;
                    if (n_idx < (int)param.outHW)
                    {
                        int oh = n_idx / param.out_w;
                        int ow = n_idx % param.out_w;
                        int ih = oh * Sh - Ph + ky;
                        int iw = ow * Sw - Pw + kx;
                        if (ih >= 0 && ih < (int)param.in_h &&
                            iw >= 0 && iw < (int)param.in_w)
                        {
                            packed |= (uint32_t)inputs[c_idx * param.inHW
                                                       + ih * param.in_w + iw]
                                      << (b * 8);
                        }
                    }
                }
            }
        }
        uint32_t smeminput_ptr = ptx::smem_u32addr(
            &reinterpret_cast<uint32_t *>(smeminput[buf])[i_k * (N_TILE / 4) + i_n4]);
        ptx::sts32(packed, smeminput_ptr);
    };

    // ---- Prologue: prefetch iteration 0 ----
    if (k_iters > 0)
    {
        load_weight(0, 0);
        load_input(0, 0);
        asm volatile("cp.async.commit_group;\n" :::);
    }

    // ---- Main accumulation loop (double-buffered) ----
    for (int k_iter = 0; k_iter < k_iters; k_iter++)
    {
        const int cur  = k_iter & 1;
        const int next = cur ^ 1;

        if (k_iter + 1 < k_iters)
        {
            load_weight(k_iter + 1, next);
            load_input(k_iter + 1, next);
            asm volatile("cp.async.commit_group;\n" :::);
        }

        asm volatile("cp.async.wait_group 1;\n" :::);
        __syncthreads();

        // Issue both LDS back-to-back before the FP loop (OPT-1 from v1)
#pragma unroll
        for (int k = 0; k < K_CHUNK; k++)
        {
            float    weight_frag[2];
            uint32_t input_frag_lo, input_frag_hi;

            uint32_t w_addr = ptx::smem_u32addr(
                &smemweight[cur][k * M_TILE + thread_m_base]);
            ptx::lds64(weight_frag[0], weight_frag[1], w_addr);

            uint32_t i_addr = ptx::smem_u32addr(
                &smeminput[cur][k * N_TILE + thread_n_base]);
            ptx::lds64(input_frag_lo, input_frag_hi, i_addr);

#pragma unroll
            for (int t = 0; t < T_STEPS; t++)
            {
#pragma unroll
                for (int i = 0; i < 2; i++)
                {
#pragma unroll
                    for (int j = 0; j < 8; j++)
                    {
                        uint32_t word  = (j < 4) ? input_frag_lo : input_frag_hi;
                        int      shift = (j % 4) * 8 + t;
                        int      spike = (word >> shift) & 1;
                        ptx::add_f32(
                            output_frag[t][i][j], weight_frag[i], spike);
                    }
                }
            }
        }

        __syncthreads();
    }

    asm volatile("cp.async.wait_all;\n" :::);

    // =========================================================================
    // LIF neuron — leaky integrate-and-fire with HardReset
    //
    // t=0: v = (output_frag[0] + v_reset) * tau
    // t>0: v += (output_frag[t] - (v - v_reset)) * tau
    // =========================================================================
    float   v_state[2][8];
    uint8_t packed_out[2][8];

#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            v_state[i][j]    = 0.f;
            packed_out[i][j] = 0;
        }

    // t = 0
#pragma unroll
    for (int i = 0; i < 2; i++)
    {
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            v_state[i][j] = (output_frag[0][i][j] + v_reset) * tau;
            int spike = (v_state[i][j] >= v_th) ? 1 : 0;
            packed_out[i][j] |= (uint8_t)spike;
            v_state[i][j] -= (float)spike * (v_state[i][j] - v_reset);
        }
    }

    // t > 0
#pragma unroll
    for (int t = 1; t < T_STEPS; t++)
    {
#pragma unroll
        for (int i = 0; i < 2; i++)
        {
#pragma unroll
            for (int j = 0; j < 8; j++)
            {
                v_state[i][j] +=
                    (output_frag[t][i][j] - (v_state[i][j] - v_reset)) * tau;
                int spike = (v_state[i][j] >= v_th) ? 1 : 0;
                packed_out[i][j] |= (uint8_t)(spike << t);
                v_state[i][j] -= (float)spike * (v_state[i][j] - v_reset);
            }
        }
    }

    // =========================================================================
    // Epilogue: reg → smem → gmem  (single-pass, one __syncthreads)
    //
    // Smem layout (reuses smemweight[0], bytes 0..4095):
    //   Per warp: 512 bytes = 16 M-rows × 32 N-cols
    //   warp_smem8 = smem + warp_id * 512
    //
    // Write: each thread writes rows mma_tid_y*2 and mma_tid_y*2+1, cols
    //        mma_tid_x*8..mma_tid_x*8+7 using 4 × sts32 (packed 4 bytes each).
    //
    // Read:  each lane reads its N-column (lane_id) across 16 M-rows and
    //        stores 16 coalesced bytes to global (all 32 lanes → 1 cache line
    //        per M-row).
    // =========================================================================
    uint8_t *warp_smem8 = reinterpret_cast<uint8_t *>(smem) + warp_id * 512;

    const int warp_m_global = m_tile_base + warp_m * 16;
    const int warp_n_global = n_tile_base + warp_n * 32;
    const int n_global      = warp_n_global + lane_id;
    const bool n_valid      = (n_global < (int)param.outHW);

    // Write packed bytes to smem in [M-row][N-col] order
#pragma unroll
    for (int i = 0; i < 2; i++)
    {
        const int smem_m = mma_tid_y * 2 + i;   // M-row in warp tile [0..15]
        // N-col start for this thread: mma_tid_x * 8  (values 0, 8, 16, 24)
        uint32_t lo = (uint32_t)packed_out[i][0]
                    | ((uint32_t)packed_out[i][1] << 8)
                    | ((uint32_t)packed_out[i][2] << 16)
                    | ((uint32_t)packed_out[i][3] << 24);
        uint32_t hi = (uint32_t)packed_out[i][4]
                    | ((uint32_t)packed_out[i][5] << 8)
                    | ((uint32_t)packed_out[i][6] << 16)
                    | ((uint32_t)packed_out[i][7] << 24);
        uint32_t addr_lo = ptx::smem_u32addr(
            &warp_smem8[smem_m * 32 + mma_tid_x * 8]);
        ptx::sts32(lo, addr_lo);
        ptx::sts32(hi, addr_lo + 4);
    }

    __syncthreads();

    // Read back and store to global (32 coalesced byte writes per M-row)
#pragma unroll
    for (int m_row = 0; m_row < 16; m_row++)
    {
        const int  m_global = warp_m_global + m_row;
        const bool m_valid  = (m_global < (int)param.out_ch);
        const uint8_t val   = warp_smem8[lane_id + m_row * 32];
        ptx::stg8(val,
                  outputs + (size_t)m_global * param.outHW + n_global,
                  m_valid && n_valid);
    }
}


// =============================================================================
// Launch wrappers
// =============================================================================

template <int T, int Kh, int Kw, int Sh, int Sw, int Ph, int Pw>
void snn_conv2d_sn_launch(
    const uint8_t *d_inputs, const float *d_weights_padded,
    const float *d_bias, uint8_t *d_outputs,
    Conv2DParam &param, int out_ch_padded,
    float v_th, float v_reset, float tau)
{
    dim3 block(256);
    dim3 grid(
        ((int)param.outHW  + 63) / 64,
        ((int)param.out_ch + 63) / 64,
        1
    );
    snn_conv2d_lif_64x64_k16<T, Kh, Kw, Sh, Sw, Ph, Pw>
        <<<grid, block>>>(d_inputs, d_weights_padded, d_bias, d_outputs,
                          param, out_ch_padded, v_th, v_reset, tau);
}

void snn_conv2d_sn_1x1_s1_launch(
    const uint8_t *d_in, const float *d_w, const float *d_bias, uint8_t *d_out,
    Conv2DParam &param, int T, int out_ch_padded,
    float v_th, float v_reset, float tau)
{
    switch (T)
    {
        case 1:
            snn_conv2d_sn_launch<1, 1, 1, 1, 1, 0, 0>(
                d_in, d_w, d_bias, d_out, param, out_ch_padded,
                v_th, v_reset, tau);
            break;
        case 2:
            snn_conv2d_sn_launch<2, 1, 1, 1, 1, 0, 0>(
                d_in, d_w, d_bias, d_out, param, out_ch_padded,
                v_th, v_reset, tau);
            break;
        case 3:
            snn_conv2d_sn_launch<3, 1, 1, 1, 1, 0, 0>(
                d_in, d_w, d_bias, d_out, param, out_ch_padded,
                v_th, v_reset, tau);
            break;
        case 4:
            snn_conv2d_sn_launch<4, 1, 1, 1, 1, 0, 0>(
                d_in, d_w, d_bias, d_out, param, out_ch_padded,
                v_th, v_reset, tau);
            break;
    }
}

void snn_conv2d_sn_3x3_s1_launch(
    const uint8_t *d_in, const float *d_w, const float *d_bias, uint8_t *d_out,
    Conv2DParam &param, int T, int out_ch_padded,
    float v_th, float v_reset, float tau)
{
    switch (T)
    {
        case 1:
            snn_conv2d_sn_launch<1, 3, 3, 1, 1, 1, 1>(
                d_in, d_w, d_bias, d_out, param, out_ch_padded,
                v_th, v_reset, tau);
            break;
        case 2:
            snn_conv2d_sn_launch<2, 3, 3, 1, 1, 1, 1>(
                d_in, d_w, d_bias, d_out, param, out_ch_padded,
                v_th, v_reset, tau);
            break;
        case 3:
            snn_conv2d_sn_launch<3, 3, 3, 1, 1, 1, 1>(
                d_in, d_w, d_bias, d_out, param, out_ch_padded,
                v_th, v_reset, tau);
            break;
        case 4:
            snn_conv2d_sn_launch<4, 3, 3, 1, 1, 1, 1>(
                d_in, d_w, d_bias, d_out, param, out_ch_padded,
                v_th, v_reset, tau);
            break;
    }
}

void snn_conv2d_sn_3x3_s2_launch(
    const uint8_t *d_in, const float *d_w, const float *d_bias, uint8_t *d_out,
    Conv2DParam &param, int T, int out_ch_padded,
    float v_th, float v_reset, float tau)
{
    switch (T)
    {
        case 1:
            snn_conv2d_sn_launch<1, 3, 3, 2, 2, 1, 1>(
                d_in, d_w, d_bias, d_out, param, out_ch_padded,
                v_th, v_reset, tau);
            break;
        case 2:
            snn_conv2d_sn_launch<2, 3, 3, 2, 2, 1, 1>(
                d_in, d_w, d_bias, d_out, param, out_ch_padded,
                v_th, v_reset, tau);
            break;
        case 3:
            snn_conv2d_sn_launch<3, 3, 3, 2, 2, 1, 1>(
                d_in, d_w, d_bias, d_out, param, out_ch_padded,
                v_th, v_reset, tau);
            break;
        case 4:
            snn_conv2d_sn_launch<4, 3, 3, 2, 2, 1, 1>(
                d_in, d_w, d_bias, d_out, param, out_ch_padded,
                v_th, v_reset, tau);
            break;
    }
}


// =============================================================================
// CPU reference: conv + LIF HardReset → packed uint8 spikes
// =============================================================================

template <int Kh, int Kw, int Sh, int Sw, int Ph, int Pw>
static void snn_conv2d_lif_cpu_ref(
    const uint8_t *inputs,    // [C_in][H][W]       packed T spike bits
    const float   *weights,   // [K_feat][C_out]     (unpadded)
    const float   *bias,
    uint8_t       *outputs,   // [C_out][H_out*W_out] packed T spike bits
    int T, int C_in, int H, int W, int C_out,
    float v_th, float v_reset, float tau)
{
    constexpr int KhKw = Kh * Kw;
    int H_out = (H + 2 * Ph - Kh) / Sh + 1;
    int W_out = (W + 2 * Pw - Kw) / Sw + 1;

    memset(outputs, 0, (size_t)C_out * H_out * W_out);

    for (int m = 0; m < C_out; m++)
    {
        for (int oh = 0; oh < H_out; oh++)
        {
            for (int ow = 0; ow < W_out; ow++)
            {
                // Compute conv output per time step
                float conv_t[8] = {};
                for (int t = 0; t < T; t++)
                {
                    float sum = bias[m];
                    for (int c = 0; c < C_in; c++)
                        for (int ky = 0; ky < Kh; ky++)
                            for (int kx = 0; kx < Kw; kx++)
                            {
                                int ih = oh * Sh - Ph + ky;
                                int iw = ow * Sw - Pw + kx;
                                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                                uint8_t pk = inputs[c * H * W + ih * W + iw];
                                if ((pk >> t) & 1)
                                    sum += weights[(c * KhKw + ky * Kw + kx) * C_out + m];
                            }
                    conv_t[t] = sum;
                }
                // Apply LIF HardReset.
                float v = (conv_t[0] + v_reset) * tau;
                int spike = (v >= v_th) ? 1 : 0;
                if (spike)
                {
                    outputs[m * H_out * W_out + oh * W_out + ow] |= 1u;
                    v = v_reset;
                }

                for (int t = 1; t < T; t++)
                {
                    v += (conv_t[t] - (v - v_reset)) * tau;
                    spike = (v >= v_th) ? 1 : 0;
                    if (spike)
                    {
                        outputs[m * H_out * W_out + oh * W_out + ow] |=
                            (uint8_t)(1 << t);
                        v = v_reset;
                    }
                }
            }
        }
    }
}


// =============================================================================
// Test function
// =============================================================================

template <int T, int Kh, int Kw, int Sh, int Sw, int Ph, int Pw>
void snn_conv2d_sn_test(int C_in, int H, int W, int C_out,
                         float v_th, float v_reset, float tau,
                         const char *label)
{
    constexpr int KhKw    = Kh * Kw;
    constexpr int K_CHUNK = 16;

    int H_out          = (H + 2*Ph - Kh) / Sh + 1;
    int W_out          = (W + 2*Pw - Kw) / Sw + 1;
    int in_features    = C_in * KhKw;
    int in_feat_padded = (in_features + K_CHUNK - 1) / K_CHUNK * K_CHUNK;
    int C_out_padded   = (C_out + 63) / 64 * 64;

    std::cout << "  [" << label << "]"
              << " T=" << T
              << " C_in=" << C_in
              << " H=" << H
              << " C_out=" << C_out
              << " -> H_out=" << H_out
              << " v_th=" << std::fixed << std::setprecision(2) << v_th
              << " v_reset=" << v_reset
              << " tau=" << tau
              << "  ";

    size_t input_sz   = (size_t)C_in * H * W;
    size_t weight_sz  = (size_t)in_features * C_out;
    size_t weightP_sz = (size_t)in_feat_padded * C_out_padded;
    size_t bias_sz    = (size_t)C_out;
    size_t output_sz  = (size_t)C_out * H_out * W_out;

    uint8_t *h_inputs   = new uint8_t[input_sz];
    float   *h_weights  = new float[weight_sz];
    float   *h_weightsP = new float[weightP_sz];
    float   *h_bias     = new float[bias_sz];
    uint8_t *h_outputs  = new uint8_t[output_sz];
    uint8_t *h_ref      = new uint8_t[output_sz];

    srand(42);
    for (size_t i = 0; i < input_sz; i++)
    {
        uint8_t packed = 0;
        for (int t = 0; t < T; t++)
            if (rand() & 1) packed |= (1u << t);
        h_inputs[i] = packed;
    }
    for (size_t i = 0; i < weight_sz; i++)
        h_weights[i] = (float)(rand() & 255) / 128.f - 1.f;  // [-1, 1)
    for (size_t i = 0; i < bias_sz; i++)
        h_bias[i] = (float)((int)(i % 17) - 8) / 16.f;

    pad_weights_sn(h_weights, h_weightsP, in_features, C_out,
                   in_feat_padded, C_out_padded);

    uint8_t *d_inputs;
    float   *d_weightsP, *d_bias;
    uint8_t *d_outputs;
    cudaMalloc(&d_inputs,   input_sz   * sizeof(uint8_t));
    cudaMalloc(&d_weightsP, weightP_sz * sizeof(float));
    cudaMalloc(&d_bias,     bias_sz    * sizeof(float));
    cudaMalloc(&d_outputs,  output_sz  * sizeof(uint8_t));
    cudaMemset(d_outputs, 0, output_sz * sizeof(uint8_t));
    cudaMemcpy(d_inputs,   h_inputs,   input_sz   * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weightsP, h_weightsP, weightP_sz * sizeof(float),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias,     h_bias,     bias_sz    * sizeof(float),   cudaMemcpyHostToDevice);

    Conv2DParam param;
    param.in_h = H; param.in_w = W; param.inHW = H * W;
    param.inChKhKw = in_feat_padded; param.inBatchNumel = C_in * H * W;
    param.out_ch = C_out; param.out_w = W_out;
    param.outHW = H_out * W_out; param.outBatchNumel = C_out * H_out * W_out;
    param.Kh = Kh; param.Kw = Kw; param.KhKw = KhKw;
    param.Sh = Sh; param.Sw = Sw; param.Ph = Ph; param.Pw = Pw;

    snn_conv2d_sn_launch<T, Kh, Kw, Sh, Sw, Ph, Pw>(
        d_inputs, d_weightsP, d_bias, d_outputs, param, C_out_padded,
        v_th, v_reset, tau);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "[FAILED] CUDA error: "
                  << cudaGetErrorString(err) << '\n';
        goto cleanup;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_outputs, d_outputs, output_sz * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    snn_conv2d_lif_cpu_ref<Kh, Kw, Sh, Sw, Ph, Pw>(
        h_inputs, h_weights, h_bias, h_ref, T, C_in, H, W, C_out,
        v_th, v_reset, tau);

    {
        int errors = 0;
        for (size_t i = 0; i < output_sz; i++)
        {
            if (h_outputs[i] != h_ref[i])
            {
                if (errors < 5)
                {
                    std::cout << "\n    Err[" << i << "]: gpu=0x"
                              << std::hex << std::setw(2) << std::setfill('0')
                              << static_cast<int>(h_outputs[i])
                              << " cpu=0x" << std::setw(2)
                              << static_cast<int>(h_ref[i])
                              << std::dec << std::setfill(' ');
                }
                errors++;
            }
        }
        std::cout << (errors == 0 ? "[SUCCESS]" : "[FAILED]")
                  << " (" << errors << " errors)\n";
    }

cleanup:
    cudaFree(d_inputs); cudaFree(d_weightsP); cudaFree(d_bias); cudaFree(d_outputs);
    delete[] h_inputs; delete[] h_weights; delete[] h_weightsP;
    delete[] h_bias; delete[] h_outputs; delete[] h_ref;
}


// =============================================================================
// Benchmark: fused kernel timing
// =============================================================================

static float cuda_elapsed_ms(cudaEvent_t start, cudaEvent_t stop)
{
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

template <int T, int Kh, int Kw, int Sh, int Sw, int Ph, int Pw>
void snn_conv2d_sn_bench(int C_in, int H, int W, int C_out,
                          float v_th, float v_reset, float tau,
                          const char *label, int warmup = 5, int iters = 50)
{
    constexpr int KhKw    = Kh * Kw;
    constexpr int K_CHUNK = 16;

    int H_out          = (H + 2*Ph - Kh) / Sh + 1;
    int W_out          = (W + 2*Pw - Kw) / Sw + 1;
    int in_features    = C_in * KhKw;
    int in_feat_padded = (in_features + K_CHUNK - 1) / K_CHUNK * K_CHUNK;
    int C_out_padded   = (C_out + 63) / 64 * 64;

    size_t input_sz   = (size_t)C_in * H * W;
    size_t weightP_sz = (size_t)in_feat_padded * C_out_padded;
    size_t bias_sz    = (size_t)C_out;
    size_t output_sz  = (size_t)C_out * H_out * W_out;  // uint8

    uint8_t *d_inputs;   cudaMalloc(&d_inputs,   input_sz   * sizeof(uint8_t));
    float   *d_weightsP; cudaMalloc(&d_weightsP, weightP_sz * sizeof(float));
    float   *d_bias;     cudaMalloc(&d_bias,     bias_sz    * sizeof(float));
    uint8_t *d_outputs;  cudaMalloc(&d_outputs,  output_sz  * sizeof(uint8_t));

    {
        uint8_t *h_in = new uint8_t[input_sz];
        float   *h_wp = new float[weightP_sz];
        float   *h_bias = new float[bias_sz];
        srand(42);
        for (size_t i = 0; i < input_sz;   i++) h_in[i] = (uint8_t)(rand() & 0xFF);
        for (size_t i = 0; i < weightP_sz; i++) h_wp[i] = (float)(rand() & 255) / 128.f - 1.f;
        for (size_t i = 0; i < bias_sz;    i++) h_bias[i] = (float)((int)(i % 17) - 8) / 16.f;
        cudaMemcpy(d_inputs,   h_in, input_sz   * sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weightsP, h_wp, weightP_sz * sizeof(float),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_bias,     h_bias, bias_sz  * sizeof(float),   cudaMemcpyHostToDevice);
        delete[] h_in; delete[] h_wp; delete[] h_bias;
    }

    Conv2DParam param;
    param.in_h = H; param.in_w = W; param.inHW = H * W;
    param.inChKhKw = in_feat_padded; param.inBatchNumel = C_in * H * W;
    param.out_ch = C_out; param.out_w = W_out;
    param.outHW = H_out * W_out; param.outBatchNumel = C_out * H_out * W_out;
    param.Kh = Kh; param.Kw = Kw; param.KhKw = KhKw;
    param.Sh = Sh; param.Sw = Sw; param.Ph = Ph; param.Pw = Pw;

    auto run = [&]()
    {
        snn_conv2d_sn_launch<T, Kh, Kw, Sh, Sw, Ph, Pw>(
            d_inputs, d_weightsP, d_bias, d_outputs, param, C_out_padded,
            v_th, v_reset, tau);
    };

    for (int i = 0; i < warmup; i++) run();
    cudaDeviceSynchronize();

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start); cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start);
    for (int i = 0; i < iters; i++) run();
    cudaEventRecord(ev_stop);
    cudaDeviceSynchronize();
    float t_fused = cuda_elapsed_ms(ev_start, ev_stop) / iters;

    // Effective FLOPS: T * C_out * H_out * W_out * in_features * 2 (MAC)
    double flops   = (double)T * C_out * H_out * W_out * in_features * 2.0;
    double gflops  = flops / (t_fused * 1e-3) / 1e9;
    // Effective memory: inputs (read once) + weights (read once) + outputs (written once)
    double gmem_bytes = (double)input_sz
                      + (double)weightP_sz * sizeof(float)
                      + (double)bias_sz * sizeof(float)
                      + (double)output_sz * sizeof(uint8_t);  // 1B per output, not 4B*T
    double bw = gmem_bytes / (t_fused * 1e-3) / 1e9;

    std::cout << "  " << std::left << std::setw(30) << label
              << std::right
              << " T=" << T
              << " K=" << Kh << 'x' << Kw
              << " s=" << Sh
              << " in=" << in_features << "->" << in_feat_padded
              << " C_out=" << C_out
              << " fused: " << std::fixed << std::setprecision(3)
              << t_fused << " ms"
              << "  " << std::setprecision(1) << gflops << " GFLOPS"
              << "  " << bw << " GB/s"
              << "  (output: " << output_sz << " B vs "
              << static_cast<double>(output_sz) * T * sizeof(float)
              << " B float[T])\n";

    cudaFree(d_inputs); cudaFree(d_weightsP); cudaFree(d_bias); cudaFree(d_outputs);
    cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
}

int main()
{
    const float V_TH    = 1.0f;
    const float V_RESET = 0.0f;
    const float TAU     = 0.5f;

    std::cout << "\n=== snn_conv2d_lif_64x64_k16 correctness tests ===\n";

    std::cout << "\n--- 1x1, s=1, p=0 ---\n";
    snn_conv2d_sn_test<4, 1,1,1,1,0,0>( 64,  80, 80,  64, V_TH, V_RESET, TAU, "base");
    snn_conv2d_sn_test<4, 1,1,1,1,0,0>(128,  40, 40, 128, V_TH, V_RESET, TAU, "C128");
    snn_conv2d_sn_test<2, 1,1,1,1,0,0>( 64,  80, 80,  48, V_TH, V_RESET, TAU, "C_out boundary");
    snn_conv2d_sn_test<1, 1,1,1,1,0,0>( 64,  80, 80,  64, V_TH, V_RESET, TAU, "T=1");
    snn_conv2d_sn_test<4, 1,1,1,1,0,0>( 64,  40, 40, 128, V_TH, V_RESET, TAU, "expand x2");
    snn_conv2d_sn_test<4, 1,1,1,1,0,0>(128,  40, 40,  64, V_TH, V_RESET, TAU, "squeeze x0.5");

    std::cout << "\n--- 3x3, s=1, p=1 ---\n";
    snn_conv2d_sn_test<4, 3,3,1,1,1,1>( 64,  80, 80,  64, V_TH, V_RESET, TAU, "base");
    snn_conv2d_sn_test<4, 3,3,1,1,1,1>( 32,  40, 40,  32, V_TH, V_RESET, TAU, "small");
    snn_conv2d_sn_test<2, 3,3,1,1,1,1>( 32,  43, 43,  48, V_TH, V_RESET, TAU, "boundary");
    snn_conv2d_sn_test<1, 3,3,1,1,1,1>( 64,  80, 80,  64, V_TH, V_RESET, TAU, "T=1");
    snn_conv2d_sn_test<3, 3,3,1,1,1,1>( 64,  80, 80,  64, V_TH, V_RESET, TAU, "T=3");

    std::cout << "\n--- 3x3, s=2, p=1 ---\n";
    snn_conv2d_sn_test<4, 3,3,2,2,1,1>( 64,  80, 80,  64, V_TH, V_RESET, TAU, "H_out=40");
    snn_conv2d_sn_test<4, 3,3,2,2,1,1>( 32,  80, 80,  64, V_TH, V_RESET, TAU, "C expand");
    snn_conv2d_sn_test<4, 3,3,2,2,1,1>( 16,  40, 40,  32, V_TH, V_RESET, TAU, "small H=40");
    snn_conv2d_sn_test<2, 3,3,2,2,1,1>( 32,  43, 43, 128, V_TH, V_RESET, TAU, "boundary");

    // Test non-zero v_reset
    std::cout << "\n--- v_reset != 0 ---\n";
    snn_conv2d_sn_test<4, 3,3,1,1,1,1>( 64,  40, 40,  64, 1.0f, -0.1f, TAU, "v_reset=-0.1");
    snn_conv2d_sn_test<4, 1,1,1,1,0,0>( 64,  40, 40,  64, 0.5f,  0.0f, 0.25f, "tau=0.25");

    std::cout << "\n=== snn_conv2d_lif_64x64_k16 benchmark ===\n"
              << "  (fused conv+LIF; output is uint8[C_out][H*W] "
              << "packed T-bit spikes)\n\n";

    // 1x1
    snn_conv2d_sn_bench< 4, 1,1,1,1,0,0>( 64,  80, 80,  64, V_TH, V_RESET, TAU, "1x1 C64 H80");
    snn_conv2d_sn_bench< 4, 1,1,1,1,0,0>(128,  40, 40, 128, V_TH, V_RESET, TAU, "1x1 C128 H40");
    snn_conv2d_sn_bench< 4, 1,1,1,1,0,0>(256,  20, 20, 128, V_TH, V_RESET, TAU, "1x1 C256->128 H20");
    // 3x3 s=1
    snn_conv2d_sn_bench< 4, 3,3,1,1,1,1>( 64,  80, 80,  64, V_TH, V_RESET, TAU, "3x3s1 C64 H80");
    snn_conv2d_sn_bench< 4, 3,3,1,1,1,1>( 64,  40, 40,  64, V_TH, V_RESET, TAU, "3x3s1 C64 H40");
    snn_conv2d_sn_bench< 4, 3,3,1,1,1,1>(128,  20, 20, 128, V_TH, V_RESET, TAU, "3x3s1 C128 H20");
    // 3x3 s=2
    snn_conv2d_sn_bench< 4, 3,3,2,2,1,1>( 64,  80, 80,  64, V_TH, V_RESET, TAU, "3x3s2 C64 H80->40");
    snn_conv2d_sn_bench< 4, 3,3,2,2,1,1>( 32,  80, 80,  64, V_TH, V_RESET, TAU, "3x3s2 C32->64 H80");
    // T=1,2
    snn_conv2d_sn_bench< 1, 3,3,1,1,1,1>( 64,  80, 80,  64, V_TH, V_RESET, TAU, "T=1 3x3s1 C64 H80");
    snn_conv2d_sn_bench< 2, 3,3,1,1,1,1>( 64,  80, 80,  64, V_TH, V_RESET, TAU, "T=2 3x3s1 C64 H80");
    snn_conv2d_sn_bench< 2, 1,1,1,1,0,0>(128,  40, 40, 128, V_TH, V_RESET, TAU, "T=2 1x1 C128 H40");

    std::cout << "\n=== Done ===\n";
    return 0;
}
