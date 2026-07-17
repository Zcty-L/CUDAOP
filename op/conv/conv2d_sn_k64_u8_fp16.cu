#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "config.h"
#include "ptx_utils.cuh"

// =============================================================================
// snn_conv2d_lif_64x64_k16_fp16 — Fused SNN Conv2D + LIF HardReset (fp16)
//
// Based on conv2d_k64_specialized_fp16acc_v2, with LIF neuron fused in-register.
//
// Key differences vs fp16acc_v2:
//   output_frag[T][8]  __half2  →  consumed by LIF loop, freed before epilogue
//   v_state[8]         __half2  →  membrane potential, .x=row0  .y=row1
//   packed_row0/1[8]   uint8    →  T-bit packed spike output per output position
//   epilogue           stg8     →  uint8[C_out][H_out*W_out]  (vs stg16 half[T]...)
//
// Smem: 6 KB (identical to v2)
//   k-loop:   2KB×2 weight + 1KB×2 input
//   epilogue: reuses first 4KB (weight region) for uint8 transpose
//             8 warps × 512 B = 4 KB
// =============================================================================

template <int T_STEPS, int Kh, int Kw, int Sh, int Sw, int Ph, int Pw>
__global__ __launch_bounds__(256, 3)
void snn_conv2d_lif_64x64_k16_fp16(
    const uint8_t * __restrict__ inputs,
    const __half  * __restrict__ weights,
    const __half2 * __restrict__ bias,
    uint8_t       * __restrict__ outputs,
    Conv2DParam param,
    int   out_ch_padded,
    float v_th_f,
    float tau_f)
{
    static_assert(T_STEPS >= 1 && T_STEPS <= 8);

    constexpr int K_CHUNK = 16;
    constexpr int M_TILE  = 64;
    constexpr int N_TILE  = 64;
    constexpr int KhKw    = Kh * Kw;

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

    const int thread_m_base = warp_m * 16 + mma_tid_y * 2;
    const int thread_n_base = warp_n * 32 + mma_tid_x * 8;

    const int m_tile_base = blockIdx.y * M_TILE;
    const int n_tile_base = blockIdx.x * N_TILE;

    // --- Bias: single half2 load from pre-packed bias ---
    int m_global = m_tile_base + thread_m_base;
    __half2 bias_h2 = __float2half2_rn(0.f);
    if (m_global < (int)param.out_ch)
        bias_h2 = bias[m_global >> 1];

    // __half2 accumulators: .x = M-row 0, .y = M-row 1  (same as fp16acc_v2)
    __half2 output_frag[T_STEPS][8];
#pragma unroll
    for (int t = 0; t < T_STEPS; t++)
#pragma unroll
        for (int j = 0; j < 8; j++)
            output_frag[t][j] = bias_h2;

    const int in_features = param.inChKhKw;
    const int k_iters     = in_features / K_CHUNK;

    auto load_weight = [&](int k_iter, int buf)
    {
        const int k_base = k_iter * K_CHUNK;
        const int w_row  = tid / 16;
        const int w_col  = (tid % 16) * 4;
        uint32_t smem_ptr = ptx::smem_u32addr(
            &smemweight[buf][w_row * M_TILE + w_col]);
        const __half *src = &weights[(k_base + w_row) * out_ch_padded
                                     + m_tile_base + w_col];
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n"
                     :: "r"(smem_ptr), "l"(src));
    };

    auto load_input = [&](int k_iter, int buf)
    {
        const int k_base   = k_iter * K_CHUNK;
        const int i_k      = tid / 16;
        const int i_n4     = tid % 16;
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
                        int oh = n_idx / param.out_w, ow = n_idx % param.out_w;
                        packed |= (uint32_t)inputs[c_idx * param.inHW
                                                   + (oh*Sh-Ph)*param.in_w + (ow*Sw-Pw)]
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
                        int oh = n_idx / param.out_w, ow = n_idx % param.out_w;
                        int ih = oh*Sh - Ph + ky, iw = ow*Sw - Pw + kx;
                        if (ih >= 0 && ih < (int)param.in_h &&
                            iw >= 0 && iw < (int)param.in_w)
                            packed |= (uint32_t)inputs[c_idx * param.inHW
                                                       + ih * param.in_w + iw]
                                      << (b * 8);
                    }
                }
            }
        }
        ptx::sts32(packed, ptx::smem_u32addr(
            &reinterpret_cast<uint32_t *>(smeminput[buf])[i_k * (N_TILE/4) + i_n4]));
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
            // One lds32 → __half2{weight_row0, weight_row1}
            uint32_t w_raw;
            ptx::lds32(w_raw, ptx::smem_u32addr(
                &smemweight[cur][k * M_TILE + thread_m_base]));
            __half2 weight2 = *reinterpret_cast<__half2 *>(&w_raw);

            uint32_t input_lo, input_hi;
            ptx::lds64(input_lo, input_hi, ptx::smem_u32addr(
                &smeminput[cur][k * N_TILE + thread_n_base]));

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

    // =========================================================================
    // LIF neuron — in __half2 registers, v_reset fixed to 0
    //
    // t=0: v = output_frag[0] * tau
    // t>0: v += (output_frag[t] - v) * tau
    // =========================================================================
    const __half2 v_th2 = __float2half2_rn(v_th_f);
    const __half2 tau2  = __float2half2_rn(tau_f);

    __half2 v_state[8];
#pragma unroll
    for (int j = 0; j < 8; j++)
    {
        v_state[j] = __float2half2_rn(0.f);
    }

    uint32_t lo0 = 0;
    uint32_t hi0 = 0;
    uint32_t lo1 = 0;
    uint32_t hi1 = 0;

    // t = 0
#pragma unroll
    for (int j = 0; j < 8; j++)
    {
        v_state[j] = __hmul2(output_frag[0][j], tau2);

        uint32_t spike_mask = __hge2_mask(v_state[j], v_th2);
        int s0 = spike_mask & 1;
        int s1 = (spike_mask >> 16) & 1;

        if (j < 4)
        {
            lo0 |= (uint32_t)s0 << (j * 8);
            lo1 |= (uint32_t)s1 << (j * 8);
        }
        else
        {
            hi0 |= (uint32_t)s0 << ((j - 4) * 8);
            hi1 |= (uint32_t)s1 << ((j - 4) * 8);
        }

        uint32_t state_bits = *reinterpret_cast<uint32_t *>(&v_state[j]);
        state_bits &= ~spike_mask;
        v_state[j] = *reinterpret_cast<__half2 *>(&state_bits);
    }

    // t > 0
#pragma unroll
    for (int t = 1; t < T_STEPS; t++)
    {
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            v_state[j] = __hadd2(
                v_state[j],
                __hmul2(__hsub2(output_frag[t][j], v_state[j]), tau2));

            uint32_t spike_mask = __hge2_mask(v_state[j], v_th2);
            int s0 = spike_mask & 1;
            int s1 = (spike_mask >> 16) & 1;

            int shift = (j % 4) * 8 + t;
            if (j < 4)
            {
                lo0 |= (uint32_t)s0 << shift;
                lo1 |= (uint32_t)s1 << shift;
            }
            else
            {
                hi0 |= (uint32_t)s0 << shift;
                hi1 |= (uint32_t)s1 << shift;
            }

            uint32_t state_bits = *reinterpret_cast<uint32_t *>(&v_state[j]);
            state_bits &= ~spike_mask;
            v_state[j] = *reinterpret_cast<__half2 *>(&state_bits);
        }
    }

    // =========================================================================
    // Epilogue: smem[0..4KB] (reused weight region) → uint8 global stores
    //
    // Layout per warp: 16 M-rows × 32 N-cols = 512 bytes
    //   Write: thread (mma_tid_y, mma_tid_x) fills rows mma_tid_y*2 and
    //          mma_tid_y*2+1, cols mma_tid_x*8..mma_tid_x*8+7 via 4 sts32
    //   Read:  lane i reads column i, writes 16 coalesced bytes to global
    // =========================================================================
    uint8_t *warp_smem8   = reinterpret_cast<uint8_t *>(smem) + warp_id * 512;
    const int warp_m_global = m_tile_base + warp_m * 16;
    const int warp_n_global = n_tile_base + warp_n * 32;
    const int n_global      = warp_n_global + lane_id;
    const bool n_valid      = (n_global < (int)param.outHW);

    const int smem_m0 = mma_tid_y * 2;
    const int smem_m1 = mma_tid_y * 2 + 1;
    const int smem_n  = mma_tid_x * 8;

    uint32_t a0 = ptx::smem_u32addr(&warp_smem8[smem_m0 * 32 + smem_n]);
    uint32_t a1 = ptx::smem_u32addr(&warp_smem8[smem_m1 * 32 + smem_n]);
    ptx::sts32(lo0, a0);  ptx::sts32(hi0, a0 + 4);
    ptx::sts32(lo1, a1);  ptx::sts32(hi1, a1 + 4);

    __syncthreads();

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
// Weight padding
// =============================================================================

static void pad_weights_sn_fp16(
    const float *src, __half *dst,
    int in_features, int C_out,
    int in_features_padded, int C_out_padded)
{
    for (int k = 0; k < in_features_padded; k++)
        for (int m = 0; m < C_out_padded; m++)
        {
            float v = (k < in_features && m < C_out) ? src[k * C_out + m] : 0.f;
            dst[k * C_out_padded + m] = __float2half(v);
        }
}


// =============================================================================
// Launch wrappers
// =============================================================================

template <int T, int Kh, int Kw, int Sh, int Sw, int Ph, int Pw>
void snn_conv2d_sn_fp16_launch(
    const uint8_t *d_inputs, const __half *d_weights_padded,
    const __half2 *d_bias,
    uint8_t *d_outputs, Conv2DParam &param, int out_ch_padded,
    float v_th, float tau)
{
    dim3 block(256);
    dim3 grid(((int)param.outHW + 63) / 64, ((int)param.out_ch + 63) / 64, 1);
    snn_conv2d_lif_64x64_k16_fp16<T, Kh, Kw, Sh, Sw, Ph, Pw>
        <<<grid, block>>>(d_inputs, d_weights_padded, d_bias, d_outputs,
                          param, out_ch_padded, v_th, tau);
}

void snn_conv2d_sn_fp16_1x1_s1_launch(
    const uint8_t *d_in, const __half *d_w, const __half2 *d_bias, uint8_t *d_out,
    Conv2DParam &param, int T, int Co_p, float v_th, float tau)
{
    switch (T)
    {
        case 1:
            snn_conv2d_sn_fp16_launch<1, 1, 1, 1, 1, 0, 0>(
                d_in, d_w, d_bias, d_out, param, Co_p, v_th, tau);
            break;
        case 2:
            snn_conv2d_sn_fp16_launch<2, 1, 1, 1, 1, 0, 0>(
                d_in, d_w, d_bias, d_out, param, Co_p, v_th, tau);
            break;
        case 3:
            snn_conv2d_sn_fp16_launch<3, 1, 1, 1, 1, 0, 0>(
                d_in, d_w, d_bias, d_out, param, Co_p, v_th, tau);
            break;
        case 4:
            snn_conv2d_sn_fp16_launch<4, 1, 1, 1, 1, 0, 0>(
                d_in, d_w, d_bias, d_out, param, Co_p, v_th, tau);
            break;
    }
}

void snn_conv2d_sn_fp16_3x3_s1_launch(
    const uint8_t *d_in, const __half *d_w, const __half2 *d_bias, uint8_t *d_out,
    Conv2DParam &param, int T, int Co_p, float v_th, float tau)
{
    switch (T)
    {
        case 1:
            snn_conv2d_sn_fp16_launch<1, 3, 3, 1, 1, 1, 1>(
                d_in, d_w, d_bias, d_out, param, Co_p, v_th, tau);
            break;
        case 2:
            snn_conv2d_sn_fp16_launch<2, 3, 3, 1, 1, 1, 1>(
                d_in, d_w, d_bias, d_out, param, Co_p, v_th, tau);
            break;
        case 3:
            snn_conv2d_sn_fp16_launch<3, 3, 3, 1, 1, 1, 1>(
                d_in, d_w, d_bias, d_out, param, Co_p, v_th, tau);
            break;
        case 4:
            snn_conv2d_sn_fp16_launch<4, 3, 3, 1, 1, 1, 1>(
                d_in, d_w, d_bias, d_out, param, Co_p, v_th, tau);
            break;
    }
}

void snn_conv2d_sn_fp16_3x3_s2_launch(
    const uint8_t *d_in, const __half *d_w, const __half2 *d_bias, uint8_t *d_out,
    Conv2DParam &param, int T, int Co_p, float v_th, float tau)
{
    switch (T)
    {
        case 1:
            snn_conv2d_sn_fp16_launch<1, 3, 3, 2, 2, 1, 1>(
                d_in, d_w, d_bias, d_out, param, Co_p, v_th, tau);
            break;
        case 2:
            snn_conv2d_sn_fp16_launch<2, 3, 3, 2, 2, 1, 1>(
                d_in, d_w, d_bias, d_out, param, Co_p, v_th, tau);
            break;
        case 3:
            snn_conv2d_sn_fp16_launch<3, 3, 3, 2, 2, 1, 1>(
                d_in, d_w, d_bias, d_out, param, Co_p, v_th, tau);
            break;
        case 4:
            snn_conv2d_sn_fp16_launch<4, 3, 3, 2, 2, 1, 1>(
                d_in, d_w, d_bias, d_out, param, Co_p, v_th, tau);
            break;
    }
}


// =============================================================================
// CPU reference  (fp16 arithmetic to match GPU accumulation)
//
// The conv is computed using the PADDED fp16 weight matrix, iterating K in
// the same order as the GPU.  Each add is rounded to fp16 precision by going
// through __half2float → add → __float2half, which is equivalent to __hadd.
// This gives bit-identical results to the GPU provided no NaN/denormal edge
// cases arise (not expected for typical SNN weights).
// =============================================================================

template <int Kh, int Kw, int Sh, int Sw, int Ph, int Pw>
static void snn_conv2d_lif_fp16_cpu_ref(
    const uint8_t *inputs,        // [C_in][H][W]
    const __half  *h_weightsP,    // [in_feat_padded][C_out_padded]  (padded fp16)
    uint8_t       *outputs,       // [C_out][H_out*W_out]
    int T, int C_in, int H, int W, int C_out,
    int in_feat_padded, int C_out_padded,
    const float *bias,
    float v_th, float tau)
{
    constexpr int KhKw = Kh * Kw;
    int in_features = C_in * KhKw;
    int H_out = (H + 2*Ph - Kh) / Sh + 1;
    int W_out = (W + 2*Pw - Kw) / Sw + 1;
    memset(outputs, 0, (size_t)C_out * H_out * W_out);

    for (int m = 0; m < C_out; m++)
    {
        for (int oh = 0; oh < H_out; oh++)
        {
            for (int ow = 0; ow < W_out; ow++)
            {
                // Accumulate conv in fp16, same K-major order as GPU
                __half conv_t[8] = {};
                for (int t = 0; t < T; t++)
                {
                    // Track in fp16 to match GPU accumulation precision
                    __half acc_h = __float2half(bias[m]);
                    for (int k = 0; k < in_feat_padded; k++)
                    {
                        int spike = 0;
                        if (k < in_features)
                        {
                            int c_idx = k / KhKw;
                            int ky    = k % KhKw / Kw;
                            int kx    = k % KhKw % Kw;
                            int ih    = oh*Sh - Ph + ky;
                            int iw    = ow*Sw - Pw + kx;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                spike = (inputs[c_idx*H*W + ih*W + iw] >> t) & 1;
                        }
                        if (spike)
                        {
                            // fp16 add: same rounding as GPU add.f16x2
                            float new_acc = __half2float(acc_h)
                                          + __half2float(h_weightsP[k * C_out_padded + m]);
                            acc_h = __float2half(new_acc);
                        }
                    }
                    conv_t[t] = acc_h;
                }

                // LIF HardReset in fp16 (matches GPU v_state in __half).
                const __half tau_h  = __float2half(tau);
                const __half v_th_h = __float2half(v_th);

                __half v_h = __float2half(
                    __half2float(conv_t[0]) * __half2float(tau_h));
                int spike = (__half2float(v_h) >= __half2float(v_th_h)) ? 1 : 0;
                if (spike)
                {
                    outputs[m * H_out * W_out + oh * W_out + ow] |= 1u;
                    v_h = __float2half(0.f);
                }

                for (int t = 1; t < T; t++)
                {
                    __half diff_h = __float2half(
                        __half2float(conv_t[t]) - __half2float(v_h));
                    __half update_h = __float2half(
                        __half2float(diff_h) * __half2float(tau_h));
                    v_h = __float2half(
                        __half2float(v_h) + __half2float(update_h));
                    spike =
                        (__half2float(v_h) >= __half2float(v_th_h)) ? 1 : 0;
                    if (spike)
                    {
                        outputs[m * H_out * W_out + oh * W_out + ow] |=
                            (uint8_t)(1 << t);
                        v_h = __float2half(0.f);
                    }
                }
            }
        }
    }
}


// =============================================================================
// Test
// =============================================================================

template <int T, int Kh, int Kw, int Sh, int Sw, int Ph, int Pw>
void snn_conv2d_sn_fp16_test(int C_in, int H, int W, int C_out,
                              float v_th, float tau, const char *label)
{
    constexpr int KhKw = Kh * Kw, K_CHUNK = 16;
    int H_out = (H+2*Ph-Kh)/Sh+1, W_out = (W+2*Pw-Kw)/Sw+1;
    int in_feat = C_in * KhKw;
    int in_feat_p = (in_feat + K_CHUNK - 1) / K_CHUNK * K_CHUNK;
    int Co_p = (C_out + 63) / 64 * 64;

    std::cout << "  [" << label << "]"
              << " T=" << T
              << " C_in=" << C_in
              << " H=" << H
              << " C_out=" << C_out
              << " -> H_out=" << H_out
              << " v_th=" << std::fixed << std::setprecision(2) << v_th
              << " tau=" << tau
              << "  ";

    size_t isz = (size_t)C_in*H*W;
    size_t wsz = (size_t)in_feat*C_out;
    size_t wpsz = (size_t)in_feat_p*Co_p;
    size_t osz  = (size_t)C_out*H_out*W_out;

    uint8_t *h_in   = new uint8_t[isz];
    float   *h_wf   = new float[wsz];
    __half  *h_wp   = new __half[wpsz];
    float   *h_bias = new float[C_out];
    __half2 *h_bias2 = new __half2[Co_p / 2];
    uint8_t *h_out  = new uint8_t[osz];
    uint8_t *h_ref  = new uint8_t[osz];

    srand(42);
    for (size_t i = 0; i < isz; i++)
    {
        uint8_t p = 0;
        for (int t = 0; t < T; t++) if (rand()&1) p |= (1u<<t);
        h_in[i] = p;
    }
    for (size_t i = 0; i < wsz; i++)
        h_wf[i] = (float)(rand()&255) / 256.f;   // [0, 1)
    for (int m = 0; m < C_out; m++)
        h_bias[m] = (float)(rand() & 63) / 64.f;
    for (int m = 0; m < Co_p; m += 2)
        h_bias2[m >> 1] = __half2{__float2half(h_bias[m]),
            __float2half(m + 1 < C_out ? h_bias[m + 1] : 0.f)};

    pad_weights_sn_fp16(h_wf, h_wp, in_feat, C_out, in_feat_p, Co_p);

    uint8_t *d_in; __half *d_wp; __half2 *d_bias; uint8_t *d_out;
    cudaMalloc(&d_in,   isz        * sizeof(uint8_t));
    cudaMalloc(&d_wp,   wpsz       * sizeof(__half));
    cudaMalloc(&d_bias, Co_p / 2   * sizeof(__half2));
    cudaMalloc(&d_out,  osz        * sizeof(uint8_t));
    cudaMemset(d_out, 0, osz * sizeof(uint8_t));
    cudaMemcpy(d_in,   h_in,   isz  * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wp,   h_wp,   wpsz * sizeof(__half),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias2, Co_p/2 * sizeof(__half2), cudaMemcpyHostToDevice);

    Conv2DParam p;
    p.in_h=H; p.in_w=W; p.inHW=H*W; p.inChKhKw=in_feat_p; p.inBatchNumel=C_in*H*W;
    p.out_ch=C_out; p.out_w=W_out; p.outHW=H_out*W_out; p.outBatchNumel=C_out*H_out*W_out;
    p.Kh=Kh; p.Kw=Kw; p.KhKw=KhKw; p.Sh=Sh; p.Sw=Sw; p.Ph=Ph; p.Pw=Pw;

    snn_conv2d_sn_fp16_launch<T,Kh,Kw,Sh,Sw,Ph,Pw>(
        d_in, d_wp, d_bias, d_out, p, Co_p, v_th, tau);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "[FAILED] CUDA error: "
                  << cudaGetErrorString(err) << '\n';
        goto cleanup;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, osz*sizeof(uint8_t), cudaMemcpyDeviceToHost);

    snn_conv2d_lif_fp16_cpu_ref<Kh,Kw,Sh,Sw,Ph,Pw>(
        h_in, h_wp, h_ref, T, C_in, H, W, C_out,
        in_feat_p, Co_p, h_bias, v_th, tau);

    {
        int errors = 0;
        for (size_t i = 0; i < osz; i++)
        {
            if (h_out[i] != h_ref[i])
            {
                if (errors < 5)
                {
                    std::cout << "\n    Err[" << i << "] gpu=0x"
                              << std::hex << std::setw(2) << std::setfill('0')
                              << static_cast<int>(h_out[i])
                              << " ref=0x" << std::setw(2)
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
    cudaFree(d_in); cudaFree(d_wp); cudaFree(d_bias); cudaFree(d_out);
    delete[] h_in; delete[] h_wf; delete[] h_wp; delete[] h_bias; delete[] h_bias2; delete[] h_out; delete[] h_ref;
}


#ifndef CONV2D_SN_FP16_BENCH_INC
int main()
{
    const float V_TH    = 1.0f;
    const float TAU     = 0.5f;

    std::cout
        << "\n=== snn_conv2d_lif_64x64_k16_fp16 correctness tests ===\n";

    std::cout << "\n--- 1x1, s=1, p=0 ---\n";
    snn_conv2d_sn_fp16_test<4,1,1,1,1,0,0>( 64, 80, 80,  64, V_TH, TAU, "base");
    snn_conv2d_sn_fp16_test<4,1,1,1,1,0,0>(128, 40, 40, 128, V_TH, TAU, "C128");
    snn_conv2d_sn_fp16_test<2,1,1,1,1,0,0>( 64, 80, 80,  48, V_TH, TAU, "C_out boundary");
    snn_conv2d_sn_fp16_test<1,1,1,1,1,0,0>( 64, 80, 80,  64, V_TH, TAU, "T=1");
    snn_conv2d_sn_fp16_test<4,1,1,1,1,0,0>( 64, 40, 40, 128, V_TH, TAU, "expand x2");
    snn_conv2d_sn_fp16_test<4,1,1,1,1,0,0>(128, 40, 40,  64, V_TH, TAU, "squeeze x0.5");

    std::cout << "\n--- 3x3, s=1, p=1 ---\n";
    snn_conv2d_sn_fp16_test<4,3,3,1,1,1,1>( 64, 80, 80,  64, V_TH, TAU, "base");
    snn_conv2d_sn_fp16_test<4,3,3,1,1,1,1>( 32, 40, 40,  32, V_TH, TAU, "small");
    snn_conv2d_sn_fp16_test<2,3,3,1,1,1,1>( 32, 43, 43,  48, V_TH, TAU, "boundary");
    snn_conv2d_sn_fp16_test<1,3,3,1,1,1,1>( 64, 80, 80,  64, V_TH, TAU, "T=1");
    snn_conv2d_sn_fp16_test<3,3,3,1,1,1,1>( 64, 80, 80,  64, V_TH, TAU, "T=3");

    std::cout << "\n--- 3x3, s=2, p=1 ---\n";
    snn_conv2d_sn_fp16_test<4,3,3,2,2,1,1>( 64, 80, 80,  64, V_TH, TAU, "H_out=40");
    snn_conv2d_sn_fp16_test<4,3,3,2,2,1,1>( 32, 80, 80,  64, V_TH, TAU, "C expand");
    snn_conv2d_sn_fp16_test<4,3,3,2,2,1,1>( 16, 40, 40,  32, V_TH, TAU, "small H=40");
    snn_conv2d_sn_fp16_test<2,3,3,2,2,1,1>( 32, 43, 43, 128, V_TH, 0.25f, "tau=0.25");

    std::cout << "\n=== Done ===\n";
    return 0;
}
#endif
