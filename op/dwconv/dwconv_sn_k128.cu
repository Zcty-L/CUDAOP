// dwconv_sn_k128.cu
// Fused SNN Depthwise Conv2D + IF HardReset Neuron
// Based on dwconv_k128.cu — replaces float[T] output with IF neuron in registers
//
// Input  : uint8_t [C][H*W]        — T spike bits/pixel (bit t = step t)
// Weights: float   [C][128]        — padded to 128 per channel (KhKw ≤ 128)
// Bias   : float   [C]
// Output : uint8_t [C][H_out*W_out] — T spike bits/pixel after IF HardReset
//
// IF HardReset: v += conv_t; spike = (v >= v_th); v = v_reset if spike

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include "cuda_utils.cuh"
#include "ptx_utils.cuh"

// =============================================================================
// Kernel — T_STEPS timesteps, 4 channels per block, KhKw ≤ 128
// =============================================================================

template <int T_STEPS>
__global__ void
snn_dwconv_4x128x256_sn_kernel(
    const uint8_t * __restrict__ inputs,
    const float   * __restrict__ weights,
    const float   * __restrict__ bias,
    uint8_t       * __restrict__ outputs,
    Conv2DParam param,
    float v_th,
    float v_reset)
{
    static_assert(T_STEPS >= 1 && T_STEPS <= 8,
                  "T_STEPS must be in [1,8] to fit in uint8 output");

    __shared__ __align__(2 * 1024)
    char smem[4 * 128 * 4];
    auto *smemweight = reinterpret_cast<float *>(smem);

    int posh_ori = ((blockIdx.x * 256 + threadIdx.x) / (int)param.out_w) * (int)param.Sh - (int)param.Ph;
    int posw_ori = ((blockIdx.x * 256 + threadIdx.x) % (int)param.out_w) * (int)param.Sw - (int)param.Pw;

    uint32_t weights_sts_addr = ptx::smem_u32addr(smemweight + threadIdx.x * 2);
    uint32_t weights_lds_addr = ptx::smem_u32addr(smemweight);

    // Weight layout: [C][128] padded. 256 threads split into 4×64 per channel.
    const float *weight_ldg_ptr = weights + blockIdx.y * 4 * 128
                                          + threadIdx.x / 64 * 128
                                          + threadIdx.x % 64 * 2;

    const uint8_t *input_ptr = inputs + blockIdx.y * 4 * param.inHW;

    float   weight_ldg_reg[2];
    float   weight_frag[16];
    uint8_t input_frag[4][4];
    float   output_frag[T_STEPS][4];

    const float b0 = bias[blockIdx.y * 4 + 0];
    const float b1 = bias[blockIdx.y * 4 + 1];
    const float b2 = bias[blockIdx.y * 4 + 2];
    const float b3 = bias[blockIdx.y * 4 + 3];
#pragma unroll
    for (int t = 0; t < T_STEPS; t++) {
        output_frag[t][0] = b0; output_frag[t][1] = b1;
        output_frag[t][2] = b2; output_frag[t][3] = b3;
    }

    ptx::ldg_nc(weight_ldg_reg[0], weight_ldg_ptr);
    ptx::ldg_nc(weight_ldg_reg[1], weight_ldg_ptr + 1);

    ptx::sts64(weight_ldg_reg[0], weight_ldg_reg[1], weights_sts_addr);
    __syncthreads();

    for (int k = 0; k < (int)param.KhKw; k += 4)
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int curH = posh_ori + (k + i) / (int)param.Kw;
            int curW = posw_ori + (k + i) % (int)param.Kw;
            int inOff = curH * (int)param.in_w + curW;
            if (curH >= 0 && curH < (int)param.in_h &&
                curW >= 0 && curW < (int)param.in_w)
            {
                input_frag[0][i] = input_ptr[inOff];
                input_frag[1][i] = input_ptr[inOff + (int)param.inHW];
                input_frag[2][i] = input_ptr[inOff + (int)param.inHW * 2];
                input_frag[3][i] = input_ptr[inOff + (int)param.inHW * 3];
            }
            else
            {
                input_frag[0][i] = 0;
                input_frag[1][i] = 0;
                input_frag[2][i] = 0;
                input_frag[3][i] = 0;
            }
        }

        ptx::lds128(weight_frag[0],  weight_frag[1],  weight_frag[2],  weight_frag[3],
                    weights_lds_addr);
        ptx::lds128(weight_frag[4],  weight_frag[5],  weight_frag[6],  weight_frag[7],
                    weights_lds_addr + 128 * (uint32_t)sizeof(float));
        ptx::lds128(weight_frag[8],  weight_frag[9],  weight_frag[10], weight_frag[11],
                    weights_lds_addr + 256 * (uint32_t)sizeof(float));
        ptx::lds128(weight_frag[12], weight_frag[13], weight_frag[14], weight_frag[15],
                    weights_lds_addr + 384 * (uint32_t)sizeof(float));

        weights_lds_addr += 4 * (uint32_t)sizeof(float);

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
#pragma unroll
            for (int t = 0; t < T_STEPS; t++)
            {
                if ((input_frag[0][i] >> t) & 1) output_frag[t][0] += weight_frag[i +  0];
                if ((input_frag[1][i] >> t) & 1) output_frag[t][1] += weight_frag[i +  4];
                if ((input_frag[2][i] >> t) & 1) output_frag[t][2] += weight_frag[i +  8];
                if ((input_frag[3][i] >> t) & 1) output_frag[t][3] += weight_frag[i + 12];
            }
        }
    }

    // =========================================================================
    // IF HardReset neuron — entirely in registers
    // v_state[ch]: membrane potential, persists across T steps
    // packed_out[ch]: bit t = spike at timestep t
    // HardReset: v = v_reset if spike, else v unchanged
    // =========================================================================
    float   v_state[4]    = {0.f, 0.f, 0.f, 0.f};
    uint8_t packed_out[4] = {0,   0,   0,   0  };

#pragma unroll
    for (int t = 0; t < T_STEPS; t++)
    {
#pragma unroll
        for (int ch = 0; ch < 4; ch++)
        {
            v_state[ch] += output_frag[t][ch];
            int spike = (v_state[ch] >= v_th) ? 1 : 0;
            packed_out[ch] |= (uint8_t)(spike << t);
            v_state[ch] -= (float)spike * (v_state[ch] - v_reset);
        }
    }

    // Output: uint8 [C][H_out*W_out]
    bool hw_valid = (blockIdx.x * 256 + threadIdx.x < (int)param.outHW);
    int outOffset = blockIdx.y * 4 * (int)param.outHW + blockIdx.x * 256 + threadIdx.x;
    ptx::stg8(packed_out[0], outputs + outOffset + (int)param.outHW * 0, hw_valid);
    ptx::stg8(packed_out[1], outputs + outOffset + (int)param.outHW * 1, hw_valid);
    ptx::stg8(packed_out[2], outputs + outOffset + (int)param.outHW * 2, hw_valid);
    ptx::stg8(packed_out[3], outputs + outOffset + (int)param.outHW * 3, hw_valid);
}

// =============================================================================
// Launch wrappers
// =============================================================================

template <int T>
static void snn_dwconv_sn_launch(
    const uint8_t *d_in, const float *d_w, const float *d_bias,
    uint8_t *d_out, Conv2DParam &param,
    float v_th, float v_reset)
{
    uint32_t bx = (param.outHW  + 255) / 256;
    uint32_t by = param.out_ch / 4;
    snn_dwconv_4x128x256_sn_kernel<T><<<dim3(bx, by), 256>>>(
        d_in, d_w, d_bias, d_out, param, v_th, v_reset);
}

static void snn_dwconv_sn_dispatch(
    const uint8_t *d_in, const float *d_w, const float *d_bias,
    uint8_t *d_out, Conv2DParam &param, int T,
    float v_th, float v_reset)
{
    switch (T) {
        case 1: snn_dwconv_sn_launch<1>(d_in, d_w, d_bias, d_out, param, v_th, v_reset); break;
        case 2: snn_dwconv_sn_launch<2>(d_in, d_w, d_bias, d_out, param, v_th, v_reset); break;
        case 3: snn_dwconv_sn_launch<3>(d_in, d_w, d_bias, d_out, param, v_th, v_reset); break;
        case 4: snn_dwconv_sn_launch<4>(d_in, d_w, d_bias, d_out, param, v_th, v_reset); break;
    }
}

// =============================================================================
// CPU reference: dwconv per timestep + IF HardReset → packed uint8 spikes
// =============================================================================

static void snn_dwconv_sn_cpu_ref(
    const uint8_t *inputs,   // [C][H*W]
    const float   *weights,  // [C][KhKw] fp32 (unpadded)
    const float   *bias,     // [C]
    uint8_t       *outputs,  // [C][H_out*W_out] packed T bits
    int T, int C, int H, int W,
    int Kh, int Kw, int Sh, int Sw, int Ph, int Pw,
    float v_th, float v_reset)
{
    int KhKw  = Kh * Kw;
    int H_out = (H + 2*Ph - Kh) / Sh + 1;
    int W_out = (W + 2*Pw - Kw) / Sw + 1;

    memset(outputs, 0, (size_t)C * H_out * W_out);

    for (int c = 0; c < C; c++) {
        for (int oh = 0; oh < H_out; oh++) {
            for (int ow = 0; ow < W_out; ow++) {
                float conv_t[8] = {};
                for (int t = 0; t < T; t++) {
                    float sum = bias[c];
                    for (int ky = 0; ky < Kh; ky++) {
                        for (int kx = 0; kx < Kw; kx++) {
                            int ih = oh*Sh - Ph + ky;
                            int iw = ow*Sw - Pw + kx;
                            if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                            uint8_t packed = inputs[c * H*W + ih*W + iw];
                            if ((packed >> t) & 1)
                                sum += weights[c * KhKw + ky*Kw + kx];
                        }
                    }
                    conv_t[t] = sum;
                }
                // IF HardReset
                float v = 0.f;
                int out_idx = c * H_out*W_out + oh*W_out + ow;
                for (int t = 0; t < T; t++) {
                    v += conv_t[t];
                    int spike = (v >= v_th) ? 1 : 0;
                    if (spike) {
                        outputs[out_idx] |= (uint8_t)(1 << t);
                        v = v_reset;
                    }
                }
            }
        }
    }
}

// =============================================================================
// Test + benchmark
// =============================================================================

template <int T>
static void snn_dwconv_sn_test(int C, int H, int W,
                                int Kh, int Kw, int Sh, int Sw, int Ph, int Pw,
                                float v_th, float v_reset,
                                const char *label)
{
    int KhKw  = Kh * Kw;
    int H_out = (H + 2*Ph - Kh) / Sh + 1;
    int W_out = (W + 2*Pw - Kw) / Sw + 1;

    printf("  [%s] T=%d C=%d H=%d K=%dx%d out=%dx%d vth=%.1f  ",
           label, T, C, H, Kh, Kw, H_out, W_out, (double)v_th);

    size_t in_sz     = (size_t)C * H * W;
    size_t wt_sz     = (size_t)C * KhKw;
    size_t wt_sz_gpu = (size_t)C * 128;
    size_t out_sz    = (size_t)C * H_out * W_out;

    uint8_t *h_in     = new uint8_t[in_sz];
    float   *h_wt     = new float[wt_sz];
    float   *h_wt_gpu = new float[wt_sz_gpu]();
    float   *h_bs     = new float[C];
    uint8_t *h_out    = new uint8_t[out_sz];
    uint8_t *h_ref    = new uint8_t[out_sz];

    srand(42);
    for (size_t i = 0; i < in_sz; i++) {
        h_in[i] = 0;
        for (int t = 0; t < T; t++)
            if (rand() & 1) h_in[i] |= (1u << t);
    }
    for (size_t i = 0; i < wt_sz; i++)
        h_wt[i] = (float)(rand() & 255) / 128.f - 1.f;
    for (int c = 0; c < C; c++)
        for (int k = 0; k < KhKw; k++)
            h_wt_gpu[c * 128 + k] = h_wt[c * KhKw + k];
    for (int i = 0; i < C; i++)
        h_bs[i] = (float)((int)(i % 17) - 8) / 16.f;

    uint8_t *d_in;
    float   *d_wt, *d_bs;
    uint8_t *d_out;
    cudaMalloc(&d_in,  in_sz     * sizeof(uint8_t));
    cudaMalloc(&d_wt,  wt_sz_gpu * sizeof(float));
    cudaMalloc(&d_bs,  (size_t)C * sizeof(float));
    cudaMalloc(&d_out, out_sz    * sizeof(uint8_t));
    cudaMemset(d_out, 0, out_sz * sizeof(uint8_t));

    cudaMemcpy(d_in,  h_in,     in_sz     * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wt,  h_wt_gpu, wt_sz_gpu * sizeof(float),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_bs,  h_bs,     (size_t)C * sizeof(float),   cudaMemcpyHostToDevice);

    Conv2DParam param;
    param.in_h          = H;
    param.in_w          = W;
    param.inHW          = H * W;
    param.inChKhKw      = C * KhKw;
    param.inBatchNumel  = C * H * W;
    param.out_ch        = C;
    param.out_h         = H_out;
    param.out_w         = W_out;
    param.outHW         = H_out * W_out;
    param.outBatchNumel = C * H_out * W_out;
    param.Kh            = Kh;
    param.Kw            = Kw;
    param.KhKw          = KhKw;
    param.Sh            = Sh;
    param.Sw            = Sw;
    param.Ph            = Ph;
    param.Pw            = Pw;

    snn_dwconv_sn_launch<T>(d_in, d_wt, d_bs, d_out, param, v_th, v_reset);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, out_sz * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    snn_dwconv_sn_cpu_ref(h_in, h_wt, h_bs, h_ref,
                           T, C, H, W, Kh, Kw, Sh, Sw, Ph, Pw, v_th, v_reset);
    {
        int errors = 0;
        for (size_t i = 0; i < out_sz; i++) {
            if (h_out[i] != h_ref[i]) {
                if (errors < 3)
                    printf("\n    Err[%zu]: gpu=0x%02x cpu=0x%02x",
                           i, h_out[i], h_ref[i]);
                errors++;
            }
        }

        constexpr int WARMUP = 10, BENCH = 100;
        for (int i = 0; i < WARMUP; i++)
            snn_dwconv_sn_launch<T>(d_in, d_wt, d_bs, d_out, param, v_th, v_reset);
        cudaDeviceSynchronize();

        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0); cudaEventCreate(&ev1);
        cudaEventRecord(ev0);
        for (int i = 0; i < BENCH; i++)
            snn_dwconv_sn_launch<T>(d_in, d_wt, d_bs, d_out, param, v_th, v_reset);
        cudaEventRecord(ev1);
        cudaEventSynchronize(ev1);
        float ms = 0;
        cudaEventElapsedTime(&ms, ev0, ev1);
        cudaEventDestroy(ev0); cudaEventDestroy(ev1);

        printf("%s (errs=%d) avg=%.4f ms\n",
               errors == 0 ? "PASSED" : "FAILED", errors, ms / BENCH);
    }

cleanup:
    cudaFree(d_in); cudaFree(d_wt); cudaFree(d_bs); cudaFree(d_out);
    delete[] h_in; delete[] h_wt; delete[] h_wt_gpu; delete[] h_bs;
    delete[] h_out; delete[] h_ref;
}

// =============================================================================
// Main
// =============================================================================

int main()
{
    printf("\n=== snn_dwconv_4x128x256_sn — Fused Depthwise Conv2D + IF HardReset ===\n\n");

    printf("--- 3x3 s=1 p=1 ---\n");
    snn_dwconv_sn_test<4>( 32,  8,  8, 3, 3, 1, 1, 1, 1, 1.f, 0.f, "tiny");
    snn_dwconv_sn_test<4>( 32, 40, 40, 3, 3, 1, 1, 1, 1, 1.f, 0.f, "C=32 H=40");
    snn_dwconv_sn_test<4>( 64, 40, 40, 3, 3, 1, 1, 1, 1, 1.f, 0.f, "C=64 H=40");
    snn_dwconv_sn_test<4>(128, 80, 80, 3, 3, 1, 1, 1, 1, 1.f, 0.f, "C=128 H=80");
    snn_dwconv_sn_test<2>( 64, 40, 40, 3, 3, 1, 1, 1, 1, 1.f, 0.f, "T=2");
    snn_dwconv_sn_test<1>( 64, 80, 80, 3, 3, 1, 1, 1, 1, 1.f, 0.f, "T=1 H=80");

    printf("\n--- 3x3 s=2 p=1 ---\n");
    snn_dwconv_sn_test<4>( 64, 80, 80, 3, 3, 2, 2, 1, 1, 1.f, 0.f, "C=64 H=80->40");
    snn_dwconv_sn_test<4>(128, 40, 40, 3, 3, 2, 2, 1, 1, 1.f, 0.f, "C=128 H=40->20");

    printf("\n--- 7x7 s=1 p=3 ---\n");
    snn_dwconv_sn_test<4>( 32, 40, 40, 7, 7, 1, 1, 3, 3, 1.f, 0.f, "C=32");
    snn_dwconv_sn_test<4>( 64, 40, 40, 7, 7, 1, 1, 3, 3, 1.f, 0.f, "C=64");
    snn_dwconv_sn_test<4>(128, 40, 40, 7, 7, 1, 1, 3, 3, 1.f, 0.f, "C=128");

    printf("\n--- 11x11 s=1 p=5 (KhKw=121) ---\n");
    snn_dwconv_sn_test<4>( 32, 40, 40, 11, 11, 1, 1, 5, 5, 1.f, 0.f, "C=32");
    snn_dwconv_sn_test<4>( 64, 40, 40, 11, 11, 1, 1, 5, 5, 1.f, 0.f, "C=64");
    snn_dwconv_sn_test<2>( 64, 80, 80, 11, 11, 1, 1, 5, 5, 1.f, 0.f, "T=2 H=80");

    printf("\n--- odd sizes ---\n");
    snn_dwconv_sn_test<4>( 32, 43, 43, 3, 3, 1, 1, 1, 1, 1.f, 0.f, "H=43 odd");
    snn_dwconv_sn_test<4>( 64, 41, 41, 7, 7, 1, 1, 3, 3, 1.f, 0.f, "H=41 odd");

    printf("\n=== All tests complete ===\n");
    return 0;
}
