// dwconv_k128_fp16.cu
// Depthwise Conv2D — fp16 weights + fp16 accumulators, __half output
// Based on dwconv_k128.cu (fp32):
//   - weights: float* → const __half*
//   - accumulators: float[T][4] → __half[T][4]
//   - bias init: direct register load (no smem broadcast needed)
//   - weight smem: 2KB float → 1KB __half
//   - output: float* → __half*, via ptx::stg16
//
// Input layout  : uint8_t [C][H*W]       — bit t = spike at timestep t (unchanged)
// Weight layout : __half  [C][128]       — padded to 128 per channel (KhKw ≤ 128)
// Bias layout   : __half  [C]            — pre-converted before kernel call
// Output layout : __half  [T][C][H_out*W_out]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cuda_utils.cuh"
#include "ptx_utils.cuh"

// =============================================================================
// Kernel — fp16 weights/accumulators, T_STEPS timesteps, KhKw ≤ 128
// =============================================================================

template <int T_STEPS>
__global__ void
snn_dwconv_4x128x256_fp16_kernel(
    const uint8_t * __restrict__ inputs,
    const __half  * __restrict__ weights,
    const __half  * __restrict__ bias,
    __half        * __restrict__ outputs,
    Conv2DParam param)
{
    // Weight smem: 4 channels × 128 positions × 2 B = 1024 B
    __shared__ __align__(1024) char smem[4 * 128 * 2];
    auto *smemweight = reinterpret_cast<__half *>(smem);

    int posh_ori = ((blockIdx.x * 256 + threadIdx.x) / (int)param.out_w) * (int)param.Sh - (int)param.Ph;
    int posw_ori = ((blockIdx.x * 256 + threadIdx.x) % (int)param.out_w) * (int)param.Sw - (int)param.Pw;

    // Each thread stores 2 halves at smemweight[tid*2], smemweight[tid*2+1]
    uint32_t weights_sts_addr = ptx::smem_u32addr(smemweight + threadIdx.x * 2);
    uint32_t weights_lds_addr = ptx::smem_u32addr(smemweight);

    // Weight layout: [C][128] padded. 256 threads split into 4×64 per channel.
    const uint32_t *weight_ldg_ptr = (const uint32_t *)(
        weights + blockIdx.y * 4 * 128
                + threadIdx.x / 64 * 128
                + threadIdx.x % 64 * 2);

    const uint8_t *input_ptr = inputs + blockIdx.y * 4 * param.inHW;

    // Load two halves as one 32-bit word — no boundary check (weights padded to 128)
    uint32_t weight_ldg_reg;
    ptx::ldg_nc(weight_ldg_reg, weight_ldg_ptr);
    ptx::sts32(weight_ldg_reg, weights_sts_addr);
    __syncthreads();

    // Accumulators initialized with bias (same 4 values for all threads in block)
    __half output_frag[T_STEPS][4];
    {
        const __half b0 = bias[blockIdx.y * 4 + 0];
        const __half b1 = bias[blockIdx.y * 4 + 1];
        const __half b2 = bias[blockIdx.y * 4 + 2];
        const __half b3 = bias[blockIdx.y * 4 + 3];
#pragma unroll
        for (int t = 0; t < T_STEPS; t++) {
            output_frag[t][0] = b0;
            output_frag[t][1] = b1;
            output_frag[t][2] = b2;
            output_frag[t][3] = b3;
        }
    }

    for (int k = 0; k < (int)param.KhKw; k += 4)
    {
        // Load input_frag: uint8 spike packed bits for 4 kernel positions per channel
        uint8_t input_frag[4][4];
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

        // Load 4 halves per channel via two lds64 calls (each lds64 = 2×uint32 = 4 halves)
        // weight_frag[ch*4 .. ch*4+3] = 4 weights for channel ch at positions k..k+3
        __half weight_frag[16];
        auto *wfp = reinterpret_cast<uint32_t *>(weight_frag);
        ptx::lds64(wfp[0], wfp[1], weights_lds_addr);
        ptx::lds64(wfp[2], wfp[3], weights_lds_addr + 128 * (uint32_t)sizeof(__half));
        ptx::lds64(wfp[4], wfp[5], weights_lds_addr + 256 * (uint32_t)sizeof(__half));
        ptx::lds64(wfp[6], wfp[7], weights_lds_addr + 384 * (uint32_t)sizeof(__half));
        __syncthreads();  // protects smem before next iteration or bias write

        weights_lds_addr += 4 * (uint32_t)sizeof(__half);  // advance 4 half positions

        // Spike-conditional fp16 accumulation
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
#pragma unroll
            for (int t = 0; t < T_STEPS; t++)
            {
                if ((input_frag[0][i] >> t) & 1)
                    output_frag[t][0] = __hadd(output_frag[t][0], weight_frag[i]);
                if ((input_frag[1][i] >> t) & 1)
                    output_frag[t][1] = __hadd(output_frag[t][1], weight_frag[4 + i]);
                if ((input_frag[2][i] >> t) & 1)
                    output_frag[t][2] = __hadd(output_frag[t][2], weight_frag[8 + i]);
                if ((input_frag[3][i] >> t) & 1)
                    output_frag[t][3] = __hadd(output_frag[t][3], weight_frag[12 + i]);
            }
        }
    }

    // Output: __half [T][C][H_out*W_out]
    bool hw_valid = (blockIdx.x * 256 + threadIdx.x < (int)param.outHW);
    int outOffset = blockIdx.y * 4 * (int)param.outHW + blockIdx.x * 256 + threadIdx.x;
#pragma unroll
    for (int t = 0; t < T_STEPS; t++)
    {
        __half *out_t = outputs + (size_t)t * param.outBatchNumel;
        ptx::stg16(output_frag[t][0], out_t + outOffset + (int)param.outHW * 0, hw_valid);
        ptx::stg16(output_frag[t][1], out_t + outOffset + (int)param.outHW * 1, hw_valid);
        ptx::stg16(output_frag[t][2], out_t + outOffset + (int)param.outHW * 2, hw_valid);
        ptx::stg16(output_frag[t][3], out_t + outOffset + (int)param.outHW * 3, hw_valid);
    }
}

// =============================================================================
// Launch wrappers
// =============================================================================

template <int T>
static void snn_dwconv_fp16_launch(
    const uint8_t *d_in, const __half *d_w, const __half *d_bias,
    __half *d_out, Conv2DParam &param)
{
    uint32_t bx = (param.outHW  + 255) / 256;
    uint32_t by = param.out_ch / 4;
    snn_dwconv_4x128x256_fp16_kernel<T><<<dim3(bx, by), 256>>>(d_in, d_w, d_bias, d_out, param);
}

static void snn_dwconv_fp16_dispatch(
    const uint8_t *d_in, const __half *d_w, const __half *d_bias,
    __half *d_out, Conv2DParam &param, int T)
{
    switch (T) {
        case 1: snn_dwconv_fp16_launch<1>(d_in, d_w, d_bias, d_out, param); break;
        case 2: snn_dwconv_fp16_launch<2>(d_in, d_w, d_bias, d_out, param); break;
        case 3: snn_dwconv_fp16_launch<3>(d_in, d_w, d_bias, d_out, param); break;
        case 4: snn_dwconv_fp16_launch<4>(d_in, d_w, d_bias, d_out, param); break;
    }
}

// =============================================================================
// CPU reference — fp16 accumulation to match GPU
// =============================================================================

static void snn_dwconv_fp16_cpu_ref(
    const uint8_t *inputs,   // [C][H*W]
    const float   *weights,  // [C][KhKw] fp32, converted to fp16 per weight
    const __half  *bias,     // [C]
    __half        *outputs,  // [T][C][H_out*W_out]
    int T, int C, int H, int W,
    int Kh, int Kw, int Sh, int Sw, int Ph, int Pw)
{
    int KhKw  = Kh * Kw;
    int H_out = (H + 2*Ph - Kh) / Sh + 1;
    int W_out = (W + 2*Pw - Kw) / Sw + 1;

    for (int c = 0; c < C; c++) {
        for (int t = 0; t < T; t++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    __half acc = bias[c];
                    for (int ky = 0; ky < Kh; ky++) {
                        for (int kx = 0; kx < Kw; kx++) {
                            int ih = oh*Sh - Ph + ky;
                            int iw = ow*Sw - Pw + kx;
                            if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                            uint8_t packed = inputs[c * H*W + ih*W + iw];
                            if ((packed >> t) & 1)
                                acc = __hadd(acc, __float2half(weights[c * KhKw + ky*Kw + kx]));
                        }
                    }
                    int idx = t * C * H_out*W_out + c * H_out*W_out + oh*W_out + ow;
                    outputs[idx] = acc;
                }
            }
        }
    }
}

// =============================================================================
// Test + benchmark
// =============================================================================

template <int T>
static void snn_dwconv_fp16_test(int C, int H, int W,
                                  int Kh, int Kw, int Sh, int Sw, int Ph, int Pw,
                                  const char *label)
{
    int KhKw  = Kh * Kw;
    int H_out = (H + 2*Ph - Kh) / Sh + 1;
    int W_out = (W + 2*Pw - Kw) / Sw + 1;

    printf("  [%s] T=%d C=%d H=%d K=%dx%d out=%dx%d  ",
           label, T, C, H, Kh, Kw, H_out, W_out);

    size_t in_sz     = (size_t)C * H * W;
    size_t wt_sz     = (size_t)C * KhKw;       // unpadded, for CPU ref
    size_t wt_sz_gpu = (size_t)C * 128;         // padded to 128 per channel
    size_t out_sz    = (size_t)T * C * H_out * W_out;

    uint8_t *h_in   = new uint8_t[in_sz];
    float   *h_wt   = new float[wt_sz];
    float   *h_bs   = new float[C];
    __half  *h_wt_h = new __half[wt_sz_gpu]();  // zero-initialized (padding = 0)
    __half  *h_bs_h = new __half[C];
    __half  *h_out  = new __half[out_sz];
    __half  *h_ref  = new __half[out_sz];

    srand(42);
    for (size_t i = 0; i < in_sz; i++) {
        h_in[i] = 0;
        for (int t = 0; t < T; t++)
            if (rand() & 1) h_in[i] |= (1u << t);
    }
    for (size_t i = 0; i < wt_sz; i++) {
        h_wt[i] = (float)(rand() & 255) / 256.f - 0.5f;
        h_wt_h[(i / KhKw) * 128 + (i % KhKw)] = __float2half(h_wt[i]);
    }
    for (int i = 0; i < C; i++) {
        h_bs[i]   = (float)((int)(i % 17) - 8) / 16.f;
        h_bs_h[i] = __float2half(h_bs[i]);
    }

    uint8_t *d_in;
    __half  *d_wt, *d_bs, *d_out;
    cudaMalloc(&d_in,  in_sz     * sizeof(uint8_t));
    cudaMalloc(&d_wt,  wt_sz_gpu * sizeof(__half));
    cudaMalloc(&d_bs,  (size_t)C * sizeof(__half));
    cudaMalloc(&d_out, out_sz    * sizeof(__half));

    cudaMemcpy(d_in,  h_in,    in_sz     * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wt,  h_wt_h,  wt_sz_gpu * sizeof(__half),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_bs,  h_bs_h,  (size_t)C * sizeof(__half),   cudaMemcpyHostToDevice);

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

    snn_dwconv_fp16_launch<T>(d_in, d_wt, d_bs, d_out, param);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, out_sz * sizeof(__half), cudaMemcpyDeviceToHost);

    snn_dwconv_fp16_cpu_ref(h_in, h_wt, h_bs_h, h_ref,
                             T, C, H, W, Kh, Kw, Sh, Sw, Ph, Pw);
    {
        int errors = 0;
        float max_diff = 0.f;
        for (size_t i = 0; i < out_sz; i++) {
            float diff = fabsf(__half2float(h_out[i]) - __half2float(h_ref[i]));
            if (diff > 1e-2f) {
                if (errors < 3)
                    printf("\n    Err[%zu]: gpu=%.4f cpu=%.4f",
                           i, __half2float(h_out[i]), __half2float(h_ref[i]));
                errors++;
            }
            if (diff > max_diff) max_diff = diff;
        }

        constexpr int WARMUP = 10, BENCH = 100;
        for (int i = 0; i < WARMUP; i++)
            snn_dwconv_fp16_launch<T>(d_in, d_wt, d_bs, d_out, param);
        cudaDeviceSynchronize();

        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0); cudaEventCreate(&ev1);
        cudaEventRecord(ev0);
        for (int i = 0; i < BENCH; i++)
            snn_dwconv_fp16_launch<T>(d_in, d_wt, d_bs, d_out, param);
        cudaEventRecord(ev1);
        cudaEventSynchronize(ev1);
        float ms = 0;
        cudaEventElapsedTime(&ms, ev0, ev1);
        cudaEventDestroy(ev0); cudaEventDestroy(ev1);

        printf("%s (errs=%d max_diff=%.5f) avg=%.4f ms\n",
               errors == 0 ? "PASSED" : "FAILED", errors, max_diff, ms / BENCH);
    }

cleanup:
    cudaFree(d_in); cudaFree(d_wt); cudaFree(d_bs); cudaFree(d_out);
    delete[] h_in; delete[] h_wt; delete[] h_bs; delete[] h_wt_h; delete[] h_bs_h;
    delete[] h_out; delete[] h_ref;
}

// =============================================================================
// Main
// =============================================================================

int main()
{
    printf("\n=== snn_dwconv_4x128x256_fp16 — fp16 weights + fp16 accumulators ===\n\n");

    printf("--- 3x3 s=1 p=1 ---\n");
    snn_dwconv_fp16_test<4>( 32,  8,  8, 3, 3, 1, 1, 1, 1, "tiny");
    snn_dwconv_fp16_test<4>( 32, 40, 40, 3, 3, 1, 1, 1, 1, "C=32 H=40");
    snn_dwconv_fp16_test<4>( 64, 40, 40, 3, 3, 1, 1, 1, 1, "C=64 H=40");
    snn_dwconv_fp16_test<4>(128, 80, 80, 3, 3, 1, 1, 1, 1, "C=128 H=80");
    snn_dwconv_fp16_test<2>( 64, 40, 40, 3, 3, 1, 1, 1, 1, "T=2");
    snn_dwconv_fp16_test<1>( 64, 80, 80, 3, 3, 1, 1, 1, 1, "T=1 H=80");

    printf("\n--- 3x3 s=2 p=1 ---\n");
    snn_dwconv_fp16_test<4>( 64, 80, 80, 3, 3, 2, 2, 1, 1, "C=64 H=80->40");
    snn_dwconv_fp16_test<4>(128, 40, 40, 3, 3, 2, 2, 1, 1, "C=128 H=40->20");

    printf("\n--- 7x7 s=1 p=3 ---\n");
    snn_dwconv_fp16_test<4>( 32, 40, 40, 7, 7, 1, 1, 3, 3, "C=32");
    snn_dwconv_fp16_test<4>( 64, 40, 40, 7, 7, 1, 1, 3, 3, "C=64");
    snn_dwconv_fp16_test<4>(128, 40, 40, 7, 7, 1, 1, 3, 3, "C=128");

    printf("\n--- 11x11 s=1 p=5 (KhKw=121) ---\n");
    snn_dwconv_fp16_test<4>( 32, 40, 40, 11, 11, 1, 1, 5, 5, "C=32");
    snn_dwconv_fp16_test<4>( 64, 40, 40, 11, 11, 1, 1, 5, 5, "C=64");
    snn_dwconv_fp16_test<2>( 64, 80, 80, 11, 11, 1, 1, 5, 5, "T=2 H=80");

    printf("\n--- odd sizes ---\n");
    snn_dwconv_fp16_test<4>( 32, 43, 43, 3, 3, 1, 1, 1, 1, "H=43 odd");
    snn_dwconv_fp16_test<4>( 64, 41, 41, 7, 7, 1, 1, 3, 3, "H=41 odd");

    printf("\n=== All tests complete ===\n");
    return 0;
}
