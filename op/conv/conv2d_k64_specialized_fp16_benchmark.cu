#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cuda_utils.cuh"
#include "ptx_utils.cuh"

// =============================================================================
// Unified Conv2D fp16 Benchmark — three variants compared:
//
//   fp16:     fp16 weights, fp32 accumulators, float* outputs
//   fp16acc:  fp16 weights, fp16 accumulators, __half* outputs (v0 epilogue)
//   fp16acc_v2: same as fp16acc but 6KB SMEM, simplified epilogue
// =============================================================================

// --- Include all three kernels (suppress their main) ---
#define CONV2D_FP16_NO_MAIN
#include "conv2d_k64_specialized_fp16_0521.cu"
#define CONV2D_FP16ACC_NO_MAIN
#include "conv2d_k64_specialized_fp16acc_0521.cu"
#define CONV2D_FP16ACC_V2_NO_MAIN
#include "conv2d_k64_specialized_fp16acc_v2_0521.cu"


// =============================================================================
// Unified performance benchmark
// =============================================================================

static float cuda_ms(cudaEvent_t s, cudaEvent_t e)
{ float ms; cudaEventElapsedTime(&ms, s, e); return ms; }

template <int T, int Kh, int Kw, int Sh, int Sw, int Ph, int Pw>
static void benchmark_all(
    int C_in, int H, int W, int C_out, const char *label,
    int warmup = 5, int iters = 50)
{
    constexpr int KhKw = Kh * Kw, K_CHUNK = 16;
    int H_out = (H + 2*Ph - Kh) / Sh + 1;
    int W_out = (W + 2*Pw - Kw) / Sw + 1;
    int in_feat    = C_in * KhKw;
    int in_feat_p  = ((in_feat + K_CHUNK - 1) / K_CHUNK) * K_CHUNK;
    int Co_p       = (C_out + 63) / 64 * 64;

    size_t isz  = (size_t)C_in * H * W;
    size_t wpsz = (size_t)in_feat_p * Co_p;
    size_t osz  = (size_t)T * C_out * H_out * W_out;
    double flops = (double)T * C_out * H_out * W_out * in_feat * 2.0;

    // Device buffers
    uint8_t *d_in;   cudaMalloc(&d_in,   isz  * sizeof(uint8_t));
    __half  *d_wp;   cudaMalloc(&d_wp,   wpsz * sizeof(__half));
    float   *d_o_fp32;   cudaMalloc(&d_o_fp32,   osz * sizeof(float));
    __half  *d_o_acc0;   cudaMalloc(&d_o_acc0,   osz * sizeof(__half));
    __half  *d_o_acc1;   cudaMalloc(&d_o_acc1,   osz * sizeof(__half));

    // Init
    {
        uint8_t *hi = new uint8_t[isz];
        __half *hw = new __half[wpsz];
        srand(42);
        for (size_t i = 0; i < isz; i++) hi[i] = (uint8_t)(rand() & 0xFF);
        for (size_t i = 0; i < wpsz; i++)
            hw[i] = __float2half((float)(rand() & 255) / 256.f);
        cudaMemcpy(d_in, hi, isz  * sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_wp, hw, wpsz * sizeof(__half),  cudaMemcpyHostToDevice);
        delete[] hi; delete[] hw;
    }

    Conv2DParam p;
    p.in_h=H; p.in_w=W; p.inHW=H*W; p.inChKhKw=in_feat_p; p.inBatchNumel=C_in*H*W;
    p.out_ch=C_out; p.out_w=W_out; p.outHW=H_out*W_out; p.outBatchNumel=C_out*H_out*W_out;
    p.Kh=Kh; p.Kw=Kw; p.KhKw=KhKw; p.Sh=Sh; p.Sw=Sw; p.Ph=Ph; p.Pw=Pw;

    auto run_fp16  = [&](){ snn_conv2d_fp16_launch<T,Kh,Kw,Sh,Sw,Ph,Pw>(d_in, d_wp, d_o_fp32, p, Co_p); };
    auto run_acc   = [&](){ snn_conv2d_fp16acc_launch<T,Kh,Kw,Sh,Sw,Ph,Pw>(d_in, d_wp, d_o_acc0, p, Co_p); };
    auto run_v2    = [&](){ snn_conv2d_fp16acc_v2_launch<T,Kh,Kw,Sh,Sw,Ph,Pw>(d_in, d_wp, d_o_acc1, p, Co_p); };

    // Warmup
    for (int i = 0; i < warmup; i++) { run_fp16(); run_acc(); run_v2(); }
    cudaDeviceSynchronize();

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);

    auto time_kernel = [&](auto fn) {
        cudaEventRecord(e0);
        for (int i = 0; i < iters; i++) fn();
        cudaEventRecord(e1);
        cudaDeviceSynchronize();
        return cuda_ms(e0, e1) / iters;
    };

    float t_fp16  = time_kernel(run_fp16);
    float t_acc   = time_kernel(run_acc);
    float t_v2    = time_kernel(run_v2);

    auto gflops = [&](float t) { return flops / (t * 1e-3) / 1e9; };

    printf("  %-28s T=%d K=%dx%d s=%d C_in=%-4d C_out=%-4d"
           "  fp16: %6.3fms %6.1fGF"
           "  acc: %6.3fms %6.1fGF"
           "  v2: %6.3fms %6.1fGF"
           "  acc/fp16=%.2fx v2/fp16=%.2fx v2/acc=%.2fx\n",
           label, T, Kh, Kw, Sh, C_in, C_out,
           t_fp16, gflops(t_fp16),
           t_acc,  gflops(t_acc),
           t_v2,   gflops(t_v2),
           t_fp16 / t_acc, t_fp16 / t_v2, t_acc / t_v2);

    cudaFree(d_in); cudaFree(d_wp);
    cudaFree(d_o_fp32); cudaFree(d_o_acc0); cudaFree(d_o_acc1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
}


// =============================================================================
// Main
// =============================================================================

int main()
{
    int arch;
    cudaDeviceGetAttribute(&arch, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("\n=== Unified fp16 Conv2D Benchmark ===\n");
    printf("  Device: %s  SM%d.%d  %d SMs  %d KB SMEM/block\n\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, (int)(prop.sharedMemPerBlock / 1024));

    // ---- Correctness ----
    printf("--- Correctness: fp16 (fp32 accum) ---\n");
    snn_conv2d_fp16_test<4, 1,1,1,1,0,0>( 64, 80, 80,  64, "1x1 base");
    snn_conv2d_fp16_test<4, 1,1,1,1,0,0>(128, 40, 40, 128, "1x1 larger");
    snn_conv2d_fp16_test<4, 3,3,1,1,1,1>( 64, 80, 80,  64, "3x3s1 base");
    snn_conv2d_fp16_test<4, 3,3,2,2,1,1>( 64, 80, 80,  64, "3x3s2 base");
    snn_conv2d_fp16_test<1, 3,3,1,1,1,1>( 64, 80, 80,  64, "3x3s1 T=1");

    printf("\n--- Correctness: fp16acc ---\n");
    test_fp16acc<4, 1,1,1,1,0,0>( 64, 80, 80,  64, "1x1 base");
    test_fp16acc<4, 1,1,1,1,0,0>(128, 40, 40, 128, "1x1 larger");
    test_fp16acc<4, 3,3,1,1,1,1>( 64, 80, 80,  64, "3x3s1 base");
    test_fp16acc<4, 3,3,2,2,1,1>( 64, 80, 80,  64, "3x3s2 base");
    test_fp16acc<1, 3,3,1,1,1,1>( 64, 80, 80,  64, "3x3s1 T=1");

    printf("\n--- Correctness: fp16acc_v2 ---\n");
    test_fp16acc_v2<4, 1,1,1,1,0,0>( 64, 80, 80,  64, "1x1 base");
    test_fp16acc_v2<4, 1,1,1,1,0,0>(128, 40, 40, 128, "1x1 larger");
    test_fp16acc_v2<4, 3,3,1,1,1,1>( 64, 80, 80,  64, "3x3s1 base");
    test_fp16acc_v2<4, 3,3,2,2,1,1>( 64, 80, 80,  64, "3x3s2 base");
    test_fp16acc_v2<1, 3,3,1,1,1,1>( 64, 80, 80,  64, "3x3s1 T=1");

    // ---- Performance ----
    printf("\n--- Performance: fp16 vs fp16acc vs fp16acc_v2 ---\n\n");

    // 1×1 layers
    benchmark_all<4, 1,1,1,1,0,0>( 64,  80, 80,  64,  "1x1 C64 H80");
    benchmark_all<4, 1,1,1,1,0,0>(128,  40, 40, 128,  "1x1 C128 H40");
    benchmark_all<4, 1,1,1,1,0,0>(192,  40, 40, 128,  "1x1 C192->128 H40");
    benchmark_all<4, 1,1,1,1,0,0>(256,  20, 20, 128,  "1x1 C256->128 H20");
    benchmark_all<4, 1,1,1,1,0,0>(384,  20, 20, 192,  "1x1 C384->192 H20");

    // 3×3 stride=1
    benchmark_all<4, 3,3,1,1,1,1>( 64,  80, 80,  64,  "3x3s1 C64 H80");
    benchmark_all<4, 3,3,1,1,1,1>( 32,  40, 40,  32,  "3x3s1 C32 H40");
    benchmark_all<4, 3,3,1,1,1,1>( 64,  40, 40,  64,  "3x3s1 C64 H40");
    benchmark_all<4, 3,3,1,1,1,1>(128,  20, 20, 128,  "3x3s1 C128 H20");

    // 3×3 stride=2
    benchmark_all<4, 3,3,2,2,1,1>( 64,  80, 80,  64,  "3x3s2 C64 H80->40");
    benchmark_all<4, 3,3,2,2,1,1>( 32,  80, 80,  64,  "3x3s2 C32->64 H80");
    benchmark_all<4, 3,3,2,2,1,1>( 32,  40, 40,  64,  "3x3s2 C32->64 H40");

    // T=1,2 for sensitivity
    benchmark_all<1, 3,3,1,1,1,1>( 64,  80, 80,  64,  "T=1 3x3s1 C64 H80");
    benchmark_all<2, 3,3,1,1,1,1>( 64,  80, 80,  64,  "T=2 3x3s1 C64 H80");
    benchmark_all<2, 1,1,1,1,0,0>(128,  40, 40, 128,  "T=2 1x1 C128 H40");

    printf("\n=== Done ===\n");
    return 0;
}
