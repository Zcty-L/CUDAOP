#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cuda_utils.cuh"
#include "ptx_utils.cuh"

#ifndef CONV2D_K64_SPECIALIZED_FP16ACC_0521_H
#define CONV2D_K64_SPECIALIZED_FP16ACC_0521_H

// =============================================================================
// SNN Conv2D — fp16 weight + fp16 accumulator, 64×64 tile, K_chunk=16
//
// Compared to conv2d_k64_specialized_fp16.cu (fp16w + fp32acc):
//   - output_frag: __half[T][2][8]  instead of float[T][2][8]
//   - accumulate:  @p add.f16       instead of @p add.f32
//   - outputs arg: __half *         instead of float *
//   - epilogue smem: __half layout  → 8 warps × 256 half × 2B = 4 KB
//   - Total SMEM: max(6 KB k-loop, 4 KB epilogue) = 6 KB
//
// Precision trade-off:
//   fp16 has 10-bit mantissa (~3 decimal digits).  SNN accumulators can sum
//   up to in_features spikes (each adding a weight ≈0.5), so worst-case
//   accumulated value ≈ in_features × 0.5.  For large in_features (e.g. 1152
//   for 3×3 C=128) this reaches ~576, which fp16 can represent but rounding
//   errors accumulate.  The correctness test uses a loose tolerance and prints
//   the actual max_diff so users can judge acceptability for their use case.
//
// SMEM layout:
//   smemweight[2][K_CHUNK][M_TILE] __half  = 2 × 2 KB = 4 KB
//   smeminput [2][K_CHUNK][N_TILE] uint8_t = 2 × 1 KB = 2 KB
//   Epilogue: 8 warps × 512 half × 2B = 8 KB  (reuses full smem region)
//   Total: max(6 KB k-loop, 8 KB epilogue) = 8 KB
// =============================================================================

// PTX helper: predicated fp16x2 add  (@p add.f16x2 acc, acc, w)
// Processes M-row 0 (w.x) and M-row 1 (w.y) simultaneously in one instruction.
__device__ __forceinline__ void add_f16x2(__half2 &a, const __half2 &b, int guard)
{
    asm volatile(
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @p add.f16x2 %0, %0, %1;\n"
        "}"
        : "+r"(*reinterpret_cast<unsigned *>(&a))
        : "r"(*reinterpret_cast<const unsigned *>(&b)), "r"(guard)
    );
}

template <int T_STEPS, int Kh, int Kw, int Sh, int Sw, int Ph, int Pw>
__global__ __launch_bounds__(256, T_STEPS <= 2 ? 3 : 2)
void snn_conv2d_64x64_k16_fp16acc(
    const uint8_t * __restrict__ inputs,
    const __half  * __restrict__ weights,   // fp16 [in_feat_padded][C_out_padded]
    __half        * __restrict__ outputs,   // fp16 [T][C_out][H_out*W_out]
    Conv2DParam param,
    int out_ch_padded)
{
    constexpr int K_CHUNK  = 16;
    constexpr int M_TILE   = 64;
    constexpr int N_TILE   = 64;
    constexpr int KhKw     = Kh * Kw;

    __shared__ __align__(128) char smem[8 * 1024];

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

    // fp16x2 accumulators: .x = M-row 0 result, .y = M-row 1 result
    __half2 output_frag[T_STEPS][8];
#pragma unroll
    for (int t = 0; t < T_STEPS; t++)
#pragma unroll
        for (int j = 0; j < 8; j++)
            output_frag[t][j] = __float2half2_rn(0.f);

    const int in_features = param.inChKhKw;
    const int k_iters     = in_features / K_CHUNK;

    // load_weight: cp.async.ca 8B (4 halves), same as fp16 fp32-accum version
    auto load_weight = [&](int k_iter, int buf)
    {
        const int k_base = k_iter * K_CHUNK;
        const int w_row  = tid / 16;
        const int w_col  = (tid % 16) * 4;
        uint32_t smem_ptr = ptx::smem_u32addr(
            &smemweight[buf][w_row * M_TILE + w_col]);
        const __half *src = &weights[(k_base + w_row) * out_ch_padded
                                     + m_tile_base + w_col];
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 8;\n"
            :: "r"(smem_ptr), "l"(src)
        );
    };

    // load_input: identical to fp16 fp32-accum version
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
                        int oh = n_idx / param.out_w;
                        int ow = n_idx % param.out_w;
                        packed |= (uint32_t)inputs[c_idx * param.inHW
                                                   + (oh*Sh-Ph) * param.in_w
                                                   + (ow*Sw-Pw)]
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
                            packed |= (uint32_t)inputs[c_idx * param.inHW
                                                       + ih * param.in_w + iw]
                                      << (b * 8);
                    }
                }
            }
        }
        uint32_t smeminput_ptr = ptx::smem_u32addr(
            &reinterpret_cast<uint32_t *>(smeminput[buf])[i_k * (N_TILE / 4) + i_n4]);
        ptx::sts32(packed, smeminput_ptr);
    };

    if (k_iters > 0)
    {
        load_weight(0, 0);
        load_input(0, 0);
        asm volatile("cp.async.commit_group;\n" :::);
    }

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

#pragma unroll
        for (int k = 0; k < K_CHUNK; k++)
        {
            // Load weight as __half2: .x = row0, .y = row1 (same 32-bit word)
            uint32_t w_addr = ptx::smem_u32addr(
                &smemweight[cur][k * M_TILE + thread_m_base]);
            uint32_t w_raw;
            ptx::lds32(w_raw, w_addr);
            __half2 weight2 = *reinterpret_cast<__half2 *>(&w_raw);

            uint32_t i_addr = ptx::smem_u32addr(
                &smeminput[cur][k * N_TILE + thread_n_base]);
            uint32_t input_frag_lo, input_frag_hi;
            ptx::lds64(input_frag_lo, input_frag_hi, i_addr);

#pragma unroll
            for (int t = 0; t < T_STEPS; t++)
            {
#pragma unroll
                for (int j = 0; j < 8; j++)
                {
                    uint32_t word  = (j < 4) ? input_frag_lo : input_frag_hi;
                    int      shift = (j % 4) * 8 + t;
                    int      spike = (word >> shift) & 1;
                    // One add.f16x2 handles both M-rows simultaneously
                    add_f16x2(output_frag[t][j], weight2, spike);
                }
            }
        }

        __syncthreads();
    }

    asm volatile("cp.async.wait_all;\n" :::);

    // ----------------------------------------------------------------
    // Epilogue: fp16 accumulators → smem → gmem
    //
    // Smem layout (8KB total, reuses k-loop smem):
    //   8 warps × 512 halves (1KB) per warp
    //   Within each warp: region b=0 at [0..255], region b=1 at [256..511]
    //   Both b-regions written simultaneously → only 1 sync per T iteration
    //   (vs the old 2 syncs per b-iteration = 4T syncs total)
    //   New total: 2T-1 syncs (write-sync per T + read-sync for T>1)
    //
    // Thread b-assignment: b = mma_tid_y / 4 (constant per thread)
    //   b=0 threads (mma_tid_y=0..3) write smem[0..255]
    //   b=1 threads (mma_tid_y=4..7) write smem[256..511]
    // ----------------------------------------------------------------
    __half      *warp_smem      = reinterpret_cast<__half *>(smem) + warp_id * 512;
    uint32_t     warp_smem_base = ptx::smem_u32addr(warp_smem);

    const int warp_m_global = m_tile_base + warp_m * 16;
    const int warp_n_global = n_tile_base + warp_n * 32;

    const int  n_global = warp_n_global + lane_id;
    const bool n_valid  = (n_global < (int)param.outHW);

    // Each thread's b-region is determined by mma_tid_y (constant for this thread)
    const int b_self     = mma_tid_y / 4;
    const int smem_row   = (mma_tid_y % 4) * 2;
    const uint32_t base_r0 = warp_smem_base
        + (uint32_t)(b_self * 256 + smem_row * 32 + mma_tid_x * 8) * sizeof(__half);
    const uint32_t base_r1 = warp_smem_base
        + (uint32_t)(b_self * 256 + (smem_row + 1) * 32 + mma_tid_x * 8) * sizeof(__half);

    for (int t = 0; t < T_STEPS; t++)
    {
        __half *out_t = outputs + (size_t)t * param.out_ch * param.outHW;

        // All threads write to their own b-region simultaneously (no per-b sync)
        uint32_t r0[4], r1[4];
#pragma unroll
        for (int p = 0; p < 4; p++) {
            __half2 h2_r0 = __lows2half2(output_frag[t][p*2], output_frag[t][p*2+1]);
            __half2 h2_r1 = __highs2half2(output_frag[t][p*2], output_frag[t][p*2+1]);
            r0[p] = *reinterpret_cast<uint32_t *>(&h2_r0);
            r1[p] = *reinterpret_cast<uint32_t *>(&h2_r1);
        }
        ptx::sts128(r0[0], r0[1], r0[2], r0[3], base_r0);
        ptx::sts128(r1[0], r1[1], r1[2], r1[3], base_r1);

        __syncthreads();  // wait for all b-regions written before any thread reads

        // Read both b-regions and store to global in one pass (no inter-b sync)
#pragma unroll
        for (int bi = 0; bi < 2; bi++)
        {
            const __half *smemout_lds = warp_smem + bi * 256 + lane_id;
            const int     m_base      = warp_m_global + bi * 8;
#pragma unroll
            for (int row = 0; row < 8; row++)
            {
                const int  m_global = m_base + row;
                const bool m_valid  = (m_global < (int)param.out_ch);
                ptx::stg16(smemout_lds[row * 32],
                           out_t + (size_t)m_global * param.outHW + n_global,
                           m_valid && n_valid);
            }
        }

        // Sync needed before next T's writes reuse the same smem regions
        if (t + 1 < T_STEPS)
            __syncthreads();
    }
}


// =============================================================================
// Weight padding (same as fp16 fp32-accum version)
// =============================================================================

static void pad_weights_fp16acc(
    const float *src, __half *dst,
    int in_features, int C_out,
    int in_features_padded, int C_out_padded)
{
    for (int k = 0; k < in_features_padded; k++)
        for (int m = 0; m < C_out_padded; m++) {
            float v = (k < in_features && m < C_out) ? src[k * C_out + m] : 0.f;
            dst[k * C_out_padded + m] = __float2half(v);
        }
}


// =============================================================================
// Launch wrappers
// =============================================================================

template <int T, int Kh, int Kw, int Sh, int Sw, int Ph, int Pw>
void snn_conv2d_fp16acc_launch(
    const uint8_t *d_inputs, const __half *d_weights_padded,
    __half *d_outputs, Conv2DParam &param, int out_ch_padded)
{
    dim3 block(256);
    dim3 grid((param.outHW + 63) / 64, (param.out_ch + 63) / 64, 1);
    snn_conv2d_64x64_k16_fp16acc<T, Kh, Kw, Sh, Sw, Ph, Pw>
        <<<grid, block>>>(d_inputs, d_weights_padded, d_outputs, param, out_ch_padded);
}


// =============================================================================
// CPU reference — uses fp16-rounded weights, accumulates in fp32 for reference,
// then rounds to fp16.  This gives the ideal fp16-acc result if there were no
// rounding during accumulation; actual kernel may differ due to order-of-ops.
// We also provide a pure-fp16-acc CPU ref for tighter comparison.
// =============================================================================

template <int Kh, int Kw, int Sh, int Sw, int Ph, int Pw>
static void cpu_ref_fp16acc(
    const uint8_t *inputs, const float *weights_f32,
    __half *outputs,
    int T, int C_in, int H, int W, int C_out)
{
    constexpr int KhKw = Kh * Kw;
    int H_out = (H + 2*Ph - Kh) / Sh + 1;
    int W_out = (W + 2*Pw - Kw) / Sw + 1;

    for (int t = 0; t < T; t++)
        for (int m = 0; m < C_out; m++)
            for (int oh = 0; oh < H_out; oh++)
                for (int ow = 0; ow < W_out; ow++) {
                    // Accumulate in fp16 to match kernel behavior
                    __half acc = __float2half(0.f);
                    for (int c = 0; c < C_in; c++)
                        for (int ky = 0; ky < Kh; ky++)
                            for (int kx = 0; kx < Kw; kx++) {
                                int ih = oh*Sh - Ph + ky;
                                int iw = ow*Sw - Pw + kx;
                                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                                uint8_t packed = inputs[c*H*W + ih*W + iw];
                                int spike = (packed >> t) & 1;
                                if (spike) {
                                    int fi = c*KhKw + ky*Kw + kx;
                                    __half w = __float2half(weights_f32[fi*C_out + m]);
                                    acc = __hadd(acc, w);
                                }
                            }
                    outputs[t*C_out*H_out*W_out + m*H_out*W_out + oh*W_out + ow] = acc;
                }
}


// =============================================================================
// Test + benchmark
// =============================================================================

// Test fp16acc kernel correctness
template <int T, int Kh, int Kw, int Sh, int Sw, int Ph, int Pw>
static void test_fp16acc(int C_in, int H, int W, int C_out, const char *label)
{
    constexpr int KhKw = Kh * Kw, K_CHUNK = 16;
    int H_out = (H+2*Ph-Kh)/Sh+1, W_out = (W+2*Pw-Kw)/Sw+1;
    int in_feat = C_in*KhKw;
    int in_feat_p = (in_feat + K_CHUNK-1)/K_CHUNK*K_CHUNK;
    int Co_p = (C_out+63)/64*64;

    printf("  [%s] T=%d C_in=%d H=%d C_out=%d -> H_out=%d  ", label, T, C_in, H, C_out, H_out);

    size_t isz = (size_t)C_in*H*W, wsz = (size_t)in_feat*C_out;
    size_t wpsz = (size_t)in_feat_p*Co_p, osz = (size_t)T*C_out*H_out*W_out;

    uint8_t *h_in  = new uint8_t[isz];
    float   *h_wf  = new float[wsz];
    __half  *h_wp  = new __half[wpsz];
    __half  *h_out = new __half[osz];
    __half  *h_ref = new __half[osz];

    srand(42);
    for (size_t i = 0; i < isz; i++) {
        uint8_t p = 0;
        for (int t = 0; t < T; t++) if (rand()&1) p |= 1u<<t;
        h_in[i] = p;
    }
    for (size_t i = 0; i < wsz; i++)
        h_wf[i] = (float)(rand()&255)/256.f;
    pad_weights_fp16acc(h_wf, h_wp, in_feat, C_out, in_feat_p, Co_p);

    uint8_t *d_in; __half *d_wp, *d_out;
    cudaMalloc(&d_in,  isz  * sizeof(uint8_t));
    cudaMalloc(&d_wp,  wpsz * sizeof(__half));
    cudaMalloc(&d_out, osz  * sizeof(__half));
    cudaMemset(d_out, 0, osz * sizeof(__half));
    cudaMemcpy(d_in, h_in, isz*sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wp, h_wp, wpsz*sizeof(__half),  cudaMemcpyHostToDevice);

    Conv2DParam p;
    p.in_h=H; p.in_w=W; p.inHW=H*W; p.inChKhKw=in_feat_p; p.inBatchNumel=C_in*H*W;
    p.out_ch=C_out; p.out_w=W_out; p.outHW=H_out*W_out; p.outBatchNumel=C_out*H_out*W_out;
    p.Kh=Kh; p.Kw=Kw; p.KhKw=KhKw; p.Sh=Sh; p.Sw=Sw; p.Ph=Ph; p.Pw=Pw;

    snn_conv2d_fp16acc_launch<T,Kh,Kw,Sh,Sw,Ph,Pw>(d_in, d_wp, d_out, p, Co_p);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); goto cleanup; }
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, osz*sizeof(__half), cudaMemcpyDeviceToHost);

    cpu_ref_fp16acc<Kh,Kw,Sh,Sw,Ph,Pw>(h_in, h_wf, h_ref, T, C_in, H, W, C_out);

    {
        int errors = 0; float max_diff = 0.f;
        for (size_t i = 0; i < osz; i++) {
            float diff = fabsf(__half2float(h_out[i]) - __half2float(h_ref[i]));
            // fp16 acc reordering tolerance: O(in_feat * eps_fp16)
            float tol = (float)in_feat_p * 1e-3f + 1e-2f;
            if (diff > tol) { if (errors<5) printf("\n    Err[%zu] gpu=%.3f ref=%.3f diff=%.3f", i, __half2float(h_out[i]), __half2float(h_ref[i]), diff); errors++; }
            if (diff > max_diff) max_diff = diff;
        }
        printf("  %s (%d errors, max_diff=%.4f)\n", errors==0?"PASSED!":"FAILED", errors, max_diff);
    }
cleanup:
    cudaFree(d_in); cudaFree(d_wp); cudaFree(d_out);
    delete[] h_in; delete[] h_wf; delete[] h_wp; delete[] h_out; delete[] h_ref;
}

// Benchmark: fp16w+fp32acc  vs  fp16w+fp16acc
// (includes the fp16 fp32-accum kernel directly for apples-to-apples comparison)
#define CONV2D_FP16_NO_MAIN
#include "conv2d_k64_specialized_fp16_0521.cu"
#undef CONV2D_FP16_NO_MAIN

static float cuda_time_ms(cudaEvent_t s, cudaEvent_t e)
{ float ms; cudaEventElapsedTime(&ms, s, e); return ms; }

template <int T, int Kh, int Kw, int Sh, int Sw, int Ph, int Pw>
static void benchmark_fp32acc_vs_fp16acc(
    int C_in, int H, int W, int C_out,
    const char *label, int warmup=5, int iters=50)
{
    constexpr int KhKw=Kh*Kw, K_CHUNK=16;
    int H_out=(H+2*Ph-Kh)/Sh+1, W_out=(W+2*Pw-Kw)/Sw+1;
    int in_feat_p=((C_in*KhKw)+K_CHUNK-1)/K_CHUNK*K_CHUNK;
    int Co_p=(C_out+63)/64*64;

    size_t isz=(size_t)C_in*H*W, wpsz=(size_t)in_feat_p*Co_p;
    size_t osz_f32=(size_t)T*C_out*H_out*W_out;
    size_t osz_f16=osz_f32;

    uint8_t *d_in;   cudaMalloc(&d_in,   isz*sizeof(uint8_t));
    __half  *d_wp;   cudaMalloc(&d_wp,   wpsz*sizeof(__half));
    float   *d_of32; cudaMalloc(&d_of32, osz_f32*sizeof(float));
    __half  *d_of16; cudaMalloc(&d_of16, osz_f16*sizeof(__half));

    {
        uint8_t *hi=new uint8_t[isz]; __half *hw=new __half[wpsz];
        srand(42);
        for (size_t i=0;i<isz;i++) hi[i]=(uint8_t)(rand()&0xFF);
        for (size_t i=0;i<wpsz;i++) hw[i]=__float2half((float)(rand()&255)/256.f);
        cudaMemcpy(d_in,hi,isz*sizeof(uint8_t),cudaMemcpyHostToDevice);
        cudaMemcpy(d_wp,hw,wpsz*sizeof(__half), cudaMemcpyHostToDevice);
        delete[] hi; delete[] hw;
    }

    Conv2DParam p;
    p.in_h=H; p.in_w=W; p.inHW=H*W; p.inChKhKw=in_feat_p; p.inBatchNumel=C_in*H*W;
    p.out_ch=C_out; p.out_w=W_out; p.outHW=H_out*W_out; p.outBatchNumel=C_out*H_out*W_out;
    p.Kh=Kh; p.Kw=Kw; p.KhKw=Kh*Kw; p.Sh=Sh; p.Sw=Sw; p.Ph=Ph; p.Pw=Pw;

    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    auto run_f32 = [&](){ snn_conv2d_fp16_launch<T,Kh,Kw,Sh,Sw,Ph,Pw>(d_in,d_wp,d_of32,p,Co_p); };
    auto run_f16 = [&](){ snn_conv2d_fp16acc_launch<T,Kh,Kw,Sh,Sw,Ph,Pw>(d_in,d_wp,d_of16,p,Co_p); };

    for (int i=0;i<warmup;i++){run_f32();run_f16();}
    cudaDeviceSynchronize();

    cudaEventRecord(e0); for(int i=0;i<iters;i++) run_f32(); cudaEventRecord(e1);
    cudaDeviceSynchronize(); float tf32=cuda_time_ms(e0,e1)/iters;

    cudaEventRecord(e0); for(int i=0;i<iters;i++) run_f16(); cudaEventRecord(e1);
    cudaDeviceSynchronize(); float tf16=cuda_time_ms(e0,e1)/iters;

    int in_feat=C_in*KhKw;
    double flops=(double)T*C_out*H_out*W_out*in_feat*2.0;
    double gf32=flops/(tf32*1e-3)/1e9, gf16=flops/(tf16*1e-3)/1e9;
    // weight BW same (both fp16), output BW differs (float vs half)
    double ibytes=(double)isz;
    double wbytes=(double)wpsz*sizeof(__half);
    double obytes_f32=(double)osz_f32*sizeof(float);
    double obytes_f16=(double)osz_f16*sizeof(__half);
    double bwf32=(ibytes+wbytes+obytes_f32)/(tf32*1e-3)/1e9;
    double bwf16=(ibytes+wbytes+obytes_f16)/(tf16*1e-3)/1e9;

    printf("  %-26s T=%d K=%dx%d s=%d C_in=%-4d C_out=%-4d"
           "  fp32acc: %6.3f ms %7.1f GFLOPS %5.1f GB/s"
           "  fp16acc: %6.3f ms %7.1f GFLOPS %5.1f GB/s"
           "  speedup: %.2fx\n",
           label, T, Kh, Kw, Sh, C_in, C_out,
           tf32, gf32, bwf32, tf16, gf16, bwf16, tf32/tf16);

    cudaFree(d_in); cudaFree(d_wp); cudaFree(d_of32); cudaFree(d_of16);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
}


// =============================================================================
// Main
// =============================================================================

#ifndef CONV2D_FP16ACC_NO_MAIN
int main()
{
    printf("\n=== snn_conv2d_64x64_k16_fp16acc — correctness tests ===\n");

    printf("\n--- Variant A: 1x1, s=1, p=0 ---\n");
    test_fp16acc<4,1,1,1,1,0,0>( 64, 80, 80,  64, "base");
    test_fp16acc<4,1,1,1,1,0,0>(128, 40, 40, 128, "larger");
    test_fp16acc<2,1,1,1,1,0,0>( 64, 80, 80,  48, "C_out boundary");
    test_fp16acc<4,1,1,1,1,0,0>( 64, 40, 40, 128, "expansion x2");
    test_fp16acc<4,1,1,1,1,0,0>(128, 40, 40,  64, "squeeze x0.5");

    printf("\n--- Variant B: 3x3, s=1, p=1 ---\n");
    test_fp16acc<4,3,3,1,1,1,1>( 64, 80, 80,  64, "base");
    test_fp16acc<4,3,3,1,1,1,1>( 32, 40, 40,  32, "smaller");
    test_fp16acc<2,3,3,1,1,1,1>( 32, 43, 43,  48, "C_out boundary");
    test_fp16acc<1,3,3,1,1,1,1>( 64, 80, 80,  64, "T=1");

    printf("\n--- Variant C: 3x3, s=2, p=1 ---\n");
    test_fp16acc<4,3,3,2,2,1,1>( 64, 80, 80,  64, "H_out=40");
    test_fp16acc<4,3,3,2,2,1,1>( 32, 80, 80,  64, "C expand");
    test_fp16acc<4,3,3,2,2,1,1>( 16, 40, 40,  32, "small H=40");
    test_fp16acc<2,3,3,2,2,1,1>( 32, 43, 43, 128, "boundary");

    printf("\n=== fp16w+fp32acc  vs  fp16w+fp16acc  benchmark ===\n\n");

    benchmark_fp32acc_vs_fp16acc<4,1,1,1,1,0,0>( 64, 80, 80,  64, "1x1 C64 H80");
    benchmark_fp32acc_vs_fp16acc<4,1,1,1,1,0,0>(128, 40, 40, 128, "1x1 C128 H40");
    benchmark_fp32acc_vs_fp16acc<4,1,1,1,1,0,0>(192, 40, 40, 128, "1x1 C192->128 H40");
    benchmark_fp32acc_vs_fp16acc<4,1,1,1,1,0,0>(256, 20, 20, 128, "1x1 C256->128 H20");
    benchmark_fp32acc_vs_fp16acc<4,3,3,1,1,1,1>( 64, 80, 80,  64, "3x3s1 C64 H80");
    benchmark_fp32acc_vs_fp16acc<4,3,3,1,1,1,1>( 32, 40, 40,  32, "3x3s1 C32 H40");
    benchmark_fp32acc_vs_fp16acc<4,3,3,1,1,1,1>(128, 20, 20, 128, "3x3s1 C128 H20");
    benchmark_fp32acc_vs_fp16acc<4,3,3,2,2,1,1>( 64, 80, 80,  64, "3x3s2 C64 H80->40");
    benchmark_fp32acc_vs_fp16acc<2,3,3,1,1,1,1>( 64, 80, 80,  64, "T=2 3x3s1 C64 H80");
    benchmark_fp32acc_vs_fp16acc<1,3,3,1,1,1,1>( 64, 80, 80,  64, "T=1 3x3s1 C64 H80");

    printf("\n=== Done ===\n");
    return 0;
}
#endif // CONV2D_FP16ACC_NO_MAIN

#endif // CONV2D_K64_SPECIALIZED_FP16ACC_0521_H
