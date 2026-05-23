#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cuda_utils.cuh"
#include "ptx_utils.cuh"

// =============================================================================
// Optimized SNN Conv2D — fp16 weight + fp16 accumulator, 64×64 tile, K_chunk=16
//
// Opt1 vs fp16acc v0:
//   - output_frag: __half2[T][8] (vectorized) instead of __half[T][2][8]
//   - compute:     @p add.f16x2  (one op per 2 output rows) instead of 2× @p add.f16
//   - epilogue:    __lows2half2/__highs2half2 to pack row0/row1 separately
//
// Note on cp.async.cg: requires size=16 bytes. Current 8-byte weight load
// (4 halves per thread, 16 threads per K-row) cannot use .cg directly.
//
// Why Opt1 matters:
//   The original code accumulates M-row-0 and M-row-1 in two separate __half
//   scalars with two predicated adds.  Since both rows use the SAME spike value
//   (spike depends only on N-position and time-step, not on M-row), we can pack
//   the two weights into a single __half2 and use add.f16x2 — halving the number
//   of arithmetic instructions in the inner loop.
// =============================================================================


template <int T_STEPS, int Kh, int Kw, int Sh, int Sw, int Ph, int Pw>
__global__ __launch_bounds__(256, 3)
void snn_conv2d_64x64_k16_fp16_u8(
    const uint8_t * __restrict__ inputs,
    const __half  * __restrict__ weights,
    const __half2 * __restrict__ bias,
    __half        * __restrict__ outputs,
    Conv2DParam param,
    int out_ch_padded)
{
    constexpr int K_CHUNK = 16;
    constexpr int M_TILE  = 64;
    constexpr int N_TILE  = 64;
    constexpr int KhKw    = Kh * Kw;

    // SMEM layout identical to v0: max(6 KB k-loop, 4 KB epilogue) = 6 KB
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
    __half2 bias_h2 = {};
    if (m_global < (int)param.out_ch)
        bias_h2 = bias[m_global >> 1];

    // Opt1: __half2 accumulators — .x = M-row 0 result, .y = M-row 1 result
    __half2 output_frag[T_STEPS][8];
#pragma unroll
    for (int t = 0; t < T_STEPS; t++)
#pragma unroll
        for (int j = 0; j < 8; j++)
            output_frag[t][j] = bias_h2;

    const int in_features = param.inChKhKw;
    const int k_iters     = in_features / K_CHUNK;

    // Load weight: cp.async.ca (8-byte granularity; cg requires 16B, not applicable here)
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
            // Load 2 consecutive fp16 weights → one __half2 (32-bit)
            // weight2.x = weight for M-row thread_m_base
            // weight2.y = weight for M-row thread_m_base+1
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
                    // Single add.f16x2 replaces 2× add.f16 from v0
                    ptx::add_f16x2(output_frag[t][j], weight2, spike);
                }
            }
        }

        __syncthreads();
    }

    asm volatile("cp.async.wait_all;\n" :::);

    // ----------------------------------------------------------------
    // Epilogue: identical smem layout to v0 (8 warps × 256 half × 2B = 4 KB)
    //
    // output_frag[t][j] is __half2{row0_acc, row1_acc}.
    // Packing into two sts128 rows per (t,b):
    //   row0 uint32[p] = __lows2half2(frag[p*2], frag[p*2+1])
    //                  = {frag[p*2].x, frag[p*2+1].x} = {row0[j=2p], row0[j=2p+1]}
    //   row1 uint32[p] = __highs2half2(frag[p*2], frag[p*2+1])
    //                  = {frag[p*2].y, frag[p*2+1].y} = {row1[j=2p], row1[j=2p+1]}
    // ----------------------------------------------------------------
    __half      *warp_smem      = reinterpret_cast<__half *>(smem) + warp_id * 256;
    uint32_t     warp_smem_base = ptx::smem_u32addr(warp_smem);
    const __half *smemout_lds   = warp_smem + lane_id;

    const int warp_m_global = m_tile_base + warp_m * 16;
    const int warp_n_global = n_tile_base + warp_n * 32;

    const int  n_global = warp_n_global + lane_id;
    const bool n_valid  = (n_global < (int)param.outHW);

    for (int t = 0; t < T_STEPS; t++)
    {
        __half *out_t = outputs + (size_t)t * param.out_ch * param.outHW;

#pragma unroll
        for (int b = 0; b < 2; b++)
        {
            const int smem_row = (mma_tid_y % 4) * 2;

            __syncthreads();

            if (mma_tid_y / 4 == b)
            {
                const uint32_t base_r0 = warp_smem_base
                    + (uint32_t)(smem_row * 32 + mma_tid_x * 8) * sizeof(__half);
                const uint32_t base_r1 = warp_smem_base
                    + (uint32_t)((smem_row + 1) * 32 + mma_tid_x * 8) * sizeof(__half);

                uint32_t r0[4], r1[4];
#pragma unroll
                for (int p = 0; p < 4; p++) {
                    // __lows2half2(a,b)  = {a.x, b.x} = row-0 values at j=2p, 2p+1
                    // __highs2half2(a,b) = {a.y, b.y} = row-1 values at j=2p, 2p+1
                    __half2 h2_r0 = __lows2half2(output_frag[t][p*2], output_frag[t][p*2+1]);
                    __half2 h2_r1 = __highs2half2(output_frag[t][p*2], output_frag[t][p*2+1]);
                    r0[p] = *reinterpret_cast<uint32_t *>(&h2_r0);
                    r1[p] = *reinterpret_cast<uint32_t *>(&h2_r1);
                }
                ptx::sts128(r0[0], r0[1], r0[2], r0[3], base_r0);
                ptx::sts128(r1[0], r1[1], r1[2], r1[3], base_r1);
            }

            __syncthreads();

            const int m_base = warp_m_global + b * 8;
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
    }
}


// =============================================================================
// Weight padding (unchanged)
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
// CPU reference — accumulates in fp16 to match fp16acc kernel behavior
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
// Launch wrapper
// =============================================================================

template <int T, int Kh, int Kw, int Sh, int Sw, int Ph, int Pw>
void snn_conv2d_fp16acc_launch(
    const uint8_t *d_inputs, const __half *d_weights_padded,
    const __half2 *d_bias,
    __half *d_outputs, Conv2DParam &param, int out_ch_padded)
{
    dim3 block(256);
    dim3 grid((param.outHW + 63) / 64, (param.out_ch + 63) / 64, 1);
    snn_conv2d_64x64_k16_fp16_u8<T, Kh, Kw, Sh, Sw, Ph, Pw>
        <<<grid, block>>>(d_inputs, d_weights_padded, d_bias, d_outputs, param, out_ch_padded);
}


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
    float   *h_bias = new float[C_out];
    __half2 *h_bias2 = new __half2[Co_p / 2];
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
    for (int m = 0; m < C_out; m++)
        h_bias[m] = (float)(rand() & 63) / 64.f;  // small bias values
    for (int m = 0; m < Co_p; m += 2)
        h_bias2[m >> 1] = __half2{__float2half(h_bias[m]),
            __float2half(m + 1 < C_out ? h_bias[m + 1] : 0.f)};
    pad_weights_fp16acc(h_wf, h_wp, in_feat, C_out, in_feat_p, Co_p);

    uint8_t *d_in; __half *d_wp; __half2 *d_bias; __half *d_out;
    cudaMalloc(&d_in,   isz  * sizeof(uint8_t));
    cudaMalloc(&d_wp,   wpsz * sizeof(__half));
    cudaMalloc(&d_bias, Co_p / 2 * sizeof(__half2));
    cudaMalloc(&d_out,  osz  * sizeof(__half));
    cudaMemset(d_out, 0, osz * sizeof(__half));
    cudaMemcpy(d_in,   h_in,   isz   * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wp,   h_wp,   wpsz  * sizeof(__half),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias2, Co_p / 2 * sizeof(__half2), cudaMemcpyHostToDevice);

    Conv2DParam p;
    p.in_h=H; p.in_w=W; p.inHW=H*W; p.inChKhKw=in_feat_p; p.inBatchNumel=C_in*H*W;
    p.out_ch=C_out; p.out_w=W_out; p.outHW=H_out*W_out; p.outBatchNumel=C_out*H_out*W_out;
    p.Kh=Kh; p.Kw=Kw; p.KhKw=KhKw; p.Sh=Sh; p.Sw=Sw; p.Ph=Ph; p.Pw=Pw;

    snn_conv2d_fp16acc_launch<T,Kh,Kw,Sh,Sw,Ph,Pw>(d_in, d_wp, d_bias, d_out, p, Co_p);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); goto cleanup; }
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, osz*sizeof(__half), cudaMemcpyDeviceToHost);

    cpu_ref_fp16acc<Kh,Kw,Sh,Sw,Ph,Pw>(h_in, h_wf, h_ref, T, C_in, H, W, C_out);

    // Add bias to CPU reference
    { size_t hw = H_out * W_out;
        for (size_t i = 0; i < osz; i++)
            h_ref[i] = __hadd(h_ref[i], __float2half(h_bias[(i / hw) % C_out]));
    }

    {
        int errors = 0; float max_diff = 0.f;
        for (size_t i = 0; i < osz; i++) {
            float diff = fabsf(__half2float(h_out[i]) - __half2float(h_ref[i]));
            float tol = (float)in_feat_p * 1e-3f + 1e-2f;
            if (diff > tol) { if (errors<5) printf("\n    Err[%zu] gpu=%.3f ref=%.3f diff=%.3f", i, __half2float(h_out[i]), __half2float(h_ref[i]), diff); errors++; }
            if (diff > max_diff) max_diff = diff;
        }
        printf("  %s (%d errors, max_diff=%.4f)\n", errors==0?"PASSED!":"FAILED", errors, max_diff);
    }
cleanup:
    cudaFree(d_in); cudaFree(d_wp); cudaFree(d_bias); cudaFree(d_out);
    delete[] h_in; delete[] h_wf; delete[] h_wp; delete[] h_bias; delete[] h_bias2; delete[] h_out; delete[] h_ref;
}


int main()
{
    printf("\n=== conv2d_fp16acc (Opt1: __half2 SIMD vectorization) ===\n");

    printf("\nVariant A: 1x1, s=1, p=0\n");
    test_fp16acc<4,1,1,1,1,0,0>( 64, 80, 80,  64, "base");
    test_fp16acc<4,1,1,1,1,0,0>(128, 40, 40, 128, "larger");
    test_fp16acc<2,1,1,1,1,0,0>( 64, 80, 80,  48, "C_out boundary");
    test_fp16acc<4,1,1,1,1,0,0>( 64, 40, 40, 128, "expansion x2");
    test_fp16acc<4,1,1,1,1,0,0>(128, 40, 40,  64, "squeeze x0.5");

    printf("\nVariant B: 3x3, s=1, p=1\n");
    test_fp16acc<4,3,3,1,1,1,1>( 64, 80, 80,  64, "base");
    test_fp16acc<4,3,3,1,1,1,1>( 32, 40, 40,  32, "smaller");
    test_fp16acc<2,3,3,1,1,1,1>( 32, 43, 43,  48, "C_out boundary");
    test_fp16acc<1,3,3,1,1,1,1>( 64, 80, 80,  64, "T=1");

    printf("\nVariant C: 3x3, s=2, p=1\n");
    test_fp16acc<4,3,3,2,2,1,1>( 64, 80, 80,  64, "H_out=40");
    test_fp16acc<4,3,3,2,2,1,1>( 32, 80, 80,  64, "C expand");
    test_fp16acc<4,3,3,2,2,1,1>( 16, 40, 40,  32, "small H=40");
    test_fp16acc<2,3,3,2,2,1,1>( 32, 43, 43, 128, "boundary");

    printf("\n=== Done ===\n");
    return 0;
}
