#include <bitset>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cuda_utils.cuh"
#include "ptx_utils.cuh"

// =============================================================================
// SNN Conv2D Kernel (padded-weight variant): 64×64 output tile, K_chunk=16
//
// Pre-padding assumptions (offline, before deployment):
//   1. C_out padded to multiple of 64  → no M boundary check in weight cp.async
//   2. in_features (C_in*Kh*Kw) padded to multiple of K_CHUNK=16
//      → no K boundary check in weight cp.async (always emit full 16B)
//
// Optimizations over conv2d_k64.cu:
//   A. Weight cp.async: no boundary predicate, always full 16B
//      → removes 1 setp + branch per thread per K-iter
//   B. Double buffer (ping-pong): 2×5KB=10KB SMEM (still fits 48KB default)
//      → weight load of iter k+1 overlaps compute of iter k
//      → cp.async.wait_group 1 instead of 0 (allow 1 in-flight group)
//   C. cp.async.ca instead of .cg for weights
//      → weights reused across spatial tiles → benefit from L1 cache on SM87
//      → (on SM90 H20 the .cg/.ca difference is smaller; on SM87 Jetson L1=32KB)
//
// SMEM layout (double buffer, 10KB):
//   smemweight[2][K_CHUNK=16][M_TILE=64] float  = 2×4KB = 8KB
//   smeminput [2][K_CHUNK=16][N_TILE=64] uint8  = 2×1KB = 2KB
//   Total: 10KB  (Jetson 48KB default: 4 blocks/SM; reg-limited anyway)
//
// Input load (spatial boundary) is unchanged — spatial tiles still need guards.
// =============================================================================

template <int T_STEPS>
__global__ void snn_conv2d_64x64_k16_padded(
    const uint8_t * __restrict__ inputs,   // [1, C_in, H, W] uint8 packed spikes
    const float   * __restrict__ weights,  // [C_in*Kh*Kw_padded, C_out_padded] float, col-major
    float         * __restrict__ outputs,  // [T, C_out, H_out, W_out] float
    Conv2DParam param,
    int out_ch_padded)                     // padded C_out stride in weight matrix
{
    constexpr int K_CHUNK  = 16;
    constexpr int M_TILE   = 64;
    constexpr int N_TILE   = 64;

    // Double buffer SMEM:
    //   smemweight[2][K_CHUNK][M_TILE] = 2×4096 B = 8KB
    //   smeminput [2][K_CHUNK][N_TILE] = 2×1024 B = 2KB
    //   Epilogue reuses full 8KB (8 warps × 1KB each)
    //   Allocate max(10KB epilogue-reuse part, 10KB k-loop part) → just 10KB.
    //   Epilogue only needs 8KB → 10KB covers both.
    __shared__ __align__(128) char smem[10 * 1024];

    // Weight double buffer: smem[0..8191]
    float   *smemweight[2];
    smemweight[0] = reinterpret_cast<float *>(smem);
    smemweight[1] = reinterpret_cast<float *>(smem + 4 * 1024);

    // Input double buffer: smem[8192..10239]
    uint8_t *smeminput[2];
    smeminput[0] = reinterpret_cast<uint8_t *>(smem + 8 * 1024);
    smeminput[1] = reinterpret_cast<uint8_t *>(smem + 9 * 1024);

    // --- Thread indices ---
    const int tid     = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;

    const int mma_tid_x = lane_id / 16 * 2 + lane_id % 2;  // 0..3
    const int mma_tid_y = lane_id % 16 / 2;                 // 0..7

    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;

    const int thread_m_base = warp_m * 16 + mma_tid_y * 2;
    const int thread_n_base = warp_n * 32 + mma_tid_x * 8;

    const int m_tile_base = blockIdx.y * M_TILE;
    const int n_tile_base = blockIdx.x * N_TILE;

    // --- Accumulators ---
    float output_frag[T_STEPS][2][8];
#pragma unroll
    for (int t = 0; t < T_STEPS; t++)
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 8; j++)
                output_frag[t][i][j] = 0.f;

    const int in_features = param.inChKhKw;   // guaranteed multiple of K_CHUNK after padding
    const int k_iters     = in_features / K_CHUNK;

    // ----------------------------------------------------------------
    // Helper lambda (inlined): load weight tile for one K-chunk into
    // one SMEM buffer.  No boundary predicate — weights are pre-padded.
    // Uses cp.async.ca (cache-all) so weights may hit L1 on Jetson SM87.
    // ----------------------------------------------------------------
    auto load_weight = [&](int k_iter, int buf)
    {
        const int k_base = k_iter * K_CHUNK;
        const int w_row  = tid / 16;          // 0..15
        const int w_col  = (tid % 16) * 4;    // 0,4,8,...,60

        uint32_t smem_ptr = ptx::smem_u32addr(
            &smemweight[buf][w_row * M_TILE + w_col]);
        const float *src = &weights[(k_base + w_row) * out_ch_padded
                                    + m_tile_base + w_col];

        // cp.async.ca: cache weights in L1+L2 (benefits repeated block access)
        // No src_size predicate — always full 16B load
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 16;\n"
            :: "r"(smem_ptr), "l"(src)
        );
    };

    // ----------------------------------------------------------------
    // Helper lambda: load input tile for one K-chunk into one SMEM buffer.
    // Spatial boundary checks remain necessary.
    // ----------------------------------------------------------------
    auto load_input = [&](int k_iter, int buf)
    {
        const int k_base = k_iter * K_CHUNK;
        const int i_k    = tid / 16;
        const int i_n4   = tid % 16;
        const int global_k = k_base + i_k;

        uint32_t packed = 0;
        if (global_k < in_features)
        {
            int c_idx = global_k / param.KhKw;
            int ky    = global_k % param.KhKw / param.Kw;
            int kx    = global_k % param.KhKw % param.Kw;
#pragma unroll
            for (int b = 0; b < 4; b++)
            {
                int n_idx = n_tile_base + i_n4 * 4 + b;
                if (n_idx < (int)param.outHW)
                {
                    int oh = n_idx / param.out_w;
                    int ow = n_idx % param.out_w;
                    int ih = oh * param.Sh - param.Ph + ky;
                    int iw = ow * param.Sw - param.Pw + kx;
                    if (ih >= 0 && ih < (int)param.in_h &&
                        iw >= 0 && iw < (int)param.in_w)
                    {
                        packed |= (uint32_t)inputs[c_idx * param.inHW + ih * param.in_w + iw]
                                  << (b * 8);
                    }
                }
            }
        }
        uint32_t smeminput_ptr = ptx::smem_u32addr(
            &reinterpret_cast<uint32_t *>(smeminput[buf])[i_k * (N_TILE / 4) + i_n4]);
        ptx::sts32(packed, smeminput_ptr);
    };

    // ----------------------------------------------------------------
    // Double-buffer prologue: preload buffer 0 for k_iter=0
    // ----------------------------------------------------------------
    if (k_iters > 0)
    {
        load_weight(0, 0);
        load_input(0, 0);
        asm volatile("cp.async.commit_group;\n" :::);
    }

    // ----------------------------------------------------------------
    // K-loop: compute iter k while loading iter k+1 into next buffer
    //
    //   wait_group 1: allow 1 in-flight cp.async group (the next iter's)
    //   while waiting, compute current iter from smemweight[cur]/smeminput[cur]
    // ----------------------------------------------------------------
    for (int k_iter = 0; k_iter < k_iters; k_iter++)
    {
        const int cur  = k_iter & 1;          // buffer in use this iter
        const int next = cur ^ 1;

        // Issue prefetch for k_iter+1 into next buffer (if exists)
        if (k_iter + 1 < k_iters)
        {
            load_weight(k_iter + 1, next);
            load_input(k_iter + 1, next);
            asm volatile("cp.async.commit_group;\n" :::);
        }

        // Wait for *current* buffer's cp.async to complete
        // wait_group 1 → waits until at most 1 group is still in flight
        // (the one we just issued above for k_iter+1)
        asm volatile("cp.async.wait_group 1;\n" :::);
        __syncthreads();

        // Compute: [2M × 8N] outer product from SMEM
        float    weight_frag[2];
        uint32_t input_frag_lo, input_frag_hi;

#pragma unroll
        for (int k = 0; k < K_CHUNK; k++)
        {
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
                        add_f32(output_frag[t][i][j], weight_frag[i], spike);
                    }
                }
            }
        }

        __syncthreads();
    }

    // Drain remaining in-flight groups
    asm volatile("cp.async.wait_all;\n" :::);

    // ----------------------------------------------------------------
    // Epilogue: reg → smem → gmem  (same pattern as conv2d_k64.cu)
    // Epilogue reuses smem[0..8191] (8 warps × 1KB each)
    // ----------------------------------------------------------------
    float       *warp_smem      = reinterpret_cast<float *>(smem) + warp_id * 256;
    uint32_t     warp_smem_base = ptx::smem_u32addr(warp_smem);
    const float *smemout_lds    = warp_smem + lane_id;

    const int warp_m_global = m_tile_base + warp_m * 16;
    const int warp_n_global = n_tile_base + warp_n * 32;

    const int  n_global = warp_n_global + lane_id;
    const bool n_valid  = (n_global < (int)param.outHW);

    for (int t = 0; t < T_STEPS; t++)
    {
        float *out_t = outputs + (size_t)t * param.out_ch * param.outHW;

#pragma unroll
        for (int b = 0; b < 2; b++)
        {
            const int smem_row = (mma_tid_y % 4) * 2;

            __syncthreads();

            if (mma_tid_y / 4 == b)
            {
                const uint32_t sts_r0 = warp_smem_base
                    + (uint32_t)(smem_row * 32 + mma_tid_x * 8) * sizeof(float);
                const uint32_t sts_r1 = warp_smem_base
                    + (uint32_t)((smem_row + 1) * 32 + mma_tid_x * 8) * sizeof(float);

                ptx::sts128(output_frag[t][0][0], output_frag[t][0][1],
                            output_frag[t][0][2], output_frag[t][0][3], sts_r0);
                ptx::sts128(output_frag[t][0][4], output_frag[t][0][5],
                            output_frag[t][0][6], output_frag[t][0][7],
                            sts_r0 + 4 * (uint32_t)sizeof(float));

                ptx::sts128(output_frag[t][1][0], output_frag[t][1][1],
                            output_frag[t][1][2], output_frag[t][1][3], sts_r1);
                ptx::sts128(output_frag[t][1][4], output_frag[t][1][5],
                            output_frag[t][1][6], output_frag[t][1][7],
                            sts_r1 + 4 * (uint32_t)sizeof(float));
            }

            __syncthreads();

            const int m_base = warp_m_global + b * 8;
#pragma unroll
            for (int row = 0; row < 8; row++)
            {
                const int  m_global = m_base + row;
                const bool m_valid  = (m_global < (int)param.out_ch);
                ptx::stg32(smemout_lds[row * 32],
                           out_t + (size_t)m_global * param.outHW + n_global,
                           m_valid && n_valid);
            }
        }
    }
}


// =============================================================================
// Test infrastructure for the padded-weight variant
// =============================================================================

// Pad weights offline: C_out to multiple of 64, in_features to multiple of K_CHUNK=16
static void pad_weights(
    const float *src, float *dst,
    int in_features, int C_out,
    int in_features_padded, int C_out_padded)
{
    // src: [in_features][C_out], dst: [in_features_padded][C_out_padded]
    for (int k = 0; k < in_features_padded; k++)
    {
        for (int m = 0; m < C_out_padded; m++)
        {
            dst[k * C_out_padded + m] =
                (k < in_features && m < C_out) ? src[k * C_out + m] : 0.f;
        }
    }
}

void snn_conv2d_padded_launch(
    const uint8_t *d_inputs, const float *d_weights_padded,
    float *d_outputs, Conv2DParam &param, int T, int out_ch_padded)
{
    dim3 block(256);
    dim3 grid(
        (param.outHW  + 63) / 64,
        (param.out_ch + 63) / 64,
        1
    );

    switch (T) {
        case 1: snn_conv2d_64x64_k16_padded<1><<<grid, block>>>(d_inputs, d_weights_padded, d_outputs, param, out_ch_padded); break;
        case 2: snn_conv2d_64x64_k16_padded<2><<<grid, block>>>(d_inputs, d_weights_padded, d_outputs, param, out_ch_padded); break;
        case 3: snn_conv2d_64x64_k16_padded<3><<<grid, block>>>(d_inputs, d_weights_padded, d_outputs, param, out_ch_padded); break;
        case 4: snn_conv2d_64x64_k16_padded<4><<<grid, block>>>(d_inputs, d_weights_padded, d_outputs, param, out_ch_padded); break;
    }
}

// CPU reference (reused from conv2d_k64.cu logic)
static void snn_conv2d_cpu_ref_padded(
    const uint8_t *inputs,
    const float   *weights,   // [in_features][C_out] (unpadded)
    float         *outputs,
    int T, int C_in, int H, int W,
    int C_out, int Kh, int Kw,
    int stride, int pad)
{
    int H_out = (H + 2 * pad - Kh) / stride + 1;
    int W_out = (W + 2 * pad - Kw) / stride + 1;

    for (int t = 0; t < T; t++) {
        for (int k = 0; k < C_out; k++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    float sum = 0.f;
                    for (int c = 0; c < C_in; c++) {
                        for (int ky = 0; ky < Kh; ky++) {
                            for (int kx = 0; kx < Kw; kx++) {
                                int ih = oh * stride - pad + ky;
                                int iw = ow * stride - pad + kx;
                                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                                uint8_t packed = inputs[c * H * W + ih * W + iw];
                                int spike = (packed >> t) & 1;
                                if (spike) {
                                    int feat_idx = c * Kh * Kw + ky * Kw + kx;
                                    sum += weights[feat_idx * C_out + k];
                                }
                            }
                        }
                    }
                    outputs[t * C_out * H_out * W_out + k * H_out * W_out + oh * W_out + ow] = sum;
                }
            }
        }
    }
}

void snn_conv2d_padded_test(int T, int C_in, int H, int W, int C_out,
                             int Kh, int Kw, int stride, int pad)
{
    constexpr int K_CHUNK = 16;

    const int H_out           = (H + 2*pad - Kh) / stride + 1;
    const int W_out           = (W + 2*pad - Kw) / stride + 1;
    const int in_features     = C_in * Kh * Kw;
    const int in_feat_padded  = (in_features + K_CHUNK - 1) / K_CHUNK * K_CHUNK;
    const int C_out_padded    = (C_out + 63) / 64 * 64;

    printf("  T=%d C_in=%d H=%d C_out=%d K=%dx%d s=%d p=%d → H_out=%d "
           "in_feat=%d→%d C_out=%d→%d  ",
           T, C_in, H, C_out, Kh, Kw, stride, pad, H_out,
           in_features, in_feat_padded, C_out, C_out_padded);

    size_t input_sz   = (size_t)C_in * H * W;
    size_t weight_sz  = (size_t)in_features * C_out;
    size_t weightP_sz = (size_t)in_feat_padded * C_out_padded;
    size_t output_sz  = (size_t)T * C_out * H_out * W_out;

    uint8_t *h_inputs    = new uint8_t[input_sz];
    float   *h_weights   = new float[weight_sz];
    float   *h_weightsP  = new float[weightP_sz];
    float   *h_outputs   = new float[output_sz];
    float   *h_ref       = new float[output_sz];

    srand(42);
    for (size_t i = 0; i < input_sz; i++) {
        uint8_t packed = 0;
        for (int t = 0; t < T; t++) {
            if ((rand() & 1)) packed |= (1u << t);
        }
        h_inputs[i] = packed;
    }
    for (size_t i = 0; i < weight_sz; i++)
        h_weights[i] = (float)(rand() & 255) / 256.f;

    pad_weights(h_weights, h_weightsP, in_features, C_out, in_feat_padded, C_out_padded);

    uint8_t *d_inputs;
    float   *d_weightsP, *d_outputs;
    cudaMalloc(&d_inputs,   input_sz   * sizeof(uint8_t));
    cudaMalloc(&d_weightsP, weightP_sz * sizeof(float));
    cudaMalloc(&d_outputs,  output_sz  * sizeof(float));
    cudaMemset(d_outputs, 0, output_sz * sizeof(float));

    cudaMemcpy(d_inputs,   h_inputs,   input_sz   * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weightsP, h_weightsP, weightP_sz * sizeof(float),   cudaMemcpyHostToDevice);

    Conv2DParam param;
    param.in_h          = H;
    param.in_w          = W;
    param.inHW          = H * W;
    param.inChKhKw      = in_feat_padded;   // padded — no K boundary check in kernel
    param.inBatchNumel  = C_in * H * W;
    param.out_ch        = C_out;             // real C_out (used for output validity guard)
    param.out_w         = W_out;
    param.outHW         = H_out * W_out;
    param.outBatchNumel = C_out * H_out * W_out;
    param.Kh            = Kh;
    param.Kw            = Kw;
    param.KhKw          = Kh * Kw;
    param.Sh            = stride;
    param.Sw            = stride;
    param.Ph            = pad;
    param.Pw            = pad;

    snn_conv2d_padded_launch(d_inputs, d_weightsP, d_outputs, param, T, C_out_padded);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  CUDA error: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_outputs, d_outputs, output_sz * sizeof(float), cudaMemcpyDeviceToHost);

    snn_conv2d_cpu_ref_padded(h_inputs, h_weights, h_ref,
                               T, C_in, H, W, C_out, Kh, Kw, stride, pad);

    {
        int errors = 0;
        for (size_t i = 0; i < output_sz; i++) {
            if (fabsf(h_outputs[i] - h_ref[i]) > 0.001f) {
                if (errors < 5)
                    printf("  Err[%zu]: gpu=%.4f cpu=%.4f\n", i, h_outputs[i], h_ref[i]);
                errors++;
            }
        }
        printf("  %s (%d errors)\n", errors == 0 ? "PASSED!" : "FAILED", errors);
    }

cleanup:
    cudaFree(d_inputs);
    cudaFree(d_weightsP);
    cudaFree(d_outputs);
    delete[] h_inputs;
    delete[] h_weights;
    delete[] h_weightsP;
    delete[] h_outputs;
    delete[] h_ref;
}


int main()
{
    printf("\n=== snn_conv2d_64x64_k16_padded tests (double-buf, no weight boundary check) ===\n");

    // T=1..4, 3×3 stride=1 pad=1
    for (int t = 1; t <= 4; t++)
        snn_conv2d_padded_test(t,  64, 80, 80,  64, 3, 3, 1, 1);

    // C_out not multiple of 64 (padded to 64)
    snn_conv2d_padded_test(2,  64, 80, 80,  48, 3, 3, 1, 1);

    // 1×1 conv
    snn_conv2d_padded_test(4,  64, 80, 80,  64, 1, 1, 1, 0);

    // stride=2
    snn_conv2d_padded_test(4,  64, 80, 80,  64, 3, 3, 2, 1);

    // larger C_in
    snn_conv2d_padded_test(4, 128, 40, 40, 128, 3, 3, 1, 1);

    // spatial not multiple of 64
    snn_conv2d_padded_test(2,  32, 43, 43,  32, 3, 3, 1, 1);

    return 0;
}
