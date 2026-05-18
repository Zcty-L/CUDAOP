#include <bitset>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cuda_utils.cuh"
#include "ptx_utils.cuh"


// =============================================================================
// SNN Conv2D Kernel: 64×64 output tile, K_chunk=16, T time-steps
//
// Layout:
//   Input  : [1, C_in, H, W]  uint8  (T bits packed per element, bit t = time step t)
//   Weights: [C_in*Kh*Kw, C_out] float (pre-transposed, column-major)
//   Output : [T, C_out, H_out, W_out] float
//
// Block tile: M_TILE=64 (C_out) × N_TILE=64 (spatial), K_chunk=16
// Threads   : 256 (8 warps)
// Grid      : (ceil(outHW/64), ceil(C_out/64), 1)
//
// SMEM (single buffer, ~5KB):
//   smemweight [K_CHUNK=16][M_TILE=64] float  = 4KB  (cp.async, 1 float4/thread)
//   smeminput  [K_CHUNK=16][N_TILE=64] uint8  = 1KB  (ldg8×N + sts)
//
// Thread decomposition (from thread_map_test.cu, verified 0 bank conflicts):
//   lane_id   = threadIdx.x % 32
//   warp_id   = threadIdx.x / 32   (0..7)
//   mma_tid_x = lane_id / 16 * 2 + lane_id % 2   (0..3, 4 cols)
//   mma_tid_y = lane_id % 16 / 2                  (0..7, 8 rows)
//
// Per-warp tile: [16M × 32N]
//   warp_m = warp_id / 2  (0..3) → M [warp_m*16, warp_m*16+16)
//   warp_n = warp_id % 2  (0..1) → N [warp_n*32, warp_n*32+32)
//
// Per-thread fragment: [2M × 8N]  (matches thread_map_test read granularity)
//   mma_tid_y (0..7) → weight float2 broadcast (8 rows × float2 = 16 floats/row)
//     smemweight[k][warp_m*16 + mma_tid_y*2 .. +2]  — 4 threads share same float2
//   mma_tid_x (0..3) → input  uint64 broadcast (4 cols × uint64 = 32 uint8/row)
//     smeminput [k][warp_n*32 + mma_tid_x*8 .. +8]  — 8 threads share same uint64
//
//   output_frag[T][2][8]: [2M rows] × [8N cols] × T time-steps
// =============================================================================

template <int T_STEPS>
__global__ void snn_conv2d_64x64_k16(
    const uint8_t * __restrict__ inputs,   // [1, C_in, H, W] uint8 packed spikes
    const float   * __restrict__ weights,  // [C_in*Kh*Kw, C_out] float, col-major
    float         * __restrict__ outputs,  // [T, C_out, H_out, W_out] float
    Conv2DParam param)
{
    constexpr int K_CHUNK = 16;
    constexpr int M_TILE  = 64;
    constexpr int N_TILE  = 64;

    // SMEM: K-loop uses 5KB (4KB weight + 1KB input)
    //       Epilogue uses 8KB (8 warps × 1KB each)
    //       Allocate max(5KB, 8KB) = 8KB
    __shared__ __align__(128) char smem[8 * 1024];
    float   *smemweight = reinterpret_cast<float   *>(smem);
    uint8_t *smeminput  = reinterpret_cast<uint8_t *>(smem + 4 * 1024);

    // --- Thread indices ---
    const int tid     = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;

    // Thread map verified in thread_map_test.cu (0 bank conflicts)
    const int mma_tid_x = lane_id / 16 * 2 + lane_id % 2;  // 0..3
    const int mma_tid_y = lane_id % 16 / 2;                 // 0..7

    // Warp-level tile assignment
    const int warp_m = warp_id / 2;  // 0..3 → M rows [warp_m*16, warp_m*16+16)
    const int warp_n = warp_id % 2;  // 0..1 → N cols [warp_n*32, warp_n*32+32)

    // Per-thread SMEM read base (matches smem_float2_load / smem_uint64_load)
    //   weight: mma_tid_y selects 1 float2 out of 8 → m offset = mma_tid_y*2
    //   input : mma_tid_x selects 1 uint64 out of 4 → n offset = mma_tid_x*8 (bytes)
    const int thread_m_base = warp_m * 16 + mma_tid_y * 2;  // 0..62, step 2
    const int thread_n_base = warp_n * 32 + mma_tid_x * 8;  // 0..56, step 8

    // --- Global tile offsets ---
    const int m_tile_base = blockIdx.y * M_TILE;
    const int n_tile_base = blockIdx.x * N_TILE;

    // --- Accumulators: T_STEPS × [2M × 8N] ---
    float output_frag[T_STEPS][2][8];
#pragma unroll
    for (int t = 0; t < T_STEPS; t++)
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 8; j++)
                output_frag[t][i][j] = 0.f;

    // --- K-loop ---
    const int in_features = param.inChKhKw;
    const int k_iters     = (in_features + K_CHUNK - 1) / K_CHUNK;

    for (int k_iter = 0; k_iter < k_iters; k_iter++)
    {
        const int k_base = k_iter * K_CHUNK;

        // ----------------------------------------------------------------
        // Stage 1: cp.async weights [K_CHUNK, M_TILE] → smemweight
        //   256 threads × 1 float4 = 256 × 16B = 4KB ✓
        //   tid/16 → k row (0..15), tid%16 → float4 col (0..15, col*4=m)
        // ----------------------------------------------------------------
        {
            int w_row    = tid / 16;
            int w_col    = (tid % 16) * 4;
            int global_k = k_base + w_row;
            int global_m = m_tile_base + w_col;

            uint32_t smem_ptr = ptx::smem_u32addr(&smemweight[w_row * M_TILE + w_col]);
            bool valid  = (global_k < in_features) && (global_m + 3 < (int)param.out_ch);
            int  src_sz = valid ? 16 : 0;

            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                :: "r"(smem_ptr),
                   "l"(&weights[global_k * param.out_ch + global_m]),
                   "r"(src_sz)
            );
        }

        // ----------------------------------------------------------------
        // Stage 2: Load inputs [K_CHUNK, N_TILE] → smeminput (uint8)
        //   smeminput layout: [K_CHUNK][N_TILE], each row = 64 uint8
        //   Load granularity: uint64 (8 bytes) per thread
        //   tid/8 → k row (0..31, only 0..15 valid), tid%8 → uint64 col (0..7, col*8=n)
        //   32 threads × 8B × (256/32) passes needed?
        //   Simpler: 256 threads cover 16×64 = 1024 bytes as uint32 (4B each):
        //     tid/16 = k (0..15), tid%16 = n/4 (0..15)  → same mapping as weight load
        // ----------------------------------------------------------------
        {
            int i_k      = tid / 16;
            int i_n4     = tid % 16;          // covers n = [i_n4*4, i_n4*4+4)
            int global_k = k_base + i_k;

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
            // Store as uint32 into smeminput via sts32
            uint32_t smeminput_ptr = ptx::smem_u32addr(&reinterpret_cast<uint32_t *>(smeminput)[i_k * (N_TILE / 4) + i_n4]);
            ptx::sts32(packed, smeminput_ptr);
        }

        asm volatile("cp.async.commit_group;\n" :::);
        asm volatile("cp.async.wait_group 0;\n" :::);
        __syncthreads();

        // ----------------------------------------------------------------
        // Stage 3: Compute [2M × 8N] outer product for this K_chunk
        //
        //   weight read: float2 at smemweight[k][thread_m_base]
        //     mma_tid_y selects row → 4 threads with same mma_tid_y broadcast same float2
        //     → weight_frag[2] = {w[thread_m_base], w[thread_m_base+1]}
        //
        //   input read: uint64 at smeminput[k][thread_n_base]
        //     mma_tid_x selects col → 8 threads with same mma_tid_x broadcast same uint64
        //     → input_frag = 8 consecutive uint8 values (8 spatial positions)
        //
        //   outer product: output_frag[t][i][j] += weight_frag[i]  if spike_t(input_frag[j])
        // ----------------------------------------------------------------
        float    weight_frag[2];
        uint32_t input_frag_lo, input_frag_hi;  // uint64 split into two uint32

#pragma unroll
        for (int k = 0; k < K_CHUNK; k++)
        {
            // lds64: float2 from smemweight — mma_tid_y broadcast (0 bank conflict)
            uint32_t w_addr = ptx::smem_u32addr(&smemweight[k * M_TILE + thread_m_base]);
            ptx::lds64(weight_frag[0], weight_frag[1], w_addr);

            // lds64: uint64 from smeminput split as two uint32 — mma_tid_x broadcast (0 bank conflict)
            uint32_t i_addr = ptx::smem_u32addr(&smeminput[k * N_TILE + thread_n_base]);
            ptx::lds64(input_frag_lo, input_frag_hi, i_addr);

            // T-way parallel conditional accumulation
#pragma unroll
            for (int t = 0; t < T_STEPS; t++)
            {
#pragma unroll
                for (int i = 0; i < 2; i++)
                {
#pragma unroll
                    for (int j = 0; j < 8; j++)
                    {
                        // j=0..3 from lo word, j=4..7 from hi word
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

    // ----------------------------------------------------------------
    // Epilogue: reg → smem → gmem  (mirrors conv2d_32x128x8 pattern)
    //
    // Per-warp tile: [16M × 32N], split into 2 batches of [8M × 32N].
    //   Per batch: 8 × 32 = 256 floats = 1KB / warp, 8 warps = 8KB ✓
    //
    // SMEM epilogue layout (reuse smem after K-loop, 8KB total):
    //   smemout for warp: smem + warp_id * 1024B  → [8M][32N] float
    //
    // Thread (mma_tid_y, mma_tid_x) owns:
    //   M: rows {mma_tid_y*2, mma_tid_y*2+1}  within [0..15] of warp tile
    //   N: cols {mma_tid_x*8 .. mma_tid_x*8+7} within [0..31] of warp tile
    //
    // Batch b (0 or 1) covers warp-local M rows [b*8 .. b*8+7]:
    //   threads with mma_tid_y ∈ {b*4, b*4+1, b*4+2, b*4+3} contribute.
    //   Within the [8M×32N] smem tile, this thread's smem-local row = mma_tid_y%4*2
    //
    // sts128: write 4 floats at [smem_row][mma_tid_x*8+0..3] and [+4..7]
    //         for both M rows (smem_row and smem_row+1)
    //
    // lds + stg32: lane_id covers 32 N positions in one row (32 threads × 1 float)
    //   loop 8 M rows → 8 stg32 calls per batch
    // ----------------------------------------------------------------

    // Epilogue reuses smem, warp gets 256 floats = 1KB
    float       *warp_smem      = reinterpret_cast<float *>(smem) + warp_id * 256;
    uint32_t     warp_smem_base = ptx::smem_u32addr(warp_smem);
    const float *smemout_lds    = warp_smem + lane_id;

    // GMEM base for this warp's tile
    const int warp_m_global = m_tile_base + warp_m * 16;  // first M row
    const int warp_n_global = n_tile_base + warp_n * 32;  // first N col

    // lane_id-based N index and guard
    const int  n_global  = warp_n_global + lane_id;
    const bool n_valid   = (n_global < (int)param.outHW);

    for (int t = 0; t < T_STEPS; t++)
    {
        float *out_t = outputs + (size_t)t * param.out_ch * param.outHW;

#pragma unroll
        for (int b = 0; b < 2; b++)
        {
            // smem-local M row for this thread within the [8M×32N] tile
            // batch b covers warp-local rows [b*8 .. b*8+7]
            // this thread's warp-local rows = mma_tid_y*2, mma_tid_y*2+1
            // smem row = (mma_tid_y - b*4)*2 = (mma_tid_y%4)*2
            const int smem_row = (mma_tid_y % 4) * 2;  // 0,2,4,6

            // sts: write [smem_row][0..7] and [smem_row+1][0..7] for N-block mma_tid_x
            // smem stride = 32 floats/row
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

            // stg: all 32 threads stream 8 rows from smem to GMEM
            // smemout_lds + row*32 → lane_id position in row
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


// CPU reference for multi-T SNN conv (for correctness verification)
void snn_conv2d_cpu_ref(
    const uint8_t *inputs,   // [1, C_in, H, W] packed uint8
    const float   *weights,  // [C_in*Kh*Kw, C_out] float
    float         *outputs,  // [T, C_out, H_out, W_out] float
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


void snn_conv2d_launch(
    const uint8_t *d_inputs, const float *d_weights, float *d_outputs,
    Conv2DParam &param, int T)
{
    dim3 block(256);
    dim3 grid(
        (param.outHW  + 63) / 64,   // spatial tiles
        (param.out_ch + 63) / 64,   // C_out tiles
        1
    );

    switch (T) {
        case 1: snn_conv2d_64x64_k16<1><<<grid, block>>>(d_inputs, d_weights, d_outputs, param); break;
        case 2: snn_conv2d_64x64_k16<2><<<grid, block>>>(d_inputs, d_weights, d_outputs, param); break;
        case 3: snn_conv2d_64x64_k16<3><<<grid, block>>>(d_inputs, d_weights, d_outputs, param); break;
        case 4: snn_conv2d_64x64_k16<4><<<grid, block>>>(d_inputs, d_weights, d_outputs, param); break;
    }
}


void snn_conv2d_test(int T, int C_in, int H, int W, int C_out,
                     int Kh, int Kw, int stride, int pad)
{
    const int H_out      = (H + 2*pad - Kh) / stride + 1;
    const int W_out      = (W + 2*pad - Kw) / stride + 1;
    const int in_features = C_in * Kh * Kw;

    printf("  T=%d C_in=%d H=%d C_out=%d K=%dx%d s=%d p=%d → H_out=%d in_feat=%d  ",
           T, C_in, H, C_out, Kh, Kw, stride, pad, H_out, in_features);

    size_t input_sz  = (size_t)C_in * H * W;
    size_t weight_sz = (size_t)in_features * C_out;
    size_t output_sz = (size_t)T * C_out * H_out * W_out;

    uint8_t *h_inputs  = new uint8_t[input_sz];
    float   *h_weights = new float[weight_sz];
    float   *h_outputs = new float[output_sz];
    float   *h_ref     = new float[output_sz];

    srand(42);
    // Pack T bits per element: bit t = spike at time t (0 or 1)
    for (size_t i = 0; i < input_sz; i++) {
        uint8_t packed = 0;
        for (int t = 0; t < T; t++) {
            if ((rand() & 1)) packed |= (1u << t);
        }
        h_inputs[i] = packed;
    }
    for (size_t i = 0; i < weight_sz; i++) {
        h_weights[i] = (float)(rand() & 255) / 256.f;
    }

    uint8_t *d_inputs;
    float   *d_weights, *d_outputs;
    cudaMalloc(&d_inputs,  input_sz  * sizeof(uint8_t));
    cudaMalloc(&d_weights, weight_sz * sizeof(float));
    cudaMalloc(&d_outputs, output_sz * sizeof(float));
    cudaMemset(d_outputs, 0, output_sz * sizeof(float));

    cudaMemcpy(d_inputs,  h_inputs,  input_sz  * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, weight_sz * sizeof(float),   cudaMemcpyHostToDevice);

    Conv2DParam param;
    param.in_h         = H;
    param.in_w         = W;
    param.inHW         = H * W;
    param.inChKhKw     = in_features;
    param.inBatchNumel = C_in * H * W;
    param.out_ch       = C_out;
    param.out_w        = W_out;
    param.outHW        = H_out * W_out;
    param.outBatchNumel = C_out * H_out * W_out;
    param.Kh           = Kh;
    param.Kw           = Kw;
    param.KhKw         = Kh * Kw;
    param.Sh           = stride;
    param.Sw           = stride;
    param.Ph           = pad;
    param.Pw           = pad;

    snn_conv2d_launch(d_inputs, d_weights, d_outputs, param, T);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  CUDA error: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_outputs, d_outputs, output_sz * sizeof(float), cudaMemcpyDeviceToHost);

    // CPU reference
    snn_conv2d_cpu_ref(h_inputs, h_weights, h_ref,
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
    cudaFree(d_weights);
    cudaFree(d_outputs);
    delete[] h_inputs;
    delete[] h_weights;
    delete[] h_outputs;
    delete[] h_ref;
}


int main() 
{
    printf("\n=== snn_conv2d_64x64_k16 tests ===\n");
    // T=1..4, 3x3 stride=1 pad=1
    for (int t = 1; t <= 4; t++)
        snn_conv2d_test(t,  64, 80, 80,  64, 3, 3, 1, 1);

    // boundary: C_out not multiple of 64
    snn_conv2d_test(2,  64, 80, 80,  48, 3, 3, 1, 1);

    // 1x1 conv
    snn_conv2d_test(4,  64, 80, 80,  64, 1, 1, 1, 0);

    // stride=2
    snn_conv2d_test(4,  64, 80, 80,  64, 3, 3, 2, 1);

    // larger C_in
    snn_conv2d_test(4, 128, 40, 40, 128, 3, 3, 1, 1);

    // spatial not multiple of 64 (outHW=41*41=1681)
    snn_conv2d_test(2,  32, 43, 43,  32, 3, 3, 1, 1);

    return 0;
}
