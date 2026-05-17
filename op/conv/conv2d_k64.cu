#include <bitset>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cuda_utils.cuh"

void direct_conv2d_T_cpu(
        const float *input, const float *filter, const float *bias, float *output,
        int N, int C, int H, int W, int K, int R, int S, int U, int V, int P, int Q)
{
    int Oh = (H + 2 * P - R) / U + 1;
    int Ow = (W + 2 * Q - S) / V + 1;

    for (int n = 0; n < N; n++)
    {
        for (int k = 0; k < K; k++)
        {
            for (int oh = 0; oh < Oh; oh++)
            {
                for (int ow = 0; ow < Ow; ow++)
                {
                    float sum = 0;
                    for (int c = 0; c < C; c++)
                    {
                        for (int r = 0; r < R; r++)
                        {
                            for (int s = 0; s < S; s++)
                            {
                                int ih = oh * U - P + r;
                                int iw = ow * V - Q + s;
                                if (iw >= 0 && ih >= 0 && iw < W && ih < H)
                                {
                                    sum += (input[n * C * H * W + c * (W * H) + ih * W + iw] *
                                            filter[k * R * S * C + c * R * S + r * S + s]);
                                }
                            }
                        }
                    }
                    output[n * K * Oh * Ow + k * Oh * Ow + oh * Ow + ow] = sum + bias[k];
                }
            }
        }
    }
}


__global__ void conv2d_32x128x8_T_S_kernel(
        uint32_t *inputs, float *weights, float *bias, float *outputs, Conv2DParam param)
{
    __shared__ __align__ (16 * 1024) char smem[16 * 1024];
    auto *smemweight = reinterpret_cast<float *>(smem);
    auto *smeminput = reinterpret_cast<uint32_t *>(smem + 8 * 1024);

    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t mma_tid_x = lane_id / 2;
    const uint32_t mma_tid_y = lane_id % 2;

    uint32_t inputs_sts_addr = smem_u32addr(smeminput + (threadIdx.x / 32) * 128 + (threadIdx.x % 32));
    uint32_t weights_sts_addr = smem_u32addr(smemweight + threadIdx.x);

    uint32_t inputs_lds_addr = smem_u32addr(smeminput + (warp_id % 2) * 64 + mma_tid_x * 4);
    uint32_t weights_lds_addr = smem_u32addr(smemweight + (warp_id / 2) * 8 + mma_tid_y * 4);

    const char *input_ldg_ptr = (const char *) (inputs + blockIdx.z * param.inBatchNumel);
    const char *weight_ldg_ptr = (const char *) (
            weights + blockIdx.y * 32 + threadIdx.x / 32 * param.out_ch + threadIdx.x % 32);

    uint32_t input_ldg_reg[4];
    float weight_ldg_reg;

    uint32_t input_frag[2][4];
    float weight_frag[2][4];
    float output_frag[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            output_frag[i][j] = 0;
        }
    }

    int posh_ori[4];
    int posw_ori[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        posh_ori[i] = ((blockIdx.x * 128 + threadIdx.x % 32 + i * 32) / param.out_w) * param.Sh - param.Ph;
        posw_ori[i] = ((blockIdx.x * 128 + threadIdx.x % 32 + i * 32) % param.out_w) * param.Sw - param.Pw;
    }

    bool weights_ldg_guard = blockIdx.y * 32 + threadIdx.x % 32 < param.out_ch;
    ldg32_nc_0(weight_ldg_reg, weight_ldg_ptr, weights_ldg_guard && threadIdx.x / 32 < param.first_k_tile);

    int curC = threadIdx.x / 32 / param.KhKw;
    int curR = threadIdx.x / 32 % param.KhKw / param.Kw;
    int curS = threadIdx.x / 32 % param.KhKw % param.Kw;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        int curH = posh_ori[i] + curR;
        int curW = posw_ori[i] + curS;
        int inOffsetTmp = curC * param.inHW + curH * param.in_w + curW;

        bool guard = threadIdx.x / 32 < param.first_k_tile && curH >= 0 &&
                     curW >= 0 && curW < param.in_w && curH < param.in_h;
        ldg32_nc_0(input_ldg_reg[i], input_ldg_ptr + inOffsetTmp * sizeof(uint32_t), guard);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        sts32(input_ldg_reg[i], inputs_sts_addr + i * 32 * sizeof(uint32_t));
    }
    sts32(weight_ldg_reg, weights_sts_addr);
    __syncthreads();

    inputs_sts_addr ^= 0x1000;  // 0x1000=4096
    weights_sts_addr ^= 0x1000; // 0x0400=1024

    weight_ldg_ptr += param.first_k_tile * param.out_ch * sizeof(float);

    lds128(input_frag[0][0], input_frag[0][1], input_frag[0][2], input_frag[0][3], inputs_lds_addr);
    lds128(weight_frag[0][0], weight_frag[0][1], weight_frag[0][2], weight_frag[0][3], weights_lds_addr);

    for (int crs = 0; crs < param.inChKhKw - param.first_k_tile; crs += 8)
    {
#pragma unroll
        for (int k_frag = 0; k_frag < 8; ++k_frag)
        {
            if (k_frag == 7)
            {
#pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    sts32(input_ldg_reg[i], inputs_sts_addr + i * 32 * sizeof(uint32_t));
                }
                sts32(weight_ldg_reg, weights_sts_addr);
                __syncthreads();

                inputs_lds_addr ^= 0x1000;
                weights_lds_addr ^= 0x1000;
                inputs_sts_addr ^= 0x1000;
                weights_sts_addr ^= 0x1000;

                weight_ldg_ptr += 8 * param.out_ch * sizeof(float);
            }

            lds128(weight_frag[(k_frag + 1) % 2][0],
                   weight_frag[(k_frag + 1) % 2][1],
                   weight_frag[(k_frag + 1) % 2][2],
                   weight_frag[(k_frag + 1) % 2][3],
                   weights_lds_addr + (k_frag + 1) % 8 * 32 * sizeof(float));
            lds128(input_frag[(k_frag + 1) % 2][0],
                   input_frag[(k_frag + 1) % 2][1],
                   input_frag[(k_frag + 1) % 2][2],
                   input_frag[(k_frag + 1) % 2][3],
                   inputs_lds_addr + (k_frag + 1) % 8 * 128 * sizeof(uint32_t));

            if (k_frag == 0)
            {
                ldg32_nc(weight_ldg_reg, weight_ldg_ptr, weights_ldg_guard);

                curC = (crs + param.first_k_tile + threadIdx.x / 32) / param.KhKw;
                curR = (crs + param.first_k_tile + threadIdx.x / 32) % param.KhKw / param.Kw;
                curS = (crs + param.first_k_tile + threadIdx.x / 32) % param.KhKw % param.Kw;
#pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    int curH = posh_ori[i] + curR;
                    int curW = posw_ori[i] + curS;
                    int inOffsetTmp = curC * param.inHW + curH * param.in_w + curW;

                    bool guard = curH >= 0 && curW >= 0 && curW < param.in_w && curH < param.in_h;
                    ldg32_nc_0(input_ldg_reg[i], input_ldg_ptr + inOffsetTmp * sizeof(uint32_t), guard);
                }
            }

#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
#pragma unroll
                for (int j = 0; j < 4; ++j)
                {
                    // output_frag[i][j] += weight_frag[k_frag % 2][i] * input_frag[k_frag % 2][j];

                    add_f32(output_frag[i][j], weight_frag[k_frag % 2][i], (int) input_frag[k_frag % 2][j]);

                    // if (input_frag[k_frag % 2][j])
                    //     output_frag[i][j] += weight_frag[k_frag % 2][i];
                }
            }
        }
    }

#pragma unroll
    for (int k_frag = 0; k_frag < 8; ++k_frag)
    {
        if (k_frag < 7)
        {
            lds128(weight_frag[(k_frag + 1) % 2][0],
                   weight_frag[(k_frag + 1) % 2][1],
                   weight_frag[(k_frag + 1) % 2][2],
                   weight_frag[(k_frag + 1) % 2][3],
                   weights_lds_addr + (k_frag + 1) % 8 * 32 * sizeof(float));
            lds128(input_frag[(k_frag + 1) % 2][0],
                   input_frag[(k_frag + 1) % 2][1],
                   input_frag[(k_frag + 1) % 2][2],
                   input_frag[(k_frag + 1) % 2][3],
                   inputs_lds_addr + (k_frag + 1) % 8 * 128 * sizeof(uint32_t));
        }

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                // output_frag[i][j] += weight_frag[k_frag % 2][i] * input_frag[k_frag % 2][j];

                add_f32(output_frag[i][j], weight_frag[k_frag % 2][i], (int) input_frag[k_frag % 2][j]);

                // if (input_frag[k_frag % 2][j])
                //     output_frag[i][j] += weight_frag[k_frag % 2][i];
            }
        }
    }

    auto *smembias = reinterpret_cast<float *>(smem + 8 * 1024);
    if (threadIdx.x < 32)
    {
        smembias[threadIdx.x] = bias[blockIdx.y * 32 + threadIdx.x];
    }

    uint32_t outputs_sts_addr = smem_u32addr((float4 *) (smem + warp_id * 1024) + mma_tid_y * 16 * 2 + mma_tid_x);
    const float *outputs_lds_ptr = (float *) (smem + warp_id * 1024) + lane_id;
    uint32_t bias_lds_addr = warp_id / 2 * 8 + mma_tid_y * 4;

    int m_idx = blockIdx.y * 32 + warp_id / 2 * 8;
    int n_idx = blockIdx.x * 128 + warp_id % 2 * 64 + lane_id;

    float *outputs_stg_ptr = outputs + blockIdx.z * param.outBatchNumel + m_idx * param.outHW + n_idx;

    if (m_idx >= param.out_ch)
    {
        return;
    }
    else if (m_idx + 8 <= param.out_ch)
    {
        uint32_t n_guard = param.outHW < n_idx ? 0 : param.outHW - n_idx;

#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            __syncthreads();

#pragma unroll
            for (int p = 0; p < 2; ++p)
            {
                sts128(output_frag[p + i * 2][0] + smembias[bias_lds_addr + p + i * 2],
                       output_frag[p + i * 2][1] + smembias[bias_lds_addr + p + i * 2],
                       output_frag[p + i * 2][2] + smembias[bias_lds_addr + p + i * 2],
                       output_frag[p + i * 2][3] + smembias[bias_lds_addr + p + i * 2],
                       outputs_sts_addr + p * 16 * sizeof(float4));
            }
            __syncthreads();

            stg32(outputs_lds_ptr[0 * 64], outputs_stg_ptr + 0 * param.outHW, 0 < n_guard);
            stg32(outputs_lds_ptr[0 * 64 + 32], outputs_stg_ptr + 0 * param.outHW + 32, 32 < n_guard);

            stg32(outputs_lds_ptr[1 * 64], outputs_stg_ptr + 1 * param.outHW, 0 < n_guard);
            stg32(outputs_lds_ptr[1 * 64 + 32], outputs_stg_ptr + 1 * param.outHW + 32, 32 < n_guard);

            stg32(outputs_lds_ptr[2 * 64], outputs_stg_ptr + 4 * param.outHW, 0 < n_guard);
            stg32(outputs_lds_ptr[2 * 64 + 32], outputs_stg_ptr + 4 * param.outHW + 32, 32 < n_guard);

            stg32(outputs_lds_ptr[3 * 64], outputs_stg_ptr + 5 * param.outHW, 0 < n_guard);
            stg32(outputs_lds_ptr[3 * 64 + 32], outputs_stg_ptr + 5 * param.outHW + 32, 32 < n_guard);

            outputs_stg_ptr += 2 * param.outHW;
        }
    }
    else
    {
        uint32_t n_guard = param.outHW < n_idx ? 0 : param.outHW - n_idx;
        uint32_t m_guard = param.out_ch < m_idx ? 0 : param.out_ch - m_idx;

#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            __syncthreads();

#pragma unroll
            for (int p = 0; p < 2; ++p)
            {
                sts128(output_frag[p + i * 2][0] + smembias[bias_lds_addr + p + i * 2],
                       output_frag[p + i * 2][1] + smembias[bias_lds_addr + p + i * 2],
                       output_frag[p + i * 2][2] + smembias[bias_lds_addr + p + i * 2],
                       output_frag[p + i * 2][3] + smembias[bias_lds_addr + p + i * 2],
                       outputs_sts_addr + p * 16 * sizeof(float4));
            }
            __syncthreads();

            stg32(outputs_lds_ptr[0 * 64], outputs_stg_ptr + 0 * param.outHW, 0 < m_guard && 0 < n_guard);
            stg32(outputs_lds_ptr[0 * 64 + 32], outputs_stg_ptr + 0 * param.outHW + 32, 0 < m_guard && 32 < n_guard);

            stg32(outputs_lds_ptr[1 * 64], outputs_stg_ptr + 1 * param.outHW, 1 < m_guard && 0 < n_guard);
            stg32(outputs_lds_ptr[1 * 64 + 32], outputs_stg_ptr + 1 * param.outHW + 32, 1 < m_guard && 32 < n_guard);

            stg32(outputs_lds_ptr[2 * 64], outputs_stg_ptr + 4 * param.outHW, 4 < m_guard && 0 < n_guard);
            stg32(outputs_lds_ptr[2 * 64 + 32], outputs_stg_ptr + 4 * param.outHW + 32, 4 < m_guard && 32 < n_guard);

            stg32(outputs_lds_ptr[3 * 64], outputs_stg_ptr + 5 * param.outHW, 5 < m_guard && 0 < n_guard);
            stg32(outputs_lds_ptr[3 * 64 + 32], outputs_stg_ptr + 5 * param.outHW + 32, 5 < m_guard && 32 < n_guard);

            outputs_stg_ptr += 2 * param.outHW;
            m_guard -= 2;
        }
    }
}




/**
 * Mini Test Kernel: 64×64 output, single block, K_CHUNK=16
 *
 * SMEM: weight [K_CHUNK=16, M_TILE=64] float (4KB)
 *       input  [K_CHUNK=16, N_TILE=64] uint8 (1KB)
 *
 * Thread layout (from thread_map.txt, 8 rows × 4 cols per warp):
 *   256 threads, 8 warps
 *   lane_id = tid % 32,  warp_id = tid / 32
 *   mma_tid_x = lane_id / 16 * 2 + lane_id % 2  (0..3)
 *   mma_tid_y = lane_id % 16 / 2                 (0..7)
 *
 * Warp decomposition:
 *   warp_m = warp_id / 2 (0..3) → M [warp_m*16, warp_m*16+16)
 *   warp_n = warp_id % 2 (0..1) → N [warp_n*32, warp_n*32+32)
 *
 * Within warp [16M, 32N], each thread handles [4M, 4N]:
 *   M_group = mma_tid_y / 2                    (0..3)
 *   N_group = mma_tid_x * 2 + mma_tid_y % 2    (0..7)
 */
__global__ void conv2d_64x64x16_test(
    const uint8_t *inputs,   // [16, 64] uint8
    const float *weights,    // [16, 64] float
    float *outputs)          // [64, 64] float
{
    constexpr int K_CHUNK = 16;
    constexpr int M_TILE = 64;
    constexpr int N_TILE = 64;

    __shared__ __align__(128) char smem[5 * 1024];
    float *smemweight = reinterpret_cast<float *>(smem);
    uint8_t *smeminput = reinterpret_cast<uint8_t *>(smem + 4 * 1024);

    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int mma_tid_x = lane_id / 16 * 2 + lane_id % 2;  // 0..3
    const int mma_tid_y = lane_id % 16 / 2;                 // 0..7

    const int warp_m = warp_id / 2;  // 0..3
    const int warp_n = warp_id % 2;  // 0..1

    // Thread's sub-position within warp [16M, 32N] → [4M, 4N]
    const int M_group = mma_tid_y / 2;                  // 0..3
    const int N_group = mma_tid_x * 2 + mma_tid_y % 2;  // 0..7

    // --- Load weights [16, 64] float → SMEM with cp.async ---
    // 16×16=256 threads, each copies 1 float4 (16B)
    int w_row_f4 = tid / 16;      // 0..15
    int w_col_f4 = tid % 16;      // 0..15
    int w_col    = w_col_f4 * 4;  // 0,4,8,...,60

    uint32_t w_smem_ptr = smem_u32addr(&smemweight[w_row_f4 * M_TILE + w_col]);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(w_smem_ptr),
           "l"(&weights[w_row_f4 * M_TILE + w_col])
    );

    // --- Load inputs [16, 64] uint8 → SMEM ---
    // 256 threads, each copies 4 uint8 = 1 uint32
    int i_row = tid / 16;        // 0..15

    uint32_t *smeminput32 = reinterpret_cast<uint32_t *>(smeminput);
    const uint32_t *inputs32 = reinterpret_cast<const uint32_t *>(inputs);
    smeminput32[i_row * (N_TILE / 4) + (tid % 16)] =
        inputs32[i_row * (N_TILE / 4) + (tid % 16)];

    asm volatile("cp.async.commit_group;\n" :::);
    asm volatile("cp.async.wait_group 0;\n" :::);
    __syncthreads();

    // --- Register fragments ---
    float weight_frag[4];
    uint32_t input_frag;  // 4 uint8 packed
    float output_frag[4][4] = {0};

    int thread_m_base = warp_m * 16 + M_group * 4;
    int thread_n_base = warp_n * 32 + N_group * 4;

    // K-iteration: accumulate over all K_CHUNK rows
    for (int k = 0; k < K_CHUNK; k++) {
        // Load weight_frag[4] from smemweight[k][thread_m_base .. thread_m_base+4)
        // SMEM weight layout: [K_CHUNK=16][M_TILE=64], row stride = M_TILE
        for (int i = 0; i < 4; i++) {
            weight_frag[i] = smemweight[k * M_TILE + thread_m_base + i];
        }

        // Load input_frag (4 uint8 packed) from smeminput[k][thread_n_base ..]
        // SMEM input layout: [K_CHUNK=16][N_TILE=64], packed as [16][16] uint32
        input_frag = smeminput32[k * (N_TILE / 4) + thread_n_base / 4];

        // Outer product: output_frag[i][j] += weight_frag[i] * input[j]
        // Ref lines 186-190: add_f32 with spike guard
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                int spike = (input_frag >> (j * 8)) & 0xFF;
                add_f32(output_frag[i][j], weight_frag[i], spike);
                // if (spike) output_frag[i][j] += weight_frag[i];
            }
        }
    }

    // --- Epilogue: write output_frag[4][4] to GMEM ---
    // Output: [M_TILE=64][N_TILE=64], row-major
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int out_m = thread_m_base + i;
            int out_n = thread_n_base + j;
            outputs[out_m * N_TILE + out_n] = output_frag[i][j];
        }
    }
}


void conv2d_64x64_cpu_ref(
    const uint8_t *inputs, const float *weights, float *outputs,
    int K, int M, int N)
{
    // output[M, N] += weights[K, M]^T @ inputs[K, N]
    // output[m][n] = sum_k weight[k][m] * input[k][n]
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                if (inputs[k * N + n]) {
                    sum += weights[k * M + m];
                }
            }
            outputs[m * N + n] = sum;
        }
    }
}


void conv2d_64x64_mini_test()
{
    constexpr int K = 16;
    constexpr int M = 64;
    constexpr int N = 64;

    printf("\n=== conv2d_64x64x16 mini test ===\n");
    printf("  K=%d, M=%d, N=%d, T=1\n", K, M, N);

    // Host alloc
    uint8_t *h_inputs  = new uint8_t[K * N];
    float   *h_weights = new float[K * M];
    float   *h_outputs = new float[M * N];
    float   *h_ref     = new float[M * N];

    // Random init
    srand(42);
    for (int i = 0; i < K * N; i++) {
        h_inputs[i] = (uint8_t)((rand() & 1));  // 0 or 1
    }
    for (int i = 0; i < K * M; i++) {
        h_weights[i] = (float)((rand() & 255)) / 256.0f;
    }

    // Device alloc
    uint8_t *d_inputs;
    float   *d_weights;
    float   *d_outputs;
    cudaMalloc(&d_inputs,  K * N * sizeof(uint8_t));
    cudaMalloc(&d_weights, K * M * sizeof(float));
    cudaMalloc(&d_outputs, M * N * sizeof(float));

    cudaMemcpy(d_inputs,  h_inputs,  K * N * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, K * M * sizeof(float),   cudaMemcpyHostToDevice);

    // Launch
    dim3 block(256);
    dim3 grid(1, 1, 1);
    conv2d_64x64x16_test<<<grid, block>>>(d_inputs, d_weights, d_outputs);

    cudaMemcpy(h_outputs, d_outputs, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // CPU reference
    conv2d_64x64_cpu_ref(h_inputs, h_weights, h_ref, K, M, N);

    // Verify
    int errors = 0;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            if (abs(h_outputs[m * N + n] - h_ref[m * N + n]) > 0.001f) {
                if (errors < 10)
                    printf("  Err[%d,%d]: gpu=%.4f, cpu=%.4f\n",
                           m, n, h_outputs[m * N + n], h_ref[m * N + n]);
                errors++;
            }
        }
    }
    printf("  %s (%d errors)\n", errors == 0 ? "PASSED!" : "FAILED", errors);

    cudaFree(d_inputs);
    cudaFree(d_weights);
    cudaFree(d_outputs);
    delete[] h_inputs;
    delete[] h_weights;
    delete[] h_outputs;
    delete[] h_ref;
}


void conv2d_T_kernel_launch(void *inputs, void *weights, void *bias, void *outputs, Conv2DParam param, uint32_t n)
{
    uint32_t bx = (param.outHW + 127) / 128;
    uint32_t by = (param.out_ch + 63) / 64;
    uint32_t bz = n;

    dim3 block(256);
    dim3 grid(bx, by, bz);

    // conv2d_32x128x8_T_S_kernel<<<grid, block>>>((uint32_t *) inputs, (float *) weights, (float *) bias, (float *) outputs, param);
}

void conv2d_T_main()
{
    // --- Mini test: 64×64, K_CHUNK=16, T=1 ---
    conv2d_64x64_mini_test();

    int n, c, h, w, k, r, s, u, v, p, q;
    // n = 4, c = 4, h = 160, w = h, k = 4, r = 3, s = r, u = 2, v = u, p = 1, q = p;
    // n = 4, c = 128, h = 160, w = h, k = 128, r = 3, s = r, u = 2, v = u, p = 1, q = p;

    // 32
    // n = 2, c = 32, h = 160, w = h, k = 16, r = 3, s = r, u = 2, v = u, p = 1, q = p;
    // n = 1, c = 32, h = 85, w = h, k = 35, r = 3, s = r, u = 2, v = u, p = 1, q = p;

    // 64
    n = 2, c = 64, h = 160, w = h, k = 64, r = 3, s = r, u = 2, v = u, p = 1, q = p;
    // n = 2, c = 65, h = 160, w = h, k = 69, r = 3, s = r, u = 2, v = u, p = 1, q = p;

    // 128
    // n = 2, c = 128, h = 160, w = h, k = 128, r = 3, s = r, u = 2, v = u, p = 1, q = p;
    // n = 2, c = 128, h = 160, w = h, k = 128, r = 1, s = r, u = 1, v = u, p = 0, q = p;
    // n = 2, c = 128, h = 160, w = h, k = 128, r = 3, s = r, u = 1, v = u, p = 1, q = p;

    // n = 2, c = 33, h = 43, w = h, k = 128, r = 3, s = r, u = 2, v = u, p = 1, q = p;
    // n = 1, c = 3, h = 40, w = h, k = 9, r = 3, s = r, u = 2, v = u, p = 1, q = p;

    int out_h = (h - r + 2 * p) / u + 1;
    int out_w = (w - s + 2 * q) / v + 1;
    printf("outH: %4d   outW: %4d\n", out_h, out_w);

    Conv2DParam param;
    param.in_h = h;
    param.in_w = w;
    param.inHW = h * w;
    param.inChKhKw = c * r * s;
    param.inChKhKw = (c * r * s + 1) / 2; // half
    param.inBatchNumel = c * h * w;
    param.out_ch = k;
    param.out_w = out_w;
    param.outHW = out_h * out_w;
    param.outBatchNumel = k * out_h * out_w;
    param.Kw = s;
    param.KhKw = r * s;
    param.Sh = u;
    param.Sw = v;
    param.Ph = p;
    param.Pw = q;

    param.k_tiles = (param.inChKhKw + 7) / 8 - 1;
    param.first_k_tile = param.inChKhKw - param.k_tiles * 8;

    double M = k;
    double N = n * out_h * out_w;
    double K = c * r * s;
    double temp = n * out_h * out_w * 1e-9f;
    double flopsPerConv = temp * M * K * 2.0;

    uint32_t *spikes;
    uint16_t *spikes_h;
    float *inputs, *weights, *weightsT, *bias, *outputs_cpu, *outputs_host;
    half *inputs_h, *weights_h, *weightsT_h, *outputs_host_h;
    half2 *bias_h;
    cudaMallocHost((void **) &spikes, n * c * h * w * sizeof(uint32_t));
    cudaMallocHost((void **) &spikes_h, n * c * h * w * sizeof(uint16_t));
    cudaMallocHost((void **) &inputs, n * c * h * w * sizeof(float));
    cudaMallocHost((void **) &weights, k * c * r * s * sizeof(float));
    cudaMallocHost((void **) &weightsT, k * c * r * s * sizeof(float));
    cudaMallocHost((void **) &bias, k * sizeof(float));
    cudaMallocHost((void **) &inputs_h, n * c * h * w * sizeof(half));
    cudaMallocHost((void **) &weights_h, k * c * r * s * sizeof(half));
    cudaMallocHost((void **) &weightsT_h, (c * r * s + 1) / 2 * k * sizeof(half2));
    // cudaMallocHost((void **) &weightsT_h, (k + 1) / 2 * c * r * s * sizeof(half2));
    cudaMallocHost((void **) &bias_h, k * sizeof(half2));
    cudaMallocHost((void **) &outputs_cpu, n * k * out_h * out_w * sizeof(float));
    cudaMallocHost((void **) &outputs_host, n * k * out_h * out_w * sizeof(float));
    cudaMallocHost((void **) &outputs_host_h, n * k * out_h * out_w * sizeof(half));

    uint32_t *spikes_device;
    uint16_t *spikes_device_h;
    float *inputs_device, *weights_device, *weightsT_device, *bias_device, *outputs_device;
    half *inputs_device_h, *weights_device_h, *weightsT_device_h, *outputs_device_h;
    half2 *bias_device_h;
    cudaMalloc((void **) &spikes_device, n * c * h * w * sizeof(uint32_t));
    cudaMalloc((void **) &spikes_device_h, n * c * h * w * sizeof(uint16_t));
    cudaMalloc((void **) &inputs_device, n * c * h * w * sizeof(float));
    cudaMalloc((void **) &weights_device, k * c * r * s * sizeof(float));
    cudaMalloc((void **) &weightsT_device, k * c * r * s * sizeof(float));
    cudaMalloc((void **) &bias_device, k * sizeof(float));
    cudaMalloc((void **) &outputs_device, n * k * out_h * out_w * sizeof(float));
    cudaMalloc((void **) &inputs_device_h, n * c * h * w * sizeof(half));
    cudaMalloc((void **) &weights_device_h, k * c * r * s * sizeof(half));
    cudaMalloc((void **) &weightsT_device_h, (c * r * s + 1) / 2 * k * sizeof(half2));
    // cudaMalloc((void **) &weightsT_device_h, (k + 1) / 2 * c * r * s * sizeof(half2));
    cudaMalloc((void **) &bias_device_h, k * sizeof(half2));
    cudaMalloc((void **) &outputs_device_h, n * k * out_h * out_w * sizeof(half));

    for (int i = 0; i < n * c * h * w; i++)
    {
        // input[i] = 0.1f;
        inputs[i] = (rand() & 255) / 256.0f;

        if (inputs[i] > 0.5f)
        {
            spikes[i] = 1;
            spikes_h[i] = 1;
            inputs[i] = 1.0f;
        }
        else
        {
            spikes[i] = 0;
            spikes_h[i] = 0;
            inputs[i] = 0;
        }
        // spikes[i] = 0;
        // spikes_h[i] = 0;
        // inputs[i] = 0.0f;
        inputs_h[i] = __float2half_rn(inputs[i]);
    }
    for (int i = 0; i < k * c * r * s; i++)
    {
        // weights[i] = 0.25f;
        weights[i] = (rand() & 255) / 256.0f;
        weights_h[i] = __float2half_rn(weights[i]);
    }
    for (int i = 0; i < k; i++)
    {
        // bias[i] = 10.0f;
        bias[i] = (rand() & 255) / 256.0f;
        bias_h[i] = half2(bias[i], bias[i]);
        // bias_h[i] = half2(0.0f, bias[i]);
        // bias_h[i] = half2(bias[i], 0.0f);
    }

    // Transpose
    for (int j = 0; j < k; j++)
    {
        for (int i = 0; i < c * r * s; i++)
        {
            weightsT[i * k + j] = weights[j * (c * r * s) + i];
        }
    }
    // Transpose half
    auto *weightsTPtr = reinterpret_cast<half2 *>(weightsT_h);
    int kernelNumel = c * r * s;
    for (int j = 0; j < k; j++)
    {
        for (int i = 0; i < (kernelNumel + 1) / 2; i++)
        {
            half2 x = half2(0, 0);

            x.x = weights_h[j * kernelNumel + i * 2];
            if (i * 2 + 1 < kernelNumel)
                x.y = weights_h[j * kernelNumel + i * 2 + 1];

            weightsTPtr[i * k + j] = x;
        }
    }
    // for (int j = 0; j < (k + 1) / 2; j++)
    // {
    //     for (int i = 0; i < kernelNumel; i++)
    //     {
    //         half2 x = half2(0, 0);

    //         x.x = weight_h[(j * 2) * kernelNumel + i];
    //         if (j * 2 + 1 < k)
    //             x.y = weight_h[(j * 2 + 1) * kernelNumel + i];

    //         weightsTPtr[i * ((k + 1) / 2) + j] = x;
    //     }
    // }

    // printf("==================== Weights T ====================\n");
    // for (int i = 0; i < c * r * s; i++)
    // {
    //     for (int j = 0; j < k; j++)
    //     {
    //         printf("%.4lf ", weightT[i * k + j]);
    //     }
    //     printf("\n");
    // }
    // printf("==================== Weights T ====================\n");
    // printf("=================== Weights T Half ===================\n");
    // for (int i = 0; i < c * r * s; i++)
    // {
    //     for (int j = 0; j < (k + 1) / 2; j++)
    //     {
    //         printf("%.4lf %.4lf ", __half2float(weightT_h[i * ((k + 1) / 2 * 2) + j * 2]),
    //                __half2float(weightT_h[i * ((k + 1) / 2 * 2) + j * 2 + 1]));
    //     }
    //     printf("\n");
    // }
    // printf("=================== Weights T Half ===================\n");

    cudaMemcpy(spikes_device, spikes, n * c * h * w * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(spikes_device_h, spikes_h, n * c * h * w * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(inputs_device, inputs, n * c * h * w * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weights_device, weights, k * c * r * s * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weightsT_device, weightsT, k * c * r * s * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bias_device, bias, k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(inputs_device_h, inputs_h, n * c * h * w * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(weights_device_h, weights_h, k * c * r * s * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(weightsT_device_h, weightsT_h, (c * r * s + 1) / 2 * k * sizeof(half2), cudaMemcpyHostToDevice);
    // cudaMemcpy(weightsT_device_h, weightT_h, (k + 1) / 2 * c * r * s * sizeof(half2), cudaMemcpyHostToDevice);
    cudaMemcpy(bias_device_h, bias_h, k * sizeof(half2), cudaMemcpyHostToDevice);

    // conv2d_T_kernel_launch(inputs_device, weightsT_device, bias_device, outputs_device, param, n);
    // cudaMemcpy(outputs_host, outputs_device, n * k * out_h * out_w * sizeof(float), cudaMemcpyDeviceToHost);
    
    // conv2d_T_kernel_launch(inputs_device_h, weightsT_device_h, bias_device_h, outputs_device_h, param, n);
    // cudaMemcpy(outputs_host_h, outputs_device_h, n * k * out_h * out_w * sizeof(half), cudaMemcpyDeviceToHost);
    
    // conv2d_T_kernel_launch(spikes_device, weightsT_device, bias_device, outputs_device, param, n);
    // cudaMemcpy(outputs_host, outputs_device, n * k * out_h * out_w * sizeof(float), cudaMemcpyDeviceToHost);
    
    conv2d_T_kernel_launch(spikes_device_h, weightsT_device_h, bias_device_h, outputs_device_h, param, n);
    cudaMemcpy(outputs_host_h, outputs_device_h, n * k * out_h * out_w * sizeof(half), cudaMemcpyDeviceToHost);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    float time_elapsed = 0;

    int iters = 0;
    for (int i = 0; i < iters; i++)
    {
        conv2d_T_kernel_launch(inputs_device, weightsT_device, bias_device, outputs_device, param, n);
        // conv2d_T_kernel_launch(inputs_device_h, weightsT_device_h, bias_device_h, outputs_device_h, param, n);
        // conv2d_T_kernel_launch(spikes_device, weightsT_device, bias_device, outputs_device, param, n);
        // conv2d_T_kernel_launch(spikes_device_h, weightsT_device_h, bias_device_h, outputs_device_h, param, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("=================== CPU Calc ===================\n");
    direct_conv2d_T_cpu(inputs, weights, bias, outputs_cpu, n, c, h, w, k, r, s, u, v, p, q);

    printf("=================== start verfiy ===================\n");
    int error = 0;
    for (int i = 0; i < n * k * out_h * out_w; i++)
    {
        // if (abs(outputs_host[i] - outputs_cpu[i]) > 0.0001f)
        if (abs(__half2float(outputs_host_h[i]) - outputs_cpu[i]) > 0.80f) // 0.25  0.80
        {
            // printf("Error: %d, gpu: %.4lf, cpu: %.4lf\n", i, outputs_host[i], outputs_cpu[i]);

            printf("Error: %d, gpu: %.4lf, cpu: %.4lf %.4lf\n",
                   i, __half2float(outputs_host_h[i]), outputs_cpu[i],
                      __half2float(outputs_host_h[i]) - outputs_cpu[i]);

            error++;
            if (error > 10)
                break;
        }
    }
    // printf("%.4lf %.4lf\n", outputs_host[0], outputs_cpu[0]);
    // printf("%.4lf %.4lf\n", outputs_host[1], outputs_cpu[1]);
    // printf("%.4lf %.4lf\n", outputs_host[2], outputs_cpu[2]);
    // printf("%.4lf %.4lf\n", outputs_host[3], outputs_cpu[3]);

    printf("%.4lf %.4lf\n", __half2float(outputs_host_h[0]), outputs_cpu[0]);
    printf("%.4lf %.4lf\n", __half2float(outputs_host_h[1]), outputs_cpu[1]);
    printf("%.4lf %.4lf\n", __half2float(outputs_host_h[2]), outputs_cpu[2]);
    printf("%.4lf %.4lf\n", __half2float(outputs_host_h[3]), outputs_cpu[3]);

    // printf("%lf %lf %lf %lf\n", outputs_cpu[0], outputs_cpu[1], outputs_cpu[2], outputs_cpu[3]);
    printf("===================  Error: %d  =====================\n", error);

    float timePerConv = time_elapsed / iters;
    double gflops = flopsPerConv / (timePerConv / 1000.0f);
    printf("%3d %3d %3d %3d | %d %d %d\n", n, c, h, w, r, s, k);
    printf("time: %.6f ms\n", timePerConv);
    printf("Performance: %.2f GFlops\n", gflops);

#ifdef use_cudnn
    printf("=================== cudnn ===================\n");

    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    cudnnTensorDescriptor_t input_desc;
    cudnnCreateTensorDescriptor(&input_desc);
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

    cudnnFilterDescriptor_t filter_desc;
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, k, c, r, s);

    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc, p, q, u, v, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    cudnnTensorDescriptor_t output_desc;
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, k, out_h, out_w);

    // cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc, &n, &k, &out_h, &out_w);


    size_t space_size = 0;
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    cudnnGetConvolutionForwardWorkspaceSize(
            cudnn, input_desc, filter_desc, conv_desc, output_desc,
            algo, &space_size);

    void *workspace = nullptr;
    cudaMalloc(&workspace, space_size);

    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(
            cudnn, &alpha,
            input_desc, input_device, filter_desc, weight_device,
            conv_desc, algo, workspace, space_size,
            &beta, output_desc, output_device);

    cudaMemcpy(output_host, output_device, n * k * out_h * out_w * sizeof(float), cudaMemcpyDeviceToHost);

    printf("=================== start verfiy ===================\n");
    error = 0;
    for (int i = 0; i < n * k * out_h * out_w; i++)
    {
        if (abs(output_host[i] - output_cpu[i]) > 0.0001f)
        {
            printf("Error: position: %d, gpu: %lf, cpu: %lf\n", i, output_host[i], output_cpu[i]);
            error++;
            break;
        }
    }
    printf("%lf %lf\n", output_host[0], output_cpu[0]);
    printf("%lf %lf\n", output_host[1], output_cpu[1]);
    printf("%lf %lf %lf %lf\n", output_cpu[0], output_cpu[1], output_cpu[2], output_cpu[3]);

    printf("===================  Error: %d  =====================\n", error);

    cudaFree(workspace);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn);
#endif


    cudaFree(spikes_device);
    cudaFree(spikes_device_h);
    cudaFree(inputs_device);
    cudaFree(weights_device);
    cudaFree(weightsT_device);
    cudaFree(bias_device);
    cudaFree(outputs_device);
    cudaFree(inputs_device_h);
    cudaFree(weights_device_h);
    cudaFree(weightsT_device_h);
    cudaFree(outputs_device_h);
    cudaFree(bias_device_h);

    cudaFreeHost(spikes);
    cudaFreeHost(spikes_h);
    cudaFreeHost(inputs);
    cudaFreeHost(weights);
    cudaFreeHost(weightsT);
    cudaFreeHost(bias);
    cudaFreeHost(outputs_cpu);
    cudaFreeHost(outputs_host);
    cudaFreeHost(inputs_h);
    cudaFreeHost(weights_h);
    cudaFreeHost(weightsT_h);
    cudaFreeHost(outputs_host_h);
    cudaFreeHost(bias_h);
}

int main() {
    conv2d_T_main();
    return 0;
}
