#include <iostream>
#include <vector>
#include <string>
#include <bitset>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "conv2d_weightsT.cuh"
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


__global__ void conv2d_128x128x8_T_kernel(float *inputs, float *weights, float *bias, float *outputs, Conv2DParam param)
{
    // __shared__ __align__ (16 * 1024)
    // __shared__ char smem[16 * 1024 + 128 * 4]; // 128*8*2 * 2 * 4 = 16*1024
    __shared__ __align__(16 * 1024) char smem[16 * 1024 + 128 * 4];
    auto *smemweight = reinterpret_cast<float *>(smem);
    auto *smeminput = reinterpret_cast<float *>(smem + 128 * 8 * 2 * 4);

    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int mma_tid_x = (lane_id / 2) % 8;
    const int mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);

    uint32_t inputs_sts_addr = smem_u32addr(smeminput + (threadIdx.x / 32) * 128 + (threadIdx.x % 32));
    uint32_t weights_sts_addr = smem_u32addr(smemweight + (threadIdx.x / 32) * 128 + (threadIdx.x % 32));

    uint32_t inputs_lds_addr = smem_u32addr(smeminput + (warp_id % 2) * 64 + mma_tid_x * 4);
    uint32_t weights_lds_addr = smem_u32addr(smemweight + (warp_id / 2) * 32 + mma_tid_y * 4);

    const char *input_ldg_ptr = (const char *) (inputs + blockIdx.z * param.inBatchNumel);
    const char *weight_ldg_ptr = (const char *) (
            weights + blockIdx.y * 128 + threadIdx.x / 32 * param.out_ch + threadIdx.x % 32);

    float input_ldg_reg[4];
    float weight_ldg_reg[4];

    float input_frag[2][8];
    float weight_frag[2][8];
    float output_frag[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
#pragma unroll
        for (int j = 0; j < 8; ++j)
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

    uint32_t weights_ldg_guard = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        int m_idx = blockIdx.y * 128 + threadIdx.x % 32 + i * 32;
        if (m_idx < param.out_ch)
        {
            weights_ldg_guard |= (1u << i);
        }
    }

    // DEBUG: Print weights_ldg_guard calculation
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
    {
        printf("\n=== WEIGHTS_LDG_GUARD ===\n");
        printf("param.out_ch: %d\n", param.out_ch);
        printf("blockIdx.y: %d\n", blockIdx.y);
        printf("threadIdx.x: %d\n", threadIdx.x);
        for (int i = 0; i < 4; i++)
        {
            int m_idx = blockIdx.y * 128 + threadIdx.x % 32 + i * 32;
            printf("i=%d: m_idx=%d, m_idx < out_ch? %d, weights_ldg_guard|=0x%x\n",
                   i, m_idx, m_idx, m_idx < param.out_ch, weights_ldg_guard);
        }
        printf("Final weights_ldg_guard: 0x%x\n", weights_ldg_guard);
    }



#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        bool guard = (weights_ldg_guard & (1u << i)) != 0 && threadIdx.x / 32 < param.first_k_tile;
        ldg32_nc_0(weight_ldg_reg[i], weight_ldg_ptr + i * 32 * sizeof(float), guard);
    }

    int curC = threadIdx.x / 32 / param.KhKw;
    int curR = threadIdx.x / 32 % param.KhKw / param.Kw;
    int curS = threadIdx.x / 32 % param.KhKw % param.Kw;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        int curH = posh_ori[i] + curR;
        int curW = posw_ori[i] + curS;
        int inOffsetTmp = curC * param.inHW + curH * param.in_w + curW;

        bool guard = threadIdx.x / 32 < param.first_k_tile && curH >= 0 && curW >= 0 && curW < param.in_w &&
                     curH < param.in_h;
        ldg32_nc_0(input_ldg_reg[i], input_ldg_ptr + inOffsetTmp * sizeof(float), guard);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        sts32(input_ldg_reg[i], inputs_sts_addr + i * 32 * sizeof(float));
        sts32(weight_ldg_reg[i], weights_sts_addr + i * 32 * sizeof(float));
    }
    __syncthreads();

    inputs_sts_addr ^= 0x1000;  // 0x1000=4096
    weights_sts_addr ^= 0x1000; // 0x2000=8192

    weight_ldg_ptr += param.first_k_tile * param.out_ch * sizeof(float);

    lds128(input_frag[0][0], input_frag[0][1], input_frag[0][2], input_frag[0][3], inputs_lds_addr);
    lds128(input_frag[0][4], input_frag[0][5], input_frag[0][6], input_frag[0][7],
           inputs_lds_addr + 32 * sizeof(float));
    lds128(weight_frag[0][0], weight_frag[0][1], weight_frag[0][2], weight_frag[0][3], weights_lds_addr);
    lds128(weight_frag[0][4], weight_frag[0][5], weight_frag[0][6], weight_frag[0][7],
           weights_lds_addr + 16 * sizeof(float));

    // DEBUG: Print smem layout
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x < 64)
    {
        if (threadIdx.x == 0)
        {
            printf("=== SMEM LAYOUT ===\n");
            printf("smem size: %d bytes\n", 16 * 1024 + 128 * 4);
            printf("smemweight starts at offset: 0\n");
            printf("smeminput starts at offset: %d\n", 128 * 8 * 2 * 4);
            printf("smeminput size: %d bytes (128*8*2*4)\n", 128 * 8 * 2 * 4);
        }
        printf("tid=%2d (warp=%d, lane=%d, mma_tid_x=%d):\n",
               threadIdx.x, warp_id, lane_id, mma_tid_x);
        printf("  inputs_sts_addr: offset %5u from smem, %5u from smeminput\n",
               (uint32_t)((char*)inputs_sts_addr - (char*)smem),
               (uint32_t)((char*)inputs_sts_addr - (char*)smeminput));
        printf("  inputs_lds_addr: offset %5u from smem, %5u from smeminput\n",
               (uint32_t)((char*)inputs_lds_addr - (char*)smem),
               (uint32_t)((char*)inputs_lds_addr - (char*)smeminput));
        printf("  Buffer: %s\n",
               ((uint32_t)((char*)inputs_lds_addr - (char*)smeminput) < 4096) ? "LOW" : "HIGH");
        if (threadIdx.x == 0)
            printf("\n");
    }

    // DEBUG: Print input_frag and weight_frag before main loop
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
    {
        printf("\n=== INPUT_FRAG AND WEIGHT_FRAG (threadIdx.x=0) ===\n");
        printf("input_frag[0]: ");
        for (int i = 0; i < 8; i++)
            printf("%.0f ", input_frag[0][i]);
        printf("\n");
        printf("input_frag[1]: ");
        for (int i = 0; i < 8; i++)
            printf("%.0f ", input_frag[1][i]);
        printf("\n");
        printf("weight_frag[0]: ");
        for (int i = 0; i < 8; i++)
            printf("%.0f ", weight_frag[0][i]);
        printf("\n");
        printf("weight_frag[1]: ");
        for (int i = 0; i < 8; i++)
            printf("%.0f ", weight_frag[1][i]);
        printf("\n");
    }

    // DEBUG: Print weight_ldg_reg loading and storage
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x < 4)
    {
        printf("\n=== WEIGHT_LDG_REG (threadIdx.x=%d) ===\n", threadIdx.x);
        printf("weights_ldg_guard: 0x%x\n", weights_ldg_guard);
        printf("param.first_k_tile: %d\n", param.first_k_tile);
        printf("threadIdx.x/32: %d\n", threadIdx.x / 32);
        printf("weight_ldg_reg: ");
        for (int i = 0; i < 4; i++)
            printf("%.0f ", weight_ldg_reg[i]);
        printf("\n");
        printf("weights_sts_addr offset from smem: %u\n",
               (uint32_t)((char*)weights_sts_addr - (char*)smem));
    }

    // DEBUG: Print smem weights after store
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
    {
        printf("\n=== SMEM WEIGHTS AFTER INITIAL STORE ===\n");
        printf("smemweight first 32 values: ");
        for (int i = 0; i < 32; i++)
            printf("%.0f ", smemweight[i]);
        printf("\n");
    }

    for (int crs = 0; crs < param.inChKhKw - param.first_k_tile; crs += 8)
    {
#pragma unroll
        for (int k_frag = 0; k_frag < 8; ++k_frag)
        {
            if (k_frag == 7)
            {
                // DEBUG: Print before address toggle
                if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && crs == 8)
                {
                    printf("crs=8, k_frag=7: BEFORE toggle\n");
                    printf("  inputs_lds_addr offset from smeminput: %u\n",
                           (uint32_t)((char*)inputs_lds_addr - (char*)smeminput));
                    printf("  inputs_sts_addr offset from smem: %u (from smem)\n",
                           (uint32_t)((char*)inputs_sts_addr - (char*)smem));
                    printf("  inputs_sts_addr offset from smeminput: %u (from smeminput)\n",
                           (uint32_t)((char*)inputs_sts_addr - (char*)smeminput));

                    // Print smeminput values around the store and load addresses
                    // printf("  smeminput[offset 0-15]: ");
                    // for (int i = 0; i < 16 && i < (128 * 8 * 2 * 4); i++)
                    //     printf("%.0f ", smeminput[i]);
                    // printf("\n");

                    // printf("  smeminput[offset 4096-4111]: ");
                    // for (int i = 4096; i < 4112 && i < (128 * 8 * 2 * 4); i++)
                    //     printf("%.0f ", smeminput[i]);
                    // printf("\n");
                }

#pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    sts32(input_ldg_reg[i], inputs_sts_addr + i * 32 * sizeof(float));
                    sts32(weight_ldg_reg[i], weights_sts_addr + i * 32 * sizeof(float));
                }

                // DEBUG: Print first 8 values of output_frag before address toggle
                if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
                {
                    printf("DEBUG tid=0: output_frag[0][0..7] = ");
                    for(int d=0; d<8; ++d) printf("%.2f ", output_frag[0][d]);
                    printf("\n");
                }

                __syncthreads();

                inputs_lds_addr ^= 0x1000;
                weights_lds_addr ^= 0x1000;
                inputs_sts_addr ^= 0x1000;
                weights_sts_addr ^= 0x1000;

                weight_ldg_ptr += 8 * param.out_ch * sizeof(float);

                // DEBUG: Print after address toggle
                if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && crs == 8)
                {
                    printf("crs=8, k_frag=7: AFTER toggle\n");
                    printf("  inputs_lds_addr offset from smeminput: %u\n",
                           (uint32_t)((char*)inputs_lds_addr - (char*)smeminput));
                    printf("  inputs_sts_addr offset from smem: %u (from smem)\n",
                           (uint32_t)((char*)inputs_sts_addr - (char*)smem));
                    printf("  inputs_sts_addr offset from smeminput: %u (from smeminput)\n",
                           (uint32_t)((char*)inputs_sts_addr - (char*)smeminput));
                }
            }

            lds128(weight_frag[(k_frag + 1) % 2][0],
                   weight_frag[(k_frag + 1) % 2][1],
                   weight_frag[(k_frag + 1) % 2][2],
                   weight_frag[(k_frag + 1) % 2][3],
                   weights_lds_addr + (k_frag + 1) % 8 * 128 * sizeof(float));
            lds128(weight_frag[(k_frag + 1) % 2][4],
                   weight_frag[(k_frag + 1) % 2][5],
                   weight_frag[(k_frag + 1) % 2][6],
                   weight_frag[(k_frag + 1) % 2][7],
                   weights_lds_addr + ((k_frag + 1) % 8 * 128 + 16) * sizeof(float));

            // DEBUG: Print preload addresses
            if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && crs == 8)
            {
                uint32_t preload_offset = (k_frag + 1) % 8 * 128 * sizeof(float);
                uint32_t actual_addr = (uint32_t)((char*)inputs_lds_addr - (char*)smeminput) + preload_offset;
                printf("k_frag=%d: preload input_frag[%d] from offset %u (base=%u + add=%u), in_buffer=%s\n",
                       k_frag, (k_frag + 1) % 2,
                       actual_addr,
                       (uint32_t)((char*)inputs_lds_addr - (char*)smeminput),
                       preload_offset,
                       actual_addr < 4096 ? "LOW" : "HIGH");
            }

            lds128(input_frag[(k_frag + 1) % 2][0],
                   input_frag[(k_frag + 1) % 2][1],
                   input_frag[(k_frag + 1) % 2][2],
                   input_frag[(k_frag + 1) % 2][3],
                   inputs_lds_addr + (k_frag + 1) % 8 * 128 * sizeof(float));
            lds128(input_frag[(k_frag + 1) % 2][4],
                   input_frag[(k_frag + 1) % 2][5],
                   input_frag[(k_frag + 1) % 2][6],
                   input_frag[(k_frag + 1) % 2][7],
                   inputs_lds_addr + ((k_frag + 1) % 8 * 128 + 32) * sizeof(float));

            if (k_frag == 0)
            {
                // DEBUG: Print before loading new data
                if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && crs == 8)
                {
                    printf("crs=8, k_frag=0: Loading new input data\n");
                    printf("  input_ldg_ptr offset: %ld (from inputs)\n",
                           (char*)input_ldg_ptr - (char*)inputs);
                    printf("  inputs_sts_addr offset from smem: %u\n",
                           (uint32_t)((char*)inputs_sts_addr - (char*)smem));
                }

#pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    ldg32_nc(weight_ldg_reg[i], weight_ldg_ptr + i * 32 * sizeof(float),
                             (weights_ldg_guard & (1u << i)) != 0);
                }

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
                    ldg32_nc_0(input_ldg_reg[i], input_ldg_ptr + inOffsetTmp * sizeof(float), guard);
                }

                // DEBUG: Print after loading
                if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && crs == 8)
                {
                    printf("  Loaded input_ldg_reg[0]=%.0f (curC=%d)\n", input_ldg_reg[0], curC);
                }
            }

#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
#pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    output_frag[i][j] += weight_frag[k_frag % 2][i] * input_frag[k_frag % 2][j];
                }
            }

            // DEBUG: Print first k_frag accumulation in detail
            if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
            {
                if (crs == 8)
                {
                    printf("crs=%d, k_frag=%d: accumulated weight_frag[%d][0]=%.0f * input_frag[%d][1]=%.0f, output_frag[0][1] before=%.2f, after=%.2f, lds_addr_offset=%u\n",
                           crs, k_frag, k_frag % 2, weight_frag[k_frag % 2][0], k_frag % 2, input_frag[k_frag % 2][1],
                           output_frag[0][1] - weight_frag[k_frag % 2][0] * input_frag[k_frag % 2][1], output_frag[0][1],
                           (uint32_t)((char*)inputs_lds_addr - (char*)smeminput));

                    // Print all input_frag values when k_frag=6
                    if (k_frag == 6)
                    {
                        printf("  k_frag=6: input_frag[0][0..7]: ");
                        for (int i = 0; i < 8; i++)
                            printf("%.0f ", input_frag[0][i]);
                        printf("\n");
                        printf("  k_frag=6: input_frag[1][0..7]: ");
                        for (int i = 0; i < 8; i++)
                            printf("%.0f ", input_frag[1][i]);
                        printf("\n");
                        printf("  k_frag=6: weight_frag[0][0..7]: ");
                        for (int i = 0; i < 8; i++)
                            printf("%.0f ", weight_frag[0][i]);
                        printf("\n");
                    }
                }
            }

            // DEBUG: Print after each k_frag accumulation
            if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && k_frag == 7)
            {
                printf("After crs=%d main loop: output_frag[0][1]=%.2f\n", crs, output_frag[0][1]);
            }
        }
    }

    // DEBUG: Print after main loop
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
    {
        printf("=== After main loop ===\n");
        printf("output_frag[0][1]=%.2f\n", output_frag[0][1]);
        printf("Expected: 512.0 (128 channels * 4 valid positions)\n");
        printf("Missing: %.2f\n", 512.0 - output_frag[0][1]);
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
                   weights_lds_addr + (k_frag + 1) % 8 * 128 * sizeof(float));
            lds128(weight_frag[(k_frag + 1) % 2][4],
                   weight_frag[(k_frag + 1) % 2][5],
                   weight_frag[(k_frag + 1) % 2][6],
                   weight_frag[(k_frag + 1) % 2][7],
                   weights_lds_addr + ((k_frag + 1) % 8 * 128 + 16) * sizeof(float));
            lds128(input_frag[(k_frag + 1) % 2][0],
                   input_frag[(k_frag + 1) % 2][1],
                   input_frag[(k_frag + 1) % 2][2],
                   input_frag[(k_frag + 1) % 2][3],
                   inputs_lds_addr + (k_frag + 1) % 8 * 128 * sizeof(float));
            lds128(input_frag[(k_frag + 1) % 2][4],
                   input_frag[(k_frag + 1) % 2][5],
                   input_frag[(k_frag + 1) % 2][6],
                   input_frag[(k_frag + 1) % 2][7],
                   inputs_lds_addr + ((k_frag + 1) % 8 * 128 + 32) * sizeof(float));
        }

#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
#pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                output_frag[i][j] += weight_frag[k_frag % 2][i] * input_frag[k_frag % 2][j];
            }
        }
    }

    // DEBUG: Print after second accumulation loop
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
    {
        printf("=== After second loop ===\n");
        printf("output_frag[0][1]=%.2f\n", output_frag[0][1]);
    }

    auto *smembias = reinterpret_cast<float *>(smem + 16 * 1024);
    if (threadIdx.x < 128)
    {
        smembias[threadIdx.x] = bias[blockIdx.y * 128 + threadIdx.x];
    }

    uint32_t outputs_sts_addr = smem_u32addr((float4 *) (smem + warp_id * 2048) + mma_tid_y * 4 * 8 + mma_tid_x);
    const float *outputs_lds_ptr = (float *) (smem + warp_id * 2048) + lane_id;
    uint32_t bias_lds_addr = warp_id / 2 * 32 + mma_tid_y * 4;

    int m_idx = blockIdx.y * 128 + warp_id / 2 * 32;
    int n_idx = blockIdx.x * 128 + warp_id % 2 * 64 + lane_id;

    float *outputs_stg_ptr = outputs + blockIdx.z * param.outBatchNumel + m_idx * param.outHW + n_idx;

    if (m_idx >= param.out_ch)
    {
        return;
    }
    else if (m_idx + 32 <= param.out_ch)
    {
        uint32_t n_guard = param.outHW < n_idx ? 0 : param.outHW - n_idx;

        // DEBUG: Print before writing output
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
        {
            printf("=== Before writing output ===\n");
            printf("output_frag[0][1]=%.2f\n", output_frag[0][1]);
            printf("m_idx=%d, n_idx=%d\n", m_idx, n_idx);
            printf("smembias[bias_lds_addr]=%.2f\n", smembias[bias_lds_addr]);
        }

#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
#pragma unroll
            for (int j = 0; j < 2; ++j)
            {
                __syncthreads();

#pragma unroll
                for (int p = 0; p < 4; ++p)
                {
                    sts128(output_frag[i * 4 + p][j * 4 + 0] + smembias[bias_lds_addr + i * 16 + p],
                           output_frag[i * 4 + p][j * 4 + 1] + smembias[bias_lds_addr + i * 16 + p],
                           output_frag[i * 4 + p][j * 4 + 2] + smembias[bias_lds_addr + i * 16 + p],
                           output_frag[i * 4 + p][j * 4 + 3] + smembias[bias_lds_addr + i * 16 + p],
                           outputs_sts_addr + p * 8 * sizeof(float4));
                }
                __syncthreads();

#pragma unroll
                for (int p = 0; p < 16; ++p)
                {
                    stg32(outputs_lds_ptr[p * 32], outputs_stg_ptr + (i * 16 + p) * param.outHW + j * 32,
                          j * 32 < n_guard);
                }
            }
        }
    }
    else
    {
        uint32_t n_guard = param.outHW < n_idx ? 0 : param.outHW - n_idx;
        uint32_t m_guard = param.out_ch < m_idx ? 0 : param.out_ch - m_idx;

        // DEBUG: Print before writing output (else branch)
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
        {
            printf("=== Before writing output (else branch) ===\n");
            printf("output_frag[0][0]=%.2f (expected 512.0 for out[0])\n", output_frag[0][0]);
            printf("output_frag[0][1]=%.2f (expected 768.0 for out[1])\n", output_frag[0][1]);
            printf("m_idx=%d, n_idx=%d, m_guard=%d, n_guard=%d\n", m_idx, n_idx, m_guard, n_guard);
            printf("Missing for out[0]: %.0f channels (%.0f / 4)\n", (512 - output_frag[0][0]) / 4, 512 - output_frag[0][0]);
            printf("Missing for out[1]: %.0f channels (%.0f / 6)\n", (768 - output_frag[0][1]) / 6, 768 - output_frag[0][1]);
        }

#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
#pragma unroll
            for (int j = 0; j < 2; ++j)
            {
                __syncthreads();

#pragma unroll
                for (int p = 0; p < 4; ++p)
                {
                    sts128(output_frag[i * 4 + p][j * 4 + 0] + smembias[bias_lds_addr + i * 16 + p],
                           output_frag[i * 4 + p][j * 4 + 1] + smembias[bias_lds_addr + i * 16 + p],
                           output_frag[i * 4 + p][j * 4 + 2] + smembias[bias_lds_addr + i * 16 + p],
                           output_frag[i * 4 + p][j * 4 + 3] + smembias[bias_lds_addr + i * 16 + p],
                           outputs_sts_addr + p * 8 * sizeof(float4));
                }
                __syncthreads();

#pragma unroll
                for (int p = 0; p < 16; ++p)
                {
                    stg32(outputs_lds_ptr[p * 32], outputs_stg_ptr + (i * 16 + p) * param.outHW + j * 32,
                          i * 16 + p < m_guard && j * 32 < n_guard);
                }
            }
        }
    }
}


__global__ void conv2d_128x128x8_T_1_kernel(float *inputs, float *weights, float *bias, float *outputs, Conv2DParam param)
{
    // 单缓冲共享内存
    __shared__ float smem[8 * 128 * 2];

    const int lane_id  = threadIdx.x % 32;
    const int warp_id  = threadIdx.x / 32;
    const int mma_tid_x = (lane_id / 2) % 8;
    const int mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);

    // 使用T_kernel的lane_id寻址而非mma_tid寻址
    // lane_id 0-3 -> frag_m 0, lane_id 4-7 -> frag_m 1, ...
    // lane_id 0,4,8,12 -> frag_n 0, lane_id 1,5,9,13 -> frag_n 1, ...
    int lane_frag_m = lane_id / 4;
    int lane_frag_n = lane_id % 4 * 2;  // 每个lane写入2列

    float *smemweight = smem;
    float *smeminput  = smem + 8 * 128;

    uint32_t inputs_sts_addr  = smem_u32addr(smeminput  + (threadIdx.x / 32) * 128 + (threadIdx.x % 32));
    uint32_t weights_sts_addr = smem_u32addr(smemweight + (threadIdx.x / 32) * 128 + (threadIdx.x % 32));
    uint32_t inputs_lds_addr  = smem_u32addr(smeminput  + (warp_id % 2) * 64 + mma_tid_x * 4);
    uint32_t weights_lds_addr = smem_u32addr(smemweight + (warp_id / 2) * 32 + mma_tid_y * 4);

    const char *input_ldg_ptr  = (const char *)(inputs + blockIdx.z * param.inBatchNumel);
    const char *weight_ldg_ptr = (const char *)(
            weights + blockIdx.y * 128 + threadIdx.x / 32 * param.out_ch + threadIdx.x % 32);

    float input_ldg_reg[4];
    float weight_ldg_reg[4];
    float input_frag[8];
    float weight_frag[8];
    float output_frag[8][8];

    // 初始化输出累加器
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            output_frag[i][j] = 0.f;

    // 预计算空间位置
    int posh_ori[4], posw_ori[4];
    for (int i = 0; i < 4; ++i)
    {
        posh_ori[i] = ((blockIdx.x * 128 + threadIdx.x % 32 + i * 32) / param.out_w) * param.Sh - param.Ph;
        posw_ori[i] = ((blockIdx.x * 128 + threadIdx.x % 32 + i * 32) % param.out_w) * param.Sw - param.Pw;
    }

    // Guard条件: 检查输出通道是否越界
    uint32_t weights_ldg_guard = 0;
    for (int i = 0; i < 4; ++i)
    {
        int m_idx = blockIdx.y * 128 + threadIdx.x % 32 + i * 32;
        if (m_idx < param.out_ch)
            weights_ldg_guard |= (1u << i);
    }

    // 主循环: LDG -> STS -> sync -> LDS -> compute
    for (int crs = 0; crs < param.inChKhKw; crs += 8)
    {
        // LDG: 加载weights
        for (int i = 0; i < 4; ++i)
        {
            bool guard = (weights_ldg_guard & (1u << i)) != 0 &&
                         (threadIdx.x / 32) < 8 &&
                         (crs + threadIdx.x / 32) < param.inChKhKw;
            ldg32_nc_0(weight_ldg_reg[i], weight_ldg_ptr + i * 32 * sizeof(float), guard);
        }

        // LDG: 加载inputs
        int curC = (crs + threadIdx.x / 32) / param.KhKw;
        int curR = (crs + threadIdx.x / 32) % param.KhKw / param.Kw;
        int curS = (crs + threadIdx.x / 32) % param.KhKw % param.Kw;
        for (int i = 0; i < 4; ++i)
        {
            int curH = posh_ori[i] + curR;
            int curW = posw_ori[i] + curS;
            int inOffsetTmp = curC * param.inHW + curH * param.in_w + curW;
            bool guard = (crs + threadIdx.x / 32) < param.inChKhKw &&
                         curH >= 0 && curH < param.in_h && curW >= 0 && curW < param.in_w;
            ldg32_nc_0(input_ldg_reg[i], input_ldg_ptr + inOffsetTmp * sizeof(float), guard);
        }

        // STS: 写入共享内存
        for (int i = 0; i < 4; ++i)
        {
            sts32(weight_ldg_reg[i], weights_sts_addr + i * 32 * sizeof(float));
            sts32(input_ldg_reg[i],  inputs_sts_addr  + i * 32 * sizeof(float));
        }
        __syncthreads();

        // LDS + compute
        for (int k_frag = 0; k_frag < 8; ++k_frag)
        {
            lds128(weight_frag[0], weight_frag[1], weight_frag[2], weight_frag[3],
                   weights_lds_addr + k_frag * 128 * sizeof(float));
            lds128(weight_frag[4], weight_frag[5], weight_frag[6], weight_frag[7],
                   weights_lds_addr + (k_frag * 128 + 16) * sizeof(float));

            lds128(input_frag[0], input_frag[1], input_frag[2], input_frag[3],
                   inputs_lds_addr + k_frag * 128 * sizeof(float));
            lds128(input_frag[4], input_frag[5], input_frag[6], input_frag[7],
                   inputs_lds_addr + (k_frag * 128 + 32) * sizeof(float));

            // 计算: 累加到output_frag
            for (int i = 0; i < 8; ++i)
                for (int j = 0; j < 8; ++j)
                    output_frag[i][j] += weight_frag[i] * input_frag[j];
        }

        __syncthreads();
        weight_ldg_ptr += 8 * param.out_ch * sizeof(float);
    }

    // Epilogue: 使用 T_kernel 的 lane_id 到 output_frag 映射
    // T_kernel: 每个 lane 在不同 iteration 写入不同的 frag_m
    // i=0: lane 0-15 写 frag_m 0-3, lane 16-31 写 frag_m 4-7
    // i=1: lane 0-15 写 frag_m 4-7, lane 16-31 写 frag_m 0-3

    int warp_m_base = blockIdx.y * 128 + (warp_id / 2) * 32;
    int warp_n_base = blockIdx.x * 128 + (warp_id % 2) * 64;

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            // T_kernel 的映射: 每个 lane 写 16 个元素
            for (int p = 0; p < 16; ++p)
            {
                int out_m = warp_m_base + i * 16 + p;
                if (out_m >= param.out_ch || out_m >= blockIdx.y * 128 + 128)
                    continue;

                int out_n = warp_n_base + j * 32 + lane_id;
                if (out_n >= param.outHW || out_n >= blockIdx.x * 128 + 128)
                    continue;

                // 正确的 frag_m 映射（T_kernel 模式）
                int frag_m;
                if (i == 0)
                    frag_m = (lane_id < 16) ? (lane_id / 4) : (lane_id / 4 - 4);
                else
                    frag_m = (lane_id < 16) ? (lane_id / 4 + 4) : (lane_id / 4);

                int frag_n = j * 4 + (p % 4);

                int out_idx = blockIdx.z * param.outBatchNumel + out_m * param.outHW + out_n;
                outputs[out_idx] = output_frag[frag_m][frag_n] + bias[out_m];
            }
        }
    }
}


void conv2d_T_kernel_launch(void *inputs, void *weights, void *bias, void *outputs, Conv2DParam param, uint32_t n)
{
    uint32_t bx = (param.outHW + 127) / 128;
    uint32_t by = (param.out_ch + 127) / 128;
    uint32_t bz = n;

    dim3 block(256);
    dim3 grid(bx, by, bz);

    conv2d_128x128x8_T_kernel<<<grid, block>>>((float *) inputs, (float *) weights, (float *) bias, (float *) outputs, param);
}

void conv2d_T_main()
{
    printf("=== conv2d_T_main started ===\n");
    fflush(stdout);

    int n, c, h, w, k, r, s, u, v, p, q;

    // 128
    n = 2, c = 128, h = 40, w = h, k = 128, r = 1, s = r, u = 1, v = u, p = 0, q = p;
    // n = 2, c = 128, h = 160, w = h, k = 128, r = 1, s = r, u = 1, v = u, p = 0, q = p;
    // n = 2, c = 128, h = 160, w = h, k = 128, r = 3, s = r, u = 1, v = u, p = 1, q = p;


    int out_h = (h - r + 2 * p) / u + 1;
    int out_w = (w - s + 2 * q) / v + 1;
    printf("outH: %4d   outW: %4d\n", out_h, out_w);

    Conv2DParam param;
    param.in_h = h;
    param.in_w = w;
    param.inHW = h * w;
    param.inChKhKw = c * r * s;
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
    cudaMalloc((void **) &bias_device_h, k * sizeof(half2));
    cudaMalloc((void **) &outputs_device_h, n * k * out_h * out_w * sizeof(half));

    for (int i = 0; i < n * c * h * w; i++)
    {
        inputs[i] = 1.0f;
        // inputs[i] = 0.0f;
        // inputs[i] = (rand() & 255) / 256.0f;
        // inputs_h[i] = __float2half_rn(inputs[i]);
    }
    for (int i = 0; i < k * c * r * s; i++)
    {
        weights[i] = 1.0f;
        // weights[i] = (rand() & 255) / 256.0f;
        // weights_h[i] = __float2half_rn(weights[i]);
    }
    for (int i = 0; i < k; i++)
    {
        bias[i] = 0.0f;
        // bias[i] = (rand() & 255) / 256.0f;
        // bias_h[i] = half2(bias[i], bias[i]);
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
    cudaMemcpy(bias_device_h, bias_h, k * sizeof(half2), cudaMemcpyHostToDevice);

    printf("=== Launching kernel ===\n");
    fflush(stdout);

    conv2d_T_kernel_launch(inputs_device, weightsT_device, bias_device, outputs_device, param, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return;
    }
    cudaDeviceSynchronize();

    printf("=== Kernel finished, copying results ===\n");
    printf("Copying %zu bytes (%d * %d * %d * %d * 4)\n",
           n * k * out_h * out_w * sizeof(float), n, k, out_h, out_w);
    fflush(stdout);

    cudaMemcpy(outputs_host, outputs_device, n * k * out_h * out_w * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Memcpy done\n");
    fflush(stdout);

    // Skip timing code for now
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start);
    // float time_elapsed = 0;
    // int iters = 0;
    // for (int i = 0; i < iters; i++) { ... }
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&time_elapsed, start, stop);
    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);

    printf("=================== CPU Calc ===================\n");
    fflush(stdout);
    direct_conv2d_T_cpu(inputs, weights, bias, outputs_cpu, n, c, h, w, k, r, s, u, v, p, q);
    printf("CPU calc done\n");
    fflush(stdout);

    printf("=================== start verfiy ===================\n");
    int error = 0;
    for (int i = 0; i < n * k * out_h * out_w; i++)
    {
        if (abs(outputs_host[i] - outputs_cpu[i]) > 0.0001f)
        {
            printf("Error: %d, gpu: %.4lf, cpu: %.4lf\n", i, outputs_host[i], outputs_cpu[i]);

            error++;
            if (error > 10)
                break;
        }
    }
    printf("%.4lf %.4lf\n", outputs_host[0], outputs_cpu[0]);
    printf("%.4lf %.4lf\n", outputs_host[1], outputs_cpu[1]);
    printf("%.4lf %.4lf\n", outputs_host[2], outputs_cpu[2]);
    printf("%.4lf %.4lf\n", outputs_host[3], outputs_cpu[3]);

    printf("===================  Error: %d  =====================\n", error);

    // Skip performance printing for now
    // float timePerConv = iters > 0 ? time_elapsed / iters : 0;
    // double gflops = iters > 0 ? flopsPerConv / (timePerConv / 1000.0f) : 0;
    printf("%3d %3d %3d %3d | %d %d %d\n", n, c, h, w, r, s, k);
    // if (iters > 0) {
    //     printf("time: %.6f ms\n", timePerConv);
    //     printf("Performance: %.2f GFlops\n", gflops);
    // }

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

int main() 
{
    conv2d_T_main();

    return 0;
}





