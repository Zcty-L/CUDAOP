#include <iostream>
#include <vector>
#include <string>
#include <bitset>
#include <cuda_runtime.h>
#include "conv2d_weightsT.cuh"
#include "cuda_utils.cuh"


void direct_conv2d_cpu(
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
                                            filter[k * C * R * S + c * R * S + r * S + s]);
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


__global__ void conv2d_mma_128x128x8_kernel(float *inputs, float *weights, float *bias, float *outputs, Conv2DParam param)
{
    // Shared Memory Layout:
    // smemweight: [128][8]  -> 1024 floats (4KB)
    // smeminput : [8][128]  -> 1024 floats (4KB)
    __shared__ __align__(128) float smem[8 * 128 * 2];
    float *smemweight = smem;
    float *smeminput  = smem + 128 * 8;

    const int tid = threadIdx.x;
    const int lane_id  = tid % 32;
    const int warp_id  = tid / 32;
    const int mma_tid_x = (lane_id / 2) % 8;
    const int mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);

    // Inputs STS (保持原有逻辑，除非后续也需要优化)
    uint32_t inputs_sts_addr  = smem_u32addr(smeminput  + (tid / 32) * 128 + (tid % 32));
    uint32_t inputs_lds_addr  = smem_u32addr(smeminput  + (warp_id % 2) * 64 + mma_tid_x * 4);
    
    // Weights LDS (针对 [128][8] 布局调整)
    // 每个 Warp 处理 M 维度的 32 行。warp_id / 2 为 0,1,2,3。
    // mma_tid_y * 4 表示该线程起始的 M 偏移。
    uint32_t weights_lds_addr_base = (warp_id / 2) * 32 + mma_tid_y * 4;

    const char *input_ldg_ptr  = (const char *)(inputs + blockIdx.z * param.inBatchNumel);

    float input_ldg_reg[4];
    float input_frag[8];
    float weight_frag[8];
    float output_frag[8][8];

    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            output_frag[i][j] = 0.f;

    int posh_ori[4], posw_ori[4];
    for (int i = 0; i < 4; ++i) {
        posh_ori[i] = ((blockIdx.x * 128 + tid % 32 + i * 32) / param.out_w) * param.Sh - param.Ph;
        posw_ori[i] = ((blockIdx.x * 128 + tid % 32 + i * 32) % param.out_w) * param.Sw - param.Pw;
    }

    for (int crs = 0; crs < param.inChKhKw; crs += 8)
    {
        // --- Weights Load using cp.async (128x8) ---
        // 256 线程平分 128*8=1024 个元素，每人搬运 4 个 float (16B)
        int w_row = tid / 2;       // 0-127 (M dimension)
        int w_col = (tid % 2) * 4; // 0 or 4 (K dimension)
        int gmem_m = blockIdx.y * 128 + w_row;
        int gmem_k = crs + w_col;
        
        uint32_t w_smem_ptr = smem_u32addr(&smemweight[w_row * 8 + w_col]);
        int src_size = (gmem_m < param.out_ch) ? 16 : 0;

        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
            :: "r"(w_smem_ptr), 
               "l"(&weights[gmem_m * param.inChKhKw + gmem_k]), 
               "r"(src_size)
        );

        // --- Inputs Load (Original ldg32 logic) ---
        int curC = (crs + tid / 32) / param.KhKw;
        int curR = (crs + tid / 32) % param.KhKw / param.Kw;
        int curS = (crs + tid / 32) % param.KhKw % param.Kw;
        for (int i = 0; i < 4; ++i) {
            int curH = posh_ori[i] + curR;
            int curW = posw_ori[i] + curS;
            int inOffsetTmp = curC * param.inHW + curH * param.in_w + curW;
            bool guard = (crs + tid / 32) < param.inChKhKw &&
                         curH >= 0 && curH < param.in_h && curW >= 0 && curW < param.in_w;
            ldg32_nc_0(input_ldg_reg[i], input_ldg_ptr + inOffsetTmp * sizeof(float), guard);
        }

        // STS Inputs
        for (int i = 0; i < 4; ++i) {
            sts32(input_ldg_reg[i], inputs_sts_addr + i * 32 * sizeof(float));
        }

        asm volatile("cp.async.commit_group;\n" :::);
        asm volatile("cp.async.wait_group 0;\n" :::);
        __syncthreads();

        // --- Compute Step ---
        for (int k_frag = 0; k_frag < 8; ++k_frag)
        {
            // LDS Weights: 从 [128][8] 加载 8 个 M 方向的权重
            // 因为布局是 M 优先，固定 k 时的 M 是不连续的（步长为 8）
            // 我们手动加载以保证正确性
            #pragma unroll
            for(int i=0; i<8; ++i) {
                weight_frag[i] = smemweight[(weights_lds_addr_base + i) * 8 + k_frag];
            }

            // LDS Inputs: [8][128] 布局，K 为首维度，连续
            lds128(input_frag[0], input_frag[1], input_frag[2], input_frag[3],
                   inputs_lds_addr + k_frag * 128 * sizeof(float));
            lds128(input_frag[4], input_frag[5], input_frag[6], input_frag[7],
                   inputs_lds_addr + (k_frag * 128 + 32) * sizeof(float));

            for (int i = 0; i < 8; ++i)
                for (int j = 0; j < 8; ++j)
                    output_frag[i][j] += weight_frag[i] * input_frag[j];
        }

        __syncthreads();
    }

    // Epilogue
    int warp_m_base = blockIdx.y * 128 + (warp_id / 2) * 32;
    int warp_n_base = blockIdx.x * 128 + (warp_id % 2) * 64;

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int p = 0; p < 16; ++p) {
                int out_m = warp_m_base + i * 16 + p;
                if (out_m >= param.out_ch) continue;

                int out_n = warp_n_base + j * 32 + lane_id;
                if (out_n >= param.outHW) continue;

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


void conv2d_mma_kernel_launch(void *inputs, void *weights, void *bias, void *outputs, Conv2DParam param, uint32_t n)
{
    dim3 block(256);
    dim3 grid((param.outHW + 127) / 128, (param.out_ch + 127) / 128, n);
    conv2d_mma_128x128x8_kernel<<<grid, block>>>((float *) inputs, (float *) weights, (float *) bias, (float *) outputs, param);
}

void conv2d_mma_main()
{
    printf("=== conv2d_mma_main with cp.async started ===\n");

    Conv2DParam param;
    uint32_t n = 2; 
    param.in_ch = 128;
    param.in_h = 40;
    param.in_w = 40;
    param.inHW = param.in_h * param.in_w;
    param.inBatchNumel = param.in_ch * param.inHW;
    param.out_ch = 128;
    param.Kh = 3;
    param.Kw = 3;
    param.KhKw = param.Kh * param.Kw;
    param.inChKhKw = param.in_ch * param.KhKw;
    param.Sh = 2;
    param.Sw = 2;
    param.Ph = 1;
    param.Pw = 1;
    param.out_h = (param.in_h - param.Kh + 2 * param.Ph) / param.Sh + 1;
    param.out_w = (param.in_w - param.Kw + 2 * param.Pw) / param.Sw + 1;
    param.outHW = param.out_h * param.out_w;
    param.outBatchNumel = param.out_ch * param.outHW;
    param.k_tiles = (param.inChKhKw + 7) / 8 - 1;
    param.first_k_tile = param.inChKhKw - param.k_tiles * 8;

    float *inputs, *weights, *bias, *outputs_cpu, *outputs_host;
    cudaMallocHost((void **) &inputs, n * param.inBatchNumel * sizeof(float));
    cudaMallocHost((void **) &weights, param.out_ch * param.inChKhKw * sizeof(float));
    cudaMallocHost((void **) &bias, param.out_ch * sizeof(float));
    cudaMallocHost((void **) &outputs_cpu, n * param.outBatchNumel * sizeof(float));
    cudaMallocHost((void **) &outputs_host, n * param.outBatchNumel * sizeof(float));

    float *inputs_device, *weights_device, *bias_device, *outputs_device;
    cudaMalloc((void **) &inputs_device, n * param.inBatchNumel * sizeof(float));
    cudaMalloc((void **) &weights_device, param.out_ch * param.inChKhKw * sizeof(float));
    cudaMalloc((void **) &bias_device, param.out_ch * sizeof(float));
    cudaMalloc((void **) &outputs_device, n * param.outBatchNumel * sizeof(float));

    for (int i = 0; i < n * param.inBatchNumel; i++) inputs[i] = 1.0f;
    for (int i = 0; i < param.out_ch * param.inChKhKw; i++) weights[i] = (float)(i % 7); 
    for (int i = 0; i < param.out_ch; i++) bias[i] = 0.0f;

    cudaMemcpy(inputs_device, inputs, n * param.inBatchNumel * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weights_device, weights, param.out_ch * param.inChKhKw * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bias_device, bias, param.out_ch * sizeof(float), cudaMemcpyHostToDevice);

    printf("=== Launching kernel with cp.async weights ===\n");
    conv2d_mma_kernel_launch(inputs_device, weights_device, bias_device, outputs_device, param, n);

    cudaDeviceSynchronize();
    cudaMemcpy(outputs_host, outputs_device, n * param.outBatchNumel * sizeof(float), cudaMemcpyDeviceToHost);

    printf("=================== CPU Calc ===================\n");
    direct_conv2d_cpu(inputs, weights, bias, outputs_cpu, 
                      n, param.in_ch, param.in_h, param.in_w, 
                      param.out_ch, param.Kh, param.Kw, 
                      param.Sh, param.Sw, param.Ph, param.Pw);

    printf("=================== start verify ===================\n");
    int error = 0;
    for (int i = 0; i < n * param.outBatchNumel; i++) {
        if (abs(outputs_host[i] - outputs_cpu[i]) > 0.1f) {
            if (error < 10) printf("Error at %d: GPU=%.4f, CPU=%.4f\n", i, outputs_host[i], outputs_cpu[i]);
            error++;
        }
    }
    printf("Total Errors: %d\n", error);

    cudaFree(inputs_device); cudaFree(weights_device); cudaFree(bias_device); cudaFree(outputs_device);
    cudaFreeHost(inputs); cudaFreeHost(weights); cudaFreeHost(bias); cudaFreeHost(outputs_cpu); cudaFreeHost(outputs_host);
}

int main() {
    conv2d_mma_main();
    return 0;
}
