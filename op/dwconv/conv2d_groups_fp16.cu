#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include "cuda_utils.cuh"
#include "ptx_utils.cuh"

// #define use_cudnn

void direct_conv2d_groups_cpu(
        const float *input, const float *filter, const float *bias, float *output,
        int N, int C, int H, int W, int R, int S, int U, int V, int P, int Q)
{
    int Oh = (H + 2 * P - R) / U + 1;
    int Ow = (W + 2 * Q - S) / V + 1;

    for (int n = 0; n < N; n++)
    {
        for (int c = 0; c < C; c++)
        {
            for (int oh = 0; oh < Oh; oh++)
            {
                for (int ow = 0; ow < Ow; ow++)
                {
                    float sum = 0;
                    for (int r = 0; r < R; r++)
                    {
                        for (int s = 0; s < S; s++)
                        {
                            int ih = oh * U - P + r;
                            int iw = ow * V - Q + s;
                            if (iw >= 0 && ih >= 0 && iw < W && ih < H)
                            {
                                sum += (input[n * C * H * W + c * W * H + ih * W + iw] *
                                        filter[c * R * S + r * S + s]);
                            }
                        }
                    }

                    output[n * C * Oh * Ow + c * Oh * Ow + oh * Ow + ow] = sum + bias[c];
                }
            }
        }
    }
}


__global__ void
conv2d_4x128x256_FP16_groups_kernel(__half *inputs, __half2 *weights, __half2 *bias, __half *outputs, Conv2DParam param)
{
    __shared__ __align__(2 * 1024)
    __shared__ char smem[4 * 128 * 4 + 4 * 4];
    auto *smemweight = reinterpret_cast<__half2 *>(smem); // 4*128*4 = 2*1024

    int posh_ori = ((blockIdx.x * 256 + threadIdx.x) / param.out_w) * param.Sh - param.Ph;
    int posw_ori = ((blockIdx.x * 256 + threadIdx.x) % param.out_w) * param.Sw - param.Pw;

    uint32_t weights_sts_addr = ptx::smem_u32addr(smemweight + threadIdx.x * 2);
    uint32_t weights_lds_addr = ptx::smem_u32addr(smemweight);

    const char *weight_ldg_ptr = (const char *) (
            weights + blockIdx.y * 4 * param.KhKw + threadIdx.x / 64 * param.KhKw + threadIdx.x % 64 * 2);
    __half *input_ptr = inputs + blockIdx.z * param.inBatchNumel + blockIdx.y * 8 * param.inHW;

    __half2 weight_ldg_reg[2];

    __half2 weight_frag[16];
    __half2 input_frag[4][4];
    __half2 output_frag[4];

#pragma unroll
    for (int i = 0; i < 4; ++i)
    { output_frag[i] = make_half2(0, 0); }

    ptx::ldg_nc_0(weight_ldg_reg[0], weight_ldg_ptr, threadIdx.x % 64 * 2 < param.KhKw);
    ptx::ldg_nc_0(weight_ldg_reg[1], weight_ldg_ptr + 1 * sizeof(__half2), threadIdx.x % 64 * 2 + 1 < param.KhKw);

    ptx::sts64(weight_ldg_reg[0], weight_ldg_reg[1], weights_sts_addr);
    __syncthreads();

    for (int k = 0; k < param.KhKw; k += 4)
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int curH = posh_ori + (k + i) / param.Kw;
            int curW = posw_ori + (k + i) % param.Kw;
            int inOffsetTmp = curH * param.in_w + curW;
            if (curH >= 0 && curW >= 0 && curW < param.in_w && curH < param.in_h)
            {
                input_frag[0][i].x = input_ptr[inOffsetTmp + param.inHW * 0];
                input_frag[0][i].y = input_ptr[inOffsetTmp + param.inHW * 1];
                input_frag[1][i].x = input_ptr[inOffsetTmp + param.inHW * 2];
                input_frag[1][i].y = input_ptr[inOffsetTmp + param.inHW * 3];
                input_frag[2][i].x = input_ptr[inOffsetTmp + param.inHW * 4];
                input_frag[2][i].y = input_ptr[inOffsetTmp + param.inHW * 5];
                input_frag[3][i].x = input_ptr[inOffsetTmp + param.inHW * 6];
                input_frag[3][i].y = input_ptr[inOffsetTmp + param.inHW * 7];
            }
            else
            {
                input_frag[0][i] = make_half2(0, 0);
                input_frag[1][i] = make_half2(0, 0);
                input_frag[2][i] = make_half2(0, 0);
                input_frag[3][i] = make_half2(0, 0);
            }
        }

        ptx::lds128(weight_frag[0], weight_frag[1], weight_frag[2], weight_frag[3], weights_lds_addr);
        ptx::lds128(weight_frag[4], weight_frag[5], weight_frag[6], weight_frag[7],
               weights_lds_addr + 128 * sizeof(__half2));
        ptx::lds128(weight_frag[8], weight_frag[9], weight_frag[10], weight_frag[11],
               weights_lds_addr + 256 * sizeof(__half2));
        ptx::lds128(weight_frag[12], weight_frag[13], weight_frag[14], weight_frag[15],
               weights_lds_addr + 384 * sizeof(__half2));
        __syncthreads();

        weights_lds_addr += 4 * sizeof(__half2);

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            output_frag[0] += weight_frag[i + 0] * input_frag[0][i];
            output_frag[1] += weight_frag[i + 4] * input_frag[1][i];
            output_frag[2] += weight_frag[i + 8] * input_frag[2][i];
            output_frag[3] += weight_frag[i + 12] * input_frag[3][i];
        }
    }

    __half2 *smembias = reinterpret_cast<__half2 *>(smem);
    if (threadIdx.x < 4)
    { smembias[threadIdx.x] = bias[blockIdx.y * 4 + threadIdx.x]; }
    __syncthreads();

    output_frag[0] += smembias[0];
    output_frag[1] += smembias[1];
    output_frag[2] += smembias[2];
    output_frag[3] += smembias[3];

    int outOffset = blockIdx.z * param.outBatchNumel + blockIdx.y * 8 * param.outHW + blockIdx.x * 256 + threadIdx.x;
    if (blockIdx.x * 256 + threadIdx.x < param.outHW)
    {
        outputs[outOffset + param.outHW * 0] = output_frag[0].x;
        outputs[outOffset + param.outHW * 1] = output_frag[0].y;
        outputs[outOffset + param.outHW * 2] = output_frag[1].x;
        outputs[outOffset + param.outHW * 3] = output_frag[1].y;
        outputs[outOffset + param.outHW * 4] = output_frag[2].x;
        outputs[outOffset + param.outHW * 5] = output_frag[2].y;
        outputs[outOffset + param.outHW * 6] = output_frag[3].x;
        outputs[outOffset + param.outHW * 7] = output_frag[3].y;
    }
}

__global__ void
conv2d_4x128x256_FP16_S_groups_kernel(uint16_t *inputs, __half2 *weights, __half2 *bias, __half *outputs,
                                      Conv2DParam param)
{
    __shared__ __align__(2 * 1024)
    __shared__ char smem[4 * 128 * 4 + 4 * 4];
    auto *smemweight = reinterpret_cast<__half2 *>(smem); // 4*128*4 = 2*1024

    int posh_ori = ((blockIdx.x * 256 + threadIdx.x) / param.out_w) * param.Sh - param.Ph;
    int posw_ori = ((blockIdx.x * 256 + threadIdx.x) % param.out_w) * param.Sw - param.Pw;

    uint32_t weights_sts_addr = ptx::smem_u32addr(smemweight + threadIdx.x * 2);
    uint32_t weights_lds_addr = ptx::smem_u32addr(smemweight);

    const char *weight_ldg_ptr = (const char *) (
            weights + blockIdx.y * 4 * param.KhKw + threadIdx.x / 64 * param.KhKw + threadIdx.x % 64 * 2);
    uint16_t *input_ptr = inputs + blockIdx.z * param.inBatchNumel + blockIdx.y * 8 * param.inHW;

    __half2 weight_ldg_reg[2];

    __half2 weight_frag[16];
    ushort2 input_frag[4][4];
    __half2 output_frag[4];

    ushort2 uint16_t_zero;
    uint16_t_zero.x = 0, uint16_t_zero.y = 0;

#pragma unroll
    for (int i = 0; i < 4; ++i)
    { output_frag[i] = make_half2(0, 0); }

    ptx::ldg_nc_0(weight_ldg_reg[0], weight_ldg_ptr, threadIdx.x % 64 * 2 < param.KhKw);
    ptx::ldg_nc_0(weight_ldg_reg[1], weight_ldg_ptr + 1 * sizeof(__half2), threadIdx.x % 64 * 2 + 1 < param.KhKw);

    ptx::sts64(weight_ldg_reg[0], weight_ldg_reg[1], weights_sts_addr);
    __syncthreads();

    for (int k = 0; k < param.KhKw; k += 4)
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int curH = posh_ori + (k + i) / param.Kw;
            int curW = posw_ori + (k + i) % param.Kw;
            int inOffsetTmp = curH * param.in_w + curW;
            if (curH >= 0 && curW >= 0 && curW < param.in_w && curH < param.in_h)
            {
                input_frag[0][i].x = input_ptr[inOffsetTmp + param.inHW * 0];
                input_frag[0][i].y = input_ptr[inOffsetTmp + param.inHW * 1];
                input_frag[1][i].x = input_ptr[inOffsetTmp + param.inHW * 2];
                input_frag[1][i].y = input_ptr[inOffsetTmp + param.inHW * 3];
                input_frag[2][i].x = input_ptr[inOffsetTmp + param.inHW * 4];
                input_frag[2][i].y = input_ptr[inOffsetTmp + param.inHW * 5];
                input_frag[3][i].x = input_ptr[inOffsetTmp + param.inHW * 6];
                input_frag[3][i].y = input_ptr[inOffsetTmp + param.inHW * 7];
            }
            else
            {
                input_frag[0][i] = uint16_t_zero;
                input_frag[1][i] = uint16_t_zero;
                input_frag[2][i] = uint16_t_zero;
                input_frag[3][i] = uint16_t_zero;
            }
        }

        ptx::lds128(weight_frag[0], weight_frag[1], weight_frag[2], weight_frag[3], weights_lds_addr);
        ptx::lds128(weight_frag[4], weight_frag[5], weight_frag[6], weight_frag[7],
               weights_lds_addr + 128 * sizeof(__half2));
        ptx::lds128(weight_frag[8], weight_frag[9], weight_frag[10], weight_frag[11],
               weights_lds_addr + 256 * sizeof(__half2));
        ptx::lds128(weight_frag[12], weight_frag[13], weight_frag[14], weight_frag[15],
               weights_lds_addr + 384 * sizeof(__half2));
        __syncthreads();

        weights_lds_addr += 4 * sizeof(__half2);

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            if (input_frag[0][i].x > 0)
            { output_frag[0].x += weight_frag[i + 0].x; }
            if (input_frag[0][i].y > 0)
            { output_frag[0].y += weight_frag[i + 0].y; }
            if (input_frag[1][i].x > 0)
            { output_frag[1].x += weight_frag[i + 4].x; }
            if (input_frag[1][i].y > 0)
            { output_frag[1].y += weight_frag[i + 4].y; }
            if (input_frag[2][i].x > 0)
            { output_frag[2].x += weight_frag[i + 8].x; }
            if (input_frag[2][i].y > 0)
            { output_frag[2].y += weight_frag[i + 8].y; }
            if (input_frag[3][i].x > 0)
            { output_frag[3].x += weight_frag[i + 12].x; }
            if (input_frag[3][i].y > 0)
            { output_frag[3].y += weight_frag[i + 12].y; }
        }
    }

    __half2 *smembias = reinterpret_cast<__half2 *>(smem);
    if (threadIdx.x < 4)
    { smembias[threadIdx.x] = bias[blockIdx.y * 4 + threadIdx.x]; }
    __syncthreads();

    output_frag[0] += smembias[0];
    output_frag[1] += smembias[1];
    output_frag[2] += smembias[2];
    output_frag[3] += smembias[3];

    int outOffset = blockIdx.z * param.outBatchNumel + blockIdx.y * 8 * param.outHW + blockIdx.x * 256 + threadIdx.x;
    if (blockIdx.x * 256 + threadIdx.x < param.outHW)
    {
        outputs[outOffset + param.outHW * 0] = output_frag[0].x;
        outputs[outOffset + param.outHW * 1] = output_frag[0].y;
        outputs[outOffset + param.outHW * 2] = output_frag[1].x;
        outputs[outOffset + param.outHW * 3] = output_frag[1].y;
        outputs[outOffset + param.outHW * 4] = output_frag[2].x;
        outputs[outOffset + param.outHW * 5] = output_frag[2].y;
        outputs[outOffset + param.outHW * 6] = output_frag[3].x;
        outputs[outOffset + param.outHW * 7] = output_frag[3].y;
    }
}


__global__ void
conv2d_4x64x256_FP16_groups_kernel(__half *inputs, __half2 *weights, __half2 *bias, __half *outputs, Conv2DParam param)
{
    __shared__ __align__(1024)
    __shared__ char smem[4 * 64 * 4 + 4 * 4];
    auto *smemweight = reinterpret_cast<__half2 *>(smem); // 4*64 * 4 = 1024

    int posh_ori = ((blockIdx.x * 256 + threadIdx.x) / param.out_w) * param.Sh - param.Ph;
    int posw_ori = ((blockIdx.x * 256 + threadIdx.x) % param.out_w) * param.Sw - param.Pw;

    uint32_t weights_sts_addr = ptx::smem_u32addr(smemweight + threadIdx.x);
    uint32_t weights_lds_addr = ptx::smem_u32addr(smemweight);

    const char *weight_ldg_ptr = (const char *) (
            weights + blockIdx.y * param.KhKw * 4 + threadIdx.x / 64 * param.KhKw + threadIdx.x % 64);
    __half *input_ptr = inputs + blockIdx.z * param.inBatchNumel + blockIdx.y * 8 * param.inHW;

    __half2 weight_ldg_reg;

    __half2 weight_frag[16];
    __half2 input_frag[4][4];
    __half2 output_frag[4];

#pragma unroll
    for (int i = 0; i < 4; ++i)
    { output_frag[i] = make_half2(0, 0); }

    ptx::ldg_nc_0(weight_ldg_reg, weight_ldg_ptr, threadIdx.x % 64 < param.KhKw);

    ptx::sts32(weight_ldg_reg, weights_sts_addr);
    __syncthreads();

    for (int k = 0; k < param.KhKw; k += 4)
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int curH = posh_ori + (k + i) / param.Kw;
            int curW = posw_ori + (k + i) % param.Kw;
            int inOffsetTmp = curH * param.in_w + curW;
            if (curH >= 0 && curW >= 0 && curW < param.in_w && curH < param.in_h)
            {
                input_frag[0][i].x = input_ptr[inOffsetTmp + param.inHW * 0];
                input_frag[0][i].y = input_ptr[inOffsetTmp + param.inHW * 1];
                input_frag[1][i].x = input_ptr[inOffsetTmp + param.inHW * 2];
                input_frag[1][i].y = input_ptr[inOffsetTmp + param.inHW * 3];
                input_frag[2][i].x = input_ptr[inOffsetTmp + param.inHW * 4];
                input_frag[2][i].y = input_ptr[inOffsetTmp + param.inHW * 5];
                input_frag[3][i].x = input_ptr[inOffsetTmp + param.inHW * 6];
                input_frag[3][i].y = input_ptr[inOffsetTmp + param.inHW * 7];
            }
            else
            {
                input_frag[0][i] = make_half2(0, 0);
                input_frag[1][i] = make_half2(0, 0);
                input_frag[2][i] = make_half2(0, 0);
                input_frag[3][i] = make_half2(0, 0);
            }
        }

        ptx::lds128(weight_frag[0], weight_frag[1],
               weight_frag[2], weight_frag[3], weights_lds_addr);
        ptx::lds128(weight_frag[4], weight_frag[5],
               weight_frag[6], weight_frag[7], weights_lds_addr + 64 * sizeof(__half2));
        ptx::lds128(weight_frag[8], weight_frag[9],
               weight_frag[10], weight_frag[11], weights_lds_addr + 128 * sizeof(__half2));
        ptx::lds128(weight_frag[12], weight_frag[13],
               weight_frag[14], weight_frag[15], weights_lds_addr + 192 * sizeof(__half2));
        __syncthreads();

        weights_lds_addr += 4 * sizeof(__half2);

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            output_frag[0] += weight_frag[i + 0] * input_frag[0][i];
            output_frag[1] += weight_frag[i + 4] * input_frag[1][i];
            output_frag[2] += weight_frag[i + 8] * input_frag[2][i];
            output_frag[3] += weight_frag[i + 12] * input_frag[3][i];
        }
    }

    __half2 *smembias = reinterpret_cast<__half2 *>(smem);
    if (threadIdx.x < 4)
    { smembias[threadIdx.x] = bias[blockIdx.y * 4 + threadIdx.x]; }
    __syncthreads();

    output_frag[0] += smembias[0];
    output_frag[1] += smembias[1];
    output_frag[2] += smembias[2];
    output_frag[3] += smembias[3];

    int outOffset = blockIdx.z * param.outBatchNumel + blockIdx.y * 8 * param.outHW + blockIdx.x * 256 + threadIdx.x;
    if (blockIdx.x * 256 + threadIdx.x < param.outHW)
    {
        outputs[outOffset + param.outHW * 0] = output_frag[0].x;
        outputs[outOffset + param.outHW * 1] = output_frag[0].y;
        outputs[outOffset + param.outHW * 2] = output_frag[1].x;
        outputs[outOffset + param.outHW * 3] = output_frag[1].y;
        outputs[outOffset + param.outHW * 4] = output_frag[2].x;
        outputs[outOffset + param.outHW * 5] = output_frag[2].y;
        outputs[outOffset + param.outHW * 6] = output_frag[3].x;
        outputs[outOffset + param.outHW * 7] = output_frag[3].y;
    }
}

__global__ void
conv2d_4x64x256_FP16_S_groups_kernel(uint16_t *inputs, __half2 *weights, __half2 *bias, __half *outputs,
                                     Conv2DParam param)
{
    __shared__ __align__(1024)
    __shared__ char smem[4 * 64 * 4 + 4 * 4];
    auto *smemweight = reinterpret_cast<__half2 *>(smem); // 4*64 * 4 = 1024

    int posh_ori = ((blockIdx.x * 256 + threadIdx.x) / param.out_w) * param.Sh - param.Ph;
    int posw_ori = ((blockIdx.x * 256 + threadIdx.x) % param.out_w) * param.Sw - param.Pw;

    uint32_t weights_sts_addr = ptx::smem_u32addr(smemweight + threadIdx.x);
    uint32_t weights_lds_addr = ptx::smem_u32addr(smemweight);

    const char *weight_ldg_ptr = (const char *) (
            weights + blockIdx.y * param.KhKw * 4 + threadIdx.x / 64 * param.KhKw + threadIdx.x % 64);
    uint16_t *input_ptr = inputs + blockIdx.z * param.inBatchNumel + blockIdx.y * 8 * param.inHW;

    __half2 weight_ldg_reg;

    __half2 weight_frag[16];
    ushort2 input_frag[4][4];
    __half2 output_frag[4];

    ushort2 uint16_t_zero;
    uint16_t_zero.x = 0, uint16_t_zero.y = 0;

#pragma unroll
    for (int i = 0; i < 4; ++i)
    { output_frag[i] = make_half2(0, 0); }

    ptx::ldg_nc_0(weight_ldg_reg, weight_ldg_ptr, threadIdx.x % 64 < param.KhKw);

    ptx::sts32(weight_ldg_reg, weights_sts_addr);
    __syncthreads();

    for (int k = 0; k < param.KhKw; k += 4)
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int curH = posh_ori + (k + i) / param.Kw;
            int curW = posw_ori + (k + i) % param.Kw;
            int inOffsetTmp = curH * param.in_w + curW;
            if (curH >= 0 && curW >= 0 && curW < param.in_w && curH < param.in_h)
            {
                input_frag[0][i].x = input_ptr[inOffsetTmp + param.inHW * 0];
                input_frag[0][i].y = input_ptr[inOffsetTmp + param.inHW * 1];
                input_frag[1][i].x = input_ptr[inOffsetTmp + param.inHW * 2];
                input_frag[1][i].y = input_ptr[inOffsetTmp + param.inHW * 3];
                input_frag[2][i].x = input_ptr[inOffsetTmp + param.inHW * 4];
                input_frag[2][i].y = input_ptr[inOffsetTmp + param.inHW * 5];
                input_frag[3][i].x = input_ptr[inOffsetTmp + param.inHW * 6];
                input_frag[3][i].y = input_ptr[inOffsetTmp + param.inHW * 7];
            }
            else
            {
                input_frag[0][i] = uint16_t_zero;
                input_frag[1][i] = uint16_t_zero;
                input_frag[2][i] = uint16_t_zero;
                input_frag[3][i] = uint16_t_zero;
            }
        }

        ptx::lds128(weight_frag[0], weight_frag[1],
               weight_frag[2], weight_frag[3], weights_lds_addr);
        ptx::lds128(weight_frag[4], weight_frag[5],
               weight_frag[6], weight_frag[7], weights_lds_addr + 64 * sizeof(__half2));
        ptx::lds128(weight_frag[8], weight_frag[9],
               weight_frag[10], weight_frag[11], weights_lds_addr + 128 * sizeof(__half2));
        ptx::lds128(weight_frag[12], weight_frag[13],
               weight_frag[14], weight_frag[15], weights_lds_addr + 192 * sizeof(__half2));
        __syncthreads();

        weights_lds_addr += 4 * sizeof(__half2);

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            if (input_frag[0][i].x > 0)
            { output_frag[0].x += weight_frag[i + 0].x; }
            if (input_frag[0][i].y > 0)
            { output_frag[0].y += weight_frag[i + 0].y; }
            if (input_frag[1][i].x > 0)
            { output_frag[1].x += weight_frag[i + 4].x; }
            if (input_frag[1][i].y > 0)
            { output_frag[1].y += weight_frag[i + 4].y; }
            if (input_frag[2][i].x > 0)
            { output_frag[2].x += weight_frag[i + 8].x; }
            if (input_frag[2][i].y > 0)
            { output_frag[2].y += weight_frag[i + 8].y; }
            if (input_frag[3][i].x > 0)
            { output_frag[3].x += weight_frag[i + 12].x; }
            if (input_frag[3][i].y > 0)
            { output_frag[3].y += weight_frag[i + 12].y; }
        }
    }

    __half2 *smembias = reinterpret_cast<__half2 *>(smem);
    if (threadIdx.x < 4)
    { smembias[threadIdx.x] = bias[blockIdx.y * 4 + threadIdx.x]; }
    __syncthreads();

    output_frag[0] += smembias[0];
    output_frag[1] += smembias[1];
    output_frag[2] += smembias[2];
    output_frag[3] += smembias[3];

    int outOffset = blockIdx.z * param.outBatchNumel + blockIdx.y * 8 * param.outHW + blockIdx.x * 256 + threadIdx.x;
    if (blockIdx.x * 256 + threadIdx.x < param.outHW)
    {
        outputs[outOffset + param.outHW * 0] = output_frag[0].x;
        outputs[outOffset + param.outHW * 1] = output_frag[0].y;
        outputs[outOffset + param.outHW * 2] = output_frag[1].x;
        outputs[outOffset + param.outHW * 3] = output_frag[1].y;
        outputs[outOffset + param.outHW * 4] = output_frag[2].x;
        outputs[outOffset + param.outHW * 5] = output_frag[2].y;
        outputs[outOffset + param.outHW * 6] = output_frag[3].x;
        outputs[outOffset + param.outHW * 7] = output_frag[3].y;
    }
}


__global__ void
conv2d_4x32x256_FP16_groups_kernel(__half *inputs, __half2 *weights, __half2 *bias, __half *outputs, Conv2DParam param)
{
    __shared__ __align__(512)
    __shared__ char smem[4 * 32 * 4 + 4 * 4];
    auto *smemweight = reinterpret_cast<__half2 *>(smem); // 4*32 * 4 = 512

    int posh_ori = ((blockIdx.x * 256 + threadIdx.x) / param.out_w) * param.Sh - param.Ph;
    int posw_ori = ((blockIdx.x * 256 + threadIdx.x) % param.out_w) * param.Sw - param.Pw;

    uint32_t weights_sts_addr = ptx::smem_u32addr(smemweight + threadIdx.x);
    uint32_t weights_lds_addr = ptx::smem_u32addr(smemweight);

    const char *weight_ldg_ptr = (const char *) (
            weights + blockIdx.y * param.KhKw * 4 + threadIdx.x / 32 * param.KhKw + threadIdx.x % 32);
    __half *input_ptr = inputs + blockIdx.z * param.inBatchNumel + blockIdx.y * 8 * param.inHW;

    __half2 weight_ldg_reg;

    __half2 weight_frag[16];
    __half2 input_frag[4][4];
    __half2 output_frag[4];

#pragma unroll
    for (int i = 0; i < 4; ++i)
    { output_frag[i] = make_half2(0, 0); }

    if (threadIdx.x < 128)
    {
        ptx::ldg_nc_0(weight_ldg_reg, weight_ldg_ptr, threadIdx.x % 32 < param.KhKw);
        ptx::sts32(weight_ldg_reg, weights_sts_addr);
    }
    __syncthreads();

    for (int k = 0; k < param.KhKw; k += 4)
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int curH = posh_ori + (k + i) / param.Kw;
            int curW = posw_ori + (k + i) % param.Kw;
            int inOffsetTmp = curH * param.in_w + curW;
            if (curH >= 0 && curW >= 0 && curW < param.in_w && curH < param.in_h)
            {
                input_frag[0][i].x = input_ptr[inOffsetTmp + param.inHW * 0];
                input_frag[0][i].y = input_ptr[inOffsetTmp + param.inHW * 1];
                input_frag[1][i].x = input_ptr[inOffsetTmp + param.inHW * 2];
                input_frag[1][i].y = input_ptr[inOffsetTmp + param.inHW * 3];
                input_frag[2][i].x = input_ptr[inOffsetTmp + param.inHW * 4];
                input_frag[2][i].y = input_ptr[inOffsetTmp + param.inHW * 5];
                input_frag[3][i].x = input_ptr[inOffsetTmp + param.inHW * 6];
                input_frag[3][i].y = input_ptr[inOffsetTmp + param.inHW * 7];
            }
            else
            {
                input_frag[0][i] = make_half2(0, 0);
                input_frag[1][i] = make_half2(0, 0);
                input_frag[2][i] = make_half2(0, 0);
                input_frag[3][i] = make_half2(0, 0);
            }
        }

        ptx::lds128(weight_frag[0], weight_frag[1],
               weight_frag[2], weight_frag[3], weights_lds_addr);
        ptx::lds128(weight_frag[4], weight_frag[5],
               weight_frag[6], weight_frag[7], weights_lds_addr + 32 * sizeof(__half2));
        ptx::lds128(weight_frag[8], weight_frag[9],
               weight_frag[10], weight_frag[11], weights_lds_addr + 64 * sizeof(__half2));
        ptx::lds128(weight_frag[12], weight_frag[13],
               weight_frag[14], weight_frag[15], weights_lds_addr + 96 * sizeof(__half2));
        __syncthreads();

        weights_lds_addr += 4 * sizeof(__half2);

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            output_frag[0] += weight_frag[i + 0] * input_frag[0][i];
            output_frag[1] += weight_frag[i + 4] * input_frag[1][i];
            output_frag[2] += weight_frag[i + 8] * input_frag[2][i];
            output_frag[3] += weight_frag[i + 12] * input_frag[3][i];
        }
    }

    __half2 *smembias = reinterpret_cast<__half2 *>(smem);
    if (threadIdx.x < 4)
    { smembias[threadIdx.x] = bias[blockIdx.y * 4 + threadIdx.x]; }
    __syncthreads();

    output_frag[0] += smembias[0];
    output_frag[1] += smembias[1];
    output_frag[2] += smembias[2];
    output_frag[3] += smembias[3];

    int outOffset = blockIdx.z * param.outBatchNumel + blockIdx.y * 8 * param.outHW + blockIdx.x * 256 + threadIdx.x;
    if (blockIdx.x * 256 + threadIdx.x < param.outHW)
    {
        outputs[outOffset + param.outHW * 0] = output_frag[0].x;
        outputs[outOffset + param.outHW * 1] = output_frag[0].y;
        outputs[outOffset + param.outHW * 2] = output_frag[1].x;
        outputs[outOffset + param.outHW * 3] = output_frag[1].y;
        outputs[outOffset + param.outHW * 4] = output_frag[2].x;
        outputs[outOffset + param.outHW * 5] = output_frag[2].y;
        outputs[outOffset + param.outHW * 6] = output_frag[3].x;
        outputs[outOffset + param.outHW * 7] = output_frag[3].y;
    }
}

__global__ void
conv2d_4x32x256_FP16_S_groups_kernel(uint16_t *inputs, __half2 *weights, __half2 *bias, __half *outputs,
                                     Conv2DParam param)
{
    __shared__ __align__(512)
    __shared__ char smem[4 * 32 * 4 + 4 * 4];
    auto *smemweight = reinterpret_cast<__half2 *>(smem); // 4*32 * 4 = 512

    int posh_ori = ((blockIdx.x * 256 + threadIdx.x) / param.out_w) * param.Sh - param.Ph;
    int posw_ori = ((blockIdx.x * 256 + threadIdx.x) % param.out_w) * param.Sw - param.Pw;

    uint32_t weights_sts_addr = ptx::smem_u32addr(smemweight + threadIdx.x);
    uint32_t weights_lds_addr = ptx::smem_u32addr(smemweight);

    const char *weight_ldg_ptr = (const char *) (
            weights + blockIdx.y * param.KhKw * 4 + threadIdx.x / 32 * param.KhKw + threadIdx.x % 32);
    uint16_t *input_ptr = inputs + blockIdx.z * param.inBatchNumel + blockIdx.y * 8 * param.inHW;

    __half2 weight_ldg_reg;

    __half2 weight_frag[16];
    ushort2 input_frag[4][4];
    __half2 output_frag[4];

    ushort2 uint16_t_zero;
    uint16_t_zero.x = 0, uint16_t_zero.y = 0;

#pragma unroll
    for (int i = 0; i < 4; ++i)
    { output_frag[i] = make_half2(0, 0); }

    if (threadIdx.x < 128)
    {
        ptx::ldg_nc_0(weight_ldg_reg, weight_ldg_ptr, threadIdx.x % 32 < param.KhKw);
        ptx::sts32(weight_ldg_reg, weights_sts_addr);
    }
    __syncthreads();

    for (int k = 0; k < param.KhKw; k += 4)
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int curH = posh_ori + (k + i) / param.Kw;
            int curW = posw_ori + (k + i) % param.Kw;
            int inOffsetTmp = curH * param.in_w + curW;
            if (curH >= 0 && curW >= 0 && curW < param.in_w && curH < param.in_h)
            {
                input_frag[0][i].x = input_ptr[inOffsetTmp + param.inHW * 0];
                input_frag[0][i].y = input_ptr[inOffsetTmp + param.inHW * 1];
                input_frag[1][i].x = input_ptr[inOffsetTmp + param.inHW * 2];
                input_frag[1][i].y = input_ptr[inOffsetTmp + param.inHW * 3];
                input_frag[2][i].x = input_ptr[inOffsetTmp + param.inHW * 4];
                input_frag[2][i].y = input_ptr[inOffsetTmp + param.inHW * 5];
                input_frag[3][i].x = input_ptr[inOffsetTmp + param.inHW * 6];
                input_frag[3][i].y = input_ptr[inOffsetTmp + param.inHW * 7];
            }
            else
            {
                input_frag[0][i] = uint16_t_zero;
                input_frag[1][i] = uint16_t_zero;
                input_frag[2][i] = uint16_t_zero;
                input_frag[3][i] = uint16_t_zero;
            }
        }

        ptx::lds128(weight_frag[0], weight_frag[1],
               weight_frag[2], weight_frag[3], weights_lds_addr);
        ptx::lds128(weight_frag[4], weight_frag[5],
               weight_frag[6], weight_frag[7], weights_lds_addr + 32 * sizeof(__half2));
        ptx::lds128(weight_frag[8], weight_frag[9],
               weight_frag[10], weight_frag[11], weights_lds_addr + 64 * sizeof(__half2));
        ptx::lds128(weight_frag[12], weight_frag[13],
               weight_frag[14], weight_frag[15], weights_lds_addr + 96 * sizeof(__half2));
        __syncthreads();

        weights_lds_addr += 4 * sizeof(__half2);

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            if (input_frag[0][i].x > 0)
            { output_frag[0].x += weight_frag[i + 0].x; }
            if (input_frag[0][i].y > 0)
            { output_frag[0].y += weight_frag[i + 0].y; }
            if (input_frag[1][i].x > 0)
            { output_frag[1].x += weight_frag[i + 4].x; }
            if (input_frag[1][i].y > 0)
            { output_frag[1].y += weight_frag[i + 4].y; }
            if (input_frag[2][i].x > 0)
            { output_frag[2].x += weight_frag[i + 8].x; }
            if (input_frag[2][i].y > 0)
            { output_frag[2].y += weight_frag[i + 8].y; }
            if (input_frag[3][i].x > 0)
            { output_frag[3].x += weight_frag[i + 12].x; }
            if (input_frag[3][i].y > 0)
            { output_frag[3].y += weight_frag[i + 12].y; }
        }
    }

    __half2 *smembias = reinterpret_cast<__half2 *>(smem);
    if (threadIdx.x < 4)
    { smembias[threadIdx.x] = bias[blockIdx.y * 4 + threadIdx.x]; }
    __syncthreads();

    output_frag[0] += smembias[0];
    output_frag[1] += smembias[1];
    output_frag[2] += smembias[2];
    output_frag[3] += smembias[3];

    int outOffset = blockIdx.z * param.outBatchNumel + blockIdx.y * 8 * param.outHW + blockIdx.x * 256 + threadIdx.x;
    if (blockIdx.x * 256 + threadIdx.x < param.outHW)
    {
        outputs[outOffset + param.outHW * 0] = output_frag[0].x;
        outputs[outOffset + param.outHW * 1] = output_frag[0].y;
        outputs[outOffset + param.outHW * 2] = output_frag[1].x;
        outputs[outOffset + param.outHW * 3] = output_frag[1].y;
        outputs[outOffset + param.outHW * 4] = output_frag[2].x;
        outputs[outOffset + param.outHW * 5] = output_frag[2].y;
        outputs[outOffset + param.outHW * 6] = output_frag[3].x;
        outputs[outOffset + param.outHW * 7] = output_frag[3].y;
    }
}


void conv_2d_groups_launch(void *inputs, void *weights, void *bias, void *outputs, Conv2DParam param, uint32_t n)
{
    uint32_t bx = (param.outHW + 255) / 256;
    uint32_t by = param.out_ch / 8; // half
    uint32_t bz = n;

    dim3 block(256);
    dim3 grid(bx, by, bz);

    if (param.KhKw <= 32) {
        conv2d_4x32x256_FP16_S_groups_kernel<<<grid, block>>>((uint16_t *) inputs, (half2 *) weights, (half2 *) bias, (half *) outputs, param);
    } else if (param.KhKw <= 64) {
        conv2d_4x64x256_FP16_S_groups_kernel<<<grid, block>>>((uint16_t *) inputs, (half2 *) weights, (half2 *) bias, (half *) outputs, param);
    } else {
        conv2d_4x128x256_FP16_S_groups_kernel<<<grid, block>>>((uint16_t *) inputs, (half2 *) weights, (half2 *) bias, (half *) outputs, param);
    }
}


int main()
{
    int ns[] = {1, 2, 4, 8, 16, 32};
    int cs[] = {32, 64, 128, 256};
    int hs[] = {40, 80, 160};
    int r = 5, s = 5, u = 2, v = 2, p = 2, q = 2;
    int iters = 100, warmup = 10;

    std::cout << std::left << std::setw(4) << "n" << " |"
              << std::setw(4) << "c" << " |"
              << std::setw(4) << "h" << " |"
              << std::setw(12) << "kernel" << " |"
              << std::setw(12) << "time_ms" << " |"
              << std::setw(12) << "gflops" << std::endl;

    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

    for (int n : ns)
    {
        for (int c : cs)
        {
            for (int h : hs)
            {
                int w = h, k = c;
                int out_h = (h - r + 2 * p) / u + 1;
                int out_w = (w - s + 2 * q) / v + 1;
                int in_numel = n * c * h * w;
                int out_numel = n * k * out_h * out_w;
                int wt_numel = k * r * s;

                double flops = (double)n * out_h * out_w * k * c * r * s * 2.0 / 1e9;

                uint16_t *spikes_h;
                half *input_h, *weight_h, *output_h, *bias_h;
                cudaMallocHost(&spikes_h, in_numel * sizeof(uint16_t));
                cudaMallocHost(&input_h, in_numel * sizeof(half));
                cudaMallocHost(&weight_h, wt_numel * sizeof(half));
                cudaMallocHost(&bias_h, k * sizeof(half));
                cudaMallocHost(&output_h, out_numel * sizeof(half));

                uint16_t *spikes_d;
                half *in_d, *wt_d, *bias_d, *out_d, *in_nhwc_d;
                cudaMalloc(&spikes_d, in_numel * sizeof(uint16_t));
                cudaMalloc(&in_d, in_numel * sizeof(half));
                cudaMalloc(&wt_d, wt_numel * sizeof(half));
                cudaMalloc(&bias_d, k * sizeof(half));
                cudaMalloc(&out_d, out_numel * sizeof(half));
                cudaMalloc(&in_nhwc_d, in_numel * sizeof(half));

                for (int i = 0; i < in_numel; i++) {
                    spikes_h[i] = (rand() % 2);
                    input_h[i] = __float2half_rn((float)spikes_h[i]);
                }
                for (int i = 0; i < wt_numel; i++)
                    weight_h[i] = __float2half_rn((rand() & 1023) / 1024.0f);
                for (int i = 0; i < k; i++)
                    bias_h[i] = __float2half_rn((rand() & 1023) / 1024.0f);

                cudaMemcpy(spikes_d, spikes_h, in_numel * sizeof(uint16_t), cudaMemcpyHostToDevice);
                cudaMemcpy(in_d, input_h, in_numel * sizeof(half), cudaMemcpyHostToDevice);
                cudaMemcpy(wt_d, weight_h, wt_numel * sizeof(half), cudaMemcpyHostToDevice);
                cudaMemcpy(bias_d, bias_h, k * sizeof(half), cudaMemcpyHostToDevice);

                // NCHW -> NHWC
                half *nhwc_h;
                cudaMallocHost(&nhwc_h, in_numel * sizeof(half));
                for (int ni = 0; ni < n; ni++)
                    for (int hi = 0; hi < h; hi++)
                        for (int wi = 0; wi < w; wi++)
                            for (int ci = 0; ci < c; ci++)
                                nhwc_h[ni * h * w * c + hi * w * c + wi * c + ci] =
                                    input_h[ni * c * h * w + ci * h * w + hi * w + wi];
                cudaMemcpy(in_nhwc_d, nhwc_h, in_numel * sizeof(half), cudaMemcpyHostToDevice);

                Conv2DParam param;
                param.in_h = h; param.in_w = w; param.inHW = h * w;
                param.inBatchNumel = c * h * w; param.out_ch = k;
                param.out_w = out_w; param.outHW = out_h * out_w;
                param.outBatchNumel = k * out_h * out_w;
                param.Kw = s; param.KhKw = r * s;
                param.Sh = u; param.Sw = v; param.Ph = p; param.Pw = q;

                // --- custom kernel ---
                conv_2d_groups_launch(spikes_d, wt_d, bias_d, out_d, param, n);
                cudaDeviceSynchronize();
                cudaEvent_t ev0, ev1;
                cudaEventCreate(&ev0); cudaEventCreate(&ev1);
                cudaEventRecord(ev0);
                for (int i = 0; i < iters; i++)
                    conv_2d_groups_launch(spikes_d, wt_d, bias_d, out_d, param, n);
                cudaEventRecord(ev1);
                cudaEventSynchronize(ev1);
                float t = 0;
                cudaEventElapsedTime(&t, ev0, ev1);
                float t_custom = t / iters;
                double g_custom = flops / (t_custom / 1000.0);
                cudaEventDestroy(ev0); cudaEventDestroy(ev1);

                // --- cuDNN NCHW ---
                cudnnTensorDescriptor_t itd, otd, btd;
                cudnnFilterDescriptor_t ftd;
                cudnnConvolutionDescriptor_t cd;
                cudnnCreateTensorDescriptor(&itd); cudnnCreateTensorDescriptor(&otd);
                cudnnCreateTensorDescriptor(&btd); cudnnCreateFilterDescriptor(&ftd);
                cudnnCreateConvolutionDescriptor(&cd);
                cudnnSetTensor4dDescriptor(itd, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, n, c, h, w);
                cudnnSetFilter4dDescriptor(ftd, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, k, 1, r, s);
                cudnnSetTensor4dDescriptor(otd, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, n, k, out_h, out_w);
                cudnnSetTensor4dDescriptor(btd, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 1, k, 1, 1);
                cudnnSetConvolution2dDescriptor(cd, p, q, u, v, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_HALF);
                cudnnSetConvolutionGroupCount(cd, c);
                size_t wss = 0;
                cudnnGetConvolutionForwardWorkspaceSize(cudnn, itd, ftd, cd, otd, algo, &wss);
                void *ws = nullptr;
                if (wss > 0) cudaMalloc(&ws, wss);
                float alpha = 1.0f, beta = 0.0f;
                for (int i = 0; i < warmup; i++)
                    cudnnConvolutionForward(cudnn, &alpha, itd, in_d, ftd, wt_d, cd, algo, ws, wss, &beta, otd, out_d);
                cudaDeviceSynchronize();
                cudaEventCreate(&ev0); cudaEventCreate(&ev1);
                cudaEventRecord(ev0);
                for (int i = 0; i < iters; i++)
                    cudnnConvolutionForward(cudnn, &alpha, itd, in_d, ftd, wt_d, cd, algo, ws, wss, &beta, otd, out_d);
                cudaEventRecord(ev1);
                cudaEventSynchronize(ev1);
                t = 0;
                cudaEventElapsedTime(&t, ev0, ev1);
                float t_nchw = t / iters;
                double g_nchw = flops / (t_nchw / 1000.0);
                cudaEventDestroy(ev0); cudaEventDestroy(ev1);

                // --- cuDNN NHWC ---
                cudnnTensorDescriptor_t itd_n, otd_n;
                cudnnFilterDescriptor_t ftd_n;
                cudnnCreateTensorDescriptor(&itd_n); cudnnCreateTensorDescriptor(&otd_n);
                cudnnCreateFilterDescriptor(&ftd_n);
                cudnnSetTensor4dDescriptor(itd_n, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, n, c, h, w);
                cudnnSetFilter4dDescriptor(ftd_n, CUDNN_DATA_HALF, CUDNN_TENSOR_NHWC, k, 1, r, s);
                cudnnSetTensor4dDescriptor(otd_n, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, n, k, out_h, out_w);
                size_t wss_n = 0;
                cudnnGetConvolutionForwardWorkspaceSize(cudnn, itd_n, ftd_n, cd, otd_n, algo, &wss_n);
                void *ws_n = nullptr;
                if (wss_n > 0) cudaMalloc(&ws_n, wss_n);
                for (int i = 0; i < warmup; i++)
                    cudnnConvolutionForward(cudnn, &alpha, itd_n, in_nhwc_d, ftd_n, wt_d, cd, algo, ws_n, wss_n, &beta, otd_n, out_d);
                cudaDeviceSynchronize();
                cudaEventCreate(&ev0); cudaEventCreate(&ev1);
                cudaEventRecord(ev0);
                for (int i = 0; i < iters; i++)
                    cudnnConvolutionForward(cudnn, &alpha, itd_n, in_nhwc_d, ftd_n, wt_d, cd, algo, ws_n, wss_n, &beta, otd_n, out_d);
                cudaEventRecord(ev1);
                cudaEventSynchronize(ev1);
                t = 0;
                cudaEventElapsedTime(&t, ev0, ev1);
                float t_nhwc = t / iters;
                double g_nhwc = flops / (t_nhwc / 1000.0);
                cudaEventDestroy(ev0); cudaEventDestroy(ev1);

                std::cout << std::left << std::setw(4) << n << " |"
                          << std::setw(4) << c << " |"
                          << std::setw(4) << h << " |"
                          << std::setw(12) << "custom" << " |"
                          << std::fixed << std::setprecision(6) << std::setw(12) << t_custom << " |"
                          << std::setw(12) << std::setw(12) << g_custom << std::endl;
                std::cout << std::left << std::setw(4) << n << " |"
                          << std::setw(4) << c << " |"
                          << std::setw(4) << h << " |"
                          << std::setw(12) << "cudnn_nchw" << " |"
                          << std::fixed << std::setprecision(6) << std::setw(12) << t_nchw << " |"
                          << std::setw(12) << std::setw(12) << g_nchw << std::endl;
                std::cout << std::left << std::setw(4) << n << " |"
                          << std::setw(4) << c << " |"
                          << std::setw(4) << h << " |"
                          << std::setw(12) << "cudnn_nhwc" << " |"
                          << std::fixed << std::setprecision(6) << std::setw(12) << t_nhwc << " |"
                          << std::setw(12) << std::setw(12) << g_nhwc << std::endl;

                if (ws) cudaFree(ws);
                if (ws_n) cudaFree(ws_n);
                cudnnDestroyTensorDescriptor(itd); cudnnDestroyTensorDescriptor(otd);
                cudnnDestroyTensorDescriptor(btd); cudnnDestroyFilterDescriptor(ftd);
                cudnnDestroyConvolutionDescriptor(cd);
                cudnnDestroyTensorDescriptor(itd_n); cudnnDestroyTensorDescriptor(otd_n);
                cudnnDestroyFilterDescriptor(ftd_n);
                cudaFree(spikes_d); cudaFree(in_d); cudaFree(in_nhwc_d); cudaFree(wt_d); cudaFree(bias_d); cudaFree(out_d);
                cudaFreeHost(spikes_h); cudaFreeHost(input_h); cudaFreeHost(nhwc_h);
                cudaFreeHost(weight_h); cudaFreeHost(bias_h); cudaFreeHost(output_h);
            }
        }
    }
    cudnnDestroy(cudnn);
    return 0;
}