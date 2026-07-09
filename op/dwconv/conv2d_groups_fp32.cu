#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include "config.h"
#include "ptx_utils.cuh"
#include "cpu/cpu_ops.h"

using namespace ptx;

__global__ void
conv2d_4x128x256_groups_kernel(float *inputs, float *weights, float *bias, float *outputs, Conv2DParam param)
{
    __shared__ __align__(2 * 1024)
    __shared__ char smem[4 * 128 * 4 + 4 * 4];
    auto *smemweight = reinterpret_cast<float *>(smem); // 4*128*4 = 2*1024

    int posh_ori = ((blockIdx.x * 256 + threadIdx.x) / param.out_w) * param.Sh - param.Ph;
    int posw_ori = ((blockIdx.x * 256 + threadIdx.x) % param.out_w) * param.Sw - param.Pw;

    uint32_t weights_sts_addr = ptx::smem_u32addr(smemweight + threadIdx.x * 2);
    uint32_t weights_lds_addr = ptx::smem_u32addr(smemweight);

    const char *weight_ldg_ptr = (const char *) (
            weights + blockIdx.y * 4 * param.KhKw + threadIdx.x / 64 * param.KhKw + threadIdx.x % 64 * 2);
    auto *input_ptr = inputs + blockIdx.z * param.inBatchNumel + blockIdx.y * 4 * param.inHW;

    float weight_ldg_reg[2];

    float weight_frag[16];
    float input_frag[4][4];
    float output_frag[4];

#pragma unroll
    for (int i = 0; i < 4; ++i)
    { output_frag[i] = 0; }

    ptx::ldg_nc_0(weight_ldg_reg[0], weight_ldg_ptr, threadIdx.x % 64 * 2 < param.KhKw);
    ptx::ldg_nc_0(weight_ldg_reg[1], weight_ldg_ptr + 1 * sizeof(float), threadIdx.x % 64 * 2 + 1 < param.KhKw);

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
                input_frag[0][i] = input_ptr[inOffsetTmp];
                input_frag[1][i] = input_ptr[inOffsetTmp + param.inHW];
                input_frag[2][i] = input_ptr[inOffsetTmp + param.inHW * 2];
                input_frag[3][i] = input_ptr[inOffsetTmp + param.inHW * 3];
            }
            else
            {
                input_frag[0][i] = 0;
                input_frag[1][i] = 0;
                input_frag[2][i] = 0;
                input_frag[3][i] = 0;
            }
        }

        ptx::lds128(weight_frag[0], weight_frag[1], weight_frag[2], weight_frag[3], weights_lds_addr);
        ptx::lds128(weight_frag[4], weight_frag[5], weight_frag[6], weight_frag[7], weights_lds_addr + 128 * sizeof(float));
        ptx::lds128(weight_frag[8], weight_frag[9], weight_frag[10], weight_frag[11],
               weights_lds_addr + 256 * sizeof(float));
        ptx::lds128(weight_frag[12], weight_frag[13], weight_frag[14], weight_frag[15],
               weights_lds_addr + 384 * sizeof(float));
        __syncthreads();

        weights_lds_addr += 4 * sizeof(float);

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            output_frag[0] += weight_frag[i] * input_frag[0][i];
            output_frag[1] += weight_frag[i + 4] * input_frag[1][i];
            output_frag[2] += weight_frag[i + 8] * input_frag[2][i];
            output_frag[3] += weight_frag[i + 12] * input_frag[3][i];
        }
    }

    auto *smembias = reinterpret_cast<float *>(smem);
    if (threadIdx.x < 4)
    { smembias[threadIdx.x] = bias[blockIdx.y * 4 + threadIdx.x]; }
    __syncthreads();

    output_frag[0] += smembias[0];
    output_frag[1] += smembias[1];
    output_frag[2] += smembias[2];
    output_frag[3] += smembias[3];

    int outOffset = blockIdx.z * param.outBatchNumel + blockIdx.y * 4 * param.outHW + blockIdx.x * 256 + threadIdx.x;
    if (blockIdx.x * 256 + threadIdx.x < param.outHW)
    {
        outputs[outOffset + param.outHW * 0] = output_frag[0];
        outputs[outOffset + param.outHW * 1] = output_frag[1];
        outputs[outOffset + param.outHW * 2] = output_frag[2];
        outputs[outOffset + param.outHW * 3] = output_frag[3];
    }
}

__global__ void
conv2d_4x128x256_groups_kernel_biasopt(float *inputs, float *weights, float *bias, float *outputs, Conv2DParam param)
{
    __shared__ __align__(2 * 1024)
    __shared__ char smem[4 * 128 * 4 + 4 * 4];
    auto *smemweight = reinterpret_cast<float *>(smem);

    int posh_ori = ((blockIdx.x * 256 + threadIdx.x) / param.out_w) * param.Sh - param.Ph;
    int posw_ori = ((blockIdx.x * 256 + threadIdx.x) % param.out_w) * param.Sw - param.Pw;

    uint32_t weights_sts_addr = smem_u32addr(smemweight + threadIdx.x * 2);
    uint32_t weights_lds_addr = smem_u32addr(smemweight);

    const char *weight_ldg_ptr = (const char *) (
            weights + blockIdx.y * 4 * param.KhKw + threadIdx.x / 64 * param.KhKw + threadIdx.x % 64 * 2);
    auto *input_ptr = inputs + blockIdx.z * param.inBatchNumel + blockIdx.y * 4 * param.inHW;

    float weight_ldg_reg[2];

    float weight_frag[16];
    float input_frag[4][4];
    float output_frag[4];

#pragma unroll
    for (int i = 0; i < 4; ++i)
    { output_frag[i] = 0; }

    ldg32_nc_0(weight_ldg_reg[0], weight_ldg_ptr, threadIdx.x % 64 * 2 < param.KhKw);
    ldg32_nc_0(weight_ldg_reg[1], weight_ldg_ptr + 1 * sizeof(float), threadIdx.x % 64 * 2 + 1 < param.KhKw);

    sts64(weight_ldg_reg[0], weight_ldg_reg[1], weights_sts_addr);
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
                input_frag[0][i] = input_ptr[inOffsetTmp];
                input_frag[1][i] = input_ptr[inOffsetTmp + param.inHW];
                input_frag[2][i] = input_ptr[inOffsetTmp + param.inHW * 2];
                input_frag[3][i] = input_ptr[inOffsetTmp + param.inHW * 3];
            }
            else
            {
                input_frag[0][i] = 0;
                input_frag[1][i] = 0;
                input_frag[2][i] = 0;
                input_frag[3][i] = 0;
            }
        }

        lds128(weight_frag[0], weight_frag[1], weight_frag[2], weight_frag[3], weights_lds_addr);
        lds128(weight_frag[4], weight_frag[5], weight_frag[6], weight_frag[7], weights_lds_addr + 128 * sizeof(float));
        lds128(weight_frag[8], weight_frag[9], weight_frag[10], weight_frag[11],
               weights_lds_addr + 256 * sizeof(float));
        lds128(weight_frag[12], weight_frag[13], weight_frag[14], weight_frag[15],
               weights_lds_addr + 384 * sizeof(float));
        __syncthreads();

        weights_lds_addr += 4 * sizeof(float);

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            output_frag[0] += weight_frag[i] * input_frag[0][i];
            output_frag[1] += weight_frag[i + 4] * input_frag[1][i];
            output_frag[2] += weight_frag[i + 8] * input_frag[2][i];
            output_frag[3] += weight_frag[i + 12] * input_frag[3][i];
        }
    }

    // Bias via shuffle (no smem, no syncthreads)
    int lane = threadIdx.x & 31;
    float bias_val = 0;
    if (lane < 4)
        bias_val = bias[blockIdx.y * 4 + lane];
    float b0 = __shfl_sync(0xFFFFFFFF, bias_val, 0);
    float b1 = __shfl_sync(0xFFFFFFFF, bias_val, 1);
    float b2 = __shfl_sync(0xFFFFFFFF, bias_val, 2);
    float b3 = __shfl_sync(0xFFFFFFFF, bias_val, 3);

    output_frag[0] += b0;
    output_frag[1] += b1;
    output_frag[2] += b2;
    output_frag[3] += b3;

    int outOffset = blockIdx.z * param.outBatchNumel + blockIdx.y * 4 * param.outHW + blockIdx.x * 256 + threadIdx.x;
    if (blockIdx.x * 256 + threadIdx.x < param.outHW)
    {
        outputs[outOffset + param.outHW * 0] = output_frag[0];
        outputs[outOffset + param.outHW * 1] = output_frag[1];
        outputs[outOffset + param.outHW * 2] = output_frag[2];
        outputs[outOffset + param.outHW * 3] = output_frag[3];
    }
}

__global__ void
conv2d_4x128x256_S_groups_kernel(uint32_t *inputs, float *weights, float *bias, float *outputs, Conv2DParam param)
{
    __shared__ __align__(2 * 1024)
    __shared__ char smem[4 * 128 * 4 + 4 * 4];
    auto *smemweight = reinterpret_cast<float *>(smem); // 4*128*4 = 2*1024

    int posh_ori = ((blockIdx.x * 256 + threadIdx.x) / param.out_w) * param.Sh - param.Ph;
    int posw_ori = ((blockIdx.x * 256 + threadIdx.x) % param.out_w) * param.Sw - param.Pw;

    uint32_t weights_sts_addr = ptx::smem_u32addr(smemweight + threadIdx.x * 2);
    uint32_t weights_lds_addr = ptx::smem_u32addr(smemweight);

    const char *weight_ldg_ptr = (const char *) (
            weights + blockIdx.y * 4 * param.KhKw + threadIdx.x / 64 * param.KhKw + threadIdx.x % 64 * 2);
    auto *input_ptr = inputs + blockIdx.z * param.inBatchNumel + blockIdx.y * 4 * param.inHW;

    float weight_ldg_reg[2];

    float weight_frag[16];
    uint32_t input_frag[4][4];
    float output_frag[4];

#pragma unroll
    for (int i = 0; i < 4; ++i)
    { output_frag[i] = 0; }

    ptx::ldg_nc_0(weight_ldg_reg[0], weight_ldg_ptr, threadIdx.x % 64 * 2 < param.KhKw);
    ptx::ldg_nc_0(weight_ldg_reg[1], weight_ldg_ptr + 1 * sizeof(float), threadIdx.x % 64 * 2 + 1 < param.KhKw);

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
                input_frag[0][i] = input_ptr[inOffsetTmp + param.inHW * 0];
                input_frag[1][i] = input_ptr[inOffsetTmp + param.inHW * 1];
                input_frag[2][i] = input_ptr[inOffsetTmp + param.inHW * 2];
                input_frag[3][i] = input_ptr[inOffsetTmp + param.inHW * 3];
            }
            else
            {
                input_frag[0][i] = 0;
                input_frag[1][i] = 0;
                input_frag[2][i] = 0;
                input_frag[3][i] = 0;
            }
        }

        ptx::lds128(weight_frag[0], weight_frag[1], weight_frag[2], weight_frag[3], weights_lds_addr);
        ptx::lds128(weight_frag[4], weight_frag[5], weight_frag[6], weight_frag[7], weights_lds_addr + 128 * sizeof(float));
        ptx::lds128(weight_frag[8], weight_frag[9], weight_frag[10], weight_frag[11],
               weights_lds_addr + 256 * sizeof(float));
        ptx::lds128(weight_frag[12], weight_frag[13], weight_frag[14], weight_frag[15],
               weights_lds_addr + 384 * sizeof(float));
        __syncthreads();

        weights_lds_addr += 4 * sizeof(float);

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            if (input_frag[0][i])
                output_frag[0] += weight_frag[i + 0];
            if (input_frag[1][i])
                output_frag[1] += weight_frag[i + 4];
            if (input_frag[2][i])
                output_frag[2] += weight_frag[i + 8];
            if (input_frag[3][i])
                output_frag[3] += weight_frag[i + 12];
        }
    }

    auto *smembias = reinterpret_cast<float *>(smem);
    if (threadIdx.x < 4)
    { smembias[threadIdx.x] = bias[blockIdx.y * 4 + threadIdx.x]; }
    __syncthreads();

    output_frag[0] += smembias[0];
    output_frag[1] += smembias[1];
    output_frag[2] += smembias[2];
    output_frag[3] += smembias[3];

    int outOffset = blockIdx.z * param.outBatchNumel + blockIdx.y * 4 * param.outHW + blockIdx.x * 256 + threadIdx.x;
    if (blockIdx.x * 256 + threadIdx.x < param.outHW)
    {
        outputs[outOffset + param.outHW * 0] = output_frag[0];
        outputs[outOffset + param.outHW * 1] = output_frag[1];
        outputs[outOffset + param.outHW * 2] = output_frag[2];
        outputs[outOffset + param.outHW * 3] = output_frag[3];
    }
}


__global__ void
conv2d_4x64x256_groups_kernel(float *inputs, float *weights, float *bias, float *outputs, Conv2DParam param)
{
    __shared__ __align__(1024)
    __shared__ char smem[4 * 64 * 4 + 4 * 4];
    auto *smemweight = reinterpret_cast<float *>(smem); // 4*64 * 4 = 1024

    int posh_ori;
    int posw_ori;
    posh_ori = ((blockIdx.x * 256 + threadIdx.x) / param.out_w) * param.Sh - param.Ph;
    posw_ori = ((blockIdx.x * 256 + threadIdx.x) % param.out_w) * param.Sw - param.Pw;

    uint32_t weights_sts_addr = ptx::smem_u32addr(smemweight + threadIdx.x);
    uint32_t weights_lds_addr = ptx::smem_u32addr(smemweight);

    const char *weight_ldg_ptr = (const char *) (
            weights + blockIdx.y * param.KhKw * 4 + threadIdx.x / 64 * param.KhKw + threadIdx.x % 64);
    auto *input_ptr = inputs + blockIdx.z * param.inBatchNumel + blockIdx.y * param.inHW * 4;

    float weight_ldg_reg;

    float weight_frag[16];
    float input_frag[4][4];
    float output_frag[4];

#pragma unroll
    for (int i = 0; i < 4; ++i)
    { output_frag[i] = 0; }

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
                input_frag[0][i] = input_ptr[inOffsetTmp + param.inHW * 0];
                input_frag[1][i] = input_ptr[inOffsetTmp + param.inHW * 1];
                input_frag[2][i] = input_ptr[inOffsetTmp + param.inHW * 2];
                input_frag[3][i] = input_ptr[inOffsetTmp + param.inHW * 3];
            }
            else
            {
                input_frag[0][i] = 0;
                input_frag[1][i] = 0;
                input_frag[2][i] = 0;
                input_frag[3][i] = 0;
            }
        }

        ptx::lds128(weight_frag[0], weight_frag[1],
               weight_frag[2], weight_frag[3], weights_lds_addr);
        ptx::lds128(weight_frag[4], weight_frag[5],
               weight_frag[6], weight_frag[7], weights_lds_addr + 64 * sizeof(float));
        ptx::lds128(weight_frag[8], weight_frag[9],
               weight_frag[10], weight_frag[11], weights_lds_addr + 128 * sizeof(float));
        ptx::lds128(weight_frag[12], weight_frag[13],
               weight_frag[14], weight_frag[15], weights_lds_addr + 192 * sizeof(float));
        __syncthreads();

        weights_lds_addr += 4 * sizeof(float);

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            output_frag[0] += weight_frag[i + 0] * input_frag[0][i];
            output_frag[1] += weight_frag[i + 4] * input_frag[1][i];
            output_frag[2] += weight_frag[i + 8] * input_frag[2][i];
            output_frag[3] += weight_frag[i + 12] * input_frag[3][i];
        }
    }

    auto *smembias = reinterpret_cast<float *>(smem);
    if (threadIdx.x < 4)
    { smembias[threadIdx.x] = bias[blockIdx.y * 4 + threadIdx.x]; }
    __syncthreads();

    output_frag[0] += smembias[0];
    output_frag[1] += smembias[1];
    output_frag[2] += smembias[2];
    output_frag[3] += smembias[3];

    int outOffset = blockIdx.z * param.outBatchNumel + blockIdx.y * 4 * param.outHW + blockIdx.x * 256 + threadIdx.x;
    if (blockIdx.x * 256 + threadIdx.x < param.outHW)
    {
        outputs[outOffset + param.outHW * 0] = output_frag[0];
        outputs[outOffset + param.outHW * 1] = output_frag[1];
        outputs[outOffset + param.outHW * 2] = output_frag[2];
        outputs[outOffset + param.outHW * 3] = output_frag[3];
    }
}

__global__ void
conv2d_4x64x256_groups_kernel_biasopt(float *inputs, float *weights, float *bias, float *outputs, Conv2DParam param)
{
    __shared__ __align__(1024)
    __shared__ char smem[4 * 64 * 4 + 4 * 4];
    auto *smemweight = reinterpret_cast<float *>(smem);

    int posh_ori;
    int posw_ori;
    posh_ori = ((blockIdx.x * 256 + threadIdx.x) / param.out_w) * param.Sh - param.Ph;
    posw_ori = ((blockIdx.x * 256 + threadIdx.x) % param.out_w) * param.Sw - param.Pw;

    uint32_t weights_sts_addr = smem_u32addr(smemweight + threadIdx.x);
    uint32_t weights_lds_addr = smem_u32addr(smemweight);

    const char *weight_ldg_ptr = (const char *) (
            weights + blockIdx.y * param.KhKw * 4 + threadIdx.x / 64 * param.KhKw + threadIdx.x % 64);
    auto *input_ptr = inputs + blockIdx.z * param.inBatchNumel + blockIdx.y * param.inHW * 4;

    float weight_ldg_reg;

    float weight_frag[16];
    float input_frag[4][4];
    float output_frag[4];

#pragma unroll
    for (int i = 0; i < 4; ++i)
    { output_frag[i] = 0; }

    ldg32_nc_0(weight_ldg_reg, weight_ldg_ptr, threadIdx.x % 64 < param.KhKw);

    sts32(weight_ldg_reg, weights_sts_addr);
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
                input_frag[0][i] = input_ptr[inOffsetTmp + param.inHW * 0];
                input_frag[1][i] = input_ptr[inOffsetTmp + param.inHW * 1];
                input_frag[2][i] = input_ptr[inOffsetTmp + param.inHW * 2];
                input_frag[3][i] = input_ptr[inOffsetTmp + param.inHW * 3];
            }
            else
            {
                input_frag[0][i] = 0;
                input_frag[1][i] = 0;
                input_frag[2][i] = 0;
                input_frag[3][i] = 0;
            }
        }

        lds128(weight_frag[0], weight_frag[1],
               weight_frag[2], weight_frag[3], weights_lds_addr);
        lds128(weight_frag[4], weight_frag[5],
               weight_frag[6], weight_frag[7], weights_lds_addr + 64 * sizeof(float));
        lds128(weight_frag[8], weight_frag[9],
               weight_frag[10], weight_frag[11], weights_lds_addr + 128 * sizeof(float));
        lds128(weight_frag[12], weight_frag[13],
               weight_frag[14], weight_frag[15], weights_lds_addr + 192 * sizeof(float));
        __syncthreads();

        weights_lds_addr += 4 * sizeof(float);

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            output_frag[0] += weight_frag[i + 0] * input_frag[0][i];
            output_frag[1] += weight_frag[i + 4] * input_frag[1][i];
            output_frag[2] += weight_frag[i + 8] * input_frag[2][i];
            output_frag[3] += weight_frag[i + 12] * input_frag[3][i];
        }
    }

    // Bias via shuffle (no smem, no syncthreads)
    int lane = threadIdx.x & 31;
    float bias_val = 0;
    if (lane < 4)
        bias_val = bias[blockIdx.y * 4 + lane];
    float b0 = __shfl_sync(0xFFFFFFFF, bias_val, 0);
    float b1 = __shfl_sync(0xFFFFFFFF, bias_val, 1);
    float b2 = __shfl_sync(0xFFFFFFFF, bias_val, 2);
    float b3 = __shfl_sync(0xFFFFFFFF, bias_val, 3);

    output_frag[0] += b0;
    output_frag[1] += b1;
    output_frag[2] += b2;
    output_frag[3] += b3;

    int outOffset = blockIdx.z * param.outBatchNumel + blockIdx.y * 4 * param.outHW + blockIdx.x * 256 + threadIdx.x;
    if (blockIdx.x * 256 + threadIdx.x < param.outHW)
    {
        outputs[outOffset + param.outHW * 0] = output_frag[0];
        outputs[outOffset + param.outHW * 1] = output_frag[1];
        outputs[outOffset + param.outHW * 2] = output_frag[2];
        outputs[outOffset + param.outHW * 3] = output_frag[3];
    }
}

__global__ void
conv2d_4x64x256_S_groups_kernel(uint32_t *inputs, float *weights, float *bias, float *outputs, Conv2DParam param)
{
    __shared__ __align__(1024)
    __shared__ char smem[4 * 64 * 4 + 4 * 4];
    auto *smemweight = reinterpret_cast<float *>(smem); // 4*64 * 4 = 1024

    int posh_ori = ((blockIdx.x * 256 + threadIdx.x) / param.out_w) * param.Sh - param.Ph;
    int posw_ori = ((blockIdx.x * 256 + threadIdx.x) % param.out_w) * param.Sw - param.Pw;

    uint32_t weights_sts_addr = ptx::smem_u32addr(smemweight + threadIdx.x);
    uint32_t weights_lds_addr = ptx::smem_u32addr(smemweight);

    const char *weight_ldg_ptr = (const char *) (
            weights + blockIdx.y * param.KhKw * 4 + threadIdx.x / 64 * param.KhKw + threadIdx.x % 64);
    auto *input_ptr = inputs + blockIdx.z * param.inBatchNumel + blockIdx.y * 4 * param.inHW;

    float weight_ldg_reg;

    float weight_frag[16];
    uint32_t input_frag[4][4];
    float output_frag[4];

#pragma unroll
    for (int i = 0; i < 4; ++i)
    { output_frag[i] = 0; }

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
                input_frag[0][i] = input_ptr[inOffsetTmp + param.inHW * 0];
                input_frag[1][i] = input_ptr[inOffsetTmp + param.inHW * 1];
                input_frag[2][i] = input_ptr[inOffsetTmp + param.inHW * 2];
                input_frag[3][i] = input_ptr[inOffsetTmp + param.inHW * 3];
            }
            else
            {
                input_frag[0][i] = 0;
                input_frag[1][i] = 0;
                input_frag[2][i] = 0;
                input_frag[3][i] = 0;
            }
        }

        ptx::lds128(weight_frag[0], weight_frag[1],
               weight_frag[2], weight_frag[3], weights_lds_addr);
        ptx::lds128(weight_frag[4], weight_frag[5],
               weight_frag[6], weight_frag[7], weights_lds_addr + 64 * sizeof(float));
        ptx::lds128(weight_frag[8], weight_frag[9],
               weight_frag[10], weight_frag[11], weights_lds_addr + 128 * sizeof(float));
        ptx::lds128(weight_frag[12], weight_frag[13],
               weight_frag[14], weight_frag[15], weights_lds_addr + 192 * sizeof(float));
        __syncthreads();

        weights_lds_addr += 4 * sizeof(float);

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            if (input_frag[0][i])
                output_frag[0] += weight_frag[i + 0];
            if (input_frag[1][i])
                output_frag[1] += weight_frag[i + 4];
            if (input_frag[2][i])
                output_frag[2] += weight_frag[i + 8];
            if (input_frag[3][i])
                output_frag[3] += weight_frag[i + 12];
        }
    }

    auto *smembias = reinterpret_cast<float *>(smem);
    if (threadIdx.x < 4)
    { smembias[threadIdx.x] = bias[blockIdx.y * 4 + threadIdx.x]; }
    __syncthreads();

    output_frag[0] += smembias[0];
    output_frag[1] += smembias[1];
    output_frag[2] += smembias[2];
    output_frag[3] += smembias[3];

    int outOffset = blockIdx.z * param.outBatchNumel + blockIdx.y * 4 * param.outHW + blockIdx.x * 256 + threadIdx.x;
    if (blockIdx.x * 256 + threadIdx.x < param.outHW)
    {
        outputs[outOffset + param.outHW * 0] = output_frag[0];
        outputs[outOffset + param.outHW * 1] = output_frag[1];
        outputs[outOffset + param.outHW * 2] = output_frag[2];
        outputs[outOffset + param.outHW * 3] = output_frag[3];
    }
}


__global__ void
conv2d_4x32x256_groups_kernel(float *inputs, float *weights, float *bias, float *outputs, Conv2DParam param)
{
    __shared__ __align__(128) char smem[4 * 32 * 4 + 4 * 4];
    auto *smemweight = reinterpret_cast<float *>(smem); // 4*32 * 4 = 512

    int posh_ori;
    int posw_ori;
    posh_ori = ((blockIdx.x * 256 + threadIdx.x) / param.out_w) * param.Sh - param.Ph;
    posw_ori = ((blockIdx.x * 256 + threadIdx.x) % param.out_w) * param.Sw - param.Pw;

    uint32_t weights_sts_addr = ptx::smem_u32addr(smemweight + threadIdx.x);
    uint32_t weights_lds_addr = ptx::smem_u32addr(smemweight);

    const char *weight_ldg_ptr = (const char *) (
            weights + blockIdx.y * param.KhKw * 4 + threadIdx.x / 32 * param.KhKw + threadIdx.x % 32);
    auto *input_ptr = inputs + blockIdx.z * param.inBatchNumel + blockIdx.y * param.inHW * 4;

    float weight_ldg_reg;

    float weight_frag[16];
    float input_frag[4][4];
    float output_frag[4];

#pragma unroll
    for (int i = 0; i < 4; ++i)
    { output_frag[i] = 0; }

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
                input_frag[0][i] = input_ptr[inOffsetTmp + param.inHW * 0];
                input_frag[1][i] = input_ptr[inOffsetTmp + param.inHW * 1];
                input_frag[2][i] = input_ptr[inOffsetTmp + param.inHW * 2];
                input_frag[3][i] = input_ptr[inOffsetTmp + param.inHW * 3];
            }
            else
            {
                input_frag[0][i] = 0;
                input_frag[1][i] = 0;
                input_frag[2][i] = 0;
                input_frag[3][i] = 0;
            }
        }

        ptx::lds128(weight_frag[0], weight_frag[1],
               weight_frag[2], weight_frag[3], weights_lds_addr);
        ptx::lds128(weight_frag[4], weight_frag[5],
               weight_frag[6], weight_frag[7], weights_lds_addr + 32 * sizeof(float));
        ptx::lds128(weight_frag[8], weight_frag[9],
               weight_frag[10], weight_frag[11], weights_lds_addr + 64 * sizeof(float));
        ptx::lds128(weight_frag[12], weight_frag[13],
               weight_frag[14], weight_frag[15], weights_lds_addr + 96 * sizeof(float));
        __syncthreads();

        weights_lds_addr += 4 * sizeof(float);

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            output_frag[0] += weight_frag[i + 0] * input_frag[0][i];
            output_frag[1] += weight_frag[i + 4] * input_frag[1][i];
            output_frag[2] += weight_frag[i + 8] * input_frag[2][i];
            output_frag[3] += weight_frag[i + 12] * input_frag[3][i];
        }
    }

    auto *smembias = reinterpret_cast<float *>(smem);
    if (threadIdx.x < 4)
    { smembias[threadIdx.x] = bias[blockIdx.y * 4 + threadIdx.x]; }
    __syncthreads();

    output_frag[0] += smembias[0];
    output_frag[1] += smembias[1];
    output_frag[2] += smembias[2];
    output_frag[3] += smembias[3];

    int outOffset = blockIdx.z * param.outBatchNumel + blockIdx.y * 4 * param.outHW + blockIdx.x * 256 + threadIdx.x;
    if (blockIdx.x * 256 + threadIdx.x < param.outHW)
    {
        outputs[outOffset + param.outHW * 0] = output_frag[0];
        outputs[outOffset + param.outHW * 1] = output_frag[1];
        outputs[outOffset + param.outHW * 2] = output_frag[2];
        outputs[outOffset + param.outHW * 3] = output_frag[3];
    }
}

__global__ void
conv2d_4x32x256_groups_kernel_biasopt(float *inputs, float *weights, float *bias, float *outputs, Conv2DParam param)
{
    __shared__ __align__(128) char smem[4 * 32 * 4 + 4 * 4];
    auto *smemweight = reinterpret_cast<float *>(smem);

    int posh_ori;
    int posw_ori;
    posh_ori = ((blockIdx.x * 256 + threadIdx.x) / param.out_w) * param.Sh - param.Ph;
    posw_ori = ((blockIdx.x * 256 + threadIdx.x) % param.out_w) * param.Sw - param.Pw;

    uint32_t weights_sts_addr = ptx::smem_u32addr(smemweight + threadIdx.x);
    uint32_t weights_lds_addr = ptx::smem_u32addr(smemweight);

    const char *weight_ldg_ptr = (const char *) (
            weights + blockIdx.y * param.KhKw * 4 + threadIdx.x / 32 * param.KhKw + threadIdx.x % 32);
    auto *input_ptr = inputs + blockIdx.z * param.inBatchNumel + blockIdx.y * param.inHW * 4;

    float weight_ldg_reg;

    float weight_frag[16];
    float input_frag[4][4];
    float output_frag[4];

#pragma unroll
    for (int i = 0; i < 4; ++i)
    { output_frag[i] = 0; }

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
                input_frag[0][i] = input_ptr[inOffsetTmp + param.inHW * 0];
                input_frag[1][i] = input_ptr[inOffsetTmp + param.inHW * 1];
                input_frag[2][i] = input_ptr[inOffsetTmp + param.inHW * 2];
                input_frag[3][i] = input_ptr[inOffsetTmp + param.inHW * 3];
            }
            else
            {
                input_frag[0][i] = 0;
                input_frag[1][i] = 0;
                input_frag[2][i] = 0;
                input_frag[3][i] = 0;
            }
        }

        ptx::lds128(weight_frag[0], weight_frag[1],
               weight_frag[2], weight_frag[3], weights_lds_addr);
        ptx::lds128(weight_frag[4], weight_frag[5],
               weight_frag[6], weight_frag[7], weights_lds_addr + 32 * sizeof(float));
        ptx::lds128(weight_frag[8], weight_frag[9],
               weight_frag[10], weight_frag[11], weights_lds_addr + 64 * sizeof(float));
        ptx::lds128(weight_frag[12], weight_frag[13],
               weight_frag[14], weight_frag[15], weights_lds_addr + 96 * sizeof(float));
        __syncthreads();

        weights_lds_addr += 4 * sizeof(float);

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            output_frag[0] += weight_frag[i + 0] * input_frag[0][i];
            output_frag[1] += weight_frag[i + 4] * input_frag[1][i];
            output_frag[2] += weight_frag[i + 8] * input_frag[2][i];
            output_frag[3] += weight_frag[i + 12] * input_frag[3][i];
        }
    }

    // Bias via shuffle (no smem, no syncthreads)
    int lane = threadIdx.x & 31;
    float bias_val = 0;
    if (lane < 4)
        bias_val = bias[blockIdx.y * 4 + lane];
    float b0 = __shfl_sync(0xFFFFFFFF, bias_val, 0);
    float b1 = __shfl_sync(0xFFFFFFFF, bias_val, 1);
    float b2 = __shfl_sync(0xFFFFFFFF, bias_val, 2);
    float b3 = __shfl_sync(0xFFFFFFFF, bias_val, 3);

    output_frag[0] += b0;
    output_frag[1] += b1;
    output_frag[2] += b2;
    output_frag[3] += b3;

    int outOffset = blockIdx.z * param.outBatchNumel + blockIdx.y * 4 * param.outHW + blockIdx.x * 256 + threadIdx.x;
    if (blockIdx.x * 256 + threadIdx.x < param.outHW)
    {
        outputs[outOffset + param.outHW * 0] = output_frag[0];
        outputs[outOffset + param.outHW * 1] = output_frag[1];
        outputs[outOffset + param.outHW * 2] = output_frag[2];
        outputs[outOffset + param.outHW * 3] = output_frag[3];
    }
}

// =============================================================================
// Double-buffer kernel (4x32 variant)
// Two smem buffers (512 B each), ping-pong weight loading with input loading
// =============================================================================
__global__ void
conv2d_4x32x256_groups_kernel_db(float *inputs, float *weights, float *bias, float *outputs, Conv2DParam param)
{
    __shared__ __align__(1024) char smem[2 * 128 * sizeof(float)];  // 1024 bytes, 2 buffers

    uint32_t buf_addr[2] = {
        ptx::smem_u32addr((float *)smem),
        ptx::smem_u32addr((float *)(smem + 512))
    };

    int posh_ori = ((blockIdx.x * 256 + threadIdx.x) / param.out_w) * param.Sh - param.Ph;
    int posw_ori = ((blockIdx.x * 256 + threadIdx.x) % param.out_w) * param.Sw - param.Pw;

    const char *weight_group_base = (const char *)(
        weights + blockIdx.y * param.KhKw * 4);
    auto *input_ptr = inputs + blockIdx.z * param.inBatchNumel + blockIdx.y * param.inHW * 4;

    float weight_ldg_reg;
    float weight_frag[16];
    float input_frag[4][4];
    float output_frag[4] = {0, 0, 0, 0};

    bool is_weight_loader = threadIdx.x < 128;
    int ch_id = threadIdx.x / 32;
    int pos_id = threadIdx.x % 32;

    // Pre-load first batch (positions 0..3) into buffer 0
    if (is_weight_loader && pos_id < 4)
    {
        ptx::ldg_nc_0(weight_ldg_reg,
            weight_group_base + (ch_id * param.KhKw + pos_id) * sizeof(float),
            pos_id < param.KhKw);
        ptx::sts32(weight_ldg_reg, buf_addr[0] + threadIdx.x * sizeof(float));
    }
    __syncthreads();

    int cur_buf = 0;
    int next_buf = 1;

    for (int k = 0; k < param.KhKw; k += 4)
    {
        // Load next batch (k+4..k+7) into next buffer — overlaps with input loading
        int next_k = k + 4;
        if (next_k < param.KhKw)
        {
            if (is_weight_loader && pos_id < 4)
            {
                int load_pos = next_k + pos_id;
                ptx::ldg_nc_0(weight_ldg_reg,
                    weight_group_base + (ch_id * param.KhKw + load_pos) * sizeof(float),
                    load_pos < param.KhKw);
                ptx::sts32(weight_ldg_reg, buf_addr[next_buf] + threadIdx.x * sizeof(float));
            }
        }

        // Load inputs for positions k..k+3
        for (int i = 0; i < 4; ++i)
        {
            int curH = posh_ori + (k + i) / param.Kw;
            int curW = posw_ori + (k + i) % param.Kw;
            int inOffsetTmp = curH * param.in_w + curW;
            if (curH >= 0 && curW >= 0 && curW < param.in_w && curH < param.in_h)
            {
                input_frag[0][i] = input_ptr[inOffsetTmp + param.inHW * 0];
                input_frag[1][i] = input_ptr[inOffsetTmp + param.inHW * 1];
                input_frag[2][i] = input_ptr[inOffsetTmp + param.inHW * 2];
                input_frag[3][i] = input_ptr[inOffsetTmp + param.inHW * 3];
            }
            else
            {
                input_frag[0][i] = 0;
                input_frag[1][i] = 0;
                input_frag[2][i] = 0;
                input_frag[3][i] = 0;
            }
        }

        // Read 4 consecutive weights (k..k+3) from current buffer
        // Each buffer stores 4 positions per channel at offsets 0..3
        ptx::lds128(weight_frag[0], weight_frag[1],
               weight_frag[2], weight_frag[3], buf_addr[cur_buf] + 0);
        ptx::lds128(weight_frag[4], weight_frag[5],
               weight_frag[6], weight_frag[7], buf_addr[cur_buf] + 32 * sizeof(float));
        ptx::lds128(weight_frag[8], weight_frag[9],
               weight_frag[10], weight_frag[11], buf_addr[cur_buf] + 64 * sizeof(float));
        ptx::lds128(weight_frag[12], weight_frag[13],
               weight_frag[14], weight_frag[15], buf_addr[cur_buf] + 96 * sizeof(float));

        __syncthreads();

        // Toggle buffers
        cur_buf ^= 1;
        next_buf ^= 1;

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            output_frag[0] += weight_frag[i + 0] * input_frag[0][i];
            output_frag[1] += weight_frag[i + 4] * input_frag[1][i];
            output_frag[2] += weight_frag[i + 8] * input_frag[2][i];
            output_frag[3] += weight_frag[i + 12] * input_frag[3][i];
        }
    }

    // Bias via shuffle (same as biasopt)
    int lane = threadIdx.x & 31;
    float bias_val = 0;
    if (lane < 4)
        bias_val = bias[blockIdx.y * 4 + lane];
    float b0 = __shfl_sync(0xFFFFFFFF, bias_val, 0);
    float b1 = __shfl_sync(0xFFFFFFFF, bias_val, 1);
    float b2 = __shfl_sync(0xFFFFFFFF, bias_val, 2);
    float b3 = __shfl_sync(0xFFFFFFFF, bias_val, 3);

    output_frag[0] += b0;
    output_frag[1] += b1;
    output_frag[2] += b2;
    output_frag[3] += b3;

    int outOffset = blockIdx.z * param.outBatchNumel + blockIdx.y * 4 * param.outHW + blockIdx.x * 256 + threadIdx.x;
    if (blockIdx.x * 256 + threadIdx.x < param.outHW)
    {
        outputs[outOffset + param.outHW * 0] = output_frag[0];
        outputs[outOffset + param.outHW * 1] = output_frag[1];
        outputs[outOffset + param.outHW * 2] = output_frag[2];
        outputs[outOffset + param.outHW * 3] = output_frag[3];
    }
}

__global__ void
conv2d_4x32x256_S_groups_kernel(uint32_t *inputs, float *weights, float *bias, float *outputs, Conv2DParam param)
{
    __shared__ __align__(512)
    __shared__ char smem[4 * 32 * 4 + 4 * 4];
    auto *smemweight = reinterpret_cast<float *>(smem); // 4*32 * 4 = 512

    int posh_ori = ((blockIdx.x * 256 + threadIdx.x) / param.out_w) * param.Sh - param.Ph;
    int posw_ori = ((blockIdx.x * 256 + threadIdx.x) % param.out_w) * param.Sw - param.Pw;

    uint32_t weights_sts_addr = ptx::smem_u32addr(smemweight + threadIdx.x);
    uint32_t weights_lds_addr = ptx::smem_u32addr(smemweight);

    const char *weight_ldg_ptr = (const char *) (
            weights + blockIdx.y * param.KhKw * 4 + threadIdx.x / 32 * param.KhKw + threadIdx.x % 32);
    auto *input_ptr = inputs + blockIdx.z * param.inBatchNumel + blockIdx.y * 4 * param.inHW;

    float weight_ldg_reg;

    float weight_frag[16];
    uint32_t input_frag[4][4];
    float output_frag[4];

#pragma unroll
    for (int i = 0; i < 4; ++i)
    { output_frag[i] = 0; }

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
                input_frag[0][i] = input_ptr[inOffsetTmp + param.inHW * 0];
                input_frag[1][i] = input_ptr[inOffsetTmp + param.inHW * 1];
                input_frag[2][i] = input_ptr[inOffsetTmp + param.inHW * 2];
                input_frag[3][i] = input_ptr[inOffsetTmp + param.inHW * 3];
            }
            else
            {
                input_frag[0][i] = 0;
                input_frag[1][i] = 0;
                input_frag[2][i] = 0;
                input_frag[3][i] = 0;
            }
        }

        ptx::lds128(weight_frag[0], weight_frag[1],
               weight_frag[2], weight_frag[3], weights_lds_addr);
        ptx::lds128(weight_frag[4], weight_frag[5],
               weight_frag[6], weight_frag[7], weights_lds_addr + 32 * sizeof(float));
        ptx::lds128(weight_frag[8], weight_frag[9],
               weight_frag[10], weight_frag[11], weights_lds_addr + 64 * sizeof(float));
        ptx::lds128(weight_frag[12], weight_frag[13],
               weight_frag[14], weight_frag[15], weights_lds_addr + 96 * sizeof(float));
        __syncthreads();

        weights_lds_addr += 4 * sizeof(float);

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            if (input_frag[0][i])
                output_frag[0] += weight_frag[i + 0];
            if (input_frag[1][i])
                output_frag[1] += weight_frag[i + 4];
            if (input_frag[2][i])
                output_frag[2] += weight_frag[i + 8];
            if (input_frag[3][i])
                output_frag[3] += weight_frag[i + 12];
        }
    }

    auto *smembias = reinterpret_cast<float *>(smem);
    if (threadIdx.x < 4)
    { smembias[threadIdx.x] = bias[blockIdx.y * 4 + threadIdx.x]; }
    __syncthreads();

    output_frag[0] += smembias[0];
    output_frag[1] += smembias[1];
    output_frag[2] += smembias[2];
    output_frag[3] += smembias[3];

    int outOffset = blockIdx.z * param.outBatchNumel + blockIdx.y * 4 * param.outHW + blockIdx.x * 256 + threadIdx.x;
    if (blockIdx.x * 256 + threadIdx.x < param.outHW)
    {
        outputs[outOffset + param.outHW * 0] = output_frag[0];
        outputs[outOffset + param.outHW * 1] = output_frag[1];
        outputs[outOffset + param.outHW * 2] = output_frag[2];
        outputs[outOffset + param.outHW * 3] = output_frag[3];
    }
}


void conv_2d_groups_launch(void *inputs, void *weights, void *bias, void *outputs, Conv2DParam param, uint32_t n)
{
    uint32_t bx = (param.outHW + 255) / 256;
    uint32_t by = param.out_ch / 4;
    uint32_t bz = n;

    dim3 block(256);
    dim3 grid(bx, by, bz);

    if (param.KhKw <= 32) 
    {
        conv2d_4x32x256_groups_kernel<<<grid, block>>>((float *) inputs, (float *) weights, (float *) bias, (float *) outputs, param);
    } 
    else if (param.KhKw <= 64) 
    {
        conv2d_4x64x256_groups_kernel<<<grid, block>>>((float *) inputs, (float *) weights, (float *) bias, (float *) outputs, param);
    } 
    else 
    {
        conv2d_4x128x256_groups_kernel<<<grid, block>>>((float *) inputs, (float *) weights, (float *) bias, (float *) outputs, param);
    }
}

void conv_2d_groups_launch_biasopt(void *inputs, void *weights, void *bias, void *outputs, Conv2DParam param, uint32_t n)
{
    uint32_t bx = (param.outHW + 255) / 256;
    uint32_t by = param.out_ch / 4;
    uint32_t bz = n;

    dim3 block(256);
    dim3 grid(bx, by, bz);

    if (param.KhKw <= 32)
    {
        conv2d_4x32x256_groups_kernel_biasopt<<<grid, block>>>((float *) inputs, (float *) weights, (float *) bias, (float *) outputs, param);
    }
    else if (param.KhKw <= 64)
    {
        conv2d_4x64x256_groups_kernel_biasopt<<<grid, block>>>((float *) inputs, (float *) weights, (float *) bias, (float *) outputs, param);
    }
    else
    {
        conv2d_4x128x256_groups_kernel_biasopt<<<grid, block>>>((float *) inputs, (float *) weights, (float *) bias, (float *) outputs, param);
    }
}

void conv_2d_groups_launch_db(void *inputs, void *weights, void *bias, void *outputs, Conv2DParam param, uint32_t n)
{
    uint32_t bx = (param.outHW + 255) / 256;
    uint32_t by = param.out_ch / 4;
    uint32_t bz = n;

    dim3 block(256);
    dim3 grid(bx, by, bz);

    if (param.KhKw <= 32)
    {
        conv2d_4x32x256_groups_kernel_db<<<grid, block>>>((float *) inputs, (float *) weights, (float *) bias, (float *) outputs, param);
    }
    else
    {
        // fallback: use biasopt kernel for unsupported sizes
        conv_2d_groups_launch_biasopt(inputs, weights, bias, outputs, param, n);
    }
}

void conv2d_groups_main(int r = 5, int n = 1, int c = 32, int h = 80)
{
    bool use_cudnn = false;
    bool verify_value = false;

    int w = h, k = c, s = r, u = 2, v = u, p = r / 2, q = p;

    int out_h = (h - r + 2 * p) / u + 1;
    int out_w = (w - s + 2 * q) / v + 1;
    std::cout << "outH: " << out_h << "  outW: " << out_w << "  " << r / 2 << std::endl;

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

    double temp = n * out_h * out_w * 1e-9f;
    // dwconv: groups=C, C_out=C, C_in_per_group=1 → FLOPS = 2 × N × C × H_out × W_out × Kh × Kw
    double flopsPerConv = temp * c * r * s * 2.0;

    float *input, *weight, *bias, *output_cpu, *output_host;
    cudaMallocHost((void **) &input, n * c * h * w * sizeof(float));
    cudaMallocHost((void **) &weight, k * r * s * sizeof(float));
    cudaMallocHost((void **) &bias, k * sizeof(float));
    cudaMallocHost((void **) &output_cpu, n * k * out_h * out_w * sizeof(float));
    cudaMallocHost((void **) &output_host, n * k * out_h * out_w * sizeof(float));

    float *input_device, *weight_device, *bias_device, *output_device, *input_device_nhwc;
    cudaMalloc((void **) &input_device, n * c * h * w * sizeof(float));
    cudaMalloc((void **) &weight_device, k * r * s * sizeof(float));
    cudaMalloc((void **) &bias_device, k * sizeof(float));
    cudaMalloc((void **) &output_device, n * k * out_h * out_w * sizeof(float));
    cudaMalloc((void **) &input_device_nhwc, n * c * h * w * sizeof(float));

    for (int i = 0; i < n * c * h * w; i++)
    {
        input[i] = (rand() & 1023) / 1024.0f;
    }
    for (int i = 0; i < k * r * s; i++)
    {
        weight[i] = (rand() & 1023) / 1024.0f;
    }
    for (int i = 0; i < k; i++)
    {
        bias[i] = (rand() & 1023) / 1024.0f;
    }

    cudaMemcpy(input_device, input, n * c * h * w * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_device, weight, k * r * s * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bias_device, bias, k * sizeof(float), cudaMemcpyHostToDevice);

    // convert NCHW→NHWC for input
    float *input_nhwc;
    cudaMallocHost(&input_nhwc, n * c * h * w * sizeof(float));
    for (int ni = 0; ni < n; ni++)
        for (int hi = 0; hi < h; hi++)
            for (int wi = 0; wi < w; wi++)
                for (int ci = 0; ci < c; ci++)
                    input_nhwc[ni * h * w * c + hi * w * c + wi * c + ci] =
                        input[ni * c * h * w + ci * h * w + hi * w + wi];
    cudaMemcpy(input_device_nhwc, input_nhwc, n * c * h * w * sizeof(float), cudaMemcpyHostToDevice);

    conv_2d_groups_launch(input_device, weight_device, bias_device, output_device, param, n);
    cudaMemcpy(output_host, output_device, n * k * out_h * out_w * sizeof(float), cudaMemcpyDeviceToHost);

    if (verify_value)
    {
        std::cout << "=================== start verify ===================" << std::endl;
        dwconv2d_cpu(input, weight, bias, output_cpu, n, c, h, w, r, s, u, v, p, q);

        int error = 0;
        for (int i = 0; i < n * k * out_h * out_w; i++)
        {
            if (abs(output_host[i] - output_cpu[i]) > 0.0001f)
            {
                if (error < 10)
                    std::cout << "Error: " << i << ", gpu: " << output_host[i] << ", cpu: " << output_cpu[i] << std::endl;
                error++;
            }
        }
        std::cout << "===================  Error: " << error << "  =====================" << std::endl;
    }

    if (use_cudnn)
    {
        std::cout << "\n\n=================== cudnn ===================\n\n";

        cudnnHandle_t cudnn;
        cudnnCreate(&cudnn);

        cudnnTensorDescriptor_t input_desc;
        cudnnCreateTensorDescriptor(&input_desc);
        cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

        cudnnFilterDescriptor_t filter_desc;
        cudnnCreateFilterDescriptor(&filter_desc);
        cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, k, 1, r, s);

        cudnnConvolutionDescriptor_t conv_desc;
        cudnnCreateConvolutionDescriptor(&conv_desc);
        cudnnSetConvolution2dDescriptor(conv_desc, p, q, u, v, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
        cudnnSetConvolutionGroupCount(conv_desc, c);

        cudnnTensorDescriptor_t bias_desc;
        cudnnCreateTensorDescriptor(&bias_desc);
        cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, k, 1, 1);

        cudnnTensorDescriptor_t output_desc;
        cudnnCreateTensorDescriptor(&output_desc);
        cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, k, out_h, out_w);

        size_t space_size = 0;
        cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_desc, filter_desc, conv_desc, output_desc, algo, &space_size);

        void *workspace = nullptr;
        cudaMalloc(&workspace, space_size);

        float alpha = 1.0f, beta = 0.0f;
        cudnnConvolutionForward(
            cudnn, &alpha,
            input_desc, input_device, filter_desc, weight_device,
            conv_desc, algo, workspace, space_size,
            &beta, output_desc, output_device);
        alpha = 1.0f, beta = 1.0f;
        cudnnAddTensor(cudnn, &alpha, bias_desc, bias_device, &beta, output_desc, output_device);

        cudaMemcpy(output_host, output_device, n * k * out_h * out_w * sizeof(float), cudaMemcpyDeviceToHost);

        if (verify_value)
        {
            std::cout << "=================== start verify ===================" << std::endl;
            int error = 0;
            for (int i = 0; i < n * k * out_h * out_w; i++)
            {
                if (abs(output_host[i] - output_cpu[i]) > 0.0001f)
                {
                    std::cout << "Error: position: " << i << ", gpu: " << output_host[i] << ", cpu: " << output_cpu[i] << std::endl;
                    error++;
                    break;
                }
            }
            std::cout << output_host[0] << " " << output_cpu[0] << std::endl;
            std::cout << output_host[1] << " " << output_cpu[1] << std::endl;
            std::cout << output_cpu[0] << " " << output_cpu[1] << " " << output_cpu[2] << " " << output_cpu[3] << std::endl;
            std::cout << "===================  Error: " << error << "  =====================" << std::endl;
        }

        int iters = 1;

        // Warmup + single launch per kernel for nsight profiling
        // --- custom ---
        for (int i = 0; i < 3; i++)
            conv_2d_groups_launch(input_device, weight_device, bias_device, output_device, param, n);
        cudaDeviceSynchronize();            // warmup done
        conv_2d_groups_launch(input_device, weight_device, bias_device, output_device, param, n); // profile this
        cudaDeviceSynchronize();

        std::cout << "launched custom" << std::endl;

        // --- biasopt ---
        for (int i = 0; i < 3; i++)
            conv_2d_groups_launch_biasopt(input_device, weight_device, bias_device, output_device, param, n);
        cudaDeviceSynchronize();
        conv_2d_groups_launch_biasopt(input_device, weight_device, bias_device, output_device, param, n);
        cudaDeviceSynchronize();

        std::cout << "launched biasopt" << std::endl;

        // --- double-buffer ---
        for (int i = 0; i < 3; i++)
            conv_2d_groups_launch_db(input_device, weight_device, bias_device, output_device, param, n);
        cudaDeviceSynchronize();
        conv_2d_groups_launch_db(input_device, weight_device, bias_device, output_device, param, n);
        cudaDeviceSynchronize();

        std::cout << "launched dbuf" << std::endl;

        // --- cuDNN NCHW ---
        for (int i = 0; i < 3; i++)
            cudnnConvolutionForward(cudnn, &alpha, input_desc, input_device, filter_desc, weight_device,
                                    conv_desc, algo, workspace, space_size, &beta, output_desc, output_device);
        cudaDeviceSynchronize();
        alpha = 1.0f; beta = 1.0f;
        cudnnAddTensor(cudnn, &alpha, bias_desc, bias_device, &beta, output_desc, output_device);
        cudnnConvolutionForward(cudnn, &alpha, input_desc, input_device, filter_desc, weight_device,
                                conv_desc, algo, workspace, space_size, &beta, output_desc, output_device);
        cudaDeviceSynchronize();

        std::cout << "launched cudnn_nchw" << std::endl;

        // --- NHWC cuDNN ---
        cudnnTensorDescriptor_t input_desc_nhwc, output_desc_nhwc;
        cudnnFilterDescriptor_t filter_desc_nhwc;
        cudnnCreateTensorDescriptor(&input_desc_nhwc);
        cudnnCreateTensorDescriptor(&output_desc_nhwc);
        cudnnCreateFilterDescriptor(&filter_desc_nhwc);
        cudnnSetTensor4dDescriptor(input_desc_nhwc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, n, c, h, w);
        cudnnSetFilter4dDescriptor(filter_desc_nhwc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NHWC, k, 1, r, s);
        cudnnSetTensor4dDescriptor(output_desc_nhwc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, n, k, out_h, out_w);

        size_t space_size_nhwc = 0;
        cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_desc_nhwc, filter_desc_nhwc, conv_desc, output_desc_nhwc, algo, &space_size_nhwc);
        void *workspace_nhwc = nullptr;
        cudaMalloc(&workspace_nhwc, space_size_nhwc);

        // single NHWC launch
        for (int i = 0; i < 3; i++)
            cudnnConvolutionForward(cudnn, &alpha, input_desc_nhwc, input_device_nhwc, filter_desc_nhwc, weight_device,
                                    conv_desc, algo, workspace_nhwc, space_size_nhwc, &beta, output_desc_nhwc, output_device);
        cudaDeviceSynchronize();
        cudnnConvolutionForward(cudnn, &alpha, input_desc_nhwc, input_device_nhwc, filter_desc_nhwc, weight_device,
                                conv_desc, algo, workspace_nhwc, space_size_nhwc, &beta, output_desc_nhwc, output_device);
        cudaDeviceSynchronize();

        std::cout << "launched cudnn_nhwc" << std::endl;

        cudaFree(workspace_nhwc);
        cudnnDestroyTensorDescriptor(input_desc_nhwc);
        cudnnDestroyTensorDescriptor(output_desc_nhwc);
        cudnnDestroyFilterDescriptor(filter_desc_nhwc);

        cudaFree(workspace);
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);
        cudnnDestroyFilterDescriptor(filter_desc);
        cudnnDestroyConvolutionDescriptor(conv_desc);
        cudnnDestroy(cudnn);
    }
    
    cudaFree(input_device);
    cudaFree(input_device_nhwc);
    cudaFree(weight_device);
    cudaFree(bias_device);
    cudaFree(output_device);
    cudaFreeHost(input);
    cudaFreeHost(input_nhwc);
    cudaFreeHost(weight);
    cudaFreeHost(bias);
    cudaFreeHost(output_cpu);
    cudaFreeHost(output_host);
}

int main(int argc, char *argv[])
{
    if (argc > 1 && strcmp(argv[1], "--profile") == 0)
    {
        int r = 7, n = 4, c = 128, h = 80;
        if (argc >= 6) { r = atoi(argv[2]); n = atoi(argv[3]); c = atoi(argv[4]); h = atoi(argv[5]); }
        conv2d_groups_main(r, n, c, h);
        return 0;
    }

    std::string csv_path = "benchmark_results_fp32.csv";
    int iters = 100;
    int warmup = 10;
    bool quick = false;

    for (int argi = 1; argi < argc; ++argi)
    {
        if (strcmp(argv[argi], "--csv") == 0 && argi + 1 < argc)
        {
            csv_path = argv[++argi];
        }
        else if (strcmp(argv[argi], "--iters") == 0 && argi + 1 < argc)
        {
            iters = std::atoi(argv[++argi]);
        }
        else if (strcmp(argv[argi], "--warmup") == 0 && argi + 1 < argc)
        {
            warmup = std::atoi(argv[++argi]);
        }
        else if (strcmp(argv[argi], "--quick") == 0)
        {
            quick = true;
        }
    }

    if (iters <= 0)
    {
        std::cout << "[ERROR] --iters must be positive" << std::endl;
        return 1;
    }

    int ns[] = {1, 2, 4, 8, 16, 32};
    int cs[] = {32, 64, 128, 256};
    int hs[] = {40, 80, 160};
    int ns_quick[] = {1, 4, 16};
    int cs_quick[] = {32, 128};
    int hs_quick[] = {40, 80};
    int u = 2, v = 2;
    int kernel_sizes[] = {3, 5, 7, 9, 11};
    int kernel_sizes_quick[] = {3, 7, 11};

    int *active_ns = quick ? ns_quick : ns;
    int *active_cs = quick ? cs_quick : cs;
    int *active_hs = quick ? hs_quick : hs;
    int *active_kernel_sizes = quick ? kernel_sizes_quick : kernel_sizes;
    int ns_count = quick ? 3 : 6;
    int cs_count = quick ? 2 : 4;
    int hs_count = quick ? 2 : 3;
    int kernel_sizes_count = quick ? 3 : 5;

    std::ofstream csv(csv_path);
    if (!csv.is_open())
    {
        std::cout << "[ERROR] failed to open csv: " << csv_path << std::endl;
        return 1;
    }
    csv << "k_size,n,c,h,kernel,time_ms,gflops,arith_intensity" << std::endl;

    std::cout << "[CONFIG] csv=" << csv_path
              << " iters=" << iters
              << " warmup=" << warmup
              << " quick=" << (quick ? "true" : "false")
              << std::endl << std::endl;

    std::cout << std::left << std::setw(6) << "k_size" << " |"
              << std::setw(4) << "n" << " |"
              << std::setw(4) << "c" << " |"
              << std::setw(4) << "h" << " |"
              << std::setw(12) << "kernel" << " |"
              << std::setw(12) << "time_ms" << " |"
              << std::setw(12) << "gflops" << " |"
              << std::setw(12) << "ai" << std::endl;

    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

    for (int r_idx = 0; r_idx < kernel_sizes_count; ++r_idx)
    {
        int r = active_kernel_sizes[r_idx];
        int s = r, p = r / 2, q = s / 2;
        for (int n_idx = 0; n_idx < ns_count; ++n_idx)
        {
            int n = active_ns[n_idx];
            for (int c_idx = 0; c_idx < cs_count; ++c_idx)
            {
                int c = active_cs[c_idx];
                for (int h_idx = 0; h_idx < hs_count; ++h_idx)
                {
                    int h = active_hs[h_idx];
                    int w = h, k = c;
                    int out_h = (h - r + 2 * p) / u + 1;
                    int out_w = (w - s + 2 * q) / v + 1;
                    int in_numel = n * c * h * w;
                    int out_numel = n * k * out_h * out_w;
                    int wt_numel = k * r * s;

                    // dwconv FLOPS: 2 × N × C_out × H_out × W_out × C_in_per_group × Kh × Kw
                    // For groups=C depthwise: C_out=C, C_in_per_group=1
                    double flops = (double)n * out_h * out_w * c * r * s * 2.0 / 1e9;
                    double total_bytes = (double)wt_numel * 4 + (double)in_numel * 4 + (double)out_numel * 4;
                    double arith_intensity = flops * 1e9 / total_bytes;

                    float *input_h, *weight_h, *bias_h, *output_h;
                    cudaMallocHost(&input_h, in_numel * sizeof(float));
                    cudaMallocHost(&weight_h, wt_numel * sizeof(float));
                    cudaMallocHost(&bias_h, k * sizeof(float));
                    cudaMallocHost(&output_h, out_numel * sizeof(float));

                    float *in_d, *wt_d, *bias_d, *out_d, *in_nhwc_d;
                    cudaMalloc(&in_d, in_numel * sizeof(float));
                    cudaMalloc(&wt_d, wt_numel * sizeof(float));
                    cudaMalloc(&bias_d, k * sizeof(float));
                    cudaMalloc(&out_d, out_numel * sizeof(float));
                    cudaMalloc(&in_nhwc_d, in_numel * sizeof(float));

                    for (int i = 0; i < in_numel; i++) input_h[i] = (rand() & 1023) / 1024.0f;
                    for (int i = 0; i < wt_numel; i++) weight_h[i] = (rand() & 1023) / 1024.0f;
                    for (int i = 0; i < k; i++) bias_h[i] = (rand() & 1023) / 1024.0f;

                    cudaMemcpy(in_d, input_h, in_numel * sizeof(float), cudaMemcpyHostToDevice);
                    cudaMemcpy(wt_d, weight_h, wt_numel * sizeof(float), cudaMemcpyHostToDevice);
                    cudaMemcpy(bias_d, bias_h, k * sizeof(float), cudaMemcpyHostToDevice);

                    // NCHW → NHWC
                    float *nhwc_h;
                    cudaMallocHost(&nhwc_h, in_numel * sizeof(float));
                    for (int ni = 0; ni < n; ni++)
                        for (int hi = 0; hi < h; hi++)
                            for (int wi = 0; wi < w; wi++)
                                for (int ci = 0; ci < c; ci++)
                                    nhwc_h[ni * h * w * c + hi * w * c + wi * c + ci] =
                                        input_h[ni * c * h * w + ci * h * w + hi * w + wi];
                    cudaMemcpy(in_nhwc_d, nhwc_h, in_numel * sizeof(float), cudaMemcpyHostToDevice);

                    Conv2DParam param;
                    param.in_h = h; param.in_w = w; param.inHW = h * w;
                    param.inBatchNumel = c * h * w; param.out_ch = k;
                    param.out_w = out_w; param.outHW = out_h * out_w;
                    param.outBatchNumel = k * out_h * out_w;
                    param.Kw = s; param.KhKw = r * s;
                    param.Sh = u; param.Sw = v; param.Ph = p; param.Pw = q;

                    // --- custom kernel ---
                    conv_2d_groups_launch(in_d, wt_d, bias_d, out_d, param, n);
                    cudaDeviceSynchronize();
                    cudaEvent_t ev0, ev1;
                    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
                    cudaEventRecord(ev0);
                    for (int i = 0; i < iters; i++)
                        conv_2d_groups_launch(in_d, wt_d, bias_d, out_d, param, n);
                    cudaEventRecord(ev1);
                    cudaEventSynchronize(ev1);
                    float t = 0;
                    cudaEventElapsedTime(&t, ev0, ev1);
                    float t_custom = t / iters;
                    double g_custom = flops / (t_custom / 1000.0);
                    cudaEventDestroy(ev0); cudaEventDestroy(ev1);

                    // --- biasopt ---
                    conv_2d_groups_launch_biasopt(in_d, wt_d, bias_d, out_d, param, n);
                    cudaDeviceSynchronize();
                    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
                    cudaEventRecord(ev0);
                    for (int i = 0; i < iters; i++)
                        conv_2d_groups_launch_biasopt(in_d, wt_d, bias_d, out_d, param, n);
                    cudaEventRecord(ev1);
                    cudaEventSynchronize(ev1);
                    cudaEventElapsedTime(&t, ev0, ev1);
                    float t_biasopt = t / iters;
                    double g_biasopt = flops / (t_biasopt / 1000.0);
                    cudaEventDestroy(ev0); cudaEventDestroy(ev1);

                    // --- double-buffer ---
                    conv_2d_groups_launch_db(in_d, wt_d, bias_d, out_d, param, n);
                    cudaDeviceSynchronize();
                    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
                    cudaEventRecord(ev0);
                    for (int i = 0; i < iters; i++)
                        conv_2d_groups_launch_db(in_d, wt_d, bias_d, out_d, param, n);
                    cudaEventRecord(ev1);
                    cudaEventSynchronize(ev1);
                    cudaEventElapsedTime(&t, ev0, ev1);
                    float t_db = t / iters;
                    double g_db = flops / (t_db / 1000.0);
                    cudaEventDestroy(ev0); cudaEventDestroy(ev1);

                    // --- cuDNN NCHW ---
                    cudnnTensorDescriptor_t itd, otd, btd;
                    cudnnFilterDescriptor_t ftd;
                    cudnnConvolutionDescriptor_t cd;
                    cudnnCreateTensorDescriptor(&itd); cudnnCreateTensorDescriptor(&otd);
                    cudnnCreateTensorDescriptor(&btd); cudnnCreateFilterDescriptor(&ftd);
                    cudnnCreateConvolutionDescriptor(&cd);
                    cudnnSetTensor4dDescriptor(itd, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
                    cudnnSetFilter4dDescriptor(ftd, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, k, 1, r, s);
                    cudnnSetTensor4dDescriptor(otd, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, k, out_h, out_w);
                    cudnnSetTensor4dDescriptor(btd, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, k, 1, 1);
                    cudnnSetConvolution2dDescriptor(cd, p, q, u, v, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
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
                    cudnnSetTensor4dDescriptor(itd_n, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, n, c, h, w);
                    cudnnSetFilter4dDescriptor(ftd_n, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NHWC, k, 1, r, s);
                    cudnnSetTensor4dDescriptor(otd_n, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, n, k, out_h, out_w);
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

                    csv << r << "," << n << "," << c << "," << h << ","
                        << "custom," << t_custom << "," << g_custom << "," << arith_intensity << std::endl;
                    csv << r << "," << n << "," << c << "," << h << ","
                        << "biasopt," << t_biasopt << "," << g_biasopt << "," << arith_intensity << std::endl;
                    csv << r << "," << n << "," << c << "," << h << ","
                        << "dbuf," << t_db << "," << g_db << "," << arith_intensity << std::endl;
                    csv << r << "," << n << "," << c << "," << h << ","
                        << "cudnn_nchw," << t_nchw << "," << g_nchw << "," << arith_intensity << std::endl;
                    csv << r << "," << n << "," << c << "," << h << ","
                        << "cudnn_nhwc," << t_nhwc << "," << g_nhwc << "," << arith_intensity << std::endl;

                    std::cout << std::left << std::setw(6) << r << " |"
                            << std::setw(4) << n << " |"
                            << std::setw(4) << c << " |"
                            << std::setw(4) << h << " |"
                            << std::setw(12) << "custom" << " |"
                            << std::fixed << std::setprecision(6) << std::setw(12) << t_custom << " |"
                            << std::setw(12) << g_custom << " |"
                            << std::setw(12) << arith_intensity << std::endl;
                    std::cout << std::left << std::setw(6) << r << " |"
                            << std::setw(4) << n << " |"
                            << std::setw(4) << c << " |"
                            << std::setw(4) << h << " |"
                            << std::setw(12) << "biasopt" << " |"
                            << std::fixed << std::setprecision(6) << std::setw(12) << t_biasopt << " |"
                            << std::setw(12) << g_biasopt << " |"
                            << std::setw(12) << arith_intensity << std::endl;
                    std::cout << std::left << std::setw(6) << r << " |"
                            << std::setw(4) << n << " |"
                            << std::setw(4) << c << " |"
                            << std::setw(4) << h << " |"
                            << std::setw(12) << "dbuf" << " |"
                            << std::fixed << std::setprecision(6) << std::setw(12) << t_db << " |"
                            << std::setw(12) << g_db << " |"
                            << std::setw(12) << arith_intensity << std::endl;
                    std::cout << std::left << std::setw(6) << r << " |"
                            << std::setw(4) << n << " |"
                            << std::setw(4) << c << " |"
                            << std::setw(4) << h << " |"
                            << std::setw(12) << "cudnn_nchw" << " |"
                            << std::fixed << std::setprecision(6) << std::setw(12) << t_nchw << " |"
                            << std::setw(12) << std::setw(12) << g_nchw << " |"
                            << std::setw(12) << arith_intensity << std::endl;
                    std::cout << std::left << std::setw(6) << r << " |"
                            << std::setw(4) << n << " |"
                            << std::setw(4) << c << " |"
                            << std::setw(4) << h << " |"
                            << std::setw(12) << "cudnn_nhwc" << " |"
                            << std::fixed << std::setprecision(6) << std::setw(12) << t_nhwc << " |"
                            << std::setw(12) << std::setw(12) << g_nhwc << " |"
                            << std::setw(12) << arith_intensity << std::endl;

                    if (ws) cudaFree(ws);
                    if (ws_n) cudaFree(ws_n);
                    cudnnDestroyTensorDescriptor(itd); cudnnDestroyTensorDescriptor(otd);
                    cudnnDestroyTensorDescriptor(btd); cudnnDestroyFilterDescriptor(ftd);
                    cudnnDestroyConvolutionDescriptor(cd);
                    cudnnDestroyTensorDescriptor(itd_n); cudnnDestroyTensorDescriptor(otd_n);
                    cudnnDestroyFilterDescriptor(ftd_n);
                    cudaFree(in_d); cudaFree(in_nhwc_d); cudaFree(wt_d); cudaFree(bias_d); cudaFree(out_d);
                    cudaFreeHost(input_h); cudaFreeHost(nhwc_h); cudaFreeHost(weight_h);
                    cudaFreeHost(bias_h); cudaFreeHost(output_h);
            }
        }
        }
    }
    csv.close();
    cudnnDestroy(cudnn);

    std::cout << std::endl
              << "[SUCCESS] benchmark finished" << std::endl;
    return 0;
}
