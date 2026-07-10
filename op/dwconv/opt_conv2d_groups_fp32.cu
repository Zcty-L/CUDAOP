#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cudnn.h>

#include "config.h"
#include "cpu/cpu_ops.h"
#include "ptx_utils.cuh"

constexpr int kConv2dBlockSize = 256;
constexpr int kConv2dChannelGroup = 4;
constexpr int kConv2dWeightSmemFloats = kConv2dChannelGroup * 128;
constexpr int kConv2dMaxInputTileH = 22;
constexpr int kConv2dMaxInputTileW = 85;
constexpr int kConv2dMaxInputTileFloats =
    kConv2dChannelGroup * kConv2dMaxInputTileH * kConv2dMaxInputTileW;
constexpr bool kUseSharedTileByDefault = false;
constexpr bool kUseInputPrefetch = false;
constexpr bool kUseInputDoubleBufferByDefault = false;
constexpr bool kUsePair2ByDefault = false;
constexpr bool kUseChannel8ByDefault = false;
constexpr int kConv2dChannelGroup8 = 8;
constexpr int kConv2dWeightSmemStride8 = 64;

#define CUDA_CHECK(call)                                                   \
    do                                                                     \
    {                                                                      \
        cudaError_t error = (call);                                        \
        if (error != cudaSuccess)                                          \
        {                                                                  \
            std::cout << "[ERROR] CUDA: " << cudaGetErrorString(error)     \
                      << std::endl;                                        \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

#define CUDNN_CHECK(call)                                                  \
    do                                                                     \
    {                                                                      \
        cudnnStatus_t status = (call);                                     \
        if (status != CUDNN_STATUS_SUCCESS)                                \
        {                                                                  \
            std::cout << "[ERROR] cuDNN: "                                 \
                      << cudnnGetErrorString(status) << std::endl;          \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

__global__ void
conv2d_4x128x256_groups_kernel(
    float *inputs,
    float *weights,
    float *bias,
    float *outputs,
    Conv2DParam param)
{
    __shared__ __align__(2 * 1024)
    __shared__ char smem[4 * 128 * 4 + 4 * 4];
    auto *smemweight = reinterpret_cast<float *>(smem);

    int posh_ori =
        ((blockIdx.x * 256 + threadIdx.x) / param.out_w) * param.Sh
        - param.Ph;
    int posw_ori =
        ((blockIdx.x * 256 + threadIdx.x) % param.out_w) * param.Sw
        - param.Pw;

    uint32_t weights_sts_addr =
        ptx::smem_u32addr(smemweight + threadIdx.x * 2);
    uint32_t weights_lds_addr = ptx::smem_u32addr(smemweight);

    const char *weight_ldg_ptr = reinterpret_cast<const char *>(
        weights + blockIdx.y * 4 * param.KhKw
        + threadIdx.x / 64 * param.KhKw
        + threadIdx.x % 64 * 2);
    auto *input_ptr =
        inputs + blockIdx.z * param.inBatchNumel
        + blockIdx.y * 4 * param.inHW;

    float weight_ldg_reg[2];
    float weight_frag[16];
    float input_frag[4][4];
    float output_frag[4];

#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        output_frag[i] = 0;
    }

    ptx::ldg_nc_0(
        weight_ldg_reg[0],
        weight_ldg_ptr,
        threadIdx.x % 64 * 2 < param.KhKw);
    ptx::ldg_nc_0(
        weight_ldg_reg[1],
        weight_ldg_ptr + sizeof(float),
        threadIdx.x % 64 * 2 + 1 < param.KhKw);

    ptx::sts64(
        weight_ldg_reg[0],
        weight_ldg_reg[1],
        weights_sts_addr);
    __syncthreads();

    for (int k = 0; k < param.KhKw; k += 4)
    {
        if (kUseInputPrefetch)
        {
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                int next_tap = k + 4 + i;
                int next_h = posh_ori + next_tap / param.Kw;
                int next_w = posw_ori + next_tap % param.Kw;
                int next_offset = next_h * param.in_w + next_w;
                bool next_valid = next_tap < static_cast<int>(param.KhKw)
                    && next_h >= 0 && next_w >= 0
                    && next_w < static_cast<int>(param.in_w)
                    && next_h < static_cast<int>(param.in_h);
                const float *prefetch_ptr = next_valid
                    ? input_ptr + next_offset
                    : input_ptr;

                ptx::prefetch_global_l1(
                    prefetch_ptr,
                    next_valid);
                ptx::prefetch_global_l1(
                    prefetch_ptr + param.inHW,
                    next_valid);
                ptx::prefetch_global_l1(
                    prefetch_ptr + param.inHW * 2,
                    next_valid);
                ptx::prefetch_global_l1(
                    prefetch_ptr + param.inHW * 3,
                    next_valid);
            }
        }

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int curH = posh_ori + (k + i) / param.Kw;
            int curW = posw_ori + (k + i) % param.Kw;
            int inOffsetTmp = curH * param.in_w + curW;
            if (curH >= 0 && curW >= 0
                && curW < param.in_w && curH < param.in_h)
            {
                input_frag[0][i] = input_ptr[inOffsetTmp];
                input_frag[1][i] =
                    input_ptr[inOffsetTmp + param.inHW];
                input_frag[2][i] =
                    input_ptr[inOffsetTmp + param.inHW * 2];
                input_frag[3][i] =
                    input_ptr[inOffsetTmp + param.inHW * 3];
            }
            else
            {
                input_frag[0][i] = 0;
                input_frag[1][i] = 0;
                input_frag[2][i] = 0;
                input_frag[3][i] = 0;
            }
        }

        ptx::lds128(
            weight_frag[0],
            weight_frag[1],
            weight_frag[2],
            weight_frag[3],
            weights_lds_addr);
        ptx::lds128(
            weight_frag[4],
            weight_frag[5],
            weight_frag[6],
            weight_frag[7],
            weights_lds_addr + 128 * sizeof(float));
        ptx::lds128(
            weight_frag[8],
            weight_frag[9],
            weight_frag[10],
            weight_frag[11],
            weights_lds_addr + 256 * sizeof(float));
        ptx::lds128(
            weight_frag[12],
            weight_frag[13],
            weight_frag[14],
            weight_frag[15],
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
    {
        smembias[threadIdx.x] = bias[blockIdx.y * 4 + threadIdx.x];
    }
    __syncthreads();

    output_frag[0] += smembias[0];
    output_frag[1] += smembias[1];
    output_frag[2] += smembias[2];
    output_frag[3] += smembias[3];

    int outOffset =
        blockIdx.z * param.outBatchNumel
        + blockIdx.y * 4 * param.outHW
        + blockIdx.x * 256 + threadIdx.x;
    if (blockIdx.x * 256 + threadIdx.x < param.outHW)
    {
        outputs[outOffset] = output_frag[0];
        outputs[outOffset + param.outHW] = output_frag[1];
        outputs[outOffset + param.outHW * 2] = output_frag[2];
        outputs[outOffset + param.outHW * 3] = output_frag[3];
    }
}

__global__ void
conv2d_8x128x256_groups_kernel(
    float *inputs,
    float *weights,
    float *bias,
    float *outputs,
    Conv2DParam param)
{
    __shared__ __align__(2 * 1024)
    __shared__ char smem[
        kConv2dChannelGroup8 * kConv2dWeightSmemStride8 * sizeof(float)];
    auto *smemweight = reinterpret_cast<float *>(smem);

    int out_index = blockIdx.x * kConv2dBlockSize + threadIdx.x;
    int posh_ori = (out_index / param.out_w) * param.Sh - param.Ph;
    int posw_ori = (out_index % param.out_w) * param.Sw - param.Pw;

    int load_channel = threadIdx.x / 32;
    int load_tap = (threadIdx.x % 32) * 2;
    const char *weight_ldg_ptr = reinterpret_cast<const char *>(
        weights + blockIdx.y * kConv2dChannelGroup8 * param.KhKw
        + load_channel * param.KhKw + load_tap);
    uint32_t weights_sts_addr =
        ptx::smem_u32addr(smemweight + threadIdx.x * 2);

    auto *input_ptr =
        inputs + blockIdx.z * param.inBatchNumel
        + blockIdx.y * kConv2dChannelGroup8 * param.inHW;

    float weight_ldg_reg[2];
    float weight_frag[kConv2dChannelGroup8 * 4];
    float input_frag[kConv2dChannelGroup8][4];
    float output_frag[kConv2dChannelGroup8];

#pragma unroll
    for (int channel = 0; channel < kConv2dChannelGroup8; ++channel)
    {
        output_frag[channel] = 0.0f;
    }

    ptx::ldg_nc_0(
        weight_ldg_reg[0],
        weight_ldg_ptr,
        load_tap < param.KhKw);
    ptx::ldg_nc_0(
        weight_ldg_reg[1],
        weight_ldg_ptr + sizeof(float),
        load_tap + 1 < param.KhKw);

    ptx::sts64(
        weight_ldg_reg[0],
        weight_ldg_reg[1],
        weights_sts_addr);
    __syncthreads();

    uint32_t weights_lds_addr = ptx::smem_u32addr(smemweight);

    for (int k = 0; k < static_cast<int>(param.KhKw); k += 4)
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int tap = k + i;
            int curH = posh_ori + tap / param.Kw;
            int curW = posw_ori + tap % param.Kw;
            int inOffsetTmp = curH * param.in_w + curW;
            bool valid_input = tap < static_cast<int>(param.KhKw)
                && curH >= 0 && curW >= 0
                && curW < static_cast<int>(param.in_w)
                && curH < static_cast<int>(param.in_h);

#pragma unroll
            for (int channel = 0; channel < kConv2dChannelGroup8; ++channel)
            {
                input_frag[channel][i] = valid_input
                    ? input_ptr[channel * param.inHW + inOffsetTmp]
                    : 0.0f;
            }
        }

#pragma unroll
        for (int channel = 0; channel < kConv2dChannelGroup8; ++channel)
        {
            ptx::lds128(
                weight_frag[channel * 4],
                weight_frag[channel * 4 + 1],
                weight_frag[channel * 4 + 2],
                weight_frag[channel * 4 + 3],
                weights_lds_addr
                    + channel * kConv2dWeightSmemStride8 * sizeof(float));
        }

        weights_lds_addr += 4 * sizeof(float);

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
#pragma unroll
            for (int channel = 0; channel < kConv2dChannelGroup8; ++channel)
            {
                output_frag[channel] +=
                    weight_frag[channel * 4 + i] * input_frag[channel][i];
            }
        }
    }

#pragma unroll
    for (int channel = 0; channel < kConv2dChannelGroup8; ++channel)
    {
        output_frag[channel] +=
            bias[blockIdx.y * kConv2dChannelGroup8 + channel];
    }

    int outOffset =
        blockIdx.z * param.outBatchNumel
        + blockIdx.y * kConv2dChannelGroup8 * param.outHW
        + out_index;
    if (out_index < static_cast<int>(param.outHW))
    {
#pragma unroll
        for (int channel = 0; channel < kConv2dChannelGroup8; ++channel)
        {
            outputs[outOffset + channel * param.outHW] =
                output_frag[channel];
        }
    }
}

__device__ __forceinline__ void
load_input_group_4x4(
    float input_frag[kConv2dChannelGroup][4],
    const float *input_ptr,
    int posh_ori,
    int posw_ori,
    int k,
    Conv2DParam param)
{
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        int tap = k + i;
        int curH = posh_ori + tap / param.Kw;
        int curW = posw_ori + tap % param.Kw;
        int inOffsetTmp = curH * param.in_w + curW;
        if (tap < static_cast<int>(param.KhKw)
            && curH >= 0 && curW >= 0
            && curW < static_cast<int>(param.in_w)
            && curH < static_cast<int>(param.in_h))
        {
            input_frag[0][i] = input_ptr[inOffsetTmp];
            input_frag[1][i] = input_ptr[inOffsetTmp + param.inHW];
            input_frag[2][i] = input_ptr[inOffsetTmp + param.inHW * 2];
            input_frag[3][i] = input_ptr[inOffsetTmp + param.inHW * 3];
        }
        else
        {
            input_frag[0][i] = 0.0f;
            input_frag[1][i] = 0.0f;
            input_frag[2][i] = 0.0f;
            input_frag[3][i] = 0.0f;
        }
    }
}

__global__ void
conv2d_4x128x256_groups_input_double_buffer_kernel(
    float *inputs,
    float *weights,
    float *bias,
    float *outputs,
    Conv2DParam param)
{
    __shared__ __align__(2 * 1024)
    __shared__ char smem[4 * 128 * 4 + 4 * 4];
    auto *smemweight = reinterpret_cast<float *>(smem);

    int out_index = blockIdx.x * kConv2dBlockSize + threadIdx.x;
    int posh_ori = (out_index / param.out_w) * param.Sh - param.Ph;
    int posw_ori = (out_index % param.out_w) * param.Sw - param.Pw;

    uint32_t weights_sts_addr =
        ptx::smem_u32addr(smemweight + threadIdx.x * 2);
    uint32_t weights_lds_addr = ptx::smem_u32addr(smemweight);

    const char *weight_ldg_ptr = reinterpret_cast<const char *>(
        weights + blockIdx.y * kConv2dChannelGroup * param.KhKw
        + threadIdx.x / 64 * param.KhKw
        + threadIdx.x % 64 * 2);
    auto *input_ptr =
        inputs + blockIdx.z * param.inBatchNumel
        + blockIdx.y * kConv2dChannelGroup * param.inHW;

    float weight_ldg_reg[2];
    float weight_frag[16];
    float input_frag[2][kConv2dChannelGroup][4];
    float output_frag[kConv2dChannelGroup];

#pragma unroll
    for (int channel = 0; channel < kConv2dChannelGroup; ++channel)
    {
        output_frag[channel] = 0.0f;
    }

    ptx::ldg_nc_0(
        weight_ldg_reg[0],
        weight_ldg_ptr,
        threadIdx.x % 64 * 2 < param.KhKw);
    ptx::ldg_nc_0(
        weight_ldg_reg[1],
        weight_ldg_ptr + sizeof(float),
        threadIdx.x % 64 * 2 + 1 < param.KhKw);

    ptx::sts64(
        weight_ldg_reg[0],
        weight_ldg_reg[1],
        weights_sts_addr);
    __syncthreads();

    int current_buffer = 0;
    load_input_group_4x4(
        input_frag[current_buffer],
        input_ptr,
        posh_ori,
        posw_ori,
        0,
        param);

    for (int k = 0; k < static_cast<int>(param.KhKw); k += 4)
    {
        ptx::lds128(
            weight_frag[0],
            weight_frag[1],
            weight_frag[2],
            weight_frag[3],
            weights_lds_addr);
        ptx::lds128(
            weight_frag[4],
            weight_frag[5],
            weight_frag[6],
            weight_frag[7],
            weights_lds_addr + 128 * sizeof(float));
        ptx::lds128(
            weight_frag[8],
            weight_frag[9],
            weight_frag[10],
            weight_frag[11],
            weights_lds_addr + 256 * sizeof(float));
        ptx::lds128(
            weight_frag[12],
            weight_frag[13],
            weight_frag[14],
            weight_frag[15],
            weights_lds_addr + 384 * sizeof(float));
        __syncthreads();

        int next_k = k + 4;
        int next_buffer = current_buffer ^ 1;
        if (next_k < static_cast<int>(param.KhKw))
        {
            load_input_group_4x4(
                input_frag[next_buffer],
                input_ptr,
                posh_ori,
                posw_ori,
                next_k,
                param);
        }

        weights_lds_addr += 4 * sizeof(float);

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            output_frag[0] +=
                weight_frag[i] * input_frag[current_buffer][0][i];
            output_frag[1] +=
                weight_frag[i + 4] * input_frag[current_buffer][1][i];
            output_frag[2] +=
                weight_frag[i + 8] * input_frag[current_buffer][2][i];
            output_frag[3] +=
                weight_frag[i + 12] * input_frag[current_buffer][3][i];
        }

        current_buffer = next_buffer;
    }

    auto *smembias = reinterpret_cast<float *>(smem);
    if (threadIdx.x < kConv2dChannelGroup)
    {
        smembias[threadIdx.x] =
            bias[blockIdx.y * kConv2dChannelGroup + threadIdx.x];
    }
    __syncthreads();

    output_frag[0] += smembias[0];
    output_frag[1] += smembias[1];
    output_frag[2] += smembias[2];
    output_frag[3] += smembias[3];

    int outOffset =
        blockIdx.z * param.outBatchNumel
        + blockIdx.y * kConv2dChannelGroup * param.outHW
        + out_index;
    if (out_index < static_cast<int>(param.outHW))
    {
        outputs[outOffset] = output_frag[0];
        outputs[outOffset + param.outHW] = output_frag[1];
        outputs[outOffset + param.outHW * 2] = output_frag[2];
        outputs[outOffset + param.outHW * 3] = output_frag[3];
    }
}

static bool supports_shared_tile(const Conv2DParam &param)
{
    if (param.Kh > 7 || param.Kw > 7)
    {
        return false;
    }

    for (uint32_t block_x = 0;
         block_x < (param.outHW + kConv2dBlockSize - 1) / kConv2dBlockSize;
         ++block_x)
    {
        uint32_t out_start = block_x * kConv2dBlockSize;
        uint32_t out_end =
            std::min(out_start + kConv2dBlockSize - 1, param.outHW - 1);
        uint32_t row_start = out_start / param.out_w;
        uint32_t row_end = out_end / param.out_w;
        uint32_t col_min =
            row_start == row_end ? out_start % param.out_w : 0;
        uint32_t col_max =
            row_start == row_end ? out_end % param.out_w : param.out_w - 1;
        uint32_t tile_h = (row_end - row_start) * param.Sh + param.Kh;
        uint32_t tile_w = (col_max - col_min) * param.Sw + param.Kw;

        if (tile_h > kConv2dMaxInputTileH
            || tile_w > kConv2dMaxInputTileW)
        {
            return false;
        }
    }
    return true;
}

__global__ void
conv2d_4x128x256_groups_shared_tile_kernel(
    float *inputs,
    float *weights,
    float *bias,
    float *outputs,
    Conv2DParam param)
{
    __shared__ __align__(16) char smem[
        (kConv2dWeightSmemFloats + kConv2dMaxInputTileFloats
         + kConv2dChannelGroup) * sizeof(float)];
    auto *smemweight = reinterpret_cast<float *>(smem);
    auto *smeminput = smemweight + kConv2dWeightSmemFloats;

    int out_start = blockIdx.x * kConv2dBlockSize;
    int out_last = min(
        out_start + kConv2dBlockSize - 1,
        static_cast<int>(param.outHW) - 1);
    int out_index = out_start + threadIdx.x;
    int row_start = out_start / param.out_w;
    int row_end = out_last / param.out_w;
    int col_min = row_start == row_end ? out_start % param.out_w : 0;
    int col_max = row_start == row_end
        ? out_last % param.out_w
        : static_cast<int>(param.out_w) - 1;
    int tile_h = (row_end - row_start) * param.Sh + param.Kh;
    int tile_w = (col_max - col_min) * param.Sw + param.Kw;
    int tile_origin_h = row_start * param.Sh - param.Ph;
    int tile_origin_w = col_min * param.Sw - param.Pw;
    int tile_area = tile_h * tile_w;

    uint32_t weights_sts_addr =
        ptx::smem_u32addr(smemweight + threadIdx.x * 2);
    uint32_t weights_lds_addr = ptx::smem_u32addr(smemweight);

    const char *weight_ldg_ptr = reinterpret_cast<const char *>(
        weights + blockIdx.y * kConv2dChannelGroup * param.KhKw
        + threadIdx.x / 64 * param.KhKw
        + threadIdx.x % 64 * 2);
    auto *input_ptr =
        inputs + blockIdx.z * param.inBatchNumel
        + blockIdx.y * kConv2dChannelGroup * param.inHW;

    float weight_ldg_reg[2];
    ptx::ldg_nc_0(
        weight_ldg_reg[0],
        weight_ldg_ptr,
        threadIdx.x % 64 * 2 < param.KhKw);
    ptx::ldg_nc_0(
        weight_ldg_reg[1],
        weight_ldg_ptr + sizeof(float),
        threadIdx.x % 64 * 2 + 1 < param.KhKw);

    ptx::sts64(
        weight_ldg_reg[0],
        weight_ldg_reg[1],
        weights_sts_addr);

    int input_tile_floats = kConv2dChannelGroup * tile_area;
    for (int index = threadIdx.x;
         index < input_tile_floats;
         index += blockDim.x)
    {
        int channel = index / tile_area;
        int offset = index - channel * tile_area;
        int tile_h_index = offset / tile_w;
        int tile_w_index = offset - tile_h_index * tile_w;
        int input_h = tile_origin_h + tile_h_index;
        int input_w = tile_origin_w + tile_w_index;

        float value = 0.0f;
        if (input_h >= 0 && input_w >= 0
            && input_h < static_cast<int>(param.in_h)
            && input_w < static_cast<int>(param.in_w))
        {
            value = input_ptr[
                channel * param.inHW + input_h * param.in_w + input_w];
        }
        smeminput[index] = value;
    }
    __syncthreads();

    float weight_frag[16];
    float input_frag[kConv2dChannelGroup][4];
    float output_frag[kConv2dChannelGroup];

#pragma unroll
    for (int i = 0; i < kConv2dChannelGroup; ++i)
    {
        output_frag[i] = 0.0f;
    }

    bool valid_output = out_index < static_cast<int>(param.outHW);
    int out_row = valid_output ? out_index / param.out_w : row_start;
    int out_col = valid_output ? out_index % param.out_w : col_min;
    int local_out_row = out_row - row_start;
    int local_out_col = out_col - col_min;

    for (int k = 0; k < static_cast<int>(param.KhKw); k += 4)
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int tap = k + i;
            int tap_h = tap / param.Kw;
            int tap_w = tap - tap_h * param.Kw;
            int tile_h_index = local_out_row * param.Sh + tap_h;
            int tile_w_index = local_out_col * param.Sw + tap_w;
            int input_offset = tile_h_index * tile_w + tile_w_index;
            bool valid_tap = valid_output && tap < static_cast<int>(param.KhKw);

#pragma unroll
            for (int channel = 0; channel < kConv2dChannelGroup; ++channel)
            {
                input_frag[channel][i] = valid_tap
                    ? smeminput[channel * tile_area + input_offset]
                    : 0.0f;
            }
        }

        ptx::lds128(
            weight_frag[0],
            weight_frag[1],
            weight_frag[2],
            weight_frag[3],
            weights_lds_addr);
        ptx::lds128(
            weight_frag[4],
            weight_frag[5],
            weight_frag[6],
            weight_frag[7],
            weights_lds_addr + 128 * sizeof(float));
        ptx::lds128(
            weight_frag[8],
            weight_frag[9],
            weight_frag[10],
            weight_frag[11],
            weights_lds_addr + 256 * sizeof(float));
        ptx::lds128(
            weight_frag[12],
            weight_frag[13],
            weight_frag[14],
            weight_frag[15],
            weights_lds_addr + 384 * sizeof(float));

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
    __syncthreads();
    if (threadIdx.x < kConv2dChannelGroup)
    {
        smembias[threadIdx.x] =
            bias[blockIdx.y * kConv2dChannelGroup + threadIdx.x];
    }
    __syncthreads();

    output_frag[0] += smembias[0];
    output_frag[1] += smembias[1];
    output_frag[2] += smembias[2];
    output_frag[3] += smembias[3];

    int outOffset =
        blockIdx.z * param.outBatchNumel
        + blockIdx.y * kConv2dChannelGroup * param.outHW
        + out_index;
    if (valid_output)
    {
        outputs[outOffset] = output_frag[0];
        outputs[outOffset + param.outHW] = output_frag[1];
        outputs[outOffset + param.outHW * 2] = output_frag[2];
        outputs[outOffset + param.outHW * 3] = output_frag[3];
    }
}

__global__ void
conv2d_4x128x256_groups_pair2_kernel(
    float *inputs,
    float *weights,
    float *bias,
    float *outputs,
    Conv2DParam param)
{
    __shared__ __align__(2 * 1024)
    __shared__ char smem[4 * 128 * 4];
    auto *smemweight = reinterpret_cast<float *>(smem);

    uint32_t weights_sts_addr =
        ptx::smem_u32addr(smemweight + threadIdx.x * 2);

    const char *weight_ldg_ptr = reinterpret_cast<const char *>(
        weights + blockIdx.y * kConv2dChannelGroup * param.KhKw
        + threadIdx.x / 64 * param.KhKw
        + threadIdx.x % 64 * 2);
    auto *input_ptr =
        inputs + blockIdx.z * param.inBatchNumel
        + blockIdx.y * kConv2dChannelGroup * param.inHW;

    float weight_ldg_reg[2];
    ptx::ldg_nc_0(
        weight_ldg_reg[0],
        weight_ldg_ptr,
        threadIdx.x % 64 * 2 < param.KhKw);
    ptx::ldg_nc_0(
        weight_ldg_reg[1],
        weight_ldg_ptr + sizeof(float),
        threadIdx.x % 64 * 2 + 1 < param.KhKw);

    ptx::sts64(
        weight_ldg_reg[0],
        weight_ldg_reg[1],
        weights_sts_addr);
    __syncthreads();

    int out0_index = blockIdx.x * kConv2dBlockSize * 2 + threadIdx.x * 2;
    int out1_index = out0_index + 1;
    bool valid0 = out0_index < static_cast<int>(param.outHW);
    bool valid1 = out1_index < static_cast<int>(param.outHW);

    int out0_h = valid0 ? out0_index / param.out_w : 0;
    int out0_w = valid0 ? out0_index % param.out_w : 0;
    int out1_h = valid1 ? out1_index / param.out_w : 0;
    int out1_w = valid1 ? out1_index % param.out_w : 0;
    int pos0_h = out0_h * param.Sh - param.Ph;
    int pos0_w = out0_w * param.Sw - param.Pw;
    int pos1_h = out1_h * param.Sh - param.Ph;
    int pos1_w = out1_w * param.Sw - param.Pw;
    bool same_row_pair = valid0 && valid1 && out0_h == out1_h
        && param.Kw <= 7 && param.Sw <= 2;

    float output0[kConv2dChannelGroup];
    float output1[kConv2dChannelGroup];

#pragma unroll
    for (int channel = 0; channel < kConv2dChannelGroup; ++channel)
    {
        output0[channel] = 0.0f;
        output1[channel] = 0.0f;
    }

    if (same_row_pair)
    {
        for (int kh = 0; kh < static_cast<int>(param.Kh); ++kh)
        {
            int input_h = pos0_h + kh;
            bool row_valid = input_h >= 0
                && input_h < static_cast<int>(param.in_h);

            for (int u = 0;
                 u < static_cast<int>(param.Kw + param.Sw);
                 ++u)
            {
                int input_w = pos0_w + u;
                bool valid = row_valid
                    && input_w >= 0
                    && input_w < static_cast<int>(param.in_w);
                int input_offset = input_h * param.in_w + input_w;
                bool contributes0 = u < static_cast<int>(param.Kw);
                int kw1 = u - param.Sw;
                bool contributes1 = kw1 >= 0
                    && kw1 < static_cast<int>(param.Kw);

#pragma unroll
                for (int channel = 0;
                     channel < kConv2dChannelGroup;
                     ++channel)
                {
                    float value = valid
                        ? input_ptr[channel * param.inHW + input_offset]
                        : 0.0f;
                    if (contributes0)
                    {
                        int tap0 = kh * param.Kw + u;
                        float weight0 = smemweight[channel * 128 + tap0];
                        output0[channel] += weight0 * value;
                    }
                    if (contributes1)
                    {
                        int tap1 = kh * param.Kw + kw1;
                        float weight1 = smemweight[channel * 128 + tap1];
                        output1[channel] += weight1 * value;
                    }
                }
            }
        }
    }
    else
    {
        for (int kh = 0; kh < static_cast<int>(param.Kh); ++kh)
        {
            int input0_h = pos0_h + kh;
            int input1_h = pos1_h + kh;

            for (int kw = 0; kw < static_cast<int>(param.Kw); ++kw)
            {
                int tap = kh * param.Kw + kw;
                int input0_w = pos0_w + kw;
                int input1_w = pos1_w + kw;
                bool input0_valid = valid0
                    && input0_h >= 0
                    && input0_h < static_cast<int>(param.in_h)
                    && input0_w >= 0
                    && input0_w < static_cast<int>(param.in_w);
                bool input1_valid = valid1
                    && input1_h >= 0
                    && input1_h < static_cast<int>(param.in_h)
                    && input1_w >= 0
                    && input1_w < static_cast<int>(param.in_w);
                int input0_offset = input0_h * param.in_w + input0_w;
                int input1_offset = input1_h * param.in_w + input1_w;

#pragma unroll
                for (int channel = 0;
                     channel < kConv2dChannelGroup;
                     ++channel)
                {
                    float weight_value = smemweight[channel * 128 + tap];
                    float value0 = input0_valid
                        ? input_ptr[channel * param.inHW + input0_offset]
                        : 0.0f;
                    float value1 = input1_valid
                        ? input_ptr[channel * param.inHW + input1_offset]
                        : 0.0f;
                    output0[channel] += weight_value * value0;
                    output1[channel] += weight_value * value1;
                }
            }
        }
    }

#pragma unroll
    for (int channel = 0; channel < kConv2dChannelGroup; ++channel)
    {
        float bias_value =
            bias[blockIdx.y * kConv2dChannelGroup + channel];
        output0[channel] += bias_value;
        output1[channel] += bias_value;
    }

    int out0_offset =
        blockIdx.z * param.outBatchNumel
        + blockIdx.y * kConv2dChannelGroup * param.outHW
        + out0_index;
    int out1_offset = out0_offset + 1;
    if (valid0)
    {
        outputs[out0_offset] = output0[0];
        outputs[out0_offset + param.outHW] = output0[1];
        outputs[out0_offset + param.outHW * 2] = output0[2];
        outputs[out0_offset + param.outHW * 3] = output0[3];
    }
    if (valid1)
    {
        outputs[out1_offset] = output1[0];
        outputs[out1_offset + param.outHW] = output1[1];
        outputs[out1_offset + param.outHW * 2] = output1[2];
        outputs[out1_offset + param.outHW * 3] = output1[3];
    }
}

static void launch_target(
    float *inputs,
    float *weights,
    float *bias,
    float *outputs,
    const Conv2DParam &param,
    int n)
{
    dim3 block(kConv2dBlockSize);

    if (kUseChannel8ByDefault
        && param.out_ch % kConv2dChannelGroup8 == 0
        && param.KhKw <= kConv2dWeightSmemStride8)
    {
        dim3 grid(
            (param.outHW + kConv2dBlockSize - 1) / kConv2dBlockSize,
            param.out_ch / kConv2dChannelGroup8,
            n);
        conv2d_8x128x256_groups_kernel<<<grid, block>>>(
            inputs,
            weights,
            bias,
            outputs,
            param);
    }
    else if (kUseInputDoubleBufferByDefault)
    {
        dim3 grid(
            (param.outHW + kConv2dBlockSize - 1) / kConv2dBlockSize,
            param.out_ch / kConv2dChannelGroup,
            n);
        conv2d_4x128x256_groups_input_double_buffer_kernel<<<grid, block>>>(
            inputs,
            weights,
            bias,
            outputs,
            param);
    }
    else if (kUsePair2ByDefault && param.Kh <= 7 && param.Kw <= 7)
    {
        dim3 grid(
            (param.outHW + kConv2dBlockSize * 2 - 1)
                / (kConv2dBlockSize * 2),
            param.out_ch / kConv2dChannelGroup,
            n);
        conv2d_4x128x256_groups_pair2_kernel<<<grid, block>>>(
            inputs,
            weights,
            bias,
            outputs,
            param);
    }
    else if (kUseSharedTileByDefault && supports_shared_tile(param))
    {
        dim3 grid(
            (param.outHW + kConv2dBlockSize - 1) / kConv2dBlockSize,
            param.out_ch / kConv2dChannelGroup,
            n);
        conv2d_4x128x256_groups_shared_tile_kernel<<<grid, block>>>(
            inputs,
            weights,
            bias,
            outputs,
            param);
    }
    else
    {
        dim3 grid(
            (param.outHW + kConv2dBlockSize - 1) / kConv2dBlockSize,
            param.out_ch / kConv2dChannelGroup,
            n);
        conv2d_4x128x256_groups_kernel<<<grid, block>>>(
            inputs,
            weights,
            bias,
            outputs,
            param);
    }
}

struct CaseConfig
{
    const char *name;
    int r;
    int n;
    int c;
    int h;
    int stride;
};

struct Stats
{
    float mean;
    float median;
    float minimum;
    float maximum;
    float stddev;
};

static Stats calculate_stats(std::vector<float> values)
{
    std::sort(values.begin(), values.end());
    double sum = 0.0;
    for (float value : values)
    {
        sum += value;
    }

    double mean = sum / values.size();
    double variance = 0.0;
    for (float value : values)
    {
        double delta = value - mean;
        variance += delta * delta;
    }
    variance /= values.size();

    return {
        static_cast<float>(mean),
        values[values.size() / 2],
        values.front(),
        values.back(),
        static_cast<float>(std::sqrt(variance))
    };
}

template <typename Launch>
static void preheat(
    Launch launch,
    int launches_per_batch,
    int duration_ms)
{
    auto start = std::chrono::steady_clock::now();
    auto duration = std::chrono::milliseconds(duration_ms);

    do
    {
        for (int i = 0; i < launches_per_batch; ++i)
        {
            launch();
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } while (std::chrono::steady_clock::now() - start < duration);
}

template <typename Launch>
static float measure_elapsed_ms(
    Launch launch,
    int launches)
{
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < launches; ++i)
    {
        launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed_ms;
}

template <typename Launch>
static int choose_launches_per_sample(
    Launch launch,
    float target_sample_ms,
    int min_launches,
    int max_launches)
{
    int launches = min_launches;
    while (launches < max_launches)
    {
        float elapsed_ms = measure_elapsed_ms(launch, launches);
        if (elapsed_ms >= target_sample_ms)
        {
            break;
        }
        launches = std::min(launches * 2, max_launches);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return launches;
}

template <typename Launch>
static Stats measure(
    Launch launch,
    int warmup,
    int iterations,
    int launches_per_sample)
{
    for (int i = 0; i < warmup; ++i)
    {
        for (int launch_index = 0;
             launch_index < launches_per_sample;
             ++launch_index)
        {
            launch();
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> samples;
    samples.reserve(iterations);
    for (int i = 0; i < iterations; ++i)
    {
        CUDA_CHECK(cudaEventRecord(start));
        for (int launch_index = 0;
             launch_index < launches_per_sample;
             ++launch_index)
        {
            launch();
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        samples.push_back(elapsed_ms / launches_per_sample);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return calculate_stats(samples);
}

template <typename Launch>
static void benchmark_implementation(
    const char *implementation,
    Launch launch,
    int groups,
    int warmup,
    int iterations,
    int preheat_ms)
{
    constexpr int preheat_launches_per_batch = 100;
    constexpr float target_sample_ms = 5.0f;
    constexpr int min_launches_per_sample = 16;
    constexpr int max_launches_per_sample = 4096;

    preheat(launch, preheat_launches_per_batch, preheat_ms);
    int launches_per_sample = choose_launches_per_sample(
        launch,
        target_sample_ms,
        min_launches_per_sample,
        max_launches_per_sample);

    std::cout << "[BENCHMARK] " << implementation
              << " preheat_ms=" << preheat_ms
              << " warmup_batches=" << warmup
              << " iterations=" << iterations
              << " groups=" << groups
              << " launches_per_sample=" << launches_per_sample
              << std::endl;
    for (int group = 1; group <= groups; ++group)
    {
        print_stats(implementation, group, measure(
            launch,
            warmup,
            iterations,
            launches_per_sample));
    }
}

static void print_stats(
    const char *implementation,
    int group,
    const Stats &stats)
{
    std::cout << "  " << std::left << std::setw(12) << implementation
              << " group=" << group
              << " mean=" << std::fixed << std::setprecision(6)
              << stats.mean
              << " median=" << stats.median
              << " min=" << stats.minimum
              << " max=" << stats.maximum
              << " stddev=" << stats.stddev
              << " ms" << std::endl;
}

static bool verify(
    const std::vector<float> &actual,
    const std::vector<float> &expected,
    float atol,
    float rtol)
{
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    size_t error_count = 0;

    for (size_t i = 0; i < actual.size(); ++i)
    {
        float abs_error = std::abs(actual[i] - expected[i]);
        float denominator = std::max(std::abs(expected[i]), 1.0e-12f);
        float rel_error = abs_error / denominator;
        max_abs_error = std::max(max_abs_error, abs_error);
        max_rel_error = std::max(max_rel_error, rel_error);
        if (abs_error > atol && rel_error > rtol)
        {
            ++error_count;
        }
    }

    std::cout << "  max_abs_error=" << std::scientific << max_abs_error
              << " max_rel_error=" << max_rel_error
              << " errors=" << error_count << "/" << actual.size()
              << std::defaultfloat << std::endl;
    return error_count == 0;
}

static Conv2DParam make_param(const CaseConfig &config)
{
    int out_h =
        (config.h - config.r + 2 * (config.r / 2)) / config.stride + 1;
    int out_w = out_h;

    Conv2DParam param{};
    param.in_h = config.h;
    param.in_w = config.h;
    param.in_ch = config.c;
    param.inHW = config.h * config.h;
    param.inChKhKw = config.c * config.r * config.r;
    param.inBatchNumel = config.c * config.h * config.h;
    param.out_ch = config.c;
    param.out_h = out_h;
    param.out_w = out_w;
    param.outHW = out_h * out_w;
    param.outBatchNumel = config.c * out_h * out_w;
    param.Kh = config.r;
    param.Kw = config.r;
    param.KhKw = config.r * config.r;
    param.Sh = config.stride;
    param.Sw = config.stride;
    param.Ph = config.r / 2;
    param.Pw = config.r / 2;
    return param;
}

static bool run_case(const CaseConfig &config)
{
    constexpr int warmup = 20;
    constexpr int iterations = 100;
    constexpr int groups = 3;
    constexpr int preheat_ms = 1000;
    constexpr float atol = 1.0e-4f;
    constexpr float rtol = 1.0e-4f;

    Conv2DParam param = make_param(config);
    size_t input_count =
        static_cast<size_t>(config.n) * param.inBatchNumel;
    size_t weight_count =
        static_cast<size_t>(config.c) * param.KhKw;
    size_t output_count =
        static_cast<size_t>(config.n) * param.outBatchNumel;

    std::vector<float> input(input_count);
    std::vector<float> weight(weight_count);
    std::vector<float> bias(config.c);
    std::vector<float> output(output_count);
    std::vector<float> reference(output_count);

    std::mt19937 generator(20260703);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (float &value : input)
    {
        value = distribution(generator);
    }
    for (float &value : weight)
    {
        value = distribution(generator);
    }
    for (float &value : bias)
    {
        value = distribution(generator);
    }

    float *input_device = nullptr;
    float *weight_device = nullptr;
    float *bias_device = nullptr;
    float *output_device = nullptr;
    CUDA_CHECK(cudaMalloc(&input_device, input_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&weight_device, weight_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bias_device, bias.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&output_device, output_count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(
        input_device,
        input.data(),
        input_count * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        weight_device,
        weight.data(),
        weight_count * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        bias_device,
        bias.data(),
        bias.size() * sizeof(float),
        cudaMemcpyHostToDevice));

    bool channel8_enabled = kUseChannel8ByDefault
        && param.out_ch % kConv2dChannelGroup8 == 0
        && param.KhKw <= kConv2dWeightSmemStride8;
    bool double_buffer_enabled = !channel8_enabled
        && kUseInputDoubleBufferByDefault;
    bool pair2_enabled = !channel8_enabled
        && !double_buffer_enabled
        && kUsePair2ByDefault && param.Kh <= 7 && param.Kw <= 7;
    uint32_t output_per_block = pair2_enabled
        ? kConv2dBlockSize * 2
        : kConv2dBlockSize;
    uint32_t channel_group = channel8_enabled
        ? kConv2dChannelGroup8
        : kConv2dChannelGroup;
    uint32_t grid_x =
        (param.outHW + output_per_block - 1) / output_per_block;

    std::cout << "\n[CONFIG] " << config.name
              << " dtype=fp32"
              << " channel8=" << channel8_enabled
              << " double_buffer=" << double_buffer_enabled
              << " pair2=" << pair2_enabled
              << " shared_tile="
              << (kUseSharedTileByDefault && supports_shared_tile(param))
              << " input_prefetch=" << kUseInputPrefetch
              << " N=" << config.n
              << " C=" << config.c
              << " H=W=" << config.h
              << " K=" << config.r
              << " stride=" << config.stride
              << " outHW=" << param.outHW
              << " grid=(" << grid_x
              << "," << param.out_ch / channel_group
              << "," << config.n << ")"
              << " block=256" << std::endl;

    std::cout << "[STAGE] correctness" << std::endl;
    launch_target(
        input_device,
        weight_device,
        bias_device,
        output_device,
        param,
        config.n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(
        output.data(),
        output_device,
        output_count * sizeof(float),
        cudaMemcpyDeviceToHost));

    dwconv2d_cpu(
        input.data(),
        weight.data(),
        bias.data(),
        reference.data(),
        config.n,
        config.c,
        config.h,
        config.h,
        config.r,
        config.r,
        config.stride,
        config.stride,
        config.r / 2,
        config.r / 2);
    bool correct = verify(output, reference, atol, rtol);

    cudnnHandle_t handle;
    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnTensorDescriptor_t bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CHECK(cudnnCreate(&handle));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        input_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        config.n,
        config.c,
        config.h,
        config.h));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        output_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        config.n,
        config.c,
        param.out_h,
        param.out_w));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        bias_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        1,
        config.c,
        1,
        1));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        filter_desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        config.c,
        1,
        config.r,
        config.r));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        conv_desc,
        config.r / 2,
        config.r / 2,
        config.stride,
        config.stride,
        1,
        1,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));
    CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc, config.c));

    cudnnConvolutionFwdAlgo_t algorithm =
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    size_t workspace_size = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        handle,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        algorithm,
        &workspace_size));
    void *workspace = nullptr;
    if (workspace_size > 0)
    {
        CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    }

    float one = 1.0f;
    float zero = 0.0f;
    auto custom_launch = [&]()
    {
        launch_target(
            input_device,
            weight_device,
            bias_device,
            output_device,
            param,
            config.n);
    };
    auto cudnn_launch = [&]()
    {
        CUDNN_CHECK(cudnnConvolutionForward(
            handle,
            &one,
            input_desc,
            input_device,
            filter_desc,
            weight_device,
            conv_desc,
            algorithm,
            workspace,
            workspace_size,
            &zero,
            output_desc,
            output_device));
        CUDNN_CHECK(cudnnAddTensor(
            handle,
            &one,
            bias_desc,
            bias_device,
            &one,
            output_desc,
            output_device));
    };

    std::cout << "[STAGE] performance separate_implementations=1"
              << std::endl;
    benchmark_implementation(
        "custom",
        custom_launch,
        groups,
        warmup,
        iterations,
        preheat_ms);
    benchmark_implementation(
        "cuDNN",
        cudnn_launch,
        groups,
        warmup,
        iterations,
        preheat_ms);

    if (workspace != nullptr)
    {
        CUDA_CHECK(cudaFree(workspace));
    }
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroy(handle));
    CUDA_CHECK(cudaFree(output_device));
    CUDA_CHECK(cudaFree(bias_device));
    CUDA_CHECK(cudaFree(weight_device));
    CUDA_CHECK(cudaFree(input_device));

    std::cout << (correct ? "[SUCCESS]" : "[FAILED]")
              << " " << config.name << std::endl;
    return correct;
}

static int run_profile(int r, int n, int c, int h)
{
    CaseConfig config{"profile", r, n, c, h, 2};
    Conv2DParam param = make_param(config);
    size_t input_count =
        static_cast<size_t>(n) * param.inBatchNumel;
    size_t weight_count =
        static_cast<size_t>(c) * param.KhKw;
    size_t output_count =
        static_cast<size_t>(n) * param.outBatchNumel;

    float *input = nullptr;
    float *weight = nullptr;
    float *bias = nullptr;
    float *output = nullptr;
    CUDA_CHECK(cudaMalloc(&input, input_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&weight, weight_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bias, c * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&output, output_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(input, 0, input_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(weight, 0, weight_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(bias, 0, c * sizeof(float)));

    launch_target(input, weight, bias, output, param, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "[SUCCESS] profile launch" << std::endl;

    CUDA_CHECK(cudaFree(output));
    CUDA_CHECK(cudaFree(bias));
    CUDA_CHECK(cudaFree(weight));
    CUDA_CHECK(cudaFree(input));
    return 0;
}

int main(int argc, char **argv)
{
    if (argc > 1 && std::strcmp(argv[1], "--profile") == 0)
    {
        int r = 7;
        int n = 4;
        int c = 128;
        int h = 80;
        if (argc == 6)
        {
            r = std::atoi(argv[2]);
            n = std::atoi(argv[3]);
            c = std::atoi(argv[4]);
            h = std::atoi(argv[5]);
        }
        return run_profile(r, n, c, h);
    }

    const CaseConfig cases[] = {
        {"throughput", 7, 16, 128, 80, 2},
        {"main", 7, 4, 128, 80, 2},
        {"common", 5, 1, 32, 80, 2},
        {"aligned", 3, 2, 64, 64, 2},
        {"boundary", 3, 1, 32, 43, 2}
    };

    bool success = true;
    for (const CaseConfig &config : cases)
    {
        success = run_case(config) && success;
    }

    std::cout << "\n"
              << (success ? "[SUCCESS] baseline" : "[FAILED] baseline")
              << std::endl;
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
