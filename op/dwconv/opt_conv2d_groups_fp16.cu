#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include "config.h"
#include "ptx_utils.cuh"

constexpr int kKernelCapacity = 128;
constexpr const char *kTargetName = "opt_conv2d_groups_fp16";
constexpr int kSharedTileH = 16;
constexpr int kSharedTileW = 16;
constexpr int kSharedBlockSize = kSharedTileH * kSharedTileW;
constexpr int kSharedChannelPairs = 4;
constexpr int kSharedWeightStride = 128;
constexpr int kSharedWeightElements =
    kSharedChannelPairs * kSharedWeightStride;

struct CaseConfig
{
    const char *name;
    int r;
    int n;
    int c;
    int h;
    int w;
    int stride;
    int launches_per_sample;
};

struct Stats
{
    float mean;
    float median;
    float minimum;
    float maximum;
    float stddev;
};

#define CUDA_CHECK(call)                                                   \
{                                                                          \
    cudaError_t status = (call);                                           \
    if (status != cudaSuccess)                                             \
    {                                                                      \
        std::cout << "[ERROR] CUDA: " << cudaGetErrorString(status)        \
                  << std::endl;                                            \
        std::exit(1);                                                      \
    }                                                                      \
}

#define CUDNN_CHECK(call)                                                  \
{                                                                          \
    cudnnStatus_t status = (call);                                         \
    if (status != CUDNN_STATUS_SUCCESS)                                    \
    {                                                                      \
        std::cout << "[ERROR] cuDNN: " << cudnnGetErrorString(status)      \
                  << std::endl;                                            \
        std::exit(1);                                                      \
    }                                                                      \
}

template <int KernelSize>
__global__ void
conv2d_4x128x256_fp16_groups_kernel(
    const __half *inputs,
    const __half2 *packed_weights,
    const __half *bias,
    __half *outputs,
    Conv2DParam param)
{
    constexpr bool specialized = KernelSize > 0;
    constexpr int specialized_elements = KernelSize * KernelSize;
    constexpr int specialized_capacity =
        (specialized_elements + 3) / 4 * 4;
    constexpr int shared_capacity = specialized
        ? specialized_capacity
        : kKernelCapacity;
    int kernel_width = specialized ? KernelSize : param.Kw;
    int kernel_elements = specialized
        ? specialized_elements
        : param.KhKw;
    __shared__ __half2 smem_weights[4 * shared_capacity];
    __shared__ __half2 smem_bias[4];

    int tid = threadIdx.x;
    int out_pos = blockIdx.x * blockDim.x + tid;
    int posh_ori = (out_pos / static_cast<int>(param.out_w))
        * static_cast<int>(param.Sh) - static_cast<int>(param.Ph);
    int posw_ori = (out_pos % static_cast<int>(param.out_w))
        * static_cast<int>(param.Sw) - static_cast<int>(param.Pw);

    for (int index = tid;
         index < 4 * shared_capacity;
         index += blockDim.x)
    {
        int channel_pair = index / shared_capacity;
        int kernel_pos = index % shared_capacity;
        __half2 value = __float2half2_rn(0.0f);
        if (kernel_pos < kernel_elements)
        {
            int weight_offset = blockIdx.y * 4 * kernel_elements
                + channel_pair * kernel_elements + kernel_pos;
            value = packed_weights[weight_offset];
        }
        smem_weights[index] = value;
    }
    __syncthreads();

    const __half *input_ptr = inputs
        + blockIdx.z * param.inBatchNumel
        + blockIdx.y * 8 * param.inHW;
    __half2 output_frag[4];

#pragma unroll
    for (int channel_pair = 0; channel_pair < 4; ++channel_pair)
    {
        output_frag[channel_pair] = __float2half2_rn(0.0f);
    }

    if (tid < 4)
    {
        int channel = blockIdx.y * 8 + tid * 2;
        smem_bias[tid] = __halves2half2(
            bias[channel],
            bias[channel + 1]);
    }
    __syncthreads();
#pragma unroll
    for (int channel_pair = 0; channel_pair < 4; ++channel_pair)
    {
        output_frag[channel_pair] = __hadd2(
            output_frag[channel_pair],
            smem_bias[channel_pair]);
    }

    for (int k = 0; k < kernel_elements; k += 4)
    {
        __half2 input_frag[4][4];

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int kernel_pos = k + i;
            int cur_h = posh_ori
                + kernel_pos / kernel_width;
            int cur_w = posw_ori
                + kernel_pos % kernel_width;
            bool valid = kernel_pos < kernel_elements
                && cur_h >= 0
                && cur_h < static_cast<int>(param.in_h)
                && cur_w >= 0
                && cur_w < static_cast<int>(param.in_w);

#pragma unroll
            for (int channel_pair = 0; channel_pair < 4; ++channel_pair)
            {
                if (valid)
                {
                    int input_offset = cur_h * param.in_w + cur_w
                        + channel_pair * 2 * param.inHW;
                    input_frag[channel_pair][i] = __halves2half2(
                        input_ptr[input_offset],
                        input_ptr[input_offset + param.inHW]);
                }
                else
                {
                    input_frag[channel_pair][i] = __float2half2_rn(0.0f);
                }
            }
        }

#pragma unroll
        for (int channel_pair = 0; channel_pair < 4; ++channel_pair)
        {
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                __half2 weight =
                    smem_weights[channel_pair * shared_capacity + k + i];
                output_frag[channel_pair] = __hfma2(
                    weight,
                    input_frag[channel_pair][i],
                    output_frag[channel_pair]);
            }
        }
    }

    if (out_pos < static_cast<int>(param.outHW))
    {
        int out_offset = blockIdx.z * param.outBatchNumel
            + blockIdx.y * 8 * param.outHW + out_pos;
#pragma unroll
        for (int channel_pair = 0; channel_pair < 4; ++channel_pair)
        {
            outputs[out_offset + channel_pair * 2 * param.outHW] =
                output_frag[channel_pair].x;
            outputs[out_offset + (channel_pair * 2 + 1) * param.outHW] =
                output_frag[channel_pair].y;
        }
    }
}

template <int KernelSize>
static void launch_specialized(
    const __half *inputs,
    const __half2 *packed_weights,
    const __half *bias,
    __half *outputs,
    const Conv2DParam &param,
    int n)
{
    dim3 block(256);
    dim3 grid(
        (param.outHW + block.x - 1) / block.x,
        param.out_ch / 8,
        n);
    conv2d_4x128x256_fp16_groups_kernel<KernelSize><<<grid, block>>>(
        inputs,
        packed_weights,
        bias,
        outputs,
        param);
}

template <int KernelSize>
__global__ void
conv2d_1x128x256_fp16_groups_kernel(
    const __half *inputs,
    const __half2 *packed_weights,
    const __half *bias,
    __half *outputs,
    Conv2DParam param)
{
    constexpr int kernel_elements = KernelSize * KernelSize;
    constexpr int shared_capacity = (kernel_elements + 3) / 4 * 4;
    __shared__ __half2 shared_weights[shared_capacity];
    __shared__ __half2 shared_bias;

    int tid = threadIdx.x;
    if (tid < shared_capacity)
    {
        shared_weights[tid] = tid < kernel_elements
            ? packed_weights[blockIdx.y * kernel_elements + tid]
            : __float2half2_rn(0.0f);
    }
    if (tid == 0)
    {
        int channel = blockIdx.y * 2;
        shared_bias = __halves2half2(
            bias[channel],
            bias[channel + 1]);
    }
    __syncthreads();

    int out_pos = blockIdx.x * blockDim.x + tid;
    int posh_origin = (out_pos / static_cast<int>(param.out_w))
        * static_cast<int>(param.Sh) - static_cast<int>(param.Ph);
    int posw_origin = (out_pos % static_cast<int>(param.out_w))
        * static_cast<int>(param.Sw) - static_cast<int>(param.Pw);
    const __half *input_pair = inputs
        + blockIdx.z * param.inBatchNumel
        + blockIdx.y * 2 * param.inHW;
    __half2 output_value = shared_bias;

    for (int tap_base = 0;
         tap_base < kernel_elements;
         tap_base += 4)
    {
        __half2 input_values[4];
#pragma unroll
        for (int index = 0; index < 4; ++index)
        {
            int tap = tap_base + index;
            int input_h = posh_origin + tap / KernelSize;
            int input_w = posw_origin + tap % KernelSize;
            bool valid = tap < kernel_elements
                && input_h >= 0
                && input_h < static_cast<int>(param.in_h)
                && input_w >= 0
                && input_w < static_cast<int>(param.in_w);
            if (valid)
            {
                int input_offset = input_h * param.in_w + input_w;
                input_values[index] = __halves2half2(
                    input_pair[input_offset],
                    input_pair[input_offset + param.inHW]);
            }
            else
            {
                input_values[index] = __float2half2_rn(0.0f);
            }
        }

#pragma unroll
        for (int index = 0; index < 4; ++index)
        {
            output_value = __hfma2(
                shared_weights[tap_base + index],
                input_values[index],
                output_value);
        }
    }

    if (out_pos < static_cast<int>(param.outHW))
    {
        int output_offset = blockIdx.z * param.outBatchNumel
            + blockIdx.y * 2 * param.outHW + out_pos;
        outputs[output_offset] = output_value.x;
        outputs[output_offset + param.outHW] = output_value.y;
    }
}

template <int KernelSize>
static void launch_pair(
    const __half *inputs,
    const __half2 *packed_weights,
    const __half *bias,
    __half *outputs,
    const Conv2DParam &param,
    int n)
{
    dim3 block(256);
    dim3 grid(
        (param.outHW + block.x - 1) / block.x,
        param.out_ch / 2,
        n);
    conv2d_1x128x256_fp16_groups_kernel<KernelSize><<<grid, block>>>(
        inputs,
        packed_weights,
        bias,
        outputs,
        param);
}

__device__ __forceinline__ int swizzle_shared_input_x(int input_x)
{
    return input_x ^ ((input_x >> 3) & 1);
}

template <int KernelSize>
__global__ void
conv2d_8x16x16_fp16_shared_input_kernel(
    const __half *inputs,
    const __half2 *packed_weights,
    const __half *bias,
    __half *outputs,
    Conv2DParam param)
{
    constexpr int kernel_elements = KernelSize * KernelSize;
    extern __shared__ __align__(16) __half2 shared[];
    __half2 *shared_weights = shared;
    __half2 *shared_bias = shared_weights + kSharedWeightElements;
    __half2 *shared_inputs = shared_bias + kSharedChannelPairs;

    int input_tile_h =
        (kSharedTileH - 1) * param.Sh + KernelSize;
    int input_tile_w =
        (kSharedTileW - 1) * param.Sw + KernelSize;
    int shared_input_stride = input_tile_w + 1;
    int input_tile_hw = input_tile_h * input_tile_w;
    int tiles_w = (param.out_w + kSharedTileW - 1) / kSharedTileW;
    int tile_h = blockIdx.x / tiles_w;
    int tile_w = blockIdx.x % tiles_w;
    int output_h_origin = tile_h * kSharedTileH;
    int output_w_origin = tile_w * kSharedTileW;
    int input_h_origin = output_h_origin * param.Sh - param.Ph;
    int input_w_origin = output_w_origin * param.Sw - param.Pw;

    for (int index = threadIdx.x;
         index < kSharedWeightElements;
         index += blockDim.x)
    {
        int tap = index / kSharedChannelPairs;
        int channel_pair = index % kSharedChannelPairs;
        __half2 value = __float2half2_rn(0.0f);
        if (tap < kernel_elements)
        {
            value = packed_weights[
                blockIdx.y * kSharedChannelPairs * kernel_elements
                + channel_pair * kernel_elements + tap];
        }
        shared_weights[index] = value;
    }

    if (threadIdx.x < kSharedChannelPairs)
    {
        int channel = blockIdx.y * 8 + threadIdx.x * 2;
        shared_bias[threadIdx.x] = __halves2half2(
            bias[channel],
            bias[channel + 1]);
    }

    const __half *input_group = inputs
        + blockIdx.z * param.inBatchNumel
        + blockIdx.y * 8 * param.inHW;
    for (int spatial = threadIdx.x;
         spatial < input_tile_hw;
         spatial += blockDim.x)
    {
        int tile_input_h = spatial / input_tile_w;
        int tile_input_w = spatial % input_tile_w;
        int input_h = input_h_origin + tile_input_h;
        int input_w = input_w_origin + tile_input_w;
        bool valid = input_h >= 0 && input_w >= 0
            && input_h < param.in_h && input_w < param.in_w;
        int input_offset = input_h * param.in_w + input_w;
        __half2 input_values[kSharedChannelPairs];
#pragma unroll
        for (int channel_pair = 0;
             channel_pair < kSharedChannelPairs;
             ++channel_pair)
        {
            if (valid)
            {
                const __half *pair_input = input_group
                    + channel_pair * 2 * param.inHW + input_offset;
                input_values[channel_pair] = __halves2half2(
                    pair_input[0],
                    pair_input[param.inHW]);
            }
            else
            {
                input_values[channel_pair] = __float2half2_rn(0.0f);
            }
        }
        int shared_input_x = swizzle_shared_input_x(tile_input_w);
        int shared_input_offset =
            tile_input_h * shared_input_stride + shared_input_x;
        uint32_t shared_input_address = ptx::smem_u32addr(
            shared_inputs
            + shared_input_offset * kSharedChannelPairs);
        ptx::sts128(
            input_values[0],
            input_values[1],
            input_values[2],
            input_values[3],
            shared_input_address);
    }
    __syncthreads();

    int local_output_h = threadIdx.x / kSharedTileW;
    int local_output_w = threadIdx.x % kSharedTileW;
    int output_h = output_h_origin + local_output_h;
    int output_w = output_w_origin + local_output_w;
    if (output_h >= param.out_h || output_w >= param.out_w)
    {
        return;
    }

    __half2 accumulators[kSharedChannelPairs] = {
        shared_bias[0],
        shared_bias[1],
        shared_bias[2],
        shared_bias[3]
    };
    for (int kernel_h = 0;
         kernel_h < KernelSize;
         ++kernel_h)
    {
        int shared_input_h = local_output_h * param.Sh + kernel_h;
        for (int kernel_w = 0;
             kernel_w < KernelSize;
             ++kernel_w)
        {
            int shared_input_w = local_output_w * param.Sw + kernel_w;
            int shared_input_x = swizzle_shared_input_x(shared_input_w);
            int input_offset =
                shared_input_h * shared_input_stride + shared_input_x;
            int tap = kernel_h * KernelSize + kernel_w;
            __half2 input_values[kSharedChannelPairs];
            __half2 weight_values[kSharedChannelPairs];
            uint32_t input_address = ptx::smem_u32addr(
                shared_inputs + input_offset * kSharedChannelPairs);
            uint32_t weight_address = ptx::smem_u32addr(
                shared_weights + tap * kSharedChannelPairs);
            ptx::lds128(
                input_values[0],
                input_values[1],
                input_values[2],
                input_values[3],
                input_address);
            ptx::lds128(
                weight_values[0],
                weight_values[1],
                weight_values[2],
                weight_values[3],
                weight_address);
#pragma unroll
            for (int channel_pair = 0;
                 channel_pair < kSharedChannelPairs;
                 ++channel_pair)
            {
                accumulators[channel_pair] = __hfma2(
                    weight_values[channel_pair],
                    input_values[channel_pair],
                    accumulators[channel_pair]);
            }
        }
    }

    int output_position = output_h * param.out_w + output_w;
    int output_base = blockIdx.z * param.outBatchNumel
        + blockIdx.y * 8 * param.outHW + output_position;
#pragma unroll
    for (int channel_pair = 0;
         channel_pair < kSharedChannelPairs;
         ++channel_pair)
    {
        outputs[output_base + channel_pair * 2 * param.outHW] =
            accumulators[channel_pair].x;
        outputs[output_base
                + (channel_pair * 2 + 1) * param.outHW] =
            accumulators[channel_pair].y;
    }
}

static size_t shared_input_bytes(const Conv2DParam &param)
{
    int input_tile_h = (kSharedTileH - 1) * param.Sh + param.Kh;
    int input_tile_w = (kSharedTileW - 1) * param.Sw + param.Kw;
    int shared_input_stride = input_tile_w + 1;
    size_t input_elements = static_cast<size_t>(kSharedChannelPairs)
        * input_tile_h * shared_input_stride;
    return (kSharedWeightElements + kSharedChannelPairs + input_elements)
        * sizeof(__half2);
}

static bool supports_shared_input(const Conv2DParam &param)
{
    constexpr size_t maximum_shared_bytes = 48 * 1024;
    return param.out_ch % 8 == 0
        && param.Kh == param.Kw
        && (param.Kh == 9 || param.Kh == 11)
        && shared_input_bytes(param) <= maximum_shared_bytes;
}

template <int KernelSize>
static void launch_shared_input(
    const __half *inputs,
    const __half2 *packed_weights,
    const __half *bias,
    __half *outputs,
    const Conv2DParam &param,
    int n)
{
    int tiles_h = (param.out_h + kSharedTileH - 1) / kSharedTileH;
    int tiles_w = (param.out_w + kSharedTileW - 1) / kSharedTileW;
    dim3 block(kSharedBlockSize);
    dim3 grid(tiles_h * tiles_w, param.out_ch / 8, n);
    conv2d_8x16x16_fp16_shared_input_kernel<KernelSize>
        <<<grid, block, shared_input_bytes(param)>>>(
            inputs,
            packed_weights,
            bias,
            outputs,
            param);
}

static void launch_baseline(
    const __half *inputs,
    const __half2 *packed_weights,
    const __half *bias,
    __half *outputs,
    const Conv2DParam &param,
    int n)
{
    if (param.Kh == 3 && param.Kw == 3)
    {
        launch_pair<3>(
            inputs, packed_weights, bias, outputs, param, n);
    }
    else if (param.Kh == 5 && param.Kw == 5)
    {
        launch_pair<5>(
            inputs, packed_weights, bias, outputs, param, n);
    }
    else if (param.Kh == 7 && param.Kw == 7)
    {
        launch_pair<7>(
            inputs, packed_weights, bias, outputs, param, n);
    }
    else if (param.Kh == 9 && param.Kw == 9)
    {
        size_t pair_blocks = static_cast<size_t>(
            (param.outHW + 255) / 256)
            * (param.out_ch / 2) * n;
        if (pair_blocks >= 140 && supports_shared_input(param))
        {
            launch_shared_input<9>(
                inputs, packed_weights, bias, outputs, param, n);
        }
        else
        {
            launch_pair<9>(
                inputs, packed_weights, bias, outputs, param, n);
        }
    }
    else if (param.Kh == 11 && param.Kw == 11)
    {
        if (supports_shared_input(param))
        {
            launch_shared_input<11>(
                inputs, packed_weights, bias, outputs, param, n);
        }
        else
        {
            launch_specialized<11>(
                inputs, packed_weights, bias, outputs, param, n);
        }
    }
    else
    {
        launch_specialized<0>(
            inputs, packed_weights, bias, outputs, param, n);
    }
}

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

static void print_stats(
    const char *implementation,
    int group,
    const Stats &stats)
{
    std::cout << "  " << std::left << std::setw(16) << implementation
              << " group=" << group
              << " mean=" << std::fixed << std::setprecision(6)
              << stats.mean
              << " median=" << stats.median
              << " min=" << stats.minimum
              << " max=" << stats.maximum
              << " stddev=" << stats.stddev
              << " ms" << std::endl;
}

static float benchmark_cudnn(
    cudnnHandle_t handle,
    cudnnTensorDescriptor_t input_desc,
    cudnnFilterDescriptor_t filter_desc,
    cudnnConvolutionDescriptor_t conv_desc,
    cudnnTensorDescriptor_t output_desc,
    cudnnConvolutionFwdAlgo_t algo,
    const __half *inputs,
    const __half *weights,
    __half *outputs,
    void *workspace,
    size_t workspace_size,
    int warmup,
    int iters)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    for (int i = 0; i < warmup; ++i)
    {
        CUDNN_CHECK(cudnnConvolutionForward(
            handle,
            &alpha,
            input_desc,
            inputs,
            filter_desc,
            weights,
            conv_desc,
            algo,
            workspace,
            workspace_size,
            &beta,
            output_desc,
            outputs));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i)
    {
        CUDNN_CHECK(cudnnConvolutionForward(
            handle,
            &alpha,
            input_desc,
            inputs,
            filter_desc,
            weights,
            conv_desc,
            algo,
            workspace,
            workspace_size,
            &beta,
            output_desc,
            outputs));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed_ms / iters;
}

template <typename Launch>
static void preheat(Launch launch, int duration_ms)
{
    auto start = std::chrono::steady_clock::now();
    auto duration = std::chrono::milliseconds(duration_ms);
    do
    {
        for (int launch_index = 0; launch_index < 100; ++launch_index)
        {
            launch();
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } while (std::chrono::steady_clock::now() - start < duration);
}

template <typename Launch>
static Stats measure_launch_stats(
    Launch launch,
    int warmup,
    int iterations,
    int launches_per_sample)
{
    auto warmup_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < warmup; ++iteration)
    {
        for (int launch_index = 0;
             launch_index < launches_per_sample;
             ++launch_index)
        {
            launch();
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto warmup_end = std::chrono::steady_clock::now();
    double warmup_ms = std::chrono::duration<double, std::milli>(
        warmup_end - warmup_start).count();
    std::cout << "  warmup_elapsed_ms=" << std::fixed
              << std::setprecision(3) << warmup_ms << std::endl;

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> samples;
    samples.reserve(iterations);
    for (int iteration = 0; iteration < iterations; ++iteration)
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

static Conv2DParam make_param(
    int c,
    int h,
    int w,
    int r,
    int stride,
    int padding)
{
    int out_h = (h - r + 2 * padding) / stride + 1;
    int out_w = (w - r + 2 * padding) / stride + 1;

    Conv2DParam param {};
    param.in_h = h;
    param.in_w = w;
    param.in_ch = c;
    param.inHW = h * w;
    param.inChKhKw = c * r * r;
    param.inBatchNumel = c * h * w;
    param.out_ch = c;
    param.out_h = out_h;
    param.out_w = out_w;
    param.outHW = out_h * out_w;
    param.outBatchNumel = c * out_h * out_w;
    param.Kh = r;
    param.Kw = r;
    param.KhKw = r * r;
    param.Sh = stride;
    param.Sw = stride;
    param.Ph = padding;
    param.Pw = padding;
    return param;
}

static void pack_weights(
    const __half *weights,
    __half2 *packed_weights,
    int c,
    int kernel_elements)
{
    for (int channel = 0; channel < c; channel += 2)
    {
        int channel_pair = channel / 2;
        for (int k = 0; k < kernel_elements; ++k)
        {
            packed_weights[channel_pair * kernel_elements + k] =
                __halves2half2(
                    weights[channel * kernel_elements + k],
                    weights[(channel + 1) * kernel_elements + k]);
        }
    }
}

static bool verify_output(
    const std::string &implementation,
    const __half *actual_output,
    const __half *cudnn_output,
    int out_numel,
    float atol,
    float rtol)
{
    std::vector<__half> actual(out_numel);
    std::vector<__half> expected(out_numel);
    CUDA_CHECK(cudaMemcpy(
        actual.data(),
        actual_output,
        out_numel * sizeof(__half),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        expected.data(),
        cudnn_output,
        out_numel * sizeof(__half),
        cudaMemcpyDeviceToHost));

    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    size_t error_count = 0;
    for (int index = 0; index < out_numel; ++index)
    {
        float actual_value = __half2float(actual[index]);
        float expected_value = __half2float(expected[index]);
        float abs_error = std::abs(actual_value - expected_value);
        float denominator = std::max(std::abs(expected_value), 1.0e-6f);
        float rel_error = abs_error / denominator;
        max_abs_error = std::max(max_abs_error, abs_error);
        max_rel_error = std::max(max_rel_error, rel_error);
        if (abs_error > atol && rel_error > rtol)
        {
            ++error_count;
        }
    }

    std::cout << "[VERIFY] " << implementation
              << " vs cuDNN NCHW"
              << " max_abs_error=" << std::scientific << max_abs_error
              << " max_rel_error=" << max_rel_error
              << " errors=" << error_count << "/" << out_numel
              << " atol=" << atol
              << " rtol=" << rtol
              << std::defaultfloat << std::endl;
    return error_count == 0;
}

static bool run_case(
    cudnnHandle_t cudnn,
    cudnnConvolutionFwdAlgo_t algo,
    const CaseConfig &config,
    bool benchmark)
{
    constexpr int warmup = 20;
    constexpr int iterations = 100;
    constexpr int groups = 3;
    constexpr float atol = 1.0e-2f;
    constexpr float rtol = 1.0e-2f;
    int r = config.r;
    int n = config.n;
    int c = config.c;
    int h = config.h;
    int w = config.w;
    int stride = config.stride;
    int padding = r / 2;
    if (r <= 0 || r * r > kKernelCapacity
        || n <= 0 || c <= 0 || c % 8 != 0
        || h <= 0 || w <= 0 || stride <= 0)
    {
        std::cout << "[ERROR] unsupported config: " << config.name
                  << ", require C % 8 == 0 and K * K <= 128"
                  << std::endl;
        return false;
    }
    Conv2DParam param = make_param(c, h, w, r, stride, padding);
    int in_numel = n * param.inBatchNumel;
    int out_numel = n * param.outBatchNumel;
    int weight_numel = c * r * r;

    double flops = static_cast<double>(out_numel) * r * r * 2.0 / 1e9;
    double total_bytes = static_cast<double>(
        in_numel + out_numel + weight_numel + c) * sizeof(__half);
    double arithmetic_intensity = flops * 1e9 / total_bytes;

    uint32_t grid_x = (param.outHW + 255) / 256;
    size_t grid_blocks = static_cast<size_t>(grid_x)
        * (c / 8) * n;
    double full_waves = static_cast<double>(grid_blocks)
        / (128.0 * 6.0);
    std::cout << "\n[CONFIG] " << config.name
              << " dtype=fp16"
              << " N=" << n
              << " C=" << c
              << " H=" << h
              << " W=" << w
              << " K=" << r
              << " stride=" << stride
              << " out=" << param.out_h << "x" << param.out_w
              << " grid=(" << grid_x << "," << c / 8 << "," << n << ")"
              << " blocks=" << grid_blocks
              << " waves_at_6_blocks_per_sm=" << std::fixed
              << std::setprecision(2) << full_waves
              << " block=256"
              << " atol=" << atol
              << " rtol=" << rtol
              << std::defaultfloat << std::endl;

    __half *input_h = nullptr;
    __half *weight_h = nullptr;
    __half2 *packed_weight_h = nullptr;
    __half *bias_h = nullptr;
    CUDA_CHECK(cudaMallocHost(&input_h, in_numel * sizeof(__half)));
    CUDA_CHECK(cudaMallocHost(&weight_h, weight_numel * sizeof(__half)));
    CUDA_CHECK(cudaMallocHost(
        &packed_weight_h,
        weight_numel / 2 * sizeof(__half2)));
    CUDA_CHECK(cudaMallocHost(&bias_h, c * sizeof(__half)));

    for (int i = 0; i < in_numel; ++i)
    {
        float value = static_cast<float>((std::rand() & 255) - 128) / 512.0f;
        input_h[i] = __float2half_rn(value);
    }
    for (int i = 0; i < weight_numel; ++i)
    {
        float value = static_cast<float>((std::rand() & 255) - 128) / 512.0f;
        weight_h[i] = __float2half_rn(value);
    }
    for (int i = 0; i < c; ++i)
    {
        float value = static_cast<float>((std::rand() & 127) - 64) / 512.0f;
        bias_h[i] = __float2half_rn(value);
    }
    pack_weights(weight_h, packed_weight_h, c, r * r);

    __half *input_d = nullptr;
    __half *weight_d = nullptr;
    __half2 *packed_weight_d = nullptr;
    __half *bias_d = nullptr;
    __half *baseline_output_d = nullptr;
    __half *cudnn_output_d = nullptr;
    CUDA_CHECK(cudaMalloc(&input_d, in_numel * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&weight_d, weight_numel * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(
        &packed_weight_d,
        weight_numel / 2 * sizeof(__half2)));
    CUDA_CHECK(cudaMalloc(&bias_d, c * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&baseline_output_d, out_numel * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&cudnn_output_d, out_numel * sizeof(__half)));

    CUDA_CHECK(cudaMemcpy(
        input_d,
        input_h,
        in_numel * sizeof(__half),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        weight_d,
        weight_h,
        weight_numel * sizeof(__half),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        packed_weight_d,
        packed_weight_h,
        weight_numel / 2 * sizeof(__half2),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        bias_d,
        bias_h,
        c * sizeof(__half),
        cudaMemcpyHostToDevice));

    std::cout << "[STAGE] correctness baseline" << std::endl;
    launch_baseline(
        input_d,
        packed_weight_d,
        bias_d,
        baseline_output_d,
        param,
        n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnTensorDescriptor_t bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        input_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_HALF,
        n,
        c,
        h,
        w));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        output_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_HALF,
        n,
        c,
        param.out_h,
        param.out_w));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        bias_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_HALF,
        1,
        c,
        1,
        1));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        filter_desc,
        CUDNN_DATA_HALF,
        CUDNN_TENSOR_NCHW,
        c,
        1,
        r,
        r));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        conv_desc,
        padding,
        padding,
        stride,
        stride,
        1,
        1,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_HALF));
    CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc, c));

    size_t workspace_size = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        algo,
        &workspace_size));
    void *workspace = nullptr;
    if (workspace_size > 0)
    {
        CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    }
    std::cout << "[STAGE] correctness reference=cuDNN_NCHW"
              << std::endl;
    benchmark_cudnn(
        cudnn,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        algo,
        input_d,
        weight_d,
        cudnn_output_d,
        workspace,
        workspace_size,
        0,
        1);
    float alpha = 1.0f;
    CUDNN_CHECK(cudnnAddTensor(
        cudnn,
        &alpha,
        bias_desc,
        bias_d,
        &alpha,
        output_desc,
        cudnn_output_d));
    bool correct = verify_output(
        "baseline",
        baseline_output_d,
        cudnn_output_d,
        out_numel,
        atol,
        rtol);

    if (benchmark)
    {
        int launches_per_sample = config.launches_per_sample;
        std::cout << "[STAGE] performance"
                  << " warmup=" << warmup
                  << " samples=" << iterations
                  << " groups=" << groups
                  << " launches_per_sample=" << launches_per_sample
                  << " arithmetic_intensity=" << std::fixed
                  << std::setprecision(3) << arithmetic_intensity
                  << std::defaultfloat << std::endl;
        auto baseline_launch = [&]()
        {
            launch_baseline(
                input_d,
                packed_weight_d,
                bias_d,
                baseline_output_d,
                param,
                n);
        };
        float beta = 0.0f;
        auto cudnn_nchw_launch = [&]()
        {
            CUDNN_CHECK(cudnnConvolutionForward(
                cudnn,
                &alpha,
                input_desc,
                input_d,
                filter_desc,
                weight_d,
                conv_desc,
                algo,
                workspace,
                workspace_size,
                &beta,
                output_desc,
                cudnn_output_d));
            CUDNN_CHECK(cudnnAddTensor(
                cudnn,
                &alpha,
                bias_desc,
                bias_d,
                &alpha,
                output_desc,
                cudnn_output_d));
        };

        constexpr int preheat_ms = 1000;
        std::cout << "[PREHEAT] baseline duration_ms="
                  << preheat_ms << std::endl;
        preheat(baseline_launch, preheat_ms);
        std::cout << "[BENCHMARK] baseline" << std::endl;
        for (int group = 1; group <= groups; ++group)
        {
            print_stats(
                "baseline",
                group,
                measure_launch_stats(
                    baseline_launch,
                    warmup,
                    iterations,
                    launches_per_sample));
        }
        std::cout << "[PREHEAT] cudnn_nchw duration_ms="
                  << preheat_ms << std::endl;
        preheat(cudnn_nchw_launch, preheat_ms);
        std::cout << "[BENCHMARK] cudnn_nchw_with_bias" << std::endl;
        for (int group = 1; group <= groups; ++group)
        {
            print_stats(
                "cudnn_nchw",
                group,
                measure_launch_stats(
                    cudnn_nchw_launch,
                    warmup,
                    iterations,
                    launches_per_sample));
        }
    }

    if (workspace != nullptr)
    {
        CUDA_CHECK(cudaFree(workspace));
    }
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));

    CUDA_CHECK(cudaFree(input_d));
    CUDA_CHECK(cudaFree(weight_d));
    CUDA_CHECK(cudaFree(packed_weight_d));
    CUDA_CHECK(cudaFree(bias_d));
    CUDA_CHECK(cudaFree(baseline_output_d));
    CUDA_CHECK(cudaFree(cudnn_output_d));
    CUDA_CHECK(cudaFreeHost(input_h));
    CUDA_CHECK(cudaFreeHost(weight_h));
    CUDA_CHECK(cudaFreeHost(packed_weight_h));
    CUDA_CHECK(cudaFreeHost(bias_h));

    std::cout << (correct ? "[SUCCESS] " : "[FAILED] ")
              << config.name << std::endl;
    return correct;
}

static int run_sweep(const char *csv_path)
{
    constexpr int warmup = 20;
    constexpr int iterations = 100;
    const int kernel_sizes[] = {3, 5, 7, 9, 11};
    const int batches[] = {1, 2, 4, 8, 16, 32};
    const int channels[] = {32, 64, 128, 256};
    const int heights[] = {40, 80, 160};

    std::ofstream csv(csv_path);
    if (!csv.is_open())
    {
        std::cout << "[ERROR] failed to open csv: "
                  << csv_path << std::endl;
        return EXIT_FAILURE;
    }
    csv << "k_size,n,c,h,kernel,time_ms,gflops,arith_intensity"
        << std::endl;

    std::cout << "[CONFIG] sweep target=skill_final_fp16"
              << " csv=" << csv_path
              << " warmup=" << warmup
              << " iterations=" << iterations
              << std::endl;
    int result_count = 0;
    for (int r : kernel_sizes)
    {
        for (int n : batches)
        {
            for (int c : channels)
            {
                for (int h : heights)
                {
                    Conv2DParam param = make_param(
                        c, h, h, r, 2, r / 2);
                    size_t input_count = static_cast<size_t>(n)
                        * param.inBatchNumel;
                    size_t packed_weight_count =
                        static_cast<size_t>(c / 2) * param.KhKw;
                    size_t output_count = static_cast<size_t>(n)
                        * param.outBatchNumel;

                    __half *input = nullptr;
                    __half2 *packed_weight = nullptr;
                    __half *bias = nullptr;
                    __half *output = nullptr;
                    CUDA_CHECK(cudaMalloc(
                        &input, input_count * sizeof(__half)));
                    CUDA_CHECK(cudaMalloc(
                        &packed_weight,
                        packed_weight_count * sizeof(__half2)));
                    CUDA_CHECK(cudaMalloc(&bias, c * sizeof(__half)));
                    CUDA_CHECK(cudaMalloc(
                        &output, output_count * sizeof(__half)));
                    CUDA_CHECK(cudaMemset(
                        input, 0, input_count * sizeof(__half)));
                    CUDA_CHECK(cudaMemset(
                        packed_weight,
                        0,
                        packed_weight_count * sizeof(__half2)));
                    CUDA_CHECK(cudaMemset(bias, 0, c * sizeof(__half)));

                    auto launch = [&]()
                    {
                        launch_baseline(
                            input,
                            packed_weight,
                            bias,
                            output,
                            param,
                            n);
                    };
                    for (int index = 0; index < warmup; ++index)
                    {
                        launch();
                    }
                    CUDA_CHECK(cudaDeviceSynchronize());

                    cudaEvent_t start;
                    cudaEvent_t stop;
                    CUDA_CHECK(cudaEventCreate(&start));
                    CUDA_CHECK(cudaEventCreate(&stop));
                    CUDA_CHECK(cudaEventRecord(start));
                    for (int index = 0; index < iterations; ++index)
                    {
                        launch();
                    }
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaEventRecord(stop));
                    CUDA_CHECK(cudaEventSynchronize(stop));
                    float elapsed_ms = 0.0f;
                    CUDA_CHECK(cudaEventElapsedTime(
                        &elapsed_ms, start, stop));
                    CUDA_CHECK(cudaEventDestroy(start));
                    CUDA_CHECK(cudaEventDestroy(stop));
                    float time_ms = elapsed_ms / iterations;

                    double flops = static_cast<double>(output_count)
                        * r * r * 2.0 / 1.0e9;
                    size_t weight_count = static_cast<size_t>(c)
                        * param.KhKw;
                    double total_bytes = static_cast<double>(
                        input_count + output_count + weight_count + c)
                        * sizeof(__half);
                    double arithmetic_intensity =
                        flops * 1.0e9 / total_bytes;
                    double gflops = flops / (time_ms / 1000.0);
                    csv << r << "," << n << "," << c << "," << h
                        << ",skill_final," << std::fixed
                        << std::setprecision(6) << time_ms
                        << "," << gflops
                        << "," << arithmetic_intensity
                        << std::endl;

                    CUDA_CHECK(cudaFree(output));
                    CUDA_CHECK(cudaFree(bias));
                    CUDA_CHECK(cudaFree(packed_weight));
                    CUDA_CHECK(cudaFree(input));
                    ++result_count;
                }
            }
        }
    }
    csv.close();
    std::cout << "[SUCCESS] sweep results=" << result_count
              << std::endl;
    return EXIT_SUCCESS;
}

int main(int argc, char *argv[])
{
    if (argc == 3 && std::strcmp(argv[1], "--sweep-csv") == 0)
    {
        return run_sweep(argv[2]);
    }
    const CaseConfig cases[] = {
        {"throughput", 7, 64, 128, 80, 80, 2, 4},
        {"main", 7, 4, 128, 80, 80, 2, 64},
        {"small", 7, 1, 32, 43, 43, 2, 512},
        {"common", 5, 1, 32, 80, 80, 2, 512},
        {"aligned", 3, 2, 64, 64, 64, 2, 512},
        {"k3_throughput", 3, 64, 128, 80, 80, 2, 4},
        {"k5_throughput", 5, 64, 128, 80, 80, 2, 4},
        {"boundary", 3, 1, 32, 43, 43, 2, 1024},
        {"k9", 9, 1, 32, 80, 80, 2, 256},
        {"k9_blocks128", 9, 1, 32, 89, 89, 2, 128},
        {"k9_blocks140", 9, 1, 40, 80, 80, 2, 128},
        {"k9_blocks168", 9, 1, 48, 80, 80, 2, 128},
        {"k9_blocks224", 9, 1, 64, 80, 80, 2, 128},
        {"k9_blocks448", 9, 1, 128, 80, 80, 2, 128},
        {"k9_main", 9, 4, 128, 80, 80, 2, 64},
        {"k9_throughput", 9, 64, 128, 80, 80, 2, 16},
        {"k11_throughput", 11, 64, 128, 80, 80, 2, 16},
        {"k11_tail", 11, 1, 32, 43, 43, 2, 256},
        {"k11_rectangular", 11, 1, 32, 65, 81, 2, 256},
        {"k11_stride1", 11, 1, 32, 43, 43, 1, 256},
        {"k11_fallback_stride3", 11, 1, 32, 43, 43, 3, 256},
        {"rectangular", 7, 1, 32, 65, 81, 2, 512},
        {"stride1", 3, 1, 32, 43, 43, 1, 512},
        {"min_channels", 7, 1, 8, 43, 43, 2, 1024},
        {"non_power_channels", 7, 1, 40, 43, 43, 2, 512},
        {"k7_blocks96", 7, 1, 32, 153, 153, 2, 256},
        {"k7_blocks104", 7, 1, 32, 161, 161, 2, 256}
    };

    const char *selected_case = nullptr;
    bool correctness_only = false;
    bool profile = false;
    for (int argument = 1; argument < argc; ++argument)
    {
        if (std::strcmp(argv[argument], "--case") == 0
            && argument + 1 < argc)
        {
            selected_case = argv[++argument];
        }
        else if (std::strcmp(argv[argument], "--profile") == 0
                 && argument + 1 < argc)
        {
            profile = true;
            selected_case = argv[++argument];
        }
        else if (std::strcmp(argv[argument], "--correctness-only") == 0)
        {
            correctness_only = true;
        }
        else
        {
            std::cout << "[ERROR] unknown argument: "
                      << argv[argument] << std::endl;
            return EXIT_FAILURE;
        }
    }

    std::cout << "[CONFIG] target=" << kTargetName
              << " capacity=" << kKernelCapacity
              << " warmup=20 samples=100 groups=3"
              << " selected_case="
              << (selected_case == nullptr ? "all" : selected_case)
              << " correctness_only="
              << (correctness_only ? "true" : "false")
              << " profile=" << (profile ? "true" : "false")
              << std::endl;

    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));
    cudnnConvolutionFwdAlgo_t algo =
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

    bool success = true;
    bool found = selected_case == nullptr;
    for (const CaseConfig &config : cases)
    {
        if (selected_case != nullptr
            && std::strcmp(config.name, selected_case) != 0)
        {
            continue;
        }
        found = true;
        std::srand(20260712);
        bool benchmark = selected_case != nullptr
            && !correctness_only && !profile;
        success = run_case(cudnn, algo, config, benchmark) && success;
    }
    CUDNN_CHECK(cudnnDestroy(cudnn));

    if (!found)
    {
        std::cout << "[ERROR] unknown case: " << selected_case << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "\n"
              << (success ? "[SUCCESS] fp16 baseline"
                          : "[FAILED] fp16 baseline")
              << std::endl;
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
