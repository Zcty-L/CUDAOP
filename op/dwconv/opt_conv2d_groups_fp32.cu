#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "config.h"
#include "cpu/cpu_ops.h"
#include "ptx_utils.cuh"

#define CUDA_CHECK(call)                                                   \
{                                                                          \
    cudaError_t status = (call);                                           \
    if (status != cudaSuccess)                                             \
    {                                                                      \
        std::cout << "[ERROR] CUDA: " << cudaGetErrorString(status)        \
                  << std::endl;                                            \
        std::exit(EXIT_FAILURE);                                            \
    }                                                                      \
}

constexpr int kSharedTileH = 16;
constexpr int kSharedTileW = 16;
constexpr int kSharedBlockSize = kSharedTileH * kSharedTileW;
constexpr int kSharedChannelGroup = 4;
constexpr int kSharedKernelSize = 11;
constexpr int kSharedKernelElements =
    kSharedKernelSize * kSharedKernelSize;
constexpr int kSharedWeightStride = 128;
constexpr int kSharedWeightElements =
    kSharedChannelGroup * kSharedWeightStride;

__device__ __forceinline__ int swizzle_shared_input_x(int input_x)
{
    return input_x ^ ((input_x >> 3) & 1);
}

template <int KernelSize>
__global__ void
conv2d_4x128x256_groups_kernel(
    float *inputs,
    float *weights,
    float *bias,
    float *outputs,
    Conv2DParam param)
{
    constexpr bool specialized_kernel = KernelSize > 0;
    constexpr int specialized_kernel_elements = KernelSize * KernelSize;
    int kernel_width = specialized_kernel ? KernelSize : param.Kw;
    int kernel_elements = specialized_kernel
        ? specialized_kernel_elements
        : param.KhKw;

    __shared__ __align__(2 * 1024)
    char smem[4 * 128 * 4 + 4 * 4];
    auto *smemweight = reinterpret_cast<float *>(smem);

    int out_pos = blockIdx.x * 256 + threadIdx.x;
    int posh_ori = (out_pos / param.out_w) * param.Sh - param.Ph;
    int posw_ori = (out_pos % param.out_w) * param.Sw - param.Pw;

    uint32_t weights_sts_addr =
        ptx::smem_u32addr(smemweight + threadIdx.x * 2);
    uint32_t weights_lds_addr = ptx::smem_u32addr(smemweight);

    const char *weight_ldg_ptr = reinterpret_cast<const char *>(
        weights + blockIdx.y * 4 * kernel_elements
        + threadIdx.x / 64 * kernel_elements
        + threadIdx.x % 64 * 2);
    auto *input_ptr = inputs
        + blockIdx.z * param.inBatchNumel
        + blockIdx.y * 4 * param.inHW;

    float weight_ldg_reg[2];
    float weight_frag[16];
    float input_frag[4][4];
    float output_frag[4];

#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        output_frag[i] = 0.0f;
    }

    ptx::ldg_nc_0(
        weight_ldg_reg[0],
        weight_ldg_ptr,
        threadIdx.x % 64 * 2 < kernel_elements);
    ptx::ldg_nc_0(
        weight_ldg_reg[1],
        weight_ldg_ptr + sizeof(float),
        threadIdx.x % 64 * 2 + 1 < kernel_elements);

    ptx::sts64(weight_ldg_reg[0], weight_ldg_reg[1], weights_sts_addr);
    __syncthreads();

    for (int k = 0; k < kernel_elements; k += 4)
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int tap = k + i;
            int curH = posh_ori + tap / kernel_width;
            int curW = posw_ori + tap % kernel_width;
            int in_offset = curH * param.in_w + curW;
            bool guard = tap < kernel_elements
                && curH >= 0 && curW >= 0
                && curW < param.in_w && curH < param.in_h;

            if constexpr (KernelSize == 3)
            {
                if (guard)
                {
                    ptx::ldg32_nc_0(
                        input_frag[0][i],
                        input_ptr + in_offset,
                        true);
                    ptx::ldg32_nc_0(
                        input_frag[1][i],
                        input_ptr + in_offset + param.inHW,
                        true);
                    ptx::ldg32_nc_0(
                        input_frag[2][i],
                        input_ptr + in_offset + param.inHW * 2,
                        true);
                    ptx::ldg32_nc_0(
                        input_frag[3][i],
                        input_ptr + in_offset + param.inHW * 3,
                        true);
                }
                else
                {
                    input_frag[0][i] = 0.0f;
                    input_frag[1][i] = 0.0f;
                    input_frag[2][i] = 0.0f;
                    input_frag[3][i] = 0.0f;
                }
            }
            else if constexpr (specialized_kernel)
            {
                ptx::ldg32_nc_0(
                    input_frag[0][i],
                    input_ptr + in_offset,
                    guard);
                ptx::ldg32_nc_0(
                    input_frag[1][i],
                    input_ptr + in_offset + param.inHW,
                    guard);
                ptx::ldg32_nc_0(
                    input_frag[2][i],
                    input_ptr + in_offset + param.inHW * 2,
                    guard);
                ptx::ldg32_nc_0(
                    input_frag[3][i],
                    input_ptr + in_offset + param.inHW * 3,
                    guard);
            }
            else
            {
                if (guard)
                {
                    input_frag[0][i] = input_ptr[in_offset];
                    input_frag[1][i] = input_ptr[
                        in_offset + param.inHW];
                    input_frag[2][i] = input_ptr[
                        in_offset + param.inHW * 2];
                    input_frag[3][i] = input_ptr[
                        in_offset + param.inHW * 3];
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

    auto *smembias = smemweight + 4 * 128;
    if (threadIdx.x < 4)
    {
        smembias[threadIdx.x] = bias[blockIdx.y * 4 + threadIdx.x];
    }
    __syncthreads();

    output_frag[0] += smembias[0];
    output_frag[1] += smembias[1];
    output_frag[2] += smembias[2];
    output_frag[3] += smembias[3];

    int outOffset = blockIdx.z * param.outBatchNumel
        + blockIdx.y * 4 * param.outHW
        + out_pos;
    if (out_pos < param.outHW)
    {
        outputs[outOffset + param.outHW * 0] = output_frag[0];
        outputs[outOffset + param.outHW * 1] = output_frag[1];
        outputs[outOffset + param.outHW * 2] = output_frag[2];
        outputs[outOffset + param.outHW * 3] = output_frag[3];
    }
}

static void launch_custom(
    void *inputs,
    void *weights,
    void *bias,
    void *outputs,
    Conv2DParam param,
    uint32_t n)
{
    dim3 block(256);
    dim3 grid((param.outHW + 255) / 256, param.out_ch / 4, n);
    conv2d_4x128x256_groups_kernel<0><<<grid, block>>>(
        static_cast<float *>(inputs),
        static_cast<float *>(weights),
        static_cast<float *>(bias),
        static_cast<float *>(outputs),
        param);
}

template <int KernelSize>
static void launch_specialized(
    void *inputs,
    void *weights,
    void *bias,
    void *outputs,
    Conv2DParam param,
    uint32_t n)
{
    dim3 block(256);
    dim3 grid((param.outHW + 255) / 256, param.out_ch / 4, n);
    conv2d_4x128x256_groups_kernel<KernelSize><<<grid, block>>>(
        static_cast<float *>(inputs),
        static_cast<float *>(weights),
        static_cast<float *>(bias),
        static_cast<float *>(outputs),
        param);
}

__global__ void
conv2d_1x128x256_groups_k7_kernel(
    const float *inputs,
    const float *weights,
    const float *bias,
    float *outputs,
    Conv2DParam param)
{
    constexpr int kernel_size = 7;
    constexpr int kernel_elements = kernel_size * kernel_size;
    __shared__ float shared_weights[kernel_elements];
    __shared__ float shared_bias;

    if (threadIdx.x < kernel_elements)
    {
        shared_weights[threadIdx.x] = weights[
            blockIdx.y * kernel_elements + threadIdx.x];
    }
    if (threadIdx.x == 0)
    {
        shared_bias = bias[blockIdx.y];
    }
    __syncthreads();

    int out_pos = blockIdx.x * blockDim.x + threadIdx.x;
    int posh_origin =
        (out_pos / param.out_w) * param.Sh - param.Ph;
    int posw_origin =
        (out_pos % param.out_w) * param.Sw - param.Pw;
    const float *input_channel = inputs
        + blockIdx.z * param.inBatchNumel
        + blockIdx.y * param.inHW;
    float accumulator = 0.0f;

    for (int tap_base = 0;
         tap_base < kernel_elements;
         tap_base += 4)
    {
        float input_values[4];
        float weight_values[4];
#pragma unroll
        for (int index = 0; index < 4; ++index)
        {
            int tap = tap_base + index;
            int kernel_h = tap / kernel_size;
            int kernel_w = tap % kernel_size;
            int input_h = posh_origin + kernel_h;
            int input_w = posw_origin + kernel_w;
            bool valid = tap < kernel_elements
                && input_h >= 0 && input_w >= 0
                && input_h < param.in_h && input_w < param.in_w;
            int input_offset = input_h * param.in_w + input_w;
            ptx::ldg32_nc_0(
                input_values[index],
                input_channel + input_offset,
                valid);
            weight_values[index] = tap < kernel_elements
                ? shared_weights[tap]
                : 0.0f;
        }
#pragma unroll
        for (int index = 0; index < 4; ++index)
        {
            accumulator += weight_values[index] * input_values[index];
        }
    }

    if (out_pos < param.outHW)
    {
        int output_offset = blockIdx.z * param.outBatchNumel
            + blockIdx.y * param.outHW + out_pos;
        outputs[output_offset] = accumulator + shared_bias;
    }
}

static void launch_k7_single_channel(
    void *inputs,
    void *weights,
    void *bias,
    void *outputs,
    Conv2DParam param,
    uint32_t n)
{
    dim3 block(256);
    dim3 grid((param.outHW + 255) / 256, param.out_ch, n);
    conv2d_1x128x256_groups_k7_kernel<<<grid, block>>>(
        static_cast<const float *>(inputs),
        static_cast<const float *>(weights),
        static_cast<const float *>(bias),
        static_cast<float *>(outputs),
        param);
}

__global__ void
conv2d_4x16x16_shared_input_kernel(
    const float *inputs,
    const float *weights,
    const float *bias,
    float *outputs,
    Conv2DParam param)
{
    extern __shared__ __align__(16) float shared[];
    float *shared_weights = shared;
    float *shared_bias = shared_weights + kSharedWeightElements;
    float *shared_inputs = shared_bias + kSharedChannelGroup;

    int input_tile_h =
        (kSharedTileH - 1) * param.Sh + kSharedKernelSize;
    int input_tile_w =
        (kSharedTileW - 1) * param.Sw + kSharedKernelSize;
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
        int channel = index / kSharedWeightStride;
        int tap = index % kSharedWeightStride;
        float value = 0.0f;
        if (tap < kSharedKernelElements)
        {
            value = weights[
                blockIdx.y * kSharedChannelGroup * kSharedKernelElements
                + channel * kSharedKernelElements + tap];
        }
        shared_weights[tap * kSharedChannelGroup + channel] = value;
    }

    if (threadIdx.x < kSharedChannelGroup)
    {
        shared_bias[threadIdx.x] = bias[
            blockIdx.y * kSharedChannelGroup + threadIdx.x];
    }

    const float *input_group = inputs
        + blockIdx.z * param.inBatchNumel
        + blockIdx.y * kSharedChannelGroup * param.inHW;
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
        float input_values[kSharedChannelGroup];
#pragma unroll
        for (int channel = 0;
             channel < kSharedChannelGroup;
             ++channel)
        {
            const float *input_address = input_group
                + channel * param.inHW + input_offset;
            ptx::ldg32_nc_0(
                input_values[channel],
                input_address,
                valid);
        }
        int shared_input_x = swizzle_shared_input_x(tile_input_w);
        int shared_input_offset =
            tile_input_h * shared_input_stride + shared_input_x;
        uint32_t shared_input_address = ptx::smem_u32addr(
            shared_inputs
            + shared_input_offset * kSharedChannelGroup);
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

    float accumulators[kSharedChannelGroup] = {
        0.0f,
        0.0f,
        0.0f,
        0.0f
    };

    for (int kernel_h = 0;
         kernel_h < kSharedKernelSize;
         ++kernel_h)
    {
        int shared_input_h = local_output_h * param.Sh + kernel_h;
        for (int kernel_w = 0;
             kernel_w < kSharedKernelSize;
             ++kernel_w)
        {
            int shared_input_w = local_output_w * param.Sw + kernel_w;
            int shared_input_x = swizzle_shared_input_x(shared_input_w);
            int input_offset =
                shared_input_h * shared_input_stride + shared_input_x;
            int tap = kernel_h * kSharedKernelSize + kernel_w;
            float input_values[kSharedChannelGroup];
            float weight_values[kSharedChannelGroup];
            uint32_t input_address = ptx::smem_u32addr(
                shared_inputs + input_offset * kSharedChannelGroup);
            uint32_t weight_address = ptx::smem_u32addr(
                shared_weights + tap * kSharedChannelGroup);
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
            for (int channel = 0;
                 channel < kSharedChannelGroup;
                 ++channel)
            {
                accumulators[channel] +=
                    weight_values[channel] * input_values[channel];
            }
        }
    }

    int output_position = output_h * param.out_w + output_w;
    int output_base = blockIdx.z * param.outBatchNumel
        + blockIdx.y * kSharedChannelGroup * param.outHW
        + output_position;
#pragma unroll
    for (int channel = 0;
         channel < kSharedChannelGroup;
         ++channel)
    {
        outputs[output_base + channel * param.outHW] =
            accumulators[channel] + shared_bias[channel];
    }
}

static size_t shared_input_bytes(const Conv2DParam &param)
{
    int input_tile_h = (kSharedTileH - 1) * param.Sh + param.Kh;
    int input_tile_w = (kSharedTileW - 1) * param.Sw + param.Kw;
    int shared_input_stride = input_tile_w + 1;
    size_t input_elements = static_cast<size_t>(kSharedChannelGroup)
        * input_tile_h * shared_input_stride;
    return (kSharedWeightElements + kSharedChannelGroup + input_elements)
        * sizeof(float);
}

static bool supports_shared_input(const Conv2DParam &param)
{
    constexpr size_t maximum_shared_bytes = 48 * 1024;
    return param.out_ch % kSharedChannelGroup == 0
        && param.Kh == 11 && param.Kw == 11
        && param.KhKw <= kSharedWeightStride
        && shared_input_bytes(param) <= maximum_shared_bytes;
}

static void launch_target(
    void *inputs,
    void *weights,
    void *bias,
    void *outputs,
    Conv2DParam param,
    uint32_t n)
{
    if (!supports_shared_input(param))
    {
        if (param.Kh == 3 && param.Kw == 3)
        {
            launch_specialized<3>(
                inputs, weights, bias, outputs, param, n);
        }
        else if (param.Kh == 5 && param.Kw == 5)
        {
            launch_specialized<5>(
                inputs, weights, bias, outputs, param, n);
        }
        else if (param.Kh == 7 && param.Kw == 7)
        {
            size_t grouped_blocks =
                static_cast<size_t>((param.outHW + 255) / 256)
                * (param.out_ch / 4) * n;
            if (grouped_blocks < 104)
            {
                launch_k7_single_channel(
                    inputs, weights, bias, outputs, param, n);
            }
            else
            {
                launch_specialized<7>(
                    inputs, weights, bias, outputs, param, n);
            }
        }
        else
        {
            launch_custom(inputs, weights, bias, outputs, param, n);
        }
        return;
    }

    int tiles_h = (param.out_h + kSharedTileH - 1) / kSharedTileH;
    int tiles_w = (param.out_w + kSharedTileW - 1) / kSharedTileW;
    dim3 block(kSharedBlockSize);
    dim3 grid(tiles_h * tiles_w, param.out_ch / kSharedChannelGroup, n);
    conv2d_4x16x16_shared_input_kernel
        <<<grid, block, shared_input_bytes(param)>>>(
            static_cast<const float *>(inputs),
            static_cast<const float *>(weights),
            static_cast<const float *>(bias),
            static_cast<float *>(outputs),
            param);
}

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
static void preheat(Launch launch, int duration_ms)
{
    auto start = std::chrono::steady_clock::now();
    auto duration = std::chrono::milliseconds(duration_ms);
    do
    {
        for (int i = 0; i < 100; ++i)
        {
            launch();
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } while (std::chrono::steady_clock::now() - start < duration);
}

template <typename Launch>
static float measure_elapsed_ms(Launch launch, int launches)
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

    std::vector<float> samples;
    samples.reserve(iterations);
    for (int i = 0; i < iterations; ++i)
    {
        samples.push_back(
            measure_elapsed_ms(launch, launches_per_sample)
            / launches_per_sample);
    }
    return calculate_stats(samples);
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
    int out_w =
        (config.w - config.r + 2 * (config.r / 2)) / config.stride + 1;

    Conv2DParam param {};
    param.in_h = config.h;
    param.in_w = config.w;
    param.in_ch = config.c;
    param.inHW = config.h * config.w;
    param.inChKhKw = config.c * config.r * config.r;
    param.inBatchNumel = config.c * param.inHW;
    param.out_ch = config.c;
    param.out_h = out_h;
    param.out_w = out_w;
    param.outHW = out_h * out_w;
    param.outBatchNumel = config.c * param.outHW;
    param.Kh = config.r;
    param.Kw = config.r;
    param.KhKw = config.r * config.r;
    param.Sh = config.stride;
    param.Sw = config.stride;
    param.Ph = config.r / 2;
    param.Pw = config.r / 2;
    return param;
}

static bool is_supported_config(const CaseConfig &config)
{
    return config.r > 0
        && config.n > 0
        && config.c > 0
        && config.c % 4 == 0
        && config.h > 0
        && config.w > 0
        && config.stride > 0
        && config.r * config.r <= 128;
}

static bool run_case(const CaseConfig &config, bool benchmark)
{
    constexpr int warmup = 20;
    constexpr int iterations = 100;
    constexpr int groups = 3;
    constexpr float atol = 1.0e-4f;
    constexpr float rtol = 1.0e-4f;

    if (!is_supported_config(config))
    {
        std::cout << "[ERROR] unsupported config: " << config.name
                  << ", require C % 4 == 0 and K * K <= 128"
                  << std::endl;
        return false;
    }

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

    uint32_t grid_x = (param.outHW + 255) / 256;
    std::cout << "\n[CONFIG] " << config.name
              << " dtype=fp32"
              << " N=" << config.n
              << " C=" << config.c
              << " H=" << config.h
              << " W=" << config.w
              << " K=" << config.r
              << " stride=" << config.stride
              << " outHW=" << param.outHW
              << " grid=(" << grid_x
              << "," << param.out_ch / 4
              << "," << config.n << ")"
              << " block=256"
              << " atol=" << atol
              << " rtol=" << rtol << std::endl;
    std::cout << "[CONFIG] target=shared_input_16x16_k11"
              << " enabled=" << supports_shared_input(param)
              << " block=" << kSharedBlockSize
              << " shared_bytes=" << shared_input_bytes(param)
              << std::endl;
    std::cout << "[STAGE] correctness baseline" << std::endl;
    launch_custom(
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
        config.w,
        config.r,
        config.r,
        config.stride,
        config.stride,
        config.r / 2,
        config.r / 2);
    bool baseline_correct = verify(output, reference, atol, rtol);

    std::cout << "[STAGE] correctness target" << std::endl;
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
    bool target_correct = verify(output, reference, atol, rtol);
    bool correct = baseline_correct && target_correct;

    if (!benchmark)
    {
        CUDA_CHECK(cudaFree(output_device));
        CUDA_CHECK(cudaFree(bias_device));
        CUDA_CHECK(cudaFree(weight_device));
        CUDA_CHECK(cudaFree(input_device));
        std::cout << (correct ? "[SUCCESS] " : "[FAILED] ")
                  << config.name << std::endl;
        return correct;
    }

    auto launch_baseline = [&]()
    {
        launch_custom(
            input_device,
            weight_device,
            bias_device,
            output_device,
            param,
            config.n);
    };
    auto launch_target_impl = [&]()
    {
        launch_target(
            input_device,
            weight_device,
            bias_device,
            output_device,
            param,
            config.n);
    };
    constexpr int preheat_ms = 1000;
    int launches_per_sample = config.launches_per_sample;
    std::cout << "[STAGE] performance separate_implementations=1"
              << " preheat_ms=" << preheat_ms
              << " warmup=" << warmup
              << " iterations=" << iterations
              << " groups=" << groups
              << " launches_per_sample=" << launches_per_sample
              << std::endl;

    preheat(launch_baseline, preheat_ms);
    std::cout << "[BENCHMARK] baseline" << std::endl;
    for (int group = 1; group <= groups; ++group)
    {
        print_stats(
            "baseline",
            group,
            measure(
                launch_baseline,
                warmup,
                iterations,
                launches_per_sample));
    }

    preheat(launch_target_impl, preheat_ms);
    std::cout << "[BENCHMARK] target" << std::endl;
    for (int group = 1; group <= groups; ++group)
    {
        print_stats(
            "target",
            group,
            measure(
                launch_target_impl,
                warmup,
                iterations,
                launches_per_sample));
    }

    CUDA_CHECK(cudaFree(output_device));
    CUDA_CHECK(cudaFree(bias_device));
    CUDA_CHECK(cudaFree(weight_device));
    CUDA_CHECK(cudaFree(input_device));

    std::cout << (correct ? "[SUCCESS] " : "[FAILED] ")
              << config.name << std::endl;
    return correct;
}

static int run_profile(
    int r,
    int n,
    int c,
    int h,
    bool target)
{
    CaseConfig config {"profile", r, n, c, h, h, 2, 1};
    if (!is_supported_config(config))
    {
        std::cout << "[ERROR] unsupported profile config"
                  << ", require C % 4 == 0 and K * K <= 128"
                  << std::endl;
        return EXIT_FAILURE;
    }
    Conv2DParam param = make_param(config);
    size_t input_count = static_cast<size_t>(n) * param.inBatchNumel;
    size_t weight_count = static_cast<size_t>(c) * param.KhKw;
    size_t output_count = static_cast<size_t>(n) * param.outBatchNumel;

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

    constexpr int profile_launches = 101;
    for (int launch_index = 0;
         launch_index < profile_launches;
         ++launch_index)
    {
        if (target)
        {
            launch_target(input, weight, bias, output, param, n);
        }
        else
        {
            launch_custom(input, weight, bias, output, param, n);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "[SUCCESS] profile launch implementation="
              << (target ? "target" : "baseline")
              << " launches=" << profile_launches
              << std::endl;

    CUDA_CHECK(cudaFree(output));
    CUDA_CHECK(cudaFree(bias));
    CUDA_CHECK(cudaFree(weight));
    CUDA_CHECK(cudaFree(input));
    return 0;
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

    std::cout << "[CONFIG] sweep target=skill_final_fp32"
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
                    CaseConfig config {
                        "sweep", r, n, c, h, h, 2, 1
                    };
                    Conv2DParam param = make_param(config);
                    size_t input_count = static_cast<size_t>(n)
                        * param.inBatchNumel;
                    size_t weight_count = static_cast<size_t>(c)
                        * param.KhKw;
                    size_t output_count = static_cast<size_t>(n)
                        * param.outBatchNumel;

                    float *input = nullptr;
                    float *weight = nullptr;
                    float *bias = nullptr;
                    float *output = nullptr;
                    CUDA_CHECK(cudaMalloc(
                        &input, input_count * sizeof(float)));
                    CUDA_CHECK(cudaMalloc(
                        &weight, weight_count * sizeof(float)));
                    CUDA_CHECK(cudaMalloc(&bias, c * sizeof(float)));
                    CUDA_CHECK(cudaMalloc(
                        &output, output_count * sizeof(float)));
                    CUDA_CHECK(cudaMemset(
                        input, 0, input_count * sizeof(float)));
                    CUDA_CHECK(cudaMemset(
                        weight, 0, weight_count * sizeof(float)));
                    CUDA_CHECK(cudaMemset(bias, 0, c * sizeof(float)));

                    auto launch = [&]()
                    {
                        launch_target(
                            input,
                            weight,
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
                    float time_ms = measure_elapsed_ms(
                        launch, iterations) / iterations;

                    double flops = static_cast<double>(output_count)
                        * r * r * 2.0 / 1.0e9;
                    double total_bytes = static_cast<double>(
                        input_count + output_count + weight_count + c)
                        * sizeof(float);
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
                    CUDA_CHECK(cudaFree(weight));
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

int main(int argc, char **argv)
{
    if (argc == 3 && std::strcmp(argv[1], "--sweep-csv") == 0)
    {
        return run_sweep(argv[2]);
    }
    bool profile = argc > 1
        && std::strcmp(argv[1], "--profile") == 0;
    bool profile_target = argc > 1
        && std::strcmp(argv[1], "--profile-target") == 0;
    if (profile || profile_target)
    {
        int r = 7;
        int n = 32;
        int c = 128;
        int h = 80;
        if (argc == 6)
        {
            r = std::atoi(argv[2]);
            n = std::atoi(argv[3]);
            c = std::atoi(argv[4]);
            h = std::atoi(argv[5]);
        }
        return run_profile(r, n, c, h, profile_target);
    }

    const CaseConfig cases[] = {
        {"throughput", 7, 32, 128, 80, 80, 2, 8},
        {"main", 7, 4, 128, 80, 80, 2, 64},
        {"k7_out32", 7, 4, 128, 64, 64, 2, 64},
        {"k7_out48", 7, 4, 128, 96, 96, 2, 32},
        {"k7_out22", 7, 1, 32, 43, 43, 2, 512},
        {"common", 5, 1, 32, 80, 80, 2, 512},
        {"aligned", 3, 2, 64, 64, 64, 2, 512},
        {"boundary", 3, 1, 32, 43, 43, 2, 1024},
        {"k11_tail", 11, 1, 32, 43, 43, 2, 256},
        {"k11_throughput", 11, 32, 128, 80, 80, 2, 8},
        {"k11_main", 11, 4, 128, 80, 80, 2, 64},
        {"k11_out32", 11, 4, 128, 64, 64, 2, 64},
        {"k11_rectangular", 11, 1, 32, 65, 81, 2, 512},
        {"k11_stride1", 11, 1, 32, 43, 43, 1, 256},
        {"stride1", 3, 1, 32, 43, 43, 1, 512},
        {"min_channels", 7, 1, 4, 43, 43, 2, 1024},
        {"non_power_channels", 7, 1, 36, 43, 43, 2, 512},
        {"dispatch_out31", 7, 1, 32, 61, 61, 2, 512},
        {"dispatch_out32", 7, 1, 32, 63, 63, 2, 512},
        {"dispatch_out33", 7, 1, 32, 65, 65, 2, 512},
        {"dispatch_blocks96", 7, 1, 32, 107, 107, 2, 256},
        {"dispatch_blocks104", 7, 1, 32, 111, 111, 2, 256},
        {"rectangular", 7, 1, 32, 65, 81, 2, 512}
    };

    const char *selected_case = nullptr;
    bool benchmark = true;
    for (int argument = 1; argument < argc; ++argument)
    {
        if (std::strcmp(argv[argument], "--correctness-only") == 0)
        {
            benchmark = false;
        }
        else if (std::strcmp(argv[argument], "--case") == 0
                 && argument + 1 < argc)
        {
            selected_case = argv[++argument];
        }
        else
        {
            std::cout << "[ERROR] usage: " << argv[0]
                      << " [--correctness-only] [--case <name>]"
                      << std::endl;
            return EXIT_FAILURE;
        }
    }

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
        success = run_case(config, benchmark) && success;
    }
    if (!found)
    {
        std::cout << "[ERROR] unknown case: " << selected_case << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "\n"
              << (success ? "[SUCCESS] baseline" : "[FAILED] baseline")
              << std::endl;
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
