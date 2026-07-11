#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
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

constexpr int kSpatialTileH = 16;
constexpr int kSpatialTileW = 16;
constexpr int kSpatialBlockSize = kSpatialTileH * kSpatialTileW;
constexpr int kSpatialChannelGroup = 4;
constexpr int kSpatialWeightStride = 128;
constexpr int kSpatialWeightElements =
    kSpatialChannelGroup * kSpatialWeightStride;

__global__ void
conv2d_4x128x256_groups_kernel(
    float *inputs,
    float *weights,
    float *bias,
    float *outputs,
    Conv2DParam param)
{
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
        weights + blockIdx.y * 4 * param.KhKw
        + threadIdx.x / 64 * param.KhKw
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
        threadIdx.x % 64 * 2 < param.KhKw);
    ptx::ldg_nc_0(
        weight_ldg_reg[1],
        weight_ldg_ptr + sizeof(float),
        threadIdx.x % 64 * 2 + 1 < param.KhKw);

    ptx::sts64(weight_ldg_reg[0], weight_ldg_reg[1], weights_sts_addr);
    __syncthreads();

    for (int k = 0; k < param.KhKw; k += 4)
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int tap = k + i;
            int curH = posh_ori + tap / param.Kw;
            int curW = posw_ori + tap % param.Kw;
            int in_offset = curH * param.in_w + curW;
            bool guard = curH >= 0 && curW >= 0
                && curW < param.in_w && curH < param.in_h;

            if (guard)
            {
                input_frag[0][i] = input_ptr[in_offset];
                input_frag[1][i] = input_ptr[in_offset + param.inHW];
                input_frag[2][i] = input_ptr[in_offset + param.inHW * 2];
                input_frag[3][i] = input_ptr[in_offset + param.inHW * 3];
            }
            else
            {
                input_frag[0][i] = 0.0f;
                input_frag[1][i] = 0.0f;
                input_frag[2][i] = 0.0f;
                input_frag[3][i] = 0.0f;
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
    conv2d_4x128x256_groups_kernel<<<grid, block>>>(
        static_cast<float *>(inputs),
        static_cast<float *>(weights),
        static_cast<float *>(bias),
        static_cast<float *>(outputs),
        param);
}

__global__ void
conv2d_4x16x16_shared_spatial_kernel(
    const float *inputs,
    const float *weights,
    const float *bias,
    float *outputs,
    Conv2DParam param)
{
    extern __shared__ __align__(16) float shared[];
    float *shared_weights = shared;
    float *shared_bias = shared_weights + kSpatialWeightElements;
    float *shared_inputs = shared_bias + kSpatialChannelGroup;

    int input_tile_h = (kSpatialTileH - 1) * param.Sh + param.Kh;
    int input_tile_w = (kSpatialTileW - 1) * param.Sw + param.Kw;
    int input_tile_hw = input_tile_h * input_tile_w;
    int tiles_w = (param.out_w + kSpatialTileW - 1) / kSpatialTileW;
    int tile_h = blockIdx.x / tiles_w;
    int tile_w = blockIdx.x % tiles_w;
    int output_h_origin = tile_h * kSpatialTileH;
    int output_w_origin = tile_w * kSpatialTileW;
    int input_h_origin = output_h_origin * param.Sh - param.Ph;
    int input_w_origin = output_w_origin * param.Sw - param.Pw;

    for (int index = threadIdx.x;
         index < kSpatialWeightElements;
         index += blockDim.x)
    {
        int tap = index / kSpatialChannelGroup;
        int channel = index % kSpatialChannelGroup;
        float value = 0.0f;
        if (tap < param.KhKw)
        {
            value = weights[
                blockIdx.y * kSpatialChannelGroup * param.KhKw
                + channel * param.KhKw + tap];
        }
        shared_weights[index] = value;
    }

    if (threadIdx.x < kSpatialChannelGroup)
    {
        shared_bias[threadIdx.x] = bias[
            blockIdx.y * kSpatialChannelGroup + threadIdx.x];
    }

    int input_elements = kSpatialChannelGroup * input_tile_hw;
    const float *input_group = inputs
        + blockIdx.z * param.inBatchNumel
        + blockIdx.y * kSpatialChannelGroup * param.inHW;
    for (int index = threadIdx.x;
         index < input_elements;
         index += blockDim.x)
    {
        int spatial = index / kSpatialChannelGroup;
        int channel = index % kSpatialChannelGroup;
        int tile_input_h = spatial / input_tile_w;
        int tile_input_w = spatial % input_tile_w;
        int input_h = input_h_origin + tile_input_h;
        int input_w = input_w_origin + tile_input_w;
        bool valid = input_h >= 0 && input_w >= 0
            && input_h < param.in_h && input_w < param.in_w;
        shared_inputs[index] = valid
            ? input_group[
                channel * param.inHW + input_h * param.in_w + input_w]
            : 0.0f;
    }
    __syncthreads();

    int local_output_h = threadIdx.x / kSpatialTileW;
    int local_output_w = threadIdx.x % kSpatialTileW;
    int output_h = output_h_origin + local_output_h;
    int output_w = output_w_origin + local_output_w;
    bool output_valid = output_h < param.out_h && output_w < param.out_w;
    if (!output_valid)
    {
        return;
    }

    float accumulators[kSpatialChannelGroup] = {
        0.0f,
        0.0f,
        0.0f,
        0.0f
    };

    for (int kernel_h = 0; kernel_h < param.Kh; ++kernel_h)
    {
        int shared_input_h = local_output_h * param.Sh + kernel_h;
        for (int kernel_w = 0; kernel_w < param.Kw; ++kernel_w)
        {
            int shared_input_w = local_output_w * param.Sw + kernel_w;
            int input_offset =
                shared_input_h * input_tile_w + shared_input_w;
            int tap = kernel_h * param.Kw + kernel_w;
            float input_values[kSpatialChannelGroup];
            float weight_values[kSpatialChannelGroup];
            uint32_t input_address = ptx::smem_u32addr(
                shared_inputs + input_offset * kSpatialChannelGroup);
            uint32_t weight_address = ptx::smem_u32addr(
                shared_weights + tap * kSpatialChannelGroup);
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
                 channel < kSpatialChannelGroup;
                 ++channel)
            {
                accumulators[channel] +=
                    weight_values[channel] * input_values[channel];
            }
        }
    }

    int output_position = output_h * param.out_w + output_w;
    int output_base = blockIdx.z * param.outBatchNumel
        + blockIdx.y * kSpatialChannelGroup * param.outHW
        + output_position;
#pragma unroll
    for (int channel = 0;
         channel < kSpatialChannelGroup;
         ++channel)
    {
        outputs[output_base + channel * param.outHW] =
            accumulators[channel] + shared_bias[channel];
    }
}

static size_t shared_spatial_bytes(const Conv2DParam &param)
{
    int input_tile_h = (kSpatialTileH - 1) * param.Sh + param.Kh;
    int input_tile_w = (kSpatialTileW - 1) * param.Sw + param.Kw;
    size_t input_elements = static_cast<size_t>(kSpatialChannelGroup)
        * input_tile_h * input_tile_w;
    return (kSpatialWeightElements + kSpatialChannelGroup + input_elements)
        * sizeof(float);
}

static bool supports_shared_spatial(const Conv2DParam &param)
{
    constexpr size_t maximum_shared_bytes = 48 * 1024;
    return param.out_ch % kSpatialChannelGroup == 0
        && param.out_h >= 32 && param.out_w >= 32
        && param.Kh == 7 && param.Kw == 7
        && param.Sh == 2 && param.Sw == 2
        && param.KhKw <= kSpatialWeightStride
        && shared_spatial_bytes(param) <= maximum_shared_bytes;
}

static void launch_target(
    void *inputs,
    void *weights,
    void *bias,
    void *outputs,
    Conv2DParam param,
    uint32_t n)
{
    if (!supports_shared_spatial(param))
    {
        launch_custom(inputs, weights, bias, outputs, param, n);
        return;
    }

    int tiles_h = (param.out_h + kSpatialTileH - 1) / kSpatialTileH;
    int tiles_w = (param.out_w + kSpatialTileW - 1) / kSpatialTileW;
    dim3 block(kSpatialBlockSize);
    dim3 grid(tiles_h * tiles_w, param.out_ch / kSpatialChannelGroup, n);
    conv2d_4x16x16_shared_spatial_kernel
        <<<grid, block, shared_spatial_bytes(param)>>>(
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

    Conv2DParam param {};
    param.in_h = config.h;
    param.in_w = config.h;
    param.in_ch = config.c;
    param.inHW = config.h * config.h;
    param.inChKhKw = config.c * config.r * config.r;
    param.inBatchNumel = config.c * config.h * config.h;
    param.out_ch = config.c;
    param.out_h = out_h;
    param.out_w = out_h;
    param.outHW = out_h * out_h;
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

static bool run_case(const CaseConfig &config)
{
    constexpr int warmup = 20;
    constexpr int iterations = 100;
    constexpr int groups = 3;
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

    uint32_t grid_x = (param.outHW + 255) / 256;
    int spatial_tiles_h =
        (param.out_h + kSpatialTileH - 1) / kSpatialTileH;
    int spatial_tiles_w =
        (param.out_w + kSpatialTileW - 1) / kSpatialTileW;
    std::cout << "\n[CONFIG] " << config.name
              << " dtype=fp32"
              << " N=" << config.n
              << " C=" << config.c
              << " H=W=" << config.h
              << " K=" << config.r
              << " stride=" << config.stride
              << " outHW=" << param.outHW
              << " grid=(" << grid_x
              << "," << param.out_ch / 4
              << "," << config.n << ")"
              << " block=256"
              << " atol=" << atol
              << " rtol=" << rtol << std::endl;
    std::cout << "[CONFIG] shared_spatial"
              << " enabled=" << supports_shared_spatial(param)
              << " tile=" << kSpatialTileH << "x" << kSpatialTileW
              << " grid=(" << spatial_tiles_h * spatial_tiles_w
              << "," << param.out_ch / kSpatialChannelGroup
              << "," << config.n << ")"
              << " block=" << kSpatialBlockSize
              << " shared_bytes=" << shared_spatial_bytes(param)
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
        config.h,
        config.r,
        config.r,
        config.stride,
        config.stride,
        config.r / 2,
        config.r / 2);
    bool baseline_correct = verify(output, reference, atol, rtol);

    std::cout << "[STAGE] correctness shared_spatial" << std::endl;
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
    bool spatial_correct = verify(output, reference, atol, rtol);
    bool correct = baseline_correct && spatial_correct;

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
    auto launch_spatial = [&]()
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

    preheat(launch_spatial, preheat_ms);
    std::cout << "[BENCHMARK] shared_spatial" << std::endl;
    for (int group = 1; group <= groups; ++group)
    {
        print_stats(
            "shared_spatial",
            group,
            measure(
                launch_spatial,
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
    bool shared_spatial)
{
    CaseConfig config {"profile", r, n, c, h, 2, 1};
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
        if (shared_spatial)
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
              << (shared_spatial ? "shared_spatial" : "baseline")
              << " launches=" << profile_launches
              << std::endl;

    CUDA_CHECK(cudaFree(output));
    CUDA_CHECK(cudaFree(bias));
    CUDA_CHECK(cudaFree(weight));
    CUDA_CHECK(cudaFree(input));
    return 0;
}

int main(int argc, char **argv)
{
    bool profile_baseline = argc > 1
        && std::strcmp(argv[1], "--profile") == 0;
    bool profile_shared = argc > 1
        && std::strcmp(argv[1], "--profile-shared") == 0;
    if (profile_baseline || profile_shared)
    {
        int r = 7;
        int n = 16;
        int c = 128;
        int h = 80;
        if (argc == 6)
        {
            r = std::atoi(argv[2]);
            n = std::atoi(argv[3]);
            c = std::atoi(argv[4]);
            h = std::atoi(argv[5]);
        }
        return run_profile(r, n, c, h, profile_shared);
    }

    const CaseConfig cases[] = {
        {"throughput", 7, 16, 128, 80, 2, 16},
        {"main", 7, 4, 128, 80, 2, 64},
        {"k7_out32", 7, 4, 128, 64, 2, 64},
        {"k7_out48", 7, 4, 128, 96, 2, 32},
        {"k7_out22", 7, 1, 32, 43, 2, 512},
        {"common", 5, 1, 32, 80, 2, 512},
        {"aligned", 3, 2, 64, 64, 2, 512},
        {"boundary", 3, 1, 32, 43, 2, 1024}
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
