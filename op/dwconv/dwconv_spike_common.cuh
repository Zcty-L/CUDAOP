#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "config.h"

#ifndef DWCONV_SPIKE_KERNEL_CAPACITY
#error "DWCONV_SPIKE_KERNEL_CAPACITY must be defined"
#endif

#ifndef DWCONV_SPIKE_KERNEL_NAME
#error "DWCONV_SPIKE_KERNEL_NAME must be defined"
#endif

#ifndef DWCONV_SPIKE_TARGET_NAME
#error "DWCONV_SPIKE_TARGET_NAME must be defined"
#endif

namespace
{

constexpr int kThreads = 256;
constexpr int kChannelsPerBlock = 4;

void check_cuda(cudaError_t status, const char *operation)
{
    if (status == cudaSuccess)
    {
        return;
    }

    std::cerr << "[CUDA ERROR] " << operation << ": "
              << cudaGetErrorString(status) << std::endl;
    std::exit(EXIT_FAILURE);
}

template <int TSteps>
__global__ void DWCONV_SPIKE_KERNEL_NAME(
    const uint8_t *__restrict__ inputs,
    const SpikeScalar *__restrict__ weights,
    const SpikeScalar *__restrict__ bias,
    SpikeScalar *__restrict__ outputs,
    Conv2DParam param)
{
    __shared__ SpikeScalar
        shared_weights[kChannelsPerBlock][DWCONV_SPIKE_KERNEL_CAPACITY];

    const int channel_base = blockIdx.y * kChannelsPerBlock;
    const int weight_count =
        kChannelsPerBlock * DWCONV_SPIKE_KERNEL_CAPACITY;

    for (int index = threadIdx.x; index < weight_count; index += blockDim.x)
    {
        const int channel = index / DWCONV_SPIKE_KERNEL_CAPACITY;
        const int kernel_index = index % DWCONV_SPIKE_KERNEL_CAPACITY;
        shared_weights[channel][kernel_index] =
            weights[(channel_base + channel) * DWCONV_SPIKE_KERNEL_CAPACITY
                    + kernel_index];
    }
    __syncthreads();

    const int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_h = output_index / static_cast<int>(param.out_w);
    const int output_w = output_index % static_cast<int>(param.out_w);
    const int input_h_origin =
        output_h * static_cast<int>(param.Sh) - static_cast<int>(param.Ph);
    const int input_w_origin =
        output_w * static_cast<int>(param.Sw) - static_cast<int>(param.Pw);

    const int lane = threadIdx.x & 31;
    SpikeScalar bias_value = spike_zero();
    if (lane < kChannelsPerBlock)
    {
        bias_value = bias[channel_base + lane];
    }

    SpikeScalar output_frag[TSteps][kChannelsPerBlock];
#pragma unroll
    for (int channel = 0; channel < kChannelsPerBlock; ++channel)
    {
        const SpikeScalar channel_bias = spike_shuffle(bias_value, channel);
#pragma unroll
        for (int timestep = 0; timestep < TSteps; ++timestep)
        {
            output_frag[timestep][channel] = channel_bias;
        }
    }

    const uint8_t *input_base =
        inputs + static_cast<size_t>(channel_base) * param.inHW;

    for (int kernel_index = 0;
         kernel_index < static_cast<int>(param.KhKw);
         ++kernel_index)
    {
        const int input_h =
            input_h_origin + kernel_index / static_cast<int>(param.Kw);
        const int input_w =
            input_w_origin + kernel_index % static_cast<int>(param.Kw);
        const bool input_valid =
            output_index < static_cast<int>(param.outHW)
            && input_h >= 0
            && input_h < static_cast<int>(param.in_h)
            && input_w >= 0
            && input_w < static_cast<int>(param.in_w);

        uint8_t packed_inputs[kChannelsPerBlock] = {};
        if (input_valid)
        {
            const int input_offset =
                input_h * static_cast<int>(param.in_w) + input_w;
#pragma unroll
            for (int channel = 0; channel < kChannelsPerBlock; ++channel)
            {
                packed_inputs[channel] =
                    input_base[channel * param.inHW + input_offset];
            }
        }

#pragma unroll
        for (int channel = 0; channel < kChannelsPerBlock; ++channel)
        {
            const SpikeScalar weight =
                shared_weights[channel][kernel_index];
#pragma unroll
            for (int timestep = 0; timestep < TSteps; ++timestep)
            {
                if ((packed_inputs[channel] >> timestep) & 1U)
                {
                    output_frag[timestep][channel] = spike_add(
                        output_frag[timestep][channel],
                        weight);
                }
            }
        }
    }

    if (output_index >= static_cast<int>(param.outHW))
    {
        return;
    }

    const int channel_output_offset = channel_base * param.outHW + output_index;
#pragma unroll
    for (int timestep = 0; timestep < TSteps; ++timestep)
    {
        SpikeScalar *timestep_output =
            outputs + static_cast<size_t>(timestep) * param.outBatchNumel;
#pragma unroll
        for (int channel = 0; channel < kChannelsPerBlock; ++channel)
        {
            timestep_output[channel_output_offset + channel * param.outHW] =
                output_frag[timestep][channel];
        }
    }
}

template <int TSteps>
void launch_spike_dwconv(
    const uint8_t *inputs,
    const SpikeScalar *weights,
    const SpikeScalar *bias,
    SpikeScalar *outputs,
    const Conv2DParam &param)
{
    const dim3 grid(
        (param.outHW + kThreads - 1) / kThreads,
        param.out_ch / kChannelsPerBlock);
    DWCONV_SPIKE_KERNEL_NAME<TSteps><<<grid, kThreads>>>(
        inputs,
        weights,
        bias,
        outputs,
        param);
}

template <int TSteps>
void cpu_reference(
    const std::vector<uint8_t> &inputs,
    const std::vector<SpikeScalar> &weights,
    const std::vector<SpikeScalar> &bias,
    std::vector<SpikeScalar> &outputs,
    const Conv2DParam &param)
{
    for (int timestep = 0; timestep < TSteps; ++timestep)
    {
        for (int channel = 0; channel < static_cast<int>(param.out_ch); ++channel)
        {
            for (int output_h = 0;
                 output_h < static_cast<int>(param.out_h);
                 ++output_h)
            {
                for (int output_w = 0;
                     output_w < static_cast<int>(param.out_w);
                     ++output_w)
                {
                    SpikeScalar accumulator = bias[channel];
                    for (int kernel_h = 0;
                         kernel_h < static_cast<int>(param.Kh);
                         ++kernel_h)
                    {
                        for (int kernel_w = 0;
                             kernel_w < static_cast<int>(param.Kw);
                             ++kernel_w)
                        {
                            const int input_h =
                                output_h * static_cast<int>(param.Sh)
                                - static_cast<int>(param.Ph) + kernel_h;
                            const int input_w =
                                output_w * static_cast<int>(param.Sw)
                                - static_cast<int>(param.Pw) + kernel_w;
                            if (input_h < 0
                                || input_h >= static_cast<int>(param.in_h)
                                || input_w < 0
                                || input_w >= static_cast<int>(param.in_w))
                            {
                                continue;
                            }

                            const size_t input_index =
                                static_cast<size_t>(channel) * param.inHW
                                + input_h * param.in_w + input_w;
                            if ((inputs[input_index] >> timestep) & 1U)
                            {
                                const int kernel_index =
                                    kernel_h * param.Kw + kernel_w;
                                accumulator = spike_host_add(
                                    accumulator,
                                    weights[channel * DWCONV_SPIKE_KERNEL_CAPACITY
                                            + kernel_index]);
                            }
                        }
                    }

                    const size_t output_index =
                        static_cast<size_t>(timestep) * param.outBatchNumel
                        + static_cast<size_t>(channel) * param.outHW
                        + output_h * param.out_w + output_w;
                    outputs[output_index] = accumulator;
                }
            }
        }
    }
}

Conv2DParam make_param(
    int channels,
    int height,
    int width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w)
{
    Conv2DParam param = {};
    param.in_h = height;
    param.in_w = width;
    param.in_ch = channels;
    param.inHW = height * width;
    param.inChKhKw = channels * kernel_h * kernel_w;
    param.inBatchNumel = channels * height * width;
    param.out_ch = channels;
    param.out_h = (height + 2 * padding_h - kernel_h) / stride_h + 1;
    param.out_w = (width + 2 * padding_w - kernel_w) / stride_w + 1;
    param.outHW = param.out_h * param.out_w;
    param.outBatchNumel = channels * param.outHW;
    param.Kh = kernel_h;
    param.Kw = kernel_w;
    param.KhKw = kernel_h * kernel_w;
    param.Sh = stride_h;
    param.Sw = stride_w;
    param.Ph = padding_h;
    param.Pw = padding_w;
    return param;
}

struct TestCase
{
    const char *name;
    int timesteps;
    int channels;
    int height;
    int width;
    int kernel_h;
    int kernel_w;
    int stride;
    int padding_h;
    int padding_w;
};

template <int TSteps>
bool run_case(const TestCase &test_case)
{
    const Conv2DParam param = make_param(
        test_case.channels,
        test_case.height,
        test_case.width,
        test_case.kernel_h,
        test_case.kernel_w,
        test_case.stride,
        test_case.stride,
        test_case.padding_h,
        test_case.padding_w);

    const size_t input_count = param.inBatchNumel;
    const size_t weight_count =
        static_cast<size_t>(param.out_ch) * DWCONV_SPIKE_KERNEL_CAPACITY;
    const size_t output_count =
        static_cast<size_t>(TSteps) * param.outBatchNumel;

    std::mt19937 generator(42 + param.KhKw + TSteps);
    std::uniform_int_distribution<int> spike_distribution(0, 1);
    std::uniform_real_distribution<float> value_distribution(-0.5F, 0.5F);

    std::vector<uint8_t> inputs(input_count, 0);
    std::vector<SpikeScalar> weights(weight_count, spike_from_float(0.0F));
    std::vector<SpikeScalar> bias(param.out_ch);
    std::vector<SpikeScalar> outputs(output_count);
    std::vector<SpikeScalar> reference(output_count);

    for (uint8_t &packed_input : inputs)
    {
        for (int timestep = 0; timestep < TSteps; ++timestep)
        {
            packed_input |= static_cast<uint8_t>(
                spike_distribution(generator) << timestep);
        }
    }
    for (int channel = 0; channel < static_cast<int>(param.out_ch); ++channel)
    {
        bias[channel] = spike_from_float(
            static_cast<float>((channel % 17) - 8) / 16.0F);
        for (int kernel_index = 0;
             kernel_index < static_cast<int>(param.KhKw);
             ++kernel_index)
        {
            weights[channel * DWCONV_SPIKE_KERNEL_CAPACITY + kernel_index] =
                spike_from_float(value_distribution(generator));
        }
    }

    uint8_t *device_inputs = nullptr;
    SpikeScalar *device_weights = nullptr;
    SpikeScalar *device_bias = nullptr;
    SpikeScalar *device_outputs = nullptr;
    check_cuda(
        cudaMalloc(&device_inputs, input_count * sizeof(uint8_t)),
        "cudaMalloc inputs");
    check_cuda(
        cudaMalloc(&device_weights, weight_count * sizeof(SpikeScalar)),
        "cudaMalloc weights");
    check_cuda(
        cudaMalloc(&device_bias, param.out_ch * sizeof(SpikeScalar)),
        "cudaMalloc bias");
    check_cuda(
        cudaMalloc(&device_outputs, output_count * sizeof(SpikeScalar)),
        "cudaMalloc outputs");

    check_cuda(
        cudaMemcpy(
            device_inputs,
            inputs.data(),
            input_count * sizeof(uint8_t),
            cudaMemcpyHostToDevice),
        "copy inputs");
    check_cuda(
        cudaMemcpy(
            device_weights,
            weights.data(),
            weight_count * sizeof(SpikeScalar),
            cudaMemcpyHostToDevice),
        "copy weights");
    check_cuda(
        cudaMemcpy(
            device_bias,
            bias.data(),
            param.out_ch * sizeof(SpikeScalar),
            cudaMemcpyHostToDevice),
        "copy bias");

    launch_spike_dwconv<TSteps>(
        device_inputs,
        device_weights,
        device_bias,
        device_outputs,
        param);
    check_cuda(cudaGetLastError(), "launch kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize kernel");
    check_cuda(
        cudaMemcpy(
            outputs.data(),
            device_outputs,
            output_count * sizeof(SpikeScalar),
            cudaMemcpyDeviceToHost),
        "copy outputs");

    cpu_reference<TSteps>(inputs, weights, bias, reference, param);

    size_t mismatch_count = 0;
    float max_abs_error = 0.0F;
    for (size_t index = 0; index < output_count; ++index)
    {
        const float error = std::abs(
            spike_to_float(outputs[index]) - spike_to_float(reference[index]));
        max_abs_error = std::max(max_abs_error, error);
        mismatch_count += error > spike_tolerance();
    }

    check_cuda(cudaFree(device_inputs), "cudaFree inputs");
    check_cuda(cudaFree(device_weights), "cudaFree weights");
    check_cuda(cudaFree(device_bias), "cudaFree bias");
    check_cuda(cudaFree(device_outputs), "cudaFree outputs");

    std::cout << "[RESULT] " << std::left << std::setw(18) << test_case.name
              << " T=" << TSteps
              << " C=" << param.out_ch
              << " K=" << param.Kh << "x" << param.Kw
              << " outHW=" << param.outHW
              << " mismatch=" << mismatch_count
              << " max_abs_error=" << std::scientific << max_abs_error
              << std::defaultfloat << std::endl;
    return mismatch_count == 0;
}

bool dispatch_case(const TestCase &test_case)
{
    switch (test_case.timesteps)
    {
        case 1:
            return run_case<1>(test_case);
        case 2:
            return run_case<2>(test_case);
        case 4:
            return run_case<4>(test_case);
        case 8:
            return run_case<8>(test_case);
        default:
            std::cerr << "[ERROR] unsupported timesteps: "
                      << test_case.timesteps << std::endl;
            return false;
    }
}

}  // namespace

int main()
{
    std::cout << "[CONFIG] target=" << DWCONV_SPIKE_TARGET_NAME
              << " capacity=" << DWCONV_SPIKE_KERNEL_CAPACITY
              << " dtype=" << spike_dtype_name() << std::endl;
    std::cout << "\n[STAGE] correctness verification" << std::endl;

    const TestCase cases[] = {
        {"bias_t1", 1, 32, 17, 19, 3, 3, 1, 1, 1},
        {"stride2_t2", 2, 64, 31, 29, 3, 3, 2, 1, 1},
        {"no_padding_t4", 4, 32, 23, 21, 3, 3, 1, 0, 0},
        {"capacity_t8", 8, 32, 19, 21,
         DWCONV_SPIKE_BOUNDARY_KH, DWCONV_SPIKE_BOUNDARY_KW,
         1, 0, 0},
    };

    bool success = true;
    for (const TestCase &test_case : cases)
    {
        success = dispatch_case(test_case) && success;
    }

    std::cout << "\n" << (success ? "[SUCCESS]" : "[FAILED]")
              << " " << DWCONV_SPIKE_TARGET_NAME
              << " correctness verification" << std::endl;
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
