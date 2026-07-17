#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "config.h"

void snn_conv2d_fp16acc_1x1_s1_launch(
    const uint8_t *d_inputs,
    const __half *d_weights_padded,
    const __half2 *d_bias,
    __half *d_outputs,
    Conv2DParam &param,
    int T,
    int out_ch_padded);

void snn_conv2d_fp16acc_3x3_s1_launch(
    const uint8_t *d_inputs,
    const __half *d_weights_padded,
    const __half2 *d_bias,
    __half *d_outputs,
    Conv2DParam &param,
    int T,
    int out_ch_padded);

void snn_conv2d_fp16acc_3x3_s2_launch(
    const uint8_t *d_inputs,
    const __half *d_weights_padded,
    const __half2 *d_bias,
    __half *d_outputs,
    Conv2DParam &param,
    int T,
    int out_ch_padded);

void snn_conv2d_sn_fp16_1x1_s1_launch(
    const uint8_t *d_in,
    const __half *d_w,
    const __half2 *d_bias,
    uint8_t *d_out,
    Conv2DParam &param,
    int T,
    int C_out_padded,
    float v_th,
    float tau);

void snn_conv2d_sn_fp16_3x3_s1_launch(
    const uint8_t *d_in,
    const __half *d_w,
    const __half2 *d_bias,
    uint8_t *d_out,
    Conv2DParam &param,
    int T,
    int C_out_padded,
    float v_th,
    float tau);

void snn_conv2d_sn_fp16_3x3_s2_launch(
    const uint8_t *d_in,
    const __half *d_w,
    const __half2 *d_bias,
    uint8_t *d_out,
    Conv2DParam &param,
    int T,
    int C_out_padded,
    float v_th,
    float tau);

namespace
{

constexpr int STABILIZATION_LAUNCHES = 5000;
constexpr int WARMUP_SAMPLES = 20;
constexpr int BENCH_SAMPLES = 100;
constexpr int BENCH_GROUPS = 3;
constexpr int BATCH_SIZE = 10;

enum class Variant
{
    K1x1S1,
    K3x3S1,
    K3x3S2
};

struct TestCase
{
    const char *name;
    Variant variant;
    int T;
    int C_in;
    int H;
    int W;
    int C_out;
    int Kh;
    int Kw;
    int Sh;
    int Sw;
    int Ph;
    int Pw;
    float v_th;
    float tau;
};

struct Stats
{
    double mean_ms = 0.0;
    double median_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double stddev_ms = 0.0;
};

struct BenchmarkResult
{
    std::array<Stats, BENCH_GROUPS> groups;
    double latency_ms = 0.0;
    double group_median_spread = 0.0;
};

struct CompareResult
{
    size_t errors = 0;
    double max_abs = 0.0;
    double max_rel = 0.0;
};

struct DeviceBuffers
{
    uint8_t *inputs = nullptr;
    __half *weights = nullptr;
    __half2 *bias = nullptr;
    __half *conv_outputs = nullptr;
    uint8_t *separate_outputs = nullptr;
    uint8_t *fused_outputs = nullptr;

    ~DeviceBuffers()
    {
        cudaFree(inputs);
        cudaFree(weights);
        cudaFree(bias);
        cudaFree(conv_outputs);
        cudaFree(separate_outputs);
        cudaFree(fused_outputs);
    }
};

template <int T_STEPS>
__global__ void lif_hard_reset_packed_fp16_kernel(
    const __half * __restrict__ conv_outputs,
    uint8_t * __restrict__ outputs,
    size_t output_numel,
    float v_th,
    float tau)
{
    static_assert(T_STEPS >= 1 && T_STEPS <= 8);

    size_t base =
        (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
    if (base >= output_numel)
    {
        return;
    }

    const __half2 zero2 = __float2half2_rn(0.0f);
    const __half2 v_th2 = __float2half2_rn(v_th);
    const __half2 tau2 = __float2half2_rn(tau);
    __half2 v[4];
    uint8_t packed[8] = {};

#pragma unroll
    for (int pair = 0; pair < 4; pair++)
    {
        size_t index = base + pair * 2;
        if (index + 1 < output_numel)
        {
            v[pair] = reinterpret_cast<const __half2 *>(conv_outputs + index)[0];
        }
        else if (index < output_numel)
        {
            v[pair] = __halves2half2(conv_outputs[index], __float2half(0.0f));
        }
        else
        {
            v[pair] = zero2;
        }

        v[pair] = __hmul2(v[pair], tau2);
        uint32_t spike_mask = __hge2_mask(v[pair], v_th2);
        packed[pair * 2] = static_cast<uint8_t>(spike_mask & 1);
        packed[pair * 2 + 1] =
            static_cast<uint8_t>((spike_mask >> 16) & 1);

        uint32_t state_bits = *reinterpret_cast<uint32_t *>(&v[pair]);
        state_bits &= ~spike_mask;
        v[pair] = *reinterpret_cast<__half2 *>(&state_bits);
    }

#pragma unroll
    for (int t = 1; t < T_STEPS; t++)
    {
#pragma unroll
        for (int pair = 0; pair < 4; pair++)
        {
            size_t index = base + pair * 2;
            size_t time_offset = static_cast<size_t>(t) * output_numel;
            __half2 input2 = zero2;
            if (index + 1 < output_numel)
            {
                input2 = reinterpret_cast<const __half2 *>(
                    conv_outputs + time_offset + index)[0];
            }
            else if (index < output_numel)
            {
                input2 = __halves2half2(
                    conv_outputs[time_offset + index],
                    __float2half(0.0f));
            }

            v[pair] = __hadd2(
                v[pair],
                __hmul2(__hsub2(input2, v[pair]), tau2));
            uint32_t spike_mask = __hge2_mask(v[pair], v_th2);
            packed[pair * 2] |= static_cast<uint8_t>(
                (spike_mask & 1) << t);
            packed[pair * 2 + 1] |= static_cast<uint8_t>(
                ((spike_mask >> 16) & 1) << t);

            uint32_t state_bits = *reinterpret_cast<uint32_t *>(&v[pair]);
            state_bits &= ~spike_mask;
            v[pair] = *reinterpret_cast<__half2 *>(&state_bits);
        }
    }

#pragma unroll
    for (int index = 0; index < 8; index++)
    {
        if (base + index < output_numel)
        {
            outputs[base + index] = packed[index];
        }
    }
}

bool check_cuda(cudaError_t error, const char *stage)
{
    if (error == cudaSuccess)
    {
        return true;
    }

    std::cout << "[FAILED] " << stage << ": "
              << cudaGetErrorString(error) << '\n';
    return false;
}

void pad_weights(
    const std::vector<float> &source,
    std::vector<__half> &destination,
    int in_features,
    int C_out,
    int in_features_padded,
    int C_out_padded)
{
    std::fill(destination.begin(), destination.end(), __float2half(0.0f));
    for (int k = 0; k < in_features; k++)
    {
        for (int m = 0; m < C_out; m++)
        {
            destination[static_cast<size_t>(k) * C_out_padded + m] =
                __float2half(source[static_cast<size_t>(k) * C_out + m]);
        }
    }
}

void pack_bias(
    const std::vector<float> &source,
    std::vector<__half2> &destination,
    int C_out_padded)
{
    for (int m = 0; m < C_out_padded; m += 2)
    {
        __half low = m < static_cast<int>(source.size())
            ? __float2half(source[m])
            : __float2half(0.0f);
        __half high = m + 1 < static_cast<int>(source.size())
            ? __float2half(source[m + 1])
            : __float2half(0.0f);
        destination[m / 2] = __halves2half2(low, high);
    }
}

void launch_conv(
    const TestCase &test_case,
    const uint8_t *inputs,
    const __half *weights,
    const __half2 *bias,
    __half *outputs,
    Conv2DParam &param,
    int C_out_padded)
{
    switch (test_case.variant)
    {
        case Variant::K1x1S1:
            snn_conv2d_fp16acc_1x1_s1_launch(
                inputs, weights, bias, outputs, param,
                test_case.T, C_out_padded);
            break;
        case Variant::K3x3S1:
            snn_conv2d_fp16acc_3x3_s1_launch(
                inputs, weights, bias, outputs, param,
                test_case.T, C_out_padded);
            break;
        case Variant::K3x3S2:
            snn_conv2d_fp16acc_3x3_s2_launch(
                inputs, weights, bias, outputs, param,
                test_case.T, C_out_padded);
            break;
    }
}

void launch_lif(
    int T,
    const __half *conv_outputs,
    uint8_t *outputs,
    size_t output_numel,
    float v_th,
    float tau)
{
    constexpr int BLOCK_SIZE = 256;
    size_t vector_count = (output_numel + 7) / 8;
    int grid_size = static_cast<int>(
        (vector_count + BLOCK_SIZE - 1) / BLOCK_SIZE);

    switch (T)
    {
        case 1:
            lif_hard_reset_packed_fp16_kernel<1>
                <<<grid_size, BLOCK_SIZE>>>(
                    conv_outputs, outputs, output_numel, v_th, tau);
            break;
        case 2:
            lif_hard_reset_packed_fp16_kernel<2>
                <<<grid_size, BLOCK_SIZE>>>(
                    conv_outputs, outputs, output_numel, v_th, tau);
            break;
        case 3:
            lif_hard_reset_packed_fp16_kernel<3>
                <<<grid_size, BLOCK_SIZE>>>(
                    conv_outputs, outputs, output_numel, v_th, tau);
            break;
        case 4:
            lif_hard_reset_packed_fp16_kernel<4>
                <<<grid_size, BLOCK_SIZE>>>(
                    conv_outputs, outputs, output_numel, v_th, tau);
            break;
        default:
            std::cout << "[FAILED] unsupported T=" << T << '\n';
            break;
    }
}

void launch_fused(
    const TestCase &test_case,
    const uint8_t *inputs,
    const __half *weights,
    const __half2 *bias,
    uint8_t *outputs,
    Conv2DParam &param,
    int C_out_padded)
{
    switch (test_case.variant)
    {
        case Variant::K1x1S1:
            snn_conv2d_sn_fp16_1x1_s1_launch(
                inputs, weights, bias, outputs, param,
                test_case.T, C_out_padded, test_case.v_th, test_case.tau);
            break;
        case Variant::K3x3S1:
            snn_conv2d_sn_fp16_3x3_s1_launch(
                inputs, weights, bias, outputs, param,
                test_case.T, C_out_padded, test_case.v_th, test_case.tau);
            break;
        case Variant::K3x3S2:
            snn_conv2d_sn_fp16_3x3_s2_launch(
                inputs, weights, bias, outputs, param,
                test_case.T, C_out_padded, test_case.v_th, test_case.tau);
            break;
    }
}

void cpu_reference(
    const TestCase &test_case,
    const std::vector<uint8_t> &inputs,
    const std::vector<__half> &padded_weights,
    const std::vector<float> &bias,
    int in_features_padded,
    int C_out_padded,
    std::vector<uint8_t> &outputs)
{
    int H_out =
        (test_case.H + 2 * test_case.Ph - test_case.Kh) / test_case.Sh + 1;
    int W_out =
        (test_case.W + 2 * test_case.Pw - test_case.Kw) / test_case.Sw + 1;
    int KhKw = test_case.Kh * test_case.Kw;
    int in_features = test_case.C_in * KhKw;
    __half tau = __float2half(test_case.tau);
    __half threshold = __float2half(test_case.v_th);

    std::fill(outputs.begin(), outputs.end(), 0);
    for (int m = 0; m < test_case.C_out; m++)
    {
        for (int oh = 0; oh < H_out; oh++)
        {
            for (int ow = 0; ow < W_out; ow++)
            {
                std::array<__half, 8> conv = {};
                for (int t = 0; t < test_case.T; t++)
                {
                    __half accumulator = __float2half(bias[m]);
                    for (int k = 0; k < in_features_padded; k++)
                    {
                        int spike = 0;
                        if (k < in_features)
                        {
                            int c = k / KhKw;
                            int ky = k % KhKw / test_case.Kw;
                            int kx = k % KhKw % test_case.Kw;
                            int ih = oh * test_case.Sh - test_case.Ph + ky;
                            int iw = ow * test_case.Sw - test_case.Pw + kx;
                            if (ih >= 0 && ih < test_case.H &&
                                iw >= 0 && iw < test_case.W)
                            {
                                uint8_t packed = inputs[
                                    static_cast<size_t>(c) * test_case.H *
                                    test_case.W + ih * test_case.W + iw];
                                spike = (packed >> t) & 1;
                            }
                        }
                        if (spike != 0)
                        {
                            accumulator = __hadd(
                                accumulator,
                                padded_weights[
                                    static_cast<size_t>(k) * C_out_padded + m]);
                        }
                    }
                    conv[t] = accumulator;
                }

                __half v = __hmul(conv[0], tau);
                int spike = __half2float(v) >= __half2float(threshold) ? 1 : 0;
                uint8_t packed = static_cast<uint8_t>(spike);
                if (spike != 0)
                {
                    v = __float2half(0.0f);
                }

                for (int t = 1; t < test_case.T; t++)
                {
                    __half difference = __hsub(conv[t], v);
                    __half update = __hmul(difference, tau);
                    v = __hadd(v, update);
                    spike =
                        __half2float(v) >= __half2float(threshold) ? 1 : 0;
                    packed |= static_cast<uint8_t>(spike << t);
                    if (spike != 0)
                    {
                        v = __float2half(0.0f);
                    }
                }

                outputs[static_cast<size_t>(m) * H_out * W_out +
                        oh * W_out + ow] = packed;
            }
        }
    }
}

CompareResult compare_outputs(
    const std::vector<uint8_t> &actual,
    const std::vector<uint8_t> &reference)
{
    CompareResult result;
    for (size_t index = 0; index < actual.size(); index++)
    {
        double difference = std::abs(
            static_cast<double>(actual[index]) -
            static_cast<double>(reference[index]));
        double relative = difference /
            std::max(1.0, std::abs(static_cast<double>(reference[index])));
        result.max_abs = std::max(result.max_abs, difference);
        result.max_rel = std::max(result.max_rel, relative);

        if (actual[index] != reference[index])
        {
            if (result.errors < 5)
            {
                std::cout << "    mismatch[" << index << "] actual=0x"
                          << std::hex << static_cast<int>(actual[index])
                          << " reference=0x"
                          << static_cast<int>(reference[index])
                          << std::dec << '\n';
            }
            result.errors++;
        }
    }
    return result;
}

Stats calculate_stats(const std::vector<float> &samples)
{
    Stats stats;
    std::vector<float> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    stats.mean_ms = std::accumulate(
        samples.begin(), samples.end(), 0.0) / samples.size();
    stats.median_ms =
        0.5 * (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]);
    stats.min_ms = sorted.front();
    stats.max_ms = sorted.back();

    double squared_sum = 0.0;
    for (float sample : samples)
    {
        double delta = sample - stats.mean_ms;
        squared_sum += delta * delta;
    }
    stats.stddev_ms = std::sqrt(squared_sum / samples.size());
    return stats;
}

template <typename Launch>
bool benchmark_implementation(
    const char *name,
    Launch launch,
    BenchmarkResult &result)
{
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (!check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)") ||
        !check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)"))
    {
        if (start != nullptr)
        {
            cudaEventDestroy(start);
        }
        return false;
    }

    std::cout << "  [性能阶段] " << name
              << " stabilization=" << STABILIZATION_LAUNCHES
              << " warmup=" << WARMUP_SAMPLES
              << " samples=" << BENCH_SAMPLES
              << " batch=" << BATCH_SIZE << '\n';

    for (int index = 0; index < STABILIZATION_LAUNCHES; index++)
    {
        launch();
    }
    if (!check_cuda(cudaGetLastError(), "stabilization launch") ||
        !check_cuda(
            cudaDeviceSynchronize(),
            "stabilization synchronize"))
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return false;
    }

    for (int group = 0; group < BENCH_GROUPS; group++)
    {
        for (int warmup = 0; warmup < WARMUP_SAMPLES; warmup++)
        {
            for (int batch = 0; batch < BATCH_SIZE; batch++)
            {
                launch();
            }
        }
        if (!check_cuda(cudaGetLastError(), "warmup launch") ||
            !check_cuda(cudaDeviceSynchronize(), "warmup synchronize"))
        {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return false;
        }

        std::vector<float> samples;
        samples.reserve(BENCH_SAMPLES);
        for (int sample = 0; sample < BENCH_SAMPLES; sample++)
        {
            if (!check_cuda(cudaEventRecord(start), "cudaEventRecord(start)"))
            {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return false;
            }
            for (int batch = 0; batch < BATCH_SIZE; batch++)
            {
                launch();
            }
            if (!check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)") ||
                !check_cuda(
                    cudaEventSynchronize(stop),
                    "cudaEventSynchronize(stop)"))
            {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return false;
            }

            float elapsed_ms = 0.0f;
            if (!check_cuda(
                    cudaEventElapsedTime(&elapsed_ms, start, stop),
                    "cudaEventElapsedTime"))
            {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return false;
            }
            samples.push_back(elapsed_ms / BATCH_SIZE);
        }

        result.groups[group] = calculate_stats(samples);
        const Stats &stats = result.groups[group];
        std::cout << "    group=" << group
                  << " mean=" << stats.mean_ms
                  << " median=" << stats.median_ms
                  << " min=" << stats.min_ms
                  << " max=" << stats.max_ms
                  << " stddev=" << stats.stddev_ms
                  << " ms\n";
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::array<double, BENCH_GROUPS> medians;
    for (int group = 0; group < BENCH_GROUPS; group++)
    {
        medians[group] = result.groups[group].median_ms;
    }
    std::sort(medians.begin(), medians.end());
    result.latency_ms = medians[BENCH_GROUPS / 2];
    result.group_median_spread =
        (medians.back() - medians.front()) / result.latency_ms;
    return true;
}

bool allocate_device_buffers(
    DeviceBuffers &buffers,
    size_t input_size,
    size_t weight_size,
    size_t bias_size,
    size_t conv_output_size,
    size_t output_size)
{
    return
        check_cuda(
            cudaMalloc(&buffers.inputs, input_size * sizeof(uint8_t)),
            "cudaMalloc(inputs)") &&
        check_cuda(
            cudaMalloc(&buffers.weights, weight_size * sizeof(__half)),
            "cudaMalloc(weights)") &&
        check_cuda(
            cudaMalloc(&buffers.bias, bias_size * sizeof(__half2)),
            "cudaMalloc(bias)") &&
        check_cuda(
            cudaMalloc(
                &buffers.conv_outputs,
                conv_output_size * sizeof(__half)),
            "cudaMalloc(conv_outputs)") &&
        check_cuda(
            cudaMalloc(
                &buffers.separate_outputs,
                output_size * sizeof(uint8_t)),
            "cudaMalloc(separate_outputs)") &&
        check_cuda(
            cudaMalloc(
                &buffers.fused_outputs,
                output_size * sizeof(uint8_t)),
            "cudaMalloc(fused_outputs)");
}

bool run_case(
    const TestCase &test_case,
    bool correctness_only,
    bool profile_only,
    const std::string &profile_impl)
{
    constexpr int K_CHUNK = 16;
    int H_out =
        (test_case.H + 2 * test_case.Ph - test_case.Kh) / test_case.Sh + 1;
    int W_out =
        (test_case.W + 2 * test_case.Pw - test_case.Kw) / test_case.Sw + 1;
    int in_features = test_case.C_in * test_case.Kh * test_case.Kw;
    int in_features_padded =
        (in_features + K_CHUNK - 1) / K_CHUNK * K_CHUNK;
    int C_out_padded = (test_case.C_out + 63) / 64 * 64;
    size_t output_numel =
        static_cast<size_t>(test_case.C_out) * H_out * W_out;
    size_t input_size =
        static_cast<size_t>(test_case.C_in) * test_case.H * test_case.W;
    size_t weight_size =
        static_cast<size_t>(in_features) * test_case.C_out;
    size_t padded_weight_size =
        static_cast<size_t>(in_features_padded) * C_out_padded;
    size_t conv_output_size = output_numel * test_case.T;
    int grid_blocks =
        ((H_out * W_out + 63) / 64) * ((test_case.C_out + 63) / 64);

    std::cout << "\n[配置] " << test_case.name
              << " dtype=input:uint8 weight:fp16 conv:fp16 output:uint8"
              << " T=" << test_case.T
              << " C_in=" << test_case.C_in
              << " HxW=" << test_case.H << 'x' << test_case.W
              << " C_out=" << test_case.C_out
              << " K=" << test_case.Kh << 'x' << test_case.Kw
              << " stride=" << test_case.Sh
              << " padding=" << test_case.Ph
              << " H_outxW_out=" << H_out << 'x' << W_out
              << " v_th=" << test_case.v_th
              << " v_reset=0.000000"
              << " tau=" << test_case.tau << '\n'
              << "  grid_blocks=" << grid_blocks << '\n';

    std::vector<uint8_t> host_inputs(input_size);
    std::vector<float> host_weights(weight_size);
    std::vector<__half> padded_weights(padded_weight_size);
    std::vector<float> host_bias(test_case.C_out);
    std::vector<__half2> packed_bias(C_out_padded / 2);
    std::vector<uint8_t> separate_outputs(output_numel);
    std::vector<uint8_t> fused_outputs(output_numel);
    std::vector<uint8_t> reference_outputs(output_numel);

    std::mt19937 random_engine(42);
    std::uniform_int_distribution<int> spike_distribution(0, 1);
    std::uniform_real_distribution<float> weight_distribution(-0.125f, 0.125f);
    std::uniform_real_distribution<float> bias_distribution(-0.25f, 0.25f);

    for (uint8_t &value : host_inputs)
    {
        uint8_t packed = 0;
        for (int t = 0; t < test_case.T; t++)
        {
            packed |= static_cast<uint8_t>(
                spike_distribution(random_engine) << t);
        }
        value = packed;
    }
    for (float &value : host_weights)
    {
        value = weight_distribution(random_engine);
    }
    for (float &value : host_bias)
    {
        value = bias_distribution(random_engine);
    }
    pad_weights(
        host_weights,
        padded_weights,
        in_features,
        test_case.C_out,
        in_features_padded,
        C_out_padded);
    pack_bias(host_bias, packed_bias, C_out_padded);

    DeviceBuffers buffers;
    if (!allocate_device_buffers(
            buffers,
            input_size,
            padded_weight_size,
            packed_bias.size(),
            conv_output_size,
            output_numel))
    {
        return false;
    }

    if (!check_cuda(
            cudaMemcpy(
                buffers.inputs,
                host_inputs.data(),
                input_size * sizeof(uint8_t),
                cudaMemcpyHostToDevice),
            "cudaMemcpy(inputs)") ||
        !check_cuda(
            cudaMemcpy(
                buffers.weights,
                padded_weights.data(),
                padded_weight_size * sizeof(__half),
                cudaMemcpyHostToDevice),
            "cudaMemcpy(weights)") ||
        !check_cuda(
            cudaMemcpy(
                buffers.bias,
                packed_bias.data(),
                packed_bias.size() * sizeof(__half2),
                cudaMemcpyHostToDevice),
            "cudaMemcpy(bias)"))
    {
        return false;
    }

    Conv2DParam param = {};
    param.in_h = test_case.H;
    param.in_w = test_case.W;
    param.inHW = test_case.H * test_case.W;
    param.inChKhKw = in_features_padded;
    param.inBatchNumel = test_case.C_in * test_case.H * test_case.W;
    param.out_ch = test_case.C_out;
    param.out_h = H_out;
    param.out_w = W_out;
    param.outHW = H_out * W_out;
    param.outBatchNumel = test_case.C_out * H_out * W_out;
    param.Kh = test_case.Kh;
    param.Kw = test_case.Kw;
    param.KhKw = test_case.Kh * test_case.Kw;
    param.Sh = test_case.Sh;
    param.Sw = test_case.Sw;
    param.Ph = test_case.Ph;
    param.Pw = test_case.Pw;

    auto separate_launch = [&]()
    {
        launch_conv(
            test_case,
            buffers.inputs,
            buffers.weights,
            buffers.bias,
            buffers.conv_outputs,
            param,
            C_out_padded);
        launch_lif(
            test_case.T,
            buffers.conv_outputs,
            buffers.separate_outputs,
            output_numel,
            test_case.v_th,
            test_case.tau);
    };

    auto fused_launch = [&]()
    {
        launch_fused(
            test_case,
            buffers.inputs,
            buffers.weights,
            buffers.bias,
            buffers.fused_outputs,
            param,
            C_out_padded);
    };

    if (profile_only)
    {
        if (profile_impl == "separate")
        {
            separate_launch();
        }
        else
        {
            fused_launch();
        }
        bool success =
            check_cuda(cudaGetLastError(), "profile launch") &&
            check_cuda(cudaDeviceSynchronize(), "profile synchronize");
        std::cout << (success ? "[SUCCESS]" : "[FAILED]")
                  << " profile launch impl=" << profile_impl << '\n';
        return success;
    }

    std::cout << "  [正确性阶段] CPU reference -> separate -> fused\n";
    cpu_reference(
        test_case,
        host_inputs,
        padded_weights,
        host_bias,
        in_features_padded,
        C_out_padded,
        reference_outputs);

    separate_launch();
    if (!check_cuda(cudaGetLastError(), "separate launch") ||
        !check_cuda(cudaDeviceSynchronize(), "separate synchronize"))
    {
        return false;
    }
    fused_launch();
    if (!check_cuda(cudaGetLastError(), "fused launch") ||
        !check_cuda(cudaDeviceSynchronize(), "fused synchronize"))
    {
        return false;
    }

    if (!check_cuda(
            cudaMemcpy(
                separate_outputs.data(),
                buffers.separate_outputs,
                output_numel * sizeof(uint8_t),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(separate_outputs)") ||
        !check_cuda(
            cudaMemcpy(
                fused_outputs.data(),
                buffers.fused_outputs,
                output_numel * sizeof(uint8_t),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(fused_outputs)"))
    {
        return false;
    }

    CompareResult separate_compare =
        compare_outputs(separate_outputs, reference_outputs);
    CompareResult fused_compare =
        compare_outputs(fused_outputs, reference_outputs);
    CompareResult cross_compare =
        compare_outputs(fused_outputs, separate_outputs);

    std::cout << "    separate errors=" << separate_compare.errors
              << '/' << output_numel
              << " max_abs=" << separate_compare.max_abs
              << " max_rel=" << separate_compare.max_rel << '\n'
              << "    fused   errors=" << fused_compare.errors
              << '/' << output_numel
              << " max_abs=" << fused_compare.max_abs
              << " max_rel=" << fused_compare.max_rel << '\n'
              << "    cross   errors=" << cross_compare.errors
              << '/' << output_numel
              << " max_abs=" << cross_compare.max_abs
              << " max_rel=" << cross_compare.max_rel << '\n';

    bool correctness_success =
        separate_compare.errors == 0 &&
        fused_compare.errors == 0 &&
        cross_compare.errors == 0;
    if (!correctness_success)
    {
        std::cout << "[FAILED] correctness " << test_case.name << '\n';
        return false;
    }
    std::cout << "  [SUCCESS] correctness " << test_case.name << '\n';

    if (correctness_only)
    {
        return true;
    }

    BenchmarkResult separate_benchmark;
    BenchmarkResult fused_benchmark;
    if (!benchmark_implementation(
            "separate FP16 Conv + LIF",
            separate_launch,
            separate_benchmark) ||
        !benchmark_implementation(
            "fused FP16 Conv+LIF",
            fused_launch,
            fused_benchmark))
    {
        return false;
    }

    double speedup =
        separate_benchmark.latency_ms / fused_benchmark.latency_ms;
    double improvement =
        (1.0 - fused_benchmark.latency_ms /
                   separate_benchmark.latency_ms) * 100.0;

    std::cout << "  [关键结果] case=" << test_case.name
              << " separate_ms=" << separate_benchmark.latency_ms
              << " fused_ms=" << fused_benchmark.latency_ms
              << " speedup=" << speedup << 'x'
              << " improvement=" << improvement << '%'
              << " separate_spread="
              << separate_benchmark.group_median_spread * 100.0 << '%'
              << " fused_spread="
              << fused_benchmark.group_median_spread * 100.0 << "%\n";

    bool stable =
        separate_benchmark.group_median_spread <= 0.02 &&
        fused_benchmark.group_median_spread <= 0.02;
    if (!stable)
    {
        std::cout << "[FAILED] group median spread exceeds 2% for "
                  << test_case.name << '\n';
        return false;
    }

    std::cout << "  [SUCCESS] benchmark " << test_case.name << '\n';
    return true;
}

bool parse_value(
    const std::string &argument,
    const std::string &prefix,
    std::string &value)
{
    if (argument.rfind(prefix, 0) != 0)
    {
        return false;
    }
    value = argument.substr(prefix.size());
    return true;
}

} // namespace

int main(int argc, char **argv)
{
    bool correctness_only = false;
    bool profile_only = false;
    std::string selected_case;
    std::string profile_impl = "fused";

    for (int index = 1; index < argc; index++)
    {
        std::string argument = argv[index];
        if (argument == "--correctness-only")
        {
            correctness_only = true;
        }
        else if (argument == "--profile-only")
        {
            profile_only = true;
        }
        else if (argument == "--case" && index + 1 < argc)
        {
            selected_case = argv[++index];
        }
        else if (argument == "--impl" && index + 1 < argc)
        {
            profile_impl = argv[++index];
        }
        else if (parse_value(argument, "--case=", selected_case))
        {
        }
        else if (parse_value(argument, "--impl=", profile_impl))
        {
        }
        else
        {
            std::cout << "[FAILED] unknown argument: " << argument << '\n';
            return EXIT_FAILURE;
        }
    }

    if (profile_impl != "separate" && profile_impl != "fused")
    {
        std::cout << "[FAILED] --impl must be separate or fused\n";
        return EXIT_FAILURE;
    }
    if (profile_only && selected_case.empty())
    {
        selected_case = "throughput_1x1";
    }

    const std::array<TestCase, 3> test_cases = {{
        {
            "throughput_1x1",
            Variant::K1x1S1,
            4,
            16,
            128,
            128,
            512,
            1,
            1,
            1,
            1,
            0,
            0,
            0.5f,
            0.5f
        },
        {
            "business_3x3_s1",
            Variant::K3x3S1,
            4,
            64,
            80,
            80,
            64,
            3,
            3,
            1,
            1,
            1,
            1,
            1.0f,
            0.5f
        },
        {
            "boundary_3x3_s2",
            Variant::K3x3S2,
            3,
            32,
            43,
            43,
            48,
            3,
            3,
            2,
            2,
            1,
            1,
            0.75f,
            0.25f
        }
    }};

    std::cout << std::fixed << std::setprecision(6)
              << "=== FP16 Conv2D + LIF fusion comparison ===\n"
              << "GPU device=0, random_seed=42\n"
              << "test1=separate snn_conv2d_64x64_k16_fp16_u8 + FP16 LIF\n"
              << "test2=fused snn_conv2d_lif_64x64_k16_fp16\n"
              << "timing=CUDA Event end-to-end, stabilization="
              << STABILIZATION_LAUNCHES
              << ", warmup=" << WARMUP_SAMPLES
              << ", samples=" << BENCH_SAMPLES
              << ", groups=" << BENCH_GROUPS
              << ", batch=" << BATCH_SIZE << '\n';

    bool found_case = selected_case.empty();
    bool success = true;
    for (const TestCase &test_case : test_cases)
    {
        if (!selected_case.empty() && selected_case != test_case.name)
        {
            continue;
        }
        found_case = true;
        success = run_case(
            test_case,
            correctness_only,
            profile_only,
            profile_impl) && success;
    }

    if (!found_case)
    {
        std::cout << "[FAILED] unknown case: " << selected_case << '\n';
        return EXIT_FAILURE;
    }

    std::cout << '\n'
              << (success ? "[SUCCESS]" : "[FAILED]")
              << " FP16 Conv2D + LIF fusion comparison\n";
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
