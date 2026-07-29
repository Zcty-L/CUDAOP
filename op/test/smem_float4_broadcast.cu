#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "ptx_utils.cuh"

namespace
{

constexpr int kWarpSize = 32;
constexpr int kValuesPerThread = 4;
constexpr int kWarmupIterations = 20;
constexpr int kBenchmarkSamples = 100;
constexpr int kBenchmarkGroups = 3;
constexpr int kLaunchesPerSample = 1000;

constexpr float kExpectedX = 1.25f;
constexpr float kExpectedY = -2.5f;
constexpr float kExpectedZ = 3.75f;
constexpr float kExpectedW = 4.5f;

enum class ProfileMode
{
    None,
    Scalar,
    Float2,
    Float4
};

void check_cuda(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

__global__ __launch_bounds__(kWarpSize, 1)
void smem_scalar_broadcast_control_kernel(float *output)
{
    __shared__ float shared_value;

    if (threadIdx.x == 0)
    {
        shared_value = kExpectedX;
    }
    __syncthreads();

    const uint32_t shared_address =
        ptx::smem_u32addr(&shared_value);
    float value;
    ptx::lds32(value, shared_address);
    output[threadIdx.x] = value;
}

__global__ __launch_bounds__(kWarpSize, 1)
void smem_float2_broadcast_control_kernel(float2 *output)
{
    __shared__ __align__(8) float2 shared_value;

    if (threadIdx.x == 0)
    {
        shared_value = make_float2(kExpectedX, kExpectedY);
    }
    __syncthreads();

    const uint32_t shared_address =
        ptx::smem_u32addr(&shared_value);
    float value_x;
    float value_y;
    ptx::lds64(value_x, value_y, shared_address);
    output[threadIdx.x] = make_float2(value_x, value_y);
}

__global__ __launch_bounds__(kWarpSize, 1)
void smem_float4_broadcast_kernel(float4 *output)
{
    __shared__ __align__(16) float4 shared_value;

    if (threadIdx.x == 0)
    {
        shared_value = make_float4(
            kExpectedX,
            kExpectedY,
            kExpectedZ,
            kExpectedW);
    }
    __syncthreads();

    const uint32_t shared_address =
        ptx::smem_u32addr(&shared_value);
    float value_x;
    float value_y;
    float value_z;
    float value_w;

    // All 32 lanes issue one LDS.128 from the same 16-byte shared address.
    ptx::lds128(
        value_x,
        value_y,
        value_z,
        value_w,
        shared_address);

    output[threadIdx.x] =
        make_float4(value_x, value_y, value_z, value_w);
}

void launch_kernel(float4 *output)
{
    smem_float4_broadcast_kernel<<<1, kWarpSize>>>(output);
}

bool validate_output(float4 *device_output)
{
    std::array<float4, kWarpSize> host_output{};
    check_cuda(
        cudaMemcpy(
            host_output.data(),
            device_output,
            sizeof(host_output),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(host_output)");

    const std::array<float, kValuesPerThread> expected =
    {
        kExpectedX,
        kExpectedY,
        kExpectedZ,
        kExpectedW
    };

    size_t error_count = 0;
    float max_absolute_error = 0.0f;
    for (int lane = 0; lane < kWarpSize; lane++)
    {
        const std::array<float, kValuesPerThread> actual =
        {
            host_output[lane].x,
            host_output[lane].y,
            host_output[lane].z,
            host_output[lane].w
        };

        for (int value = 0; value < kValuesPerThread; value++)
        {
            const float absolute_error =
                std::abs(actual[value] - expected[value]);
            max_absolute_error =
                std::max(max_absolute_error, absolute_error);
            if (absolute_error != 0.0f)
            {
                error_count++;
            }
        }
    }

    std::cout << "  [关键结果] errors=" << error_count
              << '/' << kWarpSize * kValuesPerThread
              << " max_abs=" << std::scientific
              << max_absolute_error << std::defaultfloat << '\n';
    return error_count == 0;
}

void calculate_statistics(
    std::vector<float> samples,
    double &mean,
    float &median,
    float &minimum,
    float &maximum,
    double &standard_deviation)
{
    std::sort(samples.begin(), samples.end());
    mean = std::accumulate(
        samples.begin(),
        samples.end(),
        0.0) /
        samples.size();
    median =
        0.5f *
        (samples[samples.size() / 2 - 1] +
         samples[samples.size() / 2]);
    minimum = samples.front();
    maximum = samples.back();

    double squared_sum = 0.0;
    for (float sample : samples)
    {
        const double difference = sample - mean;
        squared_sum += difference * difference;
    }
    standard_deviation =
        std::sqrt(squared_sum / samples.size());
}

float run_benchmark(float4 *device_output)
{
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    std::array<float, kBenchmarkGroups> group_medians{};
    for (int group = 0; group < kBenchmarkGroups; group++)
    {
        for (int warmup = 0;
             warmup < kWarmupIterations;
             warmup++)
        {
            for (int launch = 0;
                 launch < kLaunchesPerSample;
                 launch++)
            {
                launch_kernel(device_output);
            }
        }
        check_cuda(
            cudaGetLastError(),
            "warmup kernel launch");
        check_cuda(
            cudaDeviceSynchronize(),
            "warmup synchronization");

        std::vector<float> samples;
        samples.reserve(kBenchmarkSamples);
        for (int sample = 0;
             sample < kBenchmarkSamples;
             sample++)
        {
            check_cuda(
                cudaEventRecord(start),
                "cudaEventRecord(start)");
            for (int launch = 0;
                 launch < kLaunchesPerSample;
                 launch++)
            {
                launch_kernel(device_output);
            }
            check_cuda(
                cudaEventRecord(stop),
                "cudaEventRecord(stop)");
            check_cuda(
                cudaEventSynchronize(stop),
                "cudaEventSynchronize(stop)");

            float elapsed_ms = 0.0f;
            check_cuda(
                cudaEventElapsedTime(
                    &elapsed_ms,
                    start,
                    stop),
                "cudaEventElapsedTime");
            samples.push_back(
                elapsed_ms / kLaunchesPerSample);
        }
        check_cuda(
            cudaGetLastError(),
            "benchmark kernel launch");

        double mean = 0.0;
        float median = 0.0f;
        float minimum = 0.0f;
        float maximum = 0.0f;
        double standard_deviation = 0.0;
        calculate_statistics(
            samples,
            mean,
            median,
            minimum,
            maximum,
            standard_deviation);
        group_medians[group] = median;

        std::cout << "  group=" << group
                  << " mean=" << std::fixed
                  << std::setprecision(6) << mean
                  << " median=" << median
                  << " min=" << minimum
                  << " max=" << maximum
                  << " stddev=" << standard_deviation
                  << " ms\n";
    }

    check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");

    std::sort(group_medians.begin(), group_medians.end());
    const float final_latency =
        group_medians[kBenchmarkGroups / 2];
    const float spread =
        (group_medians.back() - group_medians.front()) /
        final_latency;

    std::cout << "  [关键结果] group_medians={"
              << group_medians[0] << ','
              << group_medians[1] << ','
              << group_medians[2] << "} ms"
              << " final_latency=" << final_latency
              << " ms spread=" << spread * 100.0f << "%\n"
              << std::defaultfloat;
    return spread;
}

} // namespace

int main(int argc, char **argv)
{
    bool correctness_only = false;
    ProfileMode profile_mode = ProfileMode::None;
    for (int index = 1; index < argc; index++)
    {
        if (std::strcmp(argv[index], "--correctness-only") == 0)
        {
            correctness_only = true;
        }
        else if (std::strcmp(argv[index], "--profile-only") == 0)
        {
            profile_mode = ProfileMode::Float4;
        }
        else if (std::strcmp(argv[index], "--profile-scalar") == 0)
        {
            profile_mode = ProfileMode::Scalar;
        }
        else if (std::strcmp(argv[index], "--profile-float2") == 0)
        {
            profile_mode = ProfileMode::Float2;
        }
        else
        {
            std::cout << "[FAILED] unknown argument: "
                      << argv[index] << '\n';
            return EXIT_FAILURE;
        }
    }

    float4 *device_output = nullptr;
    try
    {
        int device = 0;
        cudaDeviceProp properties{};
        check_cuda(cudaGetDevice(&device), "cudaGetDevice");
        check_cuda(
            cudaGetDeviceProperties(&properties, device),
            "cudaGetDeviceProperties");
        check_cuda(
            cudaMalloc(
                &device_output,
                kWarpSize * sizeof(float4)),
            "cudaMalloc(device_output)");

        std::cout << "\n=== Shared Memory float4 Broadcast Test ===\n\n"
                  << "[配置]\n"
                  << "  " << std::left << std::setw(28)
                  << "GPU" << properties.name << '\n'
                  << "  " << std::left << std::setw(28)
                  << "Compute capability"
                  << properties.major << '.'
                  << properties.minor << '\n'
                  << "  " << std::left << std::setw(28)
                  << "Threads" << kWarpSize << '\n'
                  << "  " << std::left << std::setw(28)
                  << "Shared address count" << 1 << '\n'
                  << "  " << std::left << std::setw(28)
                  << "Bytes per lane" << sizeof(float4) << '\n'
                  << "  " << std::left << std::setw(28)
                  << "Shared instruction"
                  << "ld.shared.v4.b32\n";

        if (profile_mode != ProfileMode::None)
        {
            std::cout << "\n[Profile 阶段] Launching one isolated kernel\n";
            if (profile_mode == ProfileMode::Scalar)
            {
                std::cout << "  Control: scalar broadcast\n";
                smem_scalar_broadcast_control_kernel
                    <<<1, kWarpSize>>>(
                        reinterpret_cast<float *>(device_output));
            }
            else if (profile_mode == ProfileMode::Float2)
            {
                std::cout << "  Control: float2 broadcast\n";
                smem_float2_broadcast_control_kernel
                    <<<1, kWarpSize>>>(
                        reinterpret_cast<float2 *>(device_output));
            }
            else
            {
                std::cout << "  Target: float4 broadcast\n";
                launch_kernel(device_output);
            }
            check_cuda(
                cudaGetLastError(),
                "profile kernel launch");
            check_cuda(
                cudaDeviceSynchronize(),
                "profile synchronization");
            std::cout << "\n[SUCCESS] Profile launch completed\n";
        }
        else
        {
            std::cout << "\n[正确性阶段] Launching broadcast load\n";
            launch_kernel(device_output);
            check_cuda(
                cudaGetLastError(),
                "correctness kernel launch");
            check_cuda(
                cudaDeviceSynchronize(),
                "correctness synchronization");
            if (!validate_output(device_output))
            {
                throw std::runtime_error(
                    "shared float4 broadcast validation failed");
            }
            std::cout << "  [SUCCESS] 正确性验证通过\n";

            if (!correctness_only)
            {
                std::cout
                    << "\n[性能阶段] warmup="
                    << kWarmupIterations
                    << " samples=" << kBenchmarkSamples
                    << " groups=" << kBenchmarkGroups
                    << " launches_per_sample="
                    << kLaunchesPerSample << '\n';
                const float spread =
                    run_benchmark(device_output);
                if (spread > 0.02f)
                {
                    throw std::runtime_error(
                        "group median spread exceeds 2%");
                }
                std::cout << "  [SUCCESS] 性能基线稳定\n";
            }

            std::cout
                << "\n[SUCCESS] Shared float4 broadcast test passed\n";
        }

        check_cuda(
            cudaFree(device_output),
            "cudaFree(device_output)");
        device_output = nullptr;
    }
    catch (const std::exception &error)
    {
        if (device_output != nullptr)
        {
            cudaFree(device_output);
        }
        std::cout << "\n[FAILED] " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
