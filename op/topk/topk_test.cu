#include "op/config.h"
#include "op/topk/topk.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                \
    do                                                                  \
    {                                                                   \
        const cudaError_t error = (call);                               \
        if (error != cudaSuccess)                                       \
        {                                                               \
            std::cerr << "[ERROR] " << __FILE__ << ":" << __LINE__      \
                      << " " << cudaGetErrorString(error) << std::endl;  \
            return false;                                               \
        }                                                               \
    } while (0)

namespace
{

bool values_equal(float left, float right)
{
    return (std::isnan(left) && std::isnan(right)) || left == right;
}

void fill_input(std::vector<float>& input, uint32_t seed)
{
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(
        -100.0F,
        100.0F);
    for (float& value : input)
    {
        value = distribution(generator);
    }
}

bool validate_case(const TopKParam& config, uint32_t seed)
{
    const size_t input_count =
        static_cast<size_t>(config.rows) * config.columns;
    const size_t output_count =
        static_cast<size_t>(config.rows) * config.k;
    std::vector<float> input(input_count);
    std::vector<float> expected_values(output_count);
    std::vector<int32_t> expected_indices(output_count);
    std::vector<float> actual_values(output_count);
    std::vector<int32_t> actual_indices(output_count);
    fill_input(input, seed);

    if (config.columns >= 8)
    {
        input[1] = 42.0F;
        input[3] = 42.0F;
        input[5] = std::numeric_limits<float>::infinity();
        input[7] = std::numeric_limits<float>::quiet_NaN();
    }

    cudaop::topk_cpu(
        input.data(),
        expected_values.data(),
        expected_indices.data(),
        config.rows,
        config.columns,
        config.k);

    float* device_input = nullptr;
    float* device_values = nullptr;
    int32_t* device_indices = nullptr;
    CUDA_CHECK(cudaMalloc(&device_input, input_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(
        &device_values,
        output_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(
        &device_indices,
        output_count * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(
        device_input,
        input.data(),
        input_count * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaop::topk_cuda(
        device_input,
        device_values,
        device_indices,
        config.rows,
        config.columns,
        config.k));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(
        actual_values.data(),
        device_values,
        output_count * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        actual_indices.data(),
        device_indices,
        output_count * sizeof(int32_t),
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(device_indices));
    CUDA_CHECK(cudaFree(device_values));
    CUDA_CHECK(cudaFree(device_input));

    size_t error_count = 0;
    for (size_t index = 0; index < output_count; ++index)
    {
        if (expected_indices[index] != actual_indices[index] ||
            !values_equal(expected_values[index], actual_values[index]))
        {
            if (error_count < 5)
            {
                std::cerr << "[ERROR] output=" << index
                          << " expected=(" << expected_values[index]
                          << ", " << expected_indices[index]
                          << ") actual=(" << actual_values[index]
                          << ", " << actual_indices[index]
                          << ")" << std::endl;
            }
            ++error_count;
        }
    }

    std::cout << "  " << std::left << std::setw(18)
              << (std::to_string(config.rows) + "x" +
                  std::to_string(config.columns) +
                  ", k=" + std::to_string(config.k))
              << " errors=" << error_count << std::endl;
    return error_count == 0;
}

bool validate_invalid_arguments()
{
    const cudaError_t error = cudaop::topk_cuda(
        nullptr,
        nullptr,
        nullptr,
        1,
        1,
        1);
    std::cout << "  " << std::left << std::setw(18)
              << "invalid arguments"
              << " status=" << cudaGetErrorName(error) << std::endl;
    return error == cudaErrorInvalidValue;
}

bool benchmark(const TopKParam& config)
{
    const size_t input_count =
        static_cast<size_t>(config.rows) * config.columns;
    const size_t output_count =
        static_cast<size_t>(config.rows) * config.k;
    std::vector<float> input(input_count);
    std::vector<float> cpu_values(output_count);
    std::vector<int32_t> cpu_indices(output_count);
    fill_input(input, 2026);

    float* device_input = nullptr;
    float* device_values = nullptr;
    int32_t* device_indices = nullptr;
    CUDA_CHECK(cudaMalloc(&device_input, input_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(
        &device_values,
        output_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(
        &device_indices,
        output_count * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(
        device_input,
        input.data(),
        input_count * sizeof(float),
        cudaMemcpyHostToDevice));

    for (uint32_t iteration = 0;
         iteration < config.warmup_iterations;
         ++iteration)
    {
        CUDA_CHECK(cudaop::topk_cuda(
            device_input,
            device_values,
            device_indices,
            config.rows,
            config.columns,
            config.k));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (uint32_t iteration = 0;
         iteration < config.benchmark_iterations;
         ++iteration)
    {
        CUDA_CHECK(cudaop::topk_cuda(
            device_input,
            device_values,
            device_indices,
            config.rows,
            config.columns,
            config.k));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    const auto cpu_start = std::chrono::steady_clock::now();
    cudaop::topk_cpu(
        input.data(),
        cpu_values.data(),
        cpu_indices.data(),
        config.rows,
        config.columns,
        config.k);
    const auto cpu_stop = std::chrono::steady_clock::now();
    const double cpu_ms =
        std::chrono::duration<double, std::milli>(
            cpu_stop - cpu_start)
            .count();
    const double cuda_ms =
        elapsed_ms / config.benchmark_iterations;

    std::cout << "  " << std::left << std::setw(18)
              << "CUDA average"
              << std::fixed << std::setprecision(4)
              << cuda_ms << " ms" << std::endl;
    std::cout << "  " << std::left << std::setw(18)
              << "CPU C++"
              << std::fixed << std::setprecision(4)
              << cpu_ms << " ms" << std::endl;
    std::cout << "  " << std::left << std::setw(18)
              << "speedup"
              << std::fixed << std::setprecision(2)
              << cpu_ms / cuda_ms << "x" << std::endl;

    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaFree(device_indices));
    CUDA_CHECK(cudaFree(device_values));
    CUDA_CHECK(cudaFree(device_input));
    return true;
}

}  // namespace

int main()
{
    std::cout << "\n[CONFIG] Top-K semantics" << std::endl;
    std::cout << "  " << std::left << std::setw(18)
              << "input"
              << "float32 [rows, columns]" << std::endl;
    std::cout << "  " << std::left << std::setw(18)
              << "order"
              << "value descending, index ascending" << std::endl;
    std::cout << "  " << std::left << std::setw(18)
              << "NaN"
              << "ordered before numeric values" << std::endl;

    std::cout << "\n[STAGE] Correctness validation" << std::endl;
    const std::vector<TopKParam> validation_configs = {
        {1, 1, 1, 0, 0},
        {3, 19, 5, 0, 0},
        {7, 1003, 17, 0, 0},
        {2, 257, 257, 0, 0},
    };
    bool success = true;
    for (size_t index = 0;
         index < validation_configs.size();
         ++index)
    {
        success =
            validate_case(validation_configs[index], 42 + index) &&
            success;
    }
    success = validate_invalid_arguments() && success;

    std::cout << "\n[STAGE] Performance benchmark" << std::endl;
    const TopKParam benchmark_config = {
        4096,
        1024,
        10,
        20,
        200,
    };
    std::cout << "  " << std::left << std::setw(18)
              << "shape"
              << benchmark_config.rows << "x"
              << benchmark_config.columns << std::endl;
    std::cout << "  " << std::left << std::setw(18)
              << "k"
              << benchmark_config.k << std::endl;
    std::cout << "  " << std::left << std::setw(18)
              << "measurement"
              << "kernel only, GPU-resident data" << std::endl;
    std::cout << "  " << std::left << std::setw(18)
              << "iterations"
              << benchmark_config.warmup_iterations << " warmup, "
              << benchmark_config.benchmark_iterations
              << " measured" << std::endl;
    success = benchmark(benchmark_config) && success;

    if (!success)
    {
        std::cerr << "\n[FAILED] Top-K validation failed" << std::endl;
        return 1;
    }

    std::cout << "\n[SUCCESS] Top-K validation passed" << std::endl;
    return 0;
}
