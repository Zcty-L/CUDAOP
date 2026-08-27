#include "op/softmax/softmax.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace
{

constexpr int WARP_SIZE = cudaop::kSoftmaxWarpSize;
constexpr int BLOCK_THREADS = cudaop::kSoftmaxBlockThreads;
constexpr int BLOCK_MAX_COLS = cudaop::kSoftmaxBlockMaxColumns;
constexpr int ONLINE_VECTOR_SIZE = cudaop::kSoftmaxOnlineVectorSize;
constexpr int INT8_MAX_COLS = cudaop::kSoftmaxInt8MaxColumns;
constexpr float INT8_OUTPUT_SCALE = cudaop::kSoftmaxInt8OutputScale;
constexpr int32_t INT8_OUTPUT_ZERO_POINT =
    cudaop::kSoftmaxInt8OutputZeroPoint;

constexpr int BENCHMARK_ROWS = 512;
constexpr int BENCHMARK_COLS = 1024;
constexpr int ONLINE_BENCHMARK_COLS = 4 * 4096 + 3;
constexpr int64_t ONLINE_LONG_COLS =
    static_cast<int64_t>(BLOCK_THREADS) * 10240 + 3;
constexpr int WARMUP_ITERATIONS = 20;
constexpr int BENCHMARK_ITERATIONS = 200;

using cudaop::launch_softmax;
using cudaop::launch_softmax_int8_to_float;
using cudaop::launch_softmax_int8_to_int8;

bool check_cuda(cudaError_t status, const char *operation)
{
    if (status == cudaSuccess)
    {
        return true;
    }

    std::cout << "[CUDA ERROR] " << operation << ": "
              << cudaGetErrorString(status) << '\n';
    return false;
}

void softmax_cpu(
    const std::vector<float> &source,
    std::vector<float> &destination,
    int64_t rows,
    int64_t cols)
{
    for (int64_t row = 0; row < rows; row++)
    {
        const int64_t row_offset = row * cols;
        float max_value = -std::numeric_limits<float>::infinity();
        for (int64_t col = 0; col < cols; col++)
        {
            max_value = std::max(
                max_value,
                source[static_cast<size_t>(row_offset + col)]);
        }

        float sum_value = 0.0f;
        for (int64_t col = 0; col < cols; col++)
        {
            sum_value += std::exp(
                source[static_cast<size_t>(row_offset + col)] - max_value);
        }

        const float reciprocal_sum = 1.0f / sum_value;
        for (int64_t col = 0; col < cols; col++)
        {
            destination[static_cast<size_t>(row_offset + col)] =
                std::exp(
                    source[static_cast<size_t>(row_offset + col)] -
                    max_value) *
                reciprocal_sum;
        }
    }
}

void dequantize_int8_cpu(
    const std::vector<int8_t> &source,
    std::vector<float> &destination,
    float input_scale,
    int32_t input_zero_point)
{
    for (size_t index = 0; index < source.size(); index++)
    {
        destination[index] =
            (static_cast<int32_t>(source[index]) - input_zero_point) *
            input_scale;
    }
}

int8_t quantize_probability_int8(float probability)
{
    int32_t quantized = static_cast<int32_t>(
        std::nearbyint(probability / INT8_OUTPUT_SCALE));
    quantized += INT8_OUTPUT_ZERO_POINT;
    quantized = std::clamp<int32_t>(
        quantized,
        std::numeric_limits<int8_t>::min(),
        std::numeric_limits<int8_t>::max());
    return static_cast<int8_t>(quantized);
}

void quantize_softmax_int8_cpu(
    const std::vector<float> &source,
    std::vector<int8_t> &destination)
{
    for (size_t index = 0; index < source.size(); index++)
    {
        destination[index] = quantize_probability_int8(source[index]);
    }
}

void fill_input(std::vector<float> &input, uint32_t seed)
{
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-10.0f, 10.0f);
    for (float &value : input)
    {
        value = distribution(generator);
    }
}

void fill_int8_input(std::vector<int8_t> &input, uint32_t seed)
{
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int32_t> distribution(-128, 127);
    for (int8_t &value : input)
    {
        value = static_cast<int8_t>(distribution(generator));
    }
}

bool run_accuracy_case(int64_t rows, int64_t cols, uint32_t seed)
{
    const size_t element_count =
        static_cast<size_t>(rows) * static_cast<size_t>(cols);
    const size_t byte_count = element_count * sizeof(float);

    std::vector<float> source(element_count);
    std::vector<float> reference(element_count);
    std::vector<float> actual(element_count);
    fill_input(source, seed);
    softmax_cpu(source, reference, rows, cols);

    float *device_source = nullptr;
    float *device_destination = nullptr;

    std::cout << "[精度阶段] rows=" << std::setw(4) << rows
              << " cols=" << std::setw(4) << cols << '\n';

    if (!check_cuda(
            cudaMalloc(
                reinterpret_cast<void **>(&device_source),
                byte_count),
            "cudaMalloc(source)"))
    {
        return false;
    }
    if (!check_cuda(
            cudaMalloc(
                reinterpret_cast<void **>(&device_destination),
                byte_count),
            "cudaMalloc(destination)"))
    {
        cudaFree(device_source);
        return false;
    }

    bool success =
        check_cuda(
            cudaMemcpy(
                device_source,
                source.data(),
                byte_count,
                cudaMemcpyHostToDevice),
            "cudaMemcpy(source)") &&
        check_cuda(
            launch_softmax(
                device_source,
                device_destination,
                rows,
                cols),
            "launch_softmax") &&
        check_cuda(
            cudaDeviceSynchronize(),
            "cudaDeviceSynchronize(accuracy)") &&
        check_cuda(
            cudaMemcpy(
                actual.data(),
                device_destination,
                byte_count,
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(destination)");

    cudaFree(device_source);
    cudaFree(device_destination);

    if (!success)
    {
        return false;
    }

    constexpr float ABSOLUTE_TOLERANCE = 1.0e-6f;
    constexpr float RELATIVE_TOLERANCE = 1.0e-5f;
    constexpr float ROW_SUM_TOLERANCE = 2.0e-5f;

    size_t error_count = 0;
    size_t max_error_index = 0;
    float max_absolute_error = 0.0f;
    float max_relative_error = 0.0f;
    double max_row_sum_error = 0.0;

    for (size_t index = 0; index < element_count; index++)
    {
        const float absolute_error =
            std::abs(reference[index] - actual[index]);
        const float relative_error =
            absolute_error /
            std::max(std::abs(reference[index]), ABSOLUTE_TOLERANCE);
        const float tolerance =
            ABSOLUTE_TOLERANCE +
            RELATIVE_TOLERANCE * std::abs(reference[index]);

        if (!std::isfinite(actual[index]) || absolute_error > tolerance)
        {
            error_count++;
        }
        if (absolute_error > max_absolute_error)
        {
            max_absolute_error = absolute_error;
            max_error_index = index;
        }
        max_relative_error = std::max(max_relative_error, relative_error);
    }

    for (int64_t row = 0; row < rows; row++)
    {
        double row_sum = 0.0;
        for (int64_t col = 0; col < cols; col++)
        {
            row_sum += static_cast<double>(
                actual[static_cast<size_t>(row * cols + col)]);
        }
        max_row_sum_error =
            std::max(max_row_sum_error, std::abs(row_sum - 1.0f));
    }

    const bool passed =
        error_count == 0 && max_row_sum_error <= ROW_SUM_TOLERANCE;
    std::cout << "  [关键结果] errors=" << error_count
              << '/' << element_count
              << " max_abs=" << std::scientific << max_absolute_error
              << " max_rel=" << max_relative_error
              << " row_sum_error=" << max_row_sum_error
              << std::defaultfloat << '\n';

    if (!passed)
    {
        std::cout << "  [FAILED] index=" << max_error_index
                  << " reference=" << reference[max_error_index]
                  << " actual=" << actual[max_error_index] << "\n\n";
        return false;
    }

    std::cout << "  [SUCCESS] 精度验证通过\n\n";
    return true;
}

bool run_int8_accuracy_case(
    int64_t rows,
    int64_t cols,
    float input_scale,
    int32_t input_zero_point,
    uint32_t seed)
{
    const size_t element_count = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    const size_t input_byte_count = element_count * sizeof(int8_t);
    const size_t float_byte_count = element_count * sizeof(float);

    std::vector<int8_t> source(element_count);
    std::vector<float> dequantized_source(element_count);
    std::vector<float> float_reference(element_count);
    std::vector<float> float_actual(element_count);
    std::vector<int8_t> int8_reference(element_count);
    std::vector<int8_t> int8_actual(element_count);

    fill_int8_input(source, seed);
    dequantize_int8_cpu(
        source,
        dequantized_source,
        input_scale,
        input_zero_point);
    softmax_cpu(
        dequantized_source,
        float_reference,
        rows,
        cols);
    quantize_softmax_int8_cpu(float_reference, int8_reference);

    int8_t *device_source = nullptr;
    float *device_float_destination = nullptr;
    int8_t *device_int8_destination = nullptr;

    std::cout << "[INT8 精度阶段] rows=" << std::setw(4) << rows
              << " cols=" << std::setw(4) << cols
              << " scale=" << input_scale
              << " zero_point=" << input_zero_point << '\n';

    bool success =
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void **>(&device_source),
                input_byte_count),
            "cudaMalloc(int8 source)") &&
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void **>(&device_float_destination),
                float_byte_count),
            "cudaMalloc(float destination)") &&
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void **>(&device_int8_destination),
                input_byte_count),
            "cudaMalloc(int8 destination)");

    if (!success)
    {
        if (device_source != nullptr)
        {
            cudaFree(device_source);
        }
        if (device_float_destination != nullptr)
        {
            cudaFree(device_float_destination);
        }
        if (device_int8_destination != nullptr)
        {
            cudaFree(device_int8_destination);
        }
        return false;
    }

    success =
        check_cuda(
            cudaMemcpy(
                device_source,
                source.data(),
                input_byte_count,
                cudaMemcpyHostToDevice),
            "cudaMemcpy(int8 source)") &&
        check_cuda(
            launch_softmax_int8_to_float(
                device_source,
                device_float_destination,
                rows,
                cols,
                input_scale,
                input_zero_point),
            "launch_softmax_int8_to_float") &&
        check_cuda(
            launch_softmax_int8_to_int8(
                device_source,
                device_int8_destination,
                rows,
                cols,
                input_scale,
                input_zero_point),
            "launch_softmax_int8_to_int8") &&
        check_cuda(
            cudaDeviceSynchronize(),
            "cudaDeviceSynchronize(int8 accuracy)") &&
        check_cuda(
            cudaMemcpy(
                float_actual.data(),
                device_float_destination,
                float_byte_count,
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(float destination)") &&
        check_cuda(
            cudaMemcpy(
                int8_actual.data(),
                device_int8_destination,
                input_byte_count,
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(int8 destination)");

    cudaFree(device_source);
    cudaFree(device_float_destination);
    cudaFree(device_int8_destination);

    if (!success)
    {
        return false;
    }

    constexpr float ABSOLUTE_TOLERANCE = 1.0e-6f;
    constexpr float RELATIVE_TOLERANCE = 1.0e-5f;
    size_t float_error_count = 0;
    size_t int8_error_count = 0;
    int32_t max_int8_difference = 0;
    float max_float_absolute_error = 0.0f;
    float max_int8_dequantized_error = 0.0f;
    float max_int8_row_sum_error = 0.0f;

    for (size_t index = 0; index < element_count; index++)
    {
        const float float_absolute_error =
            std::abs(float_reference[index] - float_actual[index]);
        const float tolerance =
            ABSOLUTE_TOLERANCE +
            RELATIVE_TOLERANCE * std::abs(float_reference[index]);
        if (!std::isfinite(float_actual[index]) ||
            float_absolute_error > tolerance)
        {
            float_error_count++;
        }
        max_float_absolute_error = std::max(
            max_float_absolute_error,
            float_absolute_error);

        const int32_t int8_difference = std::abs(
            static_cast<int32_t>(int8_reference[index]) -
            static_cast<int32_t>(int8_actual[index]));
        if (int8_difference > 1)
        {
            int8_error_count++;
        }
        max_int8_difference =
            std::max(max_int8_difference, int8_difference);

        const float dequantized_probability =
            (static_cast<int32_t>(int8_actual[index]) -
             INT8_OUTPUT_ZERO_POINT) *
            INT8_OUTPUT_SCALE;
        max_int8_dequantized_error = std::max(
            max_int8_dequantized_error,
            std::abs(
                float_reference[index] -
                dequantized_probability));
    }

    for (int64_t row = 0; row < rows; row++)
    {
        float row_sum = 0.0f;
        for (int64_t col = 0; col < cols; col++)
        {
            const int64_t index = row * cols + col;
            row_sum +=
                (static_cast<int32_t>(
                     int8_actual[static_cast<size_t>(index)]) -
                 INT8_OUTPUT_ZERO_POINT) *
                INT8_OUTPUT_SCALE;
        }
        max_int8_row_sum_error = std::max(
            max_int8_row_sum_error,
            std::abs(row_sum - 1.0f));
    }

    const bool passed =
        float_error_count == 0 && int8_error_count == 0;
    std::cout << "  [关键结果] float_errors=" << float_error_count
              << '/' << element_count
              << " float_max_abs=" << std::scientific
              << max_float_absolute_error
              << " int8_errors_gt_1=" << int8_error_count
              << " int8_max_diff=" << max_int8_difference
              << " int8_dequant_max_abs="
              << max_int8_dequantized_error
              << " int8_row_sum_error="
              << max_int8_row_sum_error
              << std::defaultfloat << '\n';

    std::cout << "  " << (passed ? "[SUCCESS]" : "[FAILED]")
              << " INT8 输入精度验证"
              << (passed ? "通过" : "失败") << "\n\n";
    return passed;
}

bool run_fp32_benchmark(int64_t rows, int64_t cols)
{
    const size_t element_count =
        static_cast<size_t>(rows) * static_cast<size_t>(cols);
    const size_t byte_count = element_count * sizeof(float);

    std::vector<float> source(element_count);
    fill_input(source, 2026U);

    float *device_source = nullptr;
    float *device_destination = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    std::cout << "[FP32 性能阶段] rows=" << rows
              << " cols=" << cols
              << " warmup=" << WARMUP_ITERATIONS
              << " iterations=" << BENCHMARK_ITERATIONS << '\n';

    bool success =
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void **>(&device_source),
                byte_count),
            "cudaMalloc(benchmark source)") &&
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void **>(&device_destination),
                byte_count),
            "cudaMalloc(benchmark destination)") &&
        check_cuda(
            cudaMemcpy(
                device_source,
                source.data(),
                byte_count,
                cudaMemcpyHostToDevice),
            "cudaMemcpy(benchmark source)") &&
        check_cuda(
            cudaEventCreate(&start),
            "cudaEventCreate(start)") &&
        check_cuda(
            cudaEventCreate(&stop),
            "cudaEventCreate(stop)");

    if (!success)
    {
        if (start != nullptr)
        {
            cudaEventDestroy(start);
        }
        if (stop != nullptr)
        {
            cudaEventDestroy(stop);
        }
        if (device_source != nullptr)
        {
            cudaFree(device_source);
        }
        if (device_destination != nullptr)
        {
            cudaFree(device_destination);
        }
        return false;
    }

    for (int iteration = 0; iteration < WARMUP_ITERATIONS; iteration++)
    {
        success = success &&
            check_cuda(
                launch_softmax(
                    device_source,
                    device_destination,
                    rows,
                    cols),
                "launch_softmax(warmup)");
    }
    success = success &&
        check_cuda(
            cudaDeviceSynchronize(),
            "cudaDeviceSynchronize(warmup)") &&
        check_cuda(
            cudaEventRecord(start),
            "cudaEventRecord(start)");

    for (int iteration = 0;
         iteration < BENCHMARK_ITERATIONS && success;
         iteration++)
    {
        success = check_cuda(
            launch_softmax(
                device_source,
                device_destination,
                rows,
                cols),
            "launch_softmax(benchmark)");
    }

    success = success &&
        check_cuda(
            cudaEventRecord(stop),
            "cudaEventRecord(stop)") &&
        check_cuda(
            cudaEventSynchronize(stop),
            "cudaEventSynchronize(stop)");

    float elapsed_ms = 0.0f;
    success = success &&
        check_cuda(
            cudaEventElapsedTime(&elapsed_ms, start, stop),
            "cudaEventElapsedTime");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(device_source);
    cudaFree(device_destination);

    if (!success)
    {
        return false;
    }

    const float average_ms = elapsed_ms / BENCHMARK_ITERATIONS;
    const double bytes_per_element = cols <= BLOCK_MAX_COLS ? 8.0 : 12.0;
    const double transferred_bytes =
        static_cast<double>(element_count) * bytes_per_element;
    const double effective_bandwidth =
        transferred_bytes / (static_cast<double>(average_ms) * 1.0e6);

    std::cout << "  [关键结果] latency=" << std::fixed
              << std::setprecision(6) << average_ms
              << " ms effective_bandwidth=" << std::setprecision(2)
              << effective_bandwidth << " GB/s\n"
              << std::defaultfloat
              << "  [SUCCESS] 性能测试完成\n\n";
    return true;
}

bool run_int8_benchmark()
{
    const int64_t rows = BENCHMARK_ROWS;
    const int64_t cols = BENCHMARK_COLS;
    constexpr float INPUT_SCALE = 1.0f / 32.0f;
    constexpr int32_t INPUT_ZERO_POINT = -3;

    const size_t element_count =
        static_cast<size_t>(rows) * static_cast<size_t>(cols);
    const size_t int8_byte_count = element_count * sizeof(int8_t);
    const size_t float_byte_count = element_count * sizeof(float);

    std::vector<int8_t> source(element_count);
    fill_int8_input(source, 2027U);

    int8_t *device_source = nullptr;
    float *device_float_destination = nullptr;
    int8_t *device_int8_destination = nullptr;

    std::cout << "[INT8 性能阶段] rows=" << rows
              << " cols=" << cols
              << " input_scale=" << INPUT_SCALE
              << " input_zero_point=" << INPUT_ZERO_POINT
              << " warmup=" << WARMUP_ITERATIONS
              << " iterations=" << BENCHMARK_ITERATIONS << '\n';

    bool success =
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void **>(&device_source),
                int8_byte_count),
            "cudaMalloc(int8 benchmark source)") &&
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void **>(&device_float_destination),
                float_byte_count),
            "cudaMalloc(float benchmark destination)") &&
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void **>(&device_int8_destination),
                int8_byte_count),
            "cudaMalloc(int8 benchmark destination)") &&
        check_cuda(
            cudaMemcpy(
                device_source,
                source.data(),
                int8_byte_count,
                cudaMemcpyHostToDevice),
            "cudaMemcpy(int8 benchmark source)");

    if (!success)
    {
        if (device_source != nullptr)
        {
            cudaFree(device_source);
        }
        if (device_float_destination != nullptr)
        {
            cudaFree(device_float_destination);
        }
        if (device_int8_destination != nullptr)
        {
            cudaFree(device_int8_destination);
        }
        return false;
    }

    auto measure_latency =
        [](
            const char *name,
            auto launch,
            float &average_ms)
        {
            cudaEvent_t start = nullptr;
            cudaEvent_t stop = nullptr;
            bool measured =
                check_cuda(
                    cudaEventCreate(&start),
                    "cudaEventCreate(int8 start)") &&
                check_cuda(
                    cudaEventCreate(&stop),
                    "cudaEventCreate(int8 stop)");

            if (!measured)
            {
                if (start != nullptr)
                {
                    cudaEventDestroy(start);
                }
                if (stop != nullptr)
                {
                    cudaEventDestroy(stop);
                }
                return false;
            }

            for (int iteration = 0;
                 iteration < WARMUP_ITERATIONS && measured;
                 iteration++)
            {
                measured = check_cuda(launch(), name);
            }
            measured = measured &&
                check_cuda(
                    cudaDeviceSynchronize(),
                    "cudaDeviceSynchronize(int8 warmup)") &&
                check_cuda(
                    cudaEventRecord(start),
                    "cudaEventRecord(int8 start)");

            for (int iteration = 0;
                 iteration < BENCHMARK_ITERATIONS && measured;
                 iteration++)
            {
                measured = check_cuda(launch(), name);
            }

            measured = measured &&
                check_cuda(
                    cudaEventRecord(stop),
                    "cudaEventRecord(int8 stop)") &&
                check_cuda(
                    cudaEventSynchronize(stop),
                    "cudaEventSynchronize(int8 stop)");

            float elapsed_ms = 0.0f;
            measured = measured &&
                check_cuda(
                    cudaEventElapsedTime(
                        &elapsed_ms,
                        start,
                        stop),
                    "cudaEventElapsedTime(int8)");

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            average_ms = elapsed_ms / BENCHMARK_ITERATIONS;
            return measured;
        };

    float int8_to_float_ms = 0.0f;
    float int8_to_int8_ms = 0.0f;
    success =
        measure_latency(
            "launch_softmax_int8_to_float(benchmark)",
            [&]()
            {
                return launch_softmax_int8_to_float(
                    device_source,
                    device_float_destination,
                    rows,
                    cols,
                    INPUT_SCALE,
                    INPUT_ZERO_POINT);
            },
            int8_to_float_ms) &&
        measure_latency(
            "launch_softmax_int8_to_int8(benchmark)",
            [&]()
            {
                return launch_softmax_int8_to_int8(
                    device_source,
                    device_int8_destination,
                    rows,
                    cols,
                    INPUT_SCALE,
                    INPUT_ZERO_POINT);
            },
            int8_to_int8_ms);

    cudaFree(device_source);
    cudaFree(device_float_destination);
    cudaFree(device_int8_destination);

    if (!success)
    {
        return false;
    }

    const double int8_to_float_bytes =
        static_cast<double>(int8_byte_count + float_byte_count);
    const double int8_to_int8_bytes =
        static_cast<double>(int8_byte_count) * 2.0;
    const double int8_to_float_bandwidth =
        int8_to_float_bytes /
        (static_cast<double>(int8_to_float_ms) * 1.0e6);
    const double int8_to_int8_bandwidth =
        int8_to_int8_bytes /
        (static_cast<double>(int8_to_int8_ms) * 1.0e6);

    std::cout << "  [关键结果] INT8->FP32 latency=" << std::fixed
              << std::setprecision(6) << int8_to_float_ms
              << " ms effective_bandwidth=" << std::setprecision(2)
              << int8_to_float_bandwidth << " GB/s\n"
              << "  [关键结果] INT8->INT8 latency="
              << std::setprecision(6) << int8_to_int8_ms
              << " ms effective_bandwidth=" << std::setprecision(2)
              << int8_to_int8_bandwidth << " GB/s\n"
              << std::defaultfloat
              << "  [SUCCESS] INT8 性能测试完成\n\n";
    return true;
}

} // namespace

int main()
{
    int device_id = 0;
    cudaDeviceProp device_properties{};
    if (!check_cuda(cudaGetDevice(&device_id), "cudaGetDevice") ||
        !check_cuda(
            cudaGetDeviceProperties(&device_properties, device_id),
            "cudaGetDeviceProperties"))
    {
        return 1;
    }

    std::cout << "\n=== CUDA Softmax 算子测试 ===\n\n"
              << "[配置] device=" << device_properties.name
              << " input_types={float32,int8}"
              << " output_types={float32,int8}"
              << " fp32_block_max_cols=" << BLOCK_MAX_COLS
              << " online_vector_size=" << ONLINE_VECTOR_SIZE
              << " int8_max_cols=" << INT8_MAX_COLS
              << " warp_size=" << WARP_SIZE
              << " block_threads=" << BLOCK_THREADS << '\n'
              << "[配置] INT8 output_scale=" << INT8_OUTPUT_SCALE
              << " output_zero_point=" << INT8_OUTPUT_ZERO_POINT
              << " compute=float32\n\n";

    constexpr std::array<std::pair<int64_t, int64_t>, 17> TEST_CASES =
    {{
        {1, 1},
        {3, 17},
        {7, 32},
        {9, 33},
        {17, 127},
        {31, 256},
        {33, 511},
        {65, 1023},
        {65, 1024},
        {3, 1025},
        {5, 2048},
        {7, 2049},
        {9, 4096},
        {11, 4097},
        {13, 256 * 32},
        {4, BLOCK_MAX_COLS + 1},
        {4, ONLINE_LONG_COLS},
    }};

    bool success = true;
    uint32_t seed = 42U;
    for (const auto &[rows, cols] : TEST_CASES)
    {
        success = run_accuracy_case(rows, cols, seed++) && success;
    }

    constexpr std::array<
        std::tuple<int64_t, int64_t, float, int32_t>,
        9> INT8_TEST_CASES =
    {{
        {1, 1, 1.0f / 32.0f, -3},
        {3, 17, 1.0f / 16.0f, -128},
        {7, 32, 1.0f / 32.0f, 127},
        {9, 33, 1.0f / 64.0f, 5},
        {17, 127, 1.0f / 32.0f, -17},
        {31, 256, 1.0f / 64.0f, 0},
        {33, 511, 1.0f / 32.0f, 23},
        {65, 1023, 1.0f / 16.0f, -7},
        {65, 1024, 1.0f / 32.0f, 11},
    }};

    for (const auto &[rows, cols, input_scale, input_zero_point] :
         INT8_TEST_CASES)
    {
        success =
            run_int8_accuracy_case(
                rows,
                cols,
                input_scale,
                input_zero_point,
                seed++) &&
            success;
    }

    if (success)
    {
        success =
            run_fp32_benchmark(BENCHMARK_ROWS, BENCHMARK_COLS) &&
            success;
        success =
            run_fp32_benchmark(BENCHMARK_ROWS, ONLINE_BENCHMARK_COLS) &&
            success;
        success = run_int8_benchmark() && success;
    }

    std::cout << (success ? "[SUCCESS]" : "[FAILED]")
              << " Softmax 算子测试"
              << (success ? "通过" : "失败") << '\n';
    return success ? 0 : 1;
}
