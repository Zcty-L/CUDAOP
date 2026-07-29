#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <math_constants.h>

namespace
{

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int ROWS_PER_WARP = 2;
constexpr int MAX_COLS = 1024;
constexpr int MAX_GRID_SIZE = 65535;

constexpr int BENCHMARK_ROWS = 512;
constexpr int BENCHMARK_COLS = 1024;
constexpr int WARMUP_ITERATIONS = 20;
constexpr int BENCHMARK_ITERATIONS = 200;

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

template <int COLS_PER_THREAD>
__global__ void softmax_warp_kernel(
    const float *__restrict__ source,
    float *__restrict__ destination,
    int64_t rows,
    int64_t cols)
{
    const int lane_id = threadIdx.x;
    const int64_t global_warp_id =
        static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    const int64_t warp_stride =
        static_cast<int64_t>(gridDim.x) * blockDim.y;

    for (int64_t first_row = global_warp_id * ROWS_PER_WARP;
         first_row < rows;
         first_row += warp_stride * ROWS_PER_WARP)
    {
#pragma unroll
        for (int row_index = 0; row_index < ROWS_PER_WARP; row_index++)
        {
            const int64_t row = first_row + row_index;
            if (row >= rows)
            {
                continue;
            }

            float values[COLS_PER_THREAD];
            float thread_max = -CUDART_INF_F;

#pragma unroll
            for (int index = 0; index < COLS_PER_THREAD; index++)
            {
                const int col = lane_id + index * WARP_SIZE;
                const float value =
                    col < cols ? source[row * cols + col] : -CUDART_INF_F;
                values[index] = value;
                thread_max = fmaxf(thread_max, value);
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2)
            {
                thread_max = fmaxf(
                    thread_max,
                    __shfl_xor_sync(0xffffffffU, thread_max, mask));
            }

            float thread_sum = 0.0f;
#pragma unroll
            for (int index = 0; index < COLS_PER_THREAD; index++)
            {
                const int col = lane_id + index * WARP_SIZE;
                const float value =
                    col < cols ? expf(values[index] - thread_max) : 0.0f;
                values[index] = value;
                thread_sum += value;
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2)
            {
                thread_sum +=
                    __shfl_xor_sync(0xffffffffU, thread_sum, mask);
            }

            const float reciprocal_sum = 1.0f / thread_sum;
#pragma unroll
            for (int index = 0; index < COLS_PER_THREAD; index++)
            {
                const int col = lane_id + index * WARP_SIZE;
                if (col < cols)
                {
                    destination[row * cols + col] =
                        values[index] * reciprocal_sum;
                }
            }
        }
    }
}

template <int COLS_PER_THREAD>
cudaError_t launch_softmax_kernel(
    const float *source,
    float *destination,
    int64_t rows,
    int64_t cols,
    cudaStream_t stream)
{
    constexpr int ROWS_PER_BLOCK = WARPS_PER_BLOCK * ROWS_PER_WARP;
    const int64_t required_blocks =
        (rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    const int grid_size = static_cast<int>(
        std::min<int64_t>(required_blocks, MAX_GRID_SIZE));

    const dim3 block(WARP_SIZE, WARPS_PER_BLOCK);
    softmax_warp_kernel<COLS_PER_THREAD>
        <<<grid_size, block, 0, stream>>>(
            source,
            destination,
            rows,
            cols);
    return cudaGetLastError();
}

cudaError_t launch_softmax(
    const float *source,
    float *destination,
    int64_t rows,
    int64_t cols,
    cudaStream_t stream = nullptr)
{
    if (source == nullptr || destination == nullptr ||
        rows <= 0 || cols <= 0 || cols > MAX_COLS)
    {
        return cudaErrorInvalidValue;
    }

    if (cols <= 32)
    {
        return launch_softmax_kernel<1>(
            source,
            destination,
            rows,
            cols,
            stream);
    }
    if (cols <= 64)
    {
        return launch_softmax_kernel<2>(
            source,
            destination,
            rows,
            cols,
            stream);
    }
    if (cols <= 128)
    {
        return launch_softmax_kernel<4>(
            source,
            destination,
            rows,
            cols,
            stream);
    }
    if (cols <= 256)
    {
        return launch_softmax_kernel<8>(
            source,
            destination,
            rows,
            cols,
            stream);
    }
    if (cols <= 512)
    {
        return launch_softmax_kernel<16>(
            source,
            destination,
            rows,
            cols,
            stream);
    }

    return launch_softmax_kernel<32>(
        source,
        destination,
        rows,
        cols,
        stream);
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
    float max_row_sum_error = 0.0f;

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
        float row_sum = 0.0f;
        for (int64_t col = 0; col < cols; col++)
        {
            row_sum += actual[static_cast<size_t>(row * cols + col)];
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

bool run_benchmark()
{
    const int64_t rows = BENCHMARK_ROWS;
    const int64_t cols = BENCHMARK_COLS;
    const size_t element_count =
        static_cast<size_t>(rows) * static_cast<size_t>(cols);
    const size_t byte_count = element_count * sizeof(float);

    std::vector<float> source(element_count);
    fill_input(source, 2026U);

    float *device_source = nullptr;
    float *device_destination = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    std::cout << "[性能阶段] rows=" << rows
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
    const double transferred_bytes =
        static_cast<double>(byte_count) * 2.0;
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
              << " dtype=float32"
              << " max_cols=" << MAX_COLS
              << " warp_size=" << WARP_SIZE << "\n\n";

    constexpr std::array<std::pair<int64_t, int64_t>, 9> TEST_CASES =
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
    }};

    bool success = true;
    uint32_t seed = 42U;
    for (const auto &[rows, cols] : TEST_CASES)
    {
        success = run_accuracy_case(rows, cols, seed++) && success;
    }

    if (success)
    {
        success = run_benchmark();
    }

    std::cout << (success ? "[SUCCESS]" : "[FAILED]")
              << " Softmax 算子测试"
              << (success ? "通过" : "失败") << '\n';
    return success ? 0 : 1;
}
