#include "op/topk/SortingRadixSelect.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
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

constexpr int kBlockThreads = 256;

struct RadixSelectCase
{
    int rows;
    int columns;
    int k;
    bool largest;
    bool interleaved;
    bool repeated;
};

// 一个 block 处理一个逻辑切片。interleaved=false 时输入为普通行主序；
// interleaved=true 时不同切片交错存储，用来验证 within_slice_stride。
__global__ void radix_select_rows_kernel(
    const float* input,
    float* output,
    int rows,
    int columns,
    int k,
    bool largest,
    bool interleaved)
{
    const int row = blockIdx.x;
    if (row >= rows)
    {
        return;
    }

    __shared__ int shared_storage[cudaop::topk_radix::kRadixSize];
    const float* slice =
        interleaved
        ? input + row
        : input + static_cast<size_t>(row) * columns;
    const int within_slice_stride = interleaved ? rows : 1;
    float selected = 0.0F;

    cudaop::topk_radix::radix_select(
        slice,
        k,
        largest,
        columns,
        within_slice_stride,
        shared_storage,
        &selected);

    if (threadIdx.x == 0)
    {
        output[row] = selected;
    }
}

uint32_t float_bits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// 与设备端 FloatRadixConfig::convert 相同，用于生成 CPU 参考结果。
uint32_t float_radix(float value)
{
    const uint32_t bits = float_bits(value);
    const uint32_t mask =
        (bits & 0x80000000U) != 0U
        ? 0xffffffffU
        : 0x80000000U;
    return std::isnan(value)
        ? 0xffffffffU
        : bits ^ mask;
}

bool selected_value_equal(float expected, float actual)
{
    if (std::isnan(expected))
    {
        return std::isnan(actual);
    }
    return float_bits(expected) == float_bits(actual);
}

void fill_logical_input(
    std::vector<float>& input,
    const RadixSelectCase& config,
    uint32_t seed)
{
    if (config.repeated)
    {
        std::fill(input.begin(), input.end(), 7.0F);
        return;
    }

    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(
        -1000.0F,
        1000.0F);
    for (float& value : input)
    {
        value = distribution(generator);
    }

    // 同时覆盖重复值、无穷、NaN 和带符号零。PyTorch 将所有 NaN 映射
    // 到最大 radix key；+0 的 radix key 比 -0 大。
    for (int row = 0; row < config.rows; ++row)
    {
        const size_t offset =
            static_cast<size_t>(row) * config.columns;
        if (config.columns >= 8)
        {
            input[offset] =
                std::numeric_limits<float>::quiet_NaN();
            input[offset + 1] =
                -std::numeric_limits<float>::infinity();
            input[offset + 2] =
                std::numeric_limits<float>::infinity();
            input[offset + 3] = 42.0F;
            input[offset + 4] = 42.0F;
            input[offset + 5] = -0.0F;
            input[offset + 6] = 0.0F;
            input[offset + 7] =
                -std::numeric_limits<float>::quiet_NaN();
        }
    }
}

std::vector<float> make_physical_input(
    const std::vector<float>& logical_input,
    const RadixSelectCase& config)
{
    if (!config.interleaved)
    {
        return logical_input;
    }

    std::vector<float> physical_input(logical_input.size());
    for (int row = 0; row < config.rows; ++row)
    {
        for (int column = 0;
             column < config.columns;
             ++column)
        {
            physical_input[
                static_cast<size_t>(column) * config.rows + row] =
                logical_input[
                    static_cast<size_t>(row) * config.columns + column];
        }
    }
    return physical_input;
}

std::vector<float> make_expected(
    const std::vector<float>& logical_input,
    const RadixSelectCase& config)
{
    std::vector<float> expected(config.rows);
    std::vector<float> row_values(config.columns);

    for (int row = 0; row < config.rows; ++row)
    {
        const float* row_input =
            logical_input.data() +
            static_cast<size_t>(row) * config.columns;
        std::copy(
            row_input,
            row_input + config.columns,
            row_values.begin());
        std::sort(
            row_values.begin(),
            row_values.end(),
            [&config](float left, float right)
            {
                return config.largest
                    ? float_radix(left) > float_radix(right)
                    : float_radix(left) < float_radix(right);
            });
        expected[row] = row_values[config.k - 1];
    }
    return expected;
}

bool validate_case(const RadixSelectCase& config, uint32_t seed)
{
    const size_t input_count =
        static_cast<size_t>(config.rows) * config.columns;
    std::vector<float> logical_input(input_count);
    fill_logical_input(logical_input, config, seed);
    const std::vector<float> physical_input =
        make_physical_input(logical_input, config);
    const std::vector<float> expected =
        make_expected(logical_input, config);
    std::vector<float> actual(config.rows);

    float* device_input = nullptr;
    float* device_output = nullptr;
    CUDA_CHECK(cudaMalloc(
        &device_input,
        input_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(
        &device_output,
        static_cast<size_t>(config.rows) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(
        device_input,
        physical_input.data(),
        input_count * sizeof(float),
        cudaMemcpyHostToDevice));

    radix_select_rows_kernel<<<config.rows, kBlockThreads>>>(
        device_input,
        device_output,
        config.rows,
        config.columns,
        config.k,
        config.largest,
        config.interleaved);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(
        actual.data(),
        device_output,
        static_cast<size_t>(config.rows) * sizeof(float),
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(device_output));
    CUDA_CHECK(cudaFree(device_input));

    int error_count = 0;
    for (int row = 0; row < config.rows; ++row)
    {
        if (!selected_value_equal(expected[row], actual[row]))
        {
            if (error_count < 5)
            {
                std::cerr << "[ERROR] row=" << row
                          << " expected=" << expected[row]
                          << " actual=" << actual[row]
                          << std::endl;
            }
            ++error_count;
        }
    }

    const std::string direction =
        config.largest ? "largest" : "smallest";
    const std::string layout =
        config.interleaved ? "strided" : "contiguous";
    std::cout << "  " << std::left << std::setw(24)
              << (std::to_string(config.rows) + "x" +
                  std::to_string(config.columns) +
                  ", k=" + std::to_string(config.k))
              << std::setw(10) << direction
              << std::setw(12) << layout
              << "errors=" << error_count << std::endl;
    return error_count == 0;
}

}  // namespace

int main()
{
    std::cout << "\n[CONFIG] PyTorch radix-select adaptation"
              << std::endl;
    std::cout << "  " << std::left << std::setw(24)
              << "source"
              << "PyTorch 2.9.1 SortingRadixSelect.cuh"
              << std::endl;
    std::cout << "  " << std::left << std::setw(24)
              << "radix"
              << "2 bits per pass, 4 buckets"
              << std::endl;
    std::cout << "  " << std::left << std::setw(24)
              << "result"
              << "K-th threshold only"
              << std::endl;

    std::cout << "\n[STAGE] Correctness validation"
              << std::endl;
    const std::vector<RadixSelectCase> cases = {
        {1, 1, 1, true, false, false},
        {2, 16, 1, true, false, false},
        {3, 19, 5, true, false, false},
        {7, 1003, 17, true, false, false},
        {4, 257, 129, false, false, false},
        {5, 513, 32, true, true, false},
        {2, 257, 128, true, false, true},
    };

    bool success = true;
    for (size_t index = 0; index < cases.size(); ++index)
    {
        success =
            validate_case(cases[index], 2026 + index) &&
            success;
    }

    if (!success)
    {
        std::cerr << "\n[FAILED] Radix-select validation failed"
                  << std::endl;
        return 1;
    }

    std::cout << "\n[SUCCESS] Radix-select validation passed"
              << std::endl;
    return 0;
}
