#include "op/topk/topk.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
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

struct RadixTopKCase
{
    int rows;
    int columns;
    int k;
    bool largest;
    bool sorted;
    bool repeated;
};

uint32_t float_bits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

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

bool index_is_better(
    const float* row,
    int32_t left,
    int32_t right,
    bool largest)
{
    const uint32_t left_radix = float_radix(row[left]);
    const uint32_t right_radix = float_radix(row[right]);
    if (left_radix != right_radix)
    {
        return largest
            ? left_radix > right_radix
            : left_radix < right_radix;
    }
    return left < right;
}

void fill_input(
    std::vector<float>& input,
    const RadixTopKCase& config,
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

    for (int row = 0; row < config.rows; ++row)
    {
        const size_t offset =
            static_cast<size_t>(row) * config.columns;
        if (config.columns >= 8)
        {
            input[offset] =
                std::numeric_limits<float>::quiet_NaN();
            input[offset + 1] =
                -std::numeric_limits<float>::quiet_NaN();
            input[offset + 2] =
                std::numeric_limits<float>::infinity();
            input[offset + 3] =
                -std::numeric_limits<float>::infinity();
            input[offset + 4] = 42.0F;
            input[offset + 5] = 42.0F;
            input[offset + 6] = -0.0F;
            input[offset + 7] = 0.0F;
        }
    }
}

std::vector<int32_t> make_expected_indices(
    const std::vector<float>& input,
    const RadixTopKCase& config)
{
    const size_t output_count =
        static_cast<size_t>(config.rows) * config.k;
    std::vector<int32_t> expected(output_count);
    std::vector<int32_t> order(config.columns);

    for (int row = 0; row < config.rows; ++row)
    {
        const float* row_input =
            input.data() +
            static_cast<size_t>(row) * config.columns;
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(
            order.begin(),
            order.begin() + config.k,
            order.end(),
            [row_input, &config](int32_t left, int32_t right)
            {
                return index_is_better(
                    row_input,
                    left,
                    right,
                    config.largest);
            });
        std::copy(
            order.begin(),
            order.begin() + config.k,
            expected.begin() +
                static_cast<size_t>(row) * config.k);
    }
    return expected;
}

bool validate_case(
    const RadixTopKCase& config,
    uint32_t seed)
{
    const size_t input_count =
        static_cast<size_t>(config.rows) * config.columns;
    const size_t output_count =
        static_cast<size_t>(config.rows) * config.k;
    std::vector<float> input(input_count);
    std::vector<float> actual_values(output_count);
    std::vector<int32_t> actual_indices(output_count);
    fill_input(input, config, seed);
    const std::vector<int32_t> expected_indices =
        make_expected_indices(input, config);

    float* device_input = nullptr;
    float* device_values = nullptr;
    int32_t* device_indices = nullptr;
    CUDA_CHECK(cudaMalloc(
        &device_input,
        input_count * sizeof(float)));
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

    CUDA_CHECK(cudaop::topk_radix_cuda(
        device_input,
        device_values,
        device_indices,
        config.rows,
        config.columns,
        config.k,
        config.largest,
        config.sorted));
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

    int error_count = 0;
    for (int row = 0; row < config.rows; ++row)
    {
        int32_t* actual_row =
            actual_indices.data() +
            static_cast<size_t>(row) * config.k;
        const int32_t* expected_row =
            expected_indices.data() +
            static_cast<size_t>(row) * config.k;
        const float* input_row =
            input.data() +
            static_cast<size_t>(row) * config.columns;

        // sorted=false 时先验证原始输出中的 value/index 配对，再对索引
        // 排序，只比较入选集合，不要求 kernel 的输出次序。
        if (!config.sorted)
        {
            bool pairs_valid = true;
            for (int rank = 0; rank < config.k; ++rank)
            {
                const size_t output_offset =
                    static_cast<size_t>(row) * config.k + rank;
                const int32_t actual_index = actual_row[rank];
                const bool index_in_range =
                    actual_index >= 0 &&
                    actual_index < config.columns;
                const bool value_matches =
                    index_in_range &&
                    float_bits(actual_values[output_offset]) ==
                        float_bits(input_row[actual_index]);
                if (!value_matches)
                {
                    pairs_valid = false;
                    if (error_count < 5)
                    {
                        std::cerr << "[ERROR] row=" << row
                                  << " rank=" << rank
                                  << " invalid unsorted pair"
                                  << std::endl;
                    }
                    ++error_count;
                }
            }

            if (!pairs_valid)
            {
                continue;
            }
            std::sort(
                actual_row,
                actual_row + config.k,
                [input_row, &config](
                    int32_t left,
                    int32_t right)
                {
                    return index_is_better(
                        input_row,
                        left,
                        right,
                        config.largest);
                });
        }

        for (int rank = 0; rank < config.k; ++rank)
        {
            const size_t output_offset =
                static_cast<size_t>(row) * config.k + rank;
            const int32_t actual_index = actual_row[rank];
            const bool index_in_range =
                actual_index >= 0 &&
                actual_index < config.columns;
            const bool index_matches =
                index_in_range &&
                actual_index == expected_row[rank];
            const bool value_matches =
                !config.sorted ||
                (index_in_range &&
                 float_bits(actual_values[output_offset]) ==
                     float_bits(input_row[actual_index]));

            if (!index_matches || !value_matches)
            {
                if (error_count < 5)
                {
                    std::cerr << "[ERROR] row=" << row
                              << " rank=" << rank
                              << " expected_index="
                              << expected_row[rank]
                              << " actual_index="
                              << actual_index
                              << std::endl;
                }
                ++error_count;
            }
        }
    }

    const std::string direction =
        config.largest ? "largest" : "smallest";
    const std::string order =
        config.sorted ? "sorted" : "unsorted";
    std::cout << "  " << std::left << std::setw(24)
              << (std::to_string(config.rows) + "x" +
                  std::to_string(config.columns) +
                  ", k=" + std::to_string(config.k))
              << std::setw(10) << direction
              << std::setw(10) << order
              << "errors=" << error_count << std::endl;
    return error_count == 0;
}

bool validate_invalid_arguments()
{
    const cudaError_t error = cudaop::topk_radix_cuda(
        nullptr,
        nullptr,
        nullptr,
        1,
        1,
        1);
    std::cout << "  " << std::left << std::setw(24)
              << "invalid arguments"
              << "status=" << cudaGetErrorName(error)
              << std::endl;
    return error == cudaErrorInvalidValue;
}

}  // namespace

int main()
{
    std::cout << "\n[CONFIG] Complete radix Top-K"
              << std::endl;
    std::cout << "  " << std::left << std::setw(24)
              << "input"
              << "float32 [rows, columns]"
              << std::endl;
    std::cout << "  " << std::left << std::setw(24)
              << "outputs"
              << "values/indices [rows, k]"
              << std::endl;
    std::cout << "  " << std::left << std::setw(24)
              << "selection"
              << "radix threshold + prefix scan gather"
              << std::endl;

    std::cout << "\n[STAGE] Full Top-K validation"
              << std::endl;
    const std::vector<RadixTopKCase> cases = {
        {1, 1, 1, true, true, false},
        {2, 8, 6, true, true, false},
        {3, 19, 5, true, true, false},
        {7, 1003, 17, true, true, false},
        {4, 257, 129, false, true, false},
        {3, 300, 21, false, false, false},
        {2, 257, 257, true, true, false},
        {5, 513, 32, true, false, false},
        {2, 257, 128, true, true, true},
    };

    bool success = true;
    for (size_t index = 0; index < cases.size(); ++index)
    {
        success =
            validate_case(cases[index], 4096 + index) &&
            success;
    }
    success = validate_invalid_arguments() && success;

    if (!success)
    {
        std::cerr << "\n[FAILED] Complete radix Top-K validation failed"
                  << std::endl;
        return 1;
    }

    std::cout << "\n[SUCCESS] Complete radix Top-K validation passed"
              << std::endl;
    return 0;
}
