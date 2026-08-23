#include "op/config.h"
#include "op/transpose/transpose.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace
{

using cudaop::TransposeDataType;
using cudaop::TransposeSharedMemoryLayout;

constexpr std::array<TransposeSharedMemoryLayout, 2> kLayouts = {
    TransposeSharedMemoryLayout::kPadded,
    TransposeSharedMemoryLayout::kSwizzled};

constexpr std::array<TransposeParam, 7> kValidationCases = {{
    {1, 1, 0, 0},
    {3, 5, 0, 0},
    {7, 33, 0, 0},
    {31, 32, 0, 0},
    {32, 31, 0, 0},
    {65, 97, 0, 0},
    {129, 67, 0, 0}}};

constexpr std::array<TransposeParam, 3> kBenchmarkCases = {{
    {1024, 1024, 20, 100},
    {3072, 4096, 20, 100},
    {4096, 4096, 20, 100}}};

void check_cuda(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

class DeviceBuffer
{
public:
    explicit DeviceBuffer(size_t bytes)
    {
        check_cuda(cudaMalloc(&pointer_, bytes), "cudaMalloc");
    }

    ~DeviceBuffer()
    {
        if (pointer_ != nullptr)
        {
            cudaFree(pointer_);
        }
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    void *get()
    {
        return pointer_;
    }

private:
    void *pointer_ = nullptr;
};

class CudaStream
{
public:
    CudaStream()
    {
        check_cuda(
            cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags");
    }

    ~CudaStream()
    {
        if (stream_ != nullptr)
        {
            cudaStreamDestroy(stream_);
        }
    }

    CudaStream(const CudaStream &) = delete;
    CudaStream &operator=(const CudaStream &) = delete;

    cudaStream_t get() const
    {
        return stream_;
    }

private:
    cudaStream_t stream_ = nullptr;
};

class CudaEvent
{
public:
    CudaEvent()
    {
        check_cuda(cudaEventCreate(&event_), "cudaEventCreate");
    }

    ~CudaEvent()
    {
        if (event_ != nullptr)
        {
            cudaEventDestroy(event_);
        }
    }

    CudaEvent(const CudaEvent &) = delete;
    CudaEvent &operator=(const CudaEvent &) = delete;

    cudaEvent_t get() const
    {
        return event_;
    }

private:
    cudaEvent_t event_ = nullptr;
};

template <typename Element>
Element make_value(size_t index)
{
    const int centered = static_cast<int>((index * 17U + 5U) % 29U) - 14;
    const float value = static_cast<float>(centered) * 0.25F;

    if constexpr (std::is_same_v<Element, float>)
    {
        return value;
    }
    else if constexpr (std::is_same_v<Element, __nv_bfloat16>)
    {
        return __float2bfloat16_rn(value);
    }
    else
    {
        static_assert(
            std::is_same_v<Element, __nv_fp8_e4m3>,
            "unexpected transpose test type");
        return __nv_fp8_e4m3(value);
    }
}

template <typename Element>
void transpose_cpu(
    const std::vector<Element> &input,
    std::vector<Element> &output,
    uint32_t rows,
    uint32_t columns)
{
    for (uint32_t row = 0; row < rows; ++row)
    {
        for (uint32_t column = 0; column < columns; ++column)
        {
            output[static_cast<size_t>(column) * rows + row] =
                input[static_cast<size_t>(row) * columns + column];
        }
    }
}

uint8_t get_fp4(
    const std::vector<uint8_t> &matrix,
    uint32_t columns,
    uint32_t row,
    uint32_t column)
{
    const size_t row_stride =
        (static_cast<size_t>(columns) + 1U) / 2U;
    const uint8_t packed =
        matrix[static_cast<size_t>(row) * row_stride + column / 2U];
    return static_cast<uint8_t>(
        (packed >> ((column & 1U) * 4U)) & 0x0FU);
}

void set_fp4(
    std::vector<uint8_t> &matrix,
    uint32_t columns,
    uint32_t row,
    uint32_t column,
    uint8_t value)
{
    const size_t row_stride =
        (static_cast<size_t>(columns) + 1U) / 2U;
    uint8_t &packed =
        matrix[static_cast<size_t>(row) * row_stride + column / 2U];
    const uint32_t shift = (column & 1U) * 4U;
    const uint8_t mask = static_cast<uint8_t>(0x0FU << shift);
    packed = static_cast<uint8_t>(
        (packed & static_cast<uint8_t>(~mask)) |
        static_cast<uint8_t>((value & 0x0FU) << shift));
}

bool compare_bytes(
    const void *actual,
    const void *expected,
    size_t bytes,
    size_t *first_mismatch)
{
    const auto *actual_bytes = static_cast<const uint8_t *>(actual);
    const auto *expected_bytes = static_cast<const uint8_t *>(expected);
    for (size_t index = 0; index < bytes; ++index)
    {
        if (actual_bytes[index] != expected_bytes[index])
        {
            *first_mismatch = index;
            return false;
        }
    }
    return true;
}

void print_validation_result(
    TransposeDataType data_type,
    TransposeSharedMemoryLayout layout,
    const TransposeParam &config,
    bool passed,
    size_t first_mismatch)
{
    const std::string shape =
        std::to_string(config.rows) + "x" +
        std::to_string(config.columns);
    std::cout << "  " << std::left
              << std::setw(12) << cudaop::transpose_data_type_name(data_type)
              << std::setw(10) << cudaop::transpose_layout_name(layout)
              << std::setw(12) << shape
              << std::setw(10) << (passed ? "PASS" : "FAIL");
    if (!passed)
    {
        std::cout << " first_mismatch_byte=" << first_mismatch;
    }
    std::cout << '\n';
}

template <typename Element>
bool validate_regular_case(
    const TransposeParam &config,
    TransposeDataType data_type,
    TransposeSharedMemoryLayout layout,
    cudaStream_t stream)
{
    const size_t input_elements =
        static_cast<size_t>(config.rows) * config.columns;
    const size_t output_elements =
        static_cast<size_t>(config.columns) * config.rows;
    const size_t input_bytes = input_elements * sizeof(Element);
    const size_t output_bytes = output_elements * sizeof(Element);

    std::vector<Element> input(input_elements);
    std::vector<Element> expected(output_elements);
    std::vector<Element> actual(output_elements);
    for (size_t index = 0; index < input_elements; ++index)
    {
        input[index] = make_value<Element>(index);
    }
    transpose_cpu(input, expected, config.rows, config.columns);

    DeviceBuffer device_input(input_bytes);
    DeviceBuffer device_output(output_bytes);
    check_cuda(
        cudaMemcpyAsync(
            device_input.get(),
            input.data(),
            input_bytes,
            cudaMemcpyHostToDevice,
            stream),
        "cudaMemcpyAsync validation input");
    check_cuda(
        cudaMemsetAsync(device_output.get(), 0xa5, output_bytes, stream),
        "cudaMemsetAsync validation output");
    check_cuda(
        cudaop::transpose_cuda(
            device_input.get(),
            device_output.get(),
            config.rows,
            config.columns,
            data_type,
            layout,
            stream),
        "transpose_cuda validation launch");
    check_cuda(
        cudaMemcpyAsync(
            actual.data(),
            device_output.get(),
            output_bytes,
            cudaMemcpyDeviceToHost,
            stream),
        "cudaMemcpyAsync validation output");
    check_cuda(
        cudaStreamSynchronize(stream),
        "cudaStreamSynchronize validation");

    size_t first_mismatch = 0;
    const bool passed = compare_bytes(
        actual.data(),
        expected.data(),
        output_bytes,
        &first_mismatch);
    print_validation_result(
        data_type,
        layout,
        config,
        passed,
        first_mismatch);
    return passed;
}

bool validate_fp4_case(
    const TransposeParam &config,
    TransposeSharedMemoryLayout layout,
    cudaStream_t stream)
{
    constexpr TransposeDataType data_type =
        TransposeDataType::kFloat4E2M1;
    const size_t input_bytes = cudaop::transpose_storage_bytes(
        config.rows,
        config.columns,
        data_type);
    const size_t output_bytes = cudaop::transpose_storage_bytes(
        config.columns,
        config.rows,
        data_type);
    std::vector<uint8_t> input(input_bytes, 0);
    std::vector<uint8_t> expected(output_bytes, 0);
    std::vector<uint8_t> actual(output_bytes, 0);

    for (uint32_t row = 0; row < config.rows; ++row)
    {
        for (uint32_t column = 0; column < config.columns; ++column)
        {
            const size_t index =
                static_cast<size_t>(row) * config.columns + column;
            set_fp4(
                input,
                config.columns,
                row,
                column,
                static_cast<uint8_t>((index * 7U + 3U) & 0x0FU));
        }
    }

    for (uint32_t row = 0; row < config.rows; ++row)
    {
        for (uint32_t column = 0; column < config.columns; ++column)
        {
            set_fp4(
                expected,
                config.rows,
                column,
                row,
                get_fp4(input, config.columns, row, column));
        }
    }

    DeviceBuffer device_input(input_bytes);
    DeviceBuffer device_output(output_bytes);
    check_cuda(
        cudaMemcpyAsync(
            device_input.get(),
            input.data(),
            input_bytes,
            cudaMemcpyHostToDevice,
            stream),
        "cudaMemcpyAsync FP4 validation input");
    check_cuda(
        cudaMemsetAsync(device_output.get(), 0xa5, output_bytes, stream),
        "cudaMemsetAsync FP4 validation output");
    check_cuda(
        cudaop::transpose_cuda(
            device_input.get(),
            device_output.get(),
            config.rows,
            config.columns,
            data_type,
            layout,
            stream),
        "transpose_cuda FP4 validation launch");
    check_cuda(
        cudaMemcpyAsync(
            actual.data(),
            device_output.get(),
            output_bytes,
            cudaMemcpyDeviceToHost,
            stream),
        "cudaMemcpyAsync FP4 validation output");
    check_cuda(
        cudaStreamSynchronize(stream),
        "cudaStreamSynchronize FP4 validation");

    size_t first_mismatch = 0;
    const bool passed = compare_bytes(
        actual.data(),
        expected.data(),
        output_bytes,
        &first_mismatch);
    print_validation_result(
        data_type,
        layout,
        config,
        passed,
        first_mismatch);
    return passed;
}

bool validate_arguments()
{
    DeviceBuffer buffer(16);
    bool passed = true;
    passed &= cudaop::transpose_cuda(
        nullptr,
        buffer.get(),
        1,
        1,
        TransposeDataType::kFloat32,
        TransposeSharedMemoryLayout::kPadded) == cudaErrorInvalidValue;
    passed &= cudaop::transpose_cuda(
        buffer.get(),
        buffer.get(),
        1,
        1,
        TransposeDataType::kFloat32,
        TransposeSharedMemoryLayout::kPadded) == cudaErrorInvalidValue;
    passed &= cudaop::transpose_cuda(
        buffer.get(),
        static_cast<uint8_t *>(buffer.get()) + 8,
        0,
        1,
        TransposeDataType::kFloat32,
        TransposeSharedMemoryLayout::kPadded) == cudaErrorInvalidValue;
    passed &= cudaop::transpose_cuda(
        buffer.get(),
        static_cast<uint8_t *>(buffer.get()) + 8,
        1,
        1,
        static_cast<TransposeDataType>(99),
        TransposeSharedMemoryLayout::kPadded) == cudaErrorInvalidValue;
    passed &= cudaop::transpose_cuda(
        buffer.get(),
        static_cast<uint8_t *>(buffer.get()) + 8,
        1,
        1,
        TransposeDataType::kFloat32,
        static_cast<TransposeSharedMemoryLayout>(99)) ==
        cudaErrorInvalidValue;
    passed &= cudaop::transpose_storage_bytes(
        3,
        5,
        TransposeDataType::kFloat32) == 60;
    passed &= cudaop::transpose_storage_bytes(
        3,
        5,
        TransposeDataType::kFloat4E2M1) == 9;
    passed &= cudaop::transpose_storage_bytes(
        5,
        3,
        TransposeDataType::kFloat4E2M1) == 10;

    std::cout << "  " << std::left << std::setw(40)
              << "invalid arguments and storage size"
              << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

bool validate_all(cudaStream_t stream)
{
    bool passed = true;
    for (const TransposeParam &config : kValidationCases)
    {
        for (TransposeSharedMemoryLayout layout : kLayouts)
        {
            passed &= validate_regular_case<float>(
                config,
                TransposeDataType::kFloat32,
                layout,
                stream);
            passed &= validate_regular_case<__nv_bfloat16>(
                config,
                TransposeDataType::kBfloat16,
                layout,
                stream);
            passed &= validate_regular_case<__nv_fp8_e4m3>(
                config,
                TransposeDataType::kFloat8E4M3,
                layout,
                stream);
            passed &= validate_fp4_case(config, layout, stream);
        }
    }
    return passed;
}

void benchmark_case(
    const TransposeParam &config,
    TransposeDataType data_type,
    TransposeSharedMemoryLayout layout,
    cudaStream_t stream)
{
    const size_t input_bytes = cudaop::transpose_storage_bytes(
        config.rows,
        config.columns,
        data_type);
    const size_t output_bytes = cudaop::transpose_storage_bytes(
        config.columns,
        config.rows,
        data_type);
    DeviceBuffer device_input(input_bytes);
    DeviceBuffer device_output(output_bytes);
    check_cuda(
        cudaMemsetAsync(device_input.get(), 0x35, input_bytes, stream),
        "cudaMemsetAsync benchmark input");

    for (uint32_t iteration = 0;
         iteration < config.warmup_iterations;
         ++iteration)
    {
        check_cuda(
            cudaop::transpose_cuda(
                device_input.get(),
                device_output.get(),
                config.rows,
                config.columns,
                data_type,
                layout,
                stream),
            "transpose_cuda benchmark warmup");
    }

    CudaEvent start;
    CudaEvent stop;
    check_cuda(cudaEventRecord(start.get(), stream), "cudaEventRecord start");
    for (uint32_t iteration = 0;
         iteration < config.benchmark_iterations;
         ++iteration)
    {
        check_cuda(
            cudaop::transpose_cuda(
                device_input.get(),
                device_output.get(),
                config.rows,
                config.columns,
                data_type,
                layout,
                stream),
            "transpose_cuda benchmark iteration");
    }
    check_cuda(cudaEventRecord(stop.get(), stream), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop.get()), "cudaEventSynchronize stop");

    float total_ms = 0.0F;
    check_cuda(
        cudaEventElapsedTime(&total_ms, start.get(), stop.get()),
        "cudaEventElapsedTime");
    const double average_ms =
        static_cast<double>(total_ms) / config.benchmark_iterations;
    const double effective_bandwidth_gbps =
        static_cast<double>(input_bytes + output_bytes) /
        (average_ms * 1.0e6);
    const std::string shape =
        std::to_string(config.rows) + "x" +
        std::to_string(config.columns);

    std::cout << "  " << std::left
              << std::setw(12) << cudaop::transpose_data_type_name(data_type)
              << std::setw(10) << cudaop::transpose_layout_name(layout)
              << std::setw(12) << shape
              << std::right << std::fixed << std::setprecision(4)
              << std::setw(10) << average_ms
              << std::setprecision(2)
              << std::setw(14) << effective_bandwidth_gbps << '\n';
}

void benchmark_all(cudaStream_t stream)
{
    constexpr std::array<TransposeDataType, 4> data_types = {
        TransposeDataType::kFloat32,
        TransposeDataType::kBfloat16,
        TransposeDataType::kFloat8E4M3,
        TransposeDataType::kFloat4E2M1};

    std::cout << "  " << std::left
              << std::setw(12) << "type"
              << std::setw(10) << "layout"
              << std::setw(12) << "M x N"
              << std::right << std::setw(10) << "ms"
              << std::setw(14) << "GB/s" << '\n';
    for (const TransposeParam &config : kBenchmarkCases)
    {
        for (TransposeDataType data_type : data_types)
        {
            for (TransposeSharedMemoryLayout layout : kLayouts)
            {
                benchmark_case(config, data_type, layout, stream);
            }
        }
    }
}

}  // namespace

int main(int argc, char **argv)
{
    try
    {
        bool run_benchmark = true;
        if (argc == 2 && std::string(argv[1]) == "--validate-only")
        {
            run_benchmark = false;
        }
        else if (argc != 1)
        {
            throw std::invalid_argument(
                "usage: transpose_test [--validate-only]");
        }

        int device = 0;
        cudaDeviceProp properties{};
        check_cuda(cudaGetDevice(&device), "cudaGetDevice");
        check_cuda(
            cudaGetDeviceProperties(&properties, device),
            "cudaGetDeviceProperties");
        CudaStream stream;

        std::cout << "\n[CONFIG] MxN matrix transpose" << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "GPU" << properties.name << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "Compute capability"
                  << properties.major << '.' << properties.minor << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "Logical tile" << "32x32" << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "Threads per block" << "32x8" << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "Data types"
                  << "float, bf16, fp8-e4m3, fp4-e2m1" << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "Shared-memory layouts" << "pad, XOR swizzle" << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "FP4 global packing" << "2 elements per byte" << '\n';

        std::cout << "\n[Stage 1/3] Validating API contracts" << '\n';
        if (!validate_arguments())
        {
            throw std::runtime_error("API contract validation failed");
        }

        std::cout << "\n[Stage 2/3] Running bitwise correctness tests" << '\n';
        if (!validate_all(stream.get()))
        {
            throw std::runtime_error("transpose correctness validation failed");
        }

        if (run_benchmark)
        {
            std::cout << "\n[Stage 3/3] Benchmarking CUDA kernels" << '\n';
            benchmark_all(stream.get());
        }
        else
        {
            std::cout << "\n[Stage 3/3] Benchmark skipped (--validate-only)"
                      << '\n';
        }

        std::cout << "\n[SUCCESS] All transpose variants passed" << '\n';
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cout << "\n[ERROR] " << error.what() << '\n';
        return 1;
    }
}
