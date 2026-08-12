/***************************************************************************************************
 * Copyright (c) 2026
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

// Instruction-level 2:4 structured sparse Tensor Core validation.
//
// CUTLASS/CuTe does not expose an SM80 sparse MMA Atom alongside the dense
// SM80 CuTe atoms, so this focused one-warp test uses CUTLASS's official
// cutlass::arch::SparseMma wrapper. It still executes a real mma.sp instruction.
// A dense 16x32 FP16 tile is explicitly pruned to positions {0,1} in every
// group of four, compressed to 16x16 values, and accompanied by 0x4 metadata.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cutlass/arch/mma_sparse_sm80.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

namespace
{

constexpr int kM = 16;
constexpr int kN = 8;
constexpr int kK = 32;
constexpr int kCompressedK = kK / 2;
constexpr int kWarpSize = 32;
constexpr int kDefaultWarmup = 2;
constexpr int kDefaultIterations = 20;
constexpr uint32_t kMetadataNibble = 0x4U;
constexpr uint32_t kPackedMetadata = 0x44444444U;

using SparseMma = cutlass::arch::SparseMma<
    cutlass::gemm::GemmShape<kM, kN, kK>,
    kWarpSize,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    float,
    cutlass::layout::RowMajor,
    cutlass::arch::OpMultiplyAdd,
    cutlass::arch::SPFormatType::Thread>;

void check_cuda(
    cudaError_t status,
    const char *expression,
    const char *file,
    int line)
{
    if (status == cudaSuccess)
    {
        return;
    }

    throw std::runtime_error(
        std::string("CUDA error: ") + cudaGetErrorString(status) +
        ", expression=" + expression +
        ", location=" + file + ':' + std::to_string(line));
}

void check_cublas(
    cublasStatus_t status,
    const char *expression,
    const char *file,
    int line)
{
    if (status == CUBLAS_STATUS_SUCCESS)
    {
        return;
    }

    throw std::runtime_error(
        "cuBLAS error: status=" + std::to_string(status) +
        ", expression=" + expression +
        ", location=" + file + ':' + std::to_string(line));
}

#define CUDA_CHECK(expression) \
    check_cuda((expression), #expression, __FILE__, __LINE__)

#define CUBLAS_CHECK(expression) \
    check_cublas((expression), #expression, __FILE__, __LINE__)

template <class T>
class DeviceBuffer
{
public:
    explicit DeviceBuffer(size_t count)
    {
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&pointer_),
            count * sizeof(T)));
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

    T *get()
    {
        return pointer_;
    }

private:
    T *pointer_ = nullptr;
};

class CudaStream
{
public:
    CudaStream()
    {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    }

    ~CudaStream()
    {
        if (stream_ != nullptr)
        {
            cudaStreamDestroy(stream_);
        }
    }

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
        CUDA_CHECK(cudaEventCreate(&event_));
    }

    ~CudaEvent()
    {
        if (event_ != nullptr)
        {
            cudaEventDestroy(event_);
        }
    }

    cudaEvent_t get() const
    {
        return event_;
    }

private:
    cudaEvent_t event_ = nullptr;
};

class CublasHandle
{
public:
    CublasHandle()
    {
        CUBLAS_CHECK(cublasCreate(&handle_));
        CUBLAS_CHECK(cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH));
    }

    ~CublasHandle()
    {
        if (handle_ != nullptr)
        {
            cublasDestroy(handle_);
        }
    }

    cublasHandle_t get() const
    {
        return handle_;
    }

private:
    cublasHandle_t handle_ = nullptr;
};

struct Options
{
    int warmup = kDefaultWarmup;
    int iterations = kDefaultIterations;
};

int parse_positive(const char *text, const char *name)
{
    size_t parsed = 0;
    const int value = std::stoi(text, &parsed);
    if (parsed != std::string(text).size() || value <= 0)
    {
        throw std::runtime_error(
            std::string(name) + " requires a positive integer");
    }
    return value;
}

Options parse_options(int argc, char **argv)
{
    Options options;
    for (int index = 1; index < argc; index += 2)
    {
        if (index + 1 >= argc)
        {
            throw std::runtime_error(
                std::string("missing value for ") + argv[index]);
        }

        const std::string name = argv[index];
        const int value = parse_positive(argv[index + 1], argv[index]);
        if (name == "--warmup")
        {
            options.warmup = value;
        }
        else if (name == "--iterations")
        {
            options.iterations = value;
        }
        else
        {
            throw std::runtime_error("unknown option: " + name);
        }
    }
    return options;
}

struct SparseInputs
{
    std::vector<__half> dense_a;
    std::vector<__half> compressed_a;
    std::vector<__half> b;
    std::vector<uint32_t> metadata;
};

SparseInputs make_sparse_inputs()
{
    SparseInputs inputs;
    inputs.dense_a.resize(kM * kK);
    inputs.compressed_a.resize(kM * kCompressedK);
    inputs.b.resize(kN * kK, __float2half(1.0F));
    inputs.metadata.resize(kWarpSize, kPackedMetadata);

    for (int row = 0; row < kM; ++row)
    {
        for (int group = 0; group < kK / 4; ++group)
        {
            const int dense_base = row * kK + group * 4;
            const int compressed_base = row * kCompressedK + group * 2;

            // Metadata nibble 0x4 encodes kept indices {0,1}. The other two
            // dense positions are explicit zeros, satisfying exactly 2:4.
            inputs.dense_a[dense_base + 0] = __float2half(1.0F);
            inputs.dense_a[dense_base + 1] = __float2half(1.0F);
            inputs.dense_a[dense_base + 2] = __float2half(0.0F);
            inputs.dense_a[dense_base + 3] = __float2half(0.0F);
            inputs.compressed_a[compressed_base + 0] = __float2half(1.0F);
            inputs.compressed_a[compressed_base + 1] = __float2half(1.0F);
        }
    }
    return inputs;
}

bool validate_2_to_4(const std::vector<__half> &dense_a)
{
    for (int row = 0; row < kM; ++row)
    {
        for (int group = 0; group < kK / 4; ++group)
        {
            int nonzero_count = 0;
            for (int offset = 0; offset < 4; ++offset)
            {
                const float value = __half2float(
                    dense_a[row * kK + group * 4 + offset]);
                nonzero_count += static_cast<int>(value != 0.0F);
            }
            if (nonzero_count != 2)
            {
                return false;
            }
        }
    }
    return true;
}

__global__ __launch_bounds__(kWarpSize)
void sparse_mma_tile_kernel(
    const cutlass::half_t *compressed_a,
    const cutlass::half_t *b,
    const uint32_t *metadata,
    float *c)
{
    const int lane = static_cast<int>(threadIdx.x);

    typename SparseMma::FragmentA fragment_a;
    typename SparseMma::FragmentB fragment_b;
    typename SparseMma::FragmentC accumulator;
    typename SparseMma::FragmentC source;

    // Each lane consumes eight compressed A values and eight dense B values.
    // The validation tile uses all-one retained values, which intentionally
    // isolates metadata selection and sparse MMA semantics from layout iterators.
    for (int index = 0; index < SparseMma::FragmentA::kElements; ++index)
    {
        fragment_a[index] = compressed_a[
            (lane * SparseMma::FragmentA::kElements + index) %
            (kM * kCompressedK)];
    }
    for (int index = 0; index < SparseMma::FragmentB::kElements; ++index)
    {
        fragment_b[index] = b[
            (lane * SparseMma::FragmentB::kElements + index) %
            (kN * kK)];
    }
    source.clear();

    // id2=0 selects the first metadata source group. Every participating lane
    // holds the same legal packed metadata, so the selected register is 0x44444444.
    SparseMma mma;
    mma(
        accumulator,
        fragment_a,
        fragment_b,
        source,
        metadata[lane],
        0);

    // CUTLASS/CuTe's m16n8 accumulator lane layout:
    // lane=(m_low, n_pair), value=(n_in_pair, m_high).
    const int m_low = lane % 8;
    const int n_pair = lane / 8;
    for (int value = 0; value < 4; ++value)
    {
        const int n_in_pair = value % 2;
        const int m_high = value / 2;
        const int row = m_low + m_high * 8;
        const int column = n_pair * 2 + n_in_pair;
        c[row * kN + column] = accumulator[value];
    }
}

void launch_sparse(
    const cutlass::half_t *compressed_a,
    const cutlass::half_t *b,
    const uint32_t *metadata,
    float *c,
    cudaStream_t stream)
{
    sparse_mma_tile_kernel<<<1, kWarpSize, 0, stream>>>(
        compressed_a,
        b,
        metadata,
        c);
    CUDA_CHECK(cudaGetLastError());
}

void launch_cublas(
    cublasHandle_t handle,
    const __half *dense_a,
    const __half *b,
    float *c)
{
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        kN,
        kM,
        kK,
        &alpha,
        b,
        CUDA_R_16F,
        kK,
        dense_a,
        CUDA_R_16F,
        kK,
        &beta,
        c,
        CUDA_R_32F,
        kN,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

template <class Launch>
float benchmark(
    Launch launch,
    int warmup,
    int iterations,
    cudaStream_t stream)
{
    for (int index = 0; index < warmup; ++index)
    {
        launch();
    }

    CudaEvent start;
    CudaEvent stop;
    CUDA_CHECK(cudaEventRecord(start.get(), stream));
    for (int index = 0; index < iterations; ++index)
    {
        launch();
    }
    CUDA_CHECK(cudaEventRecord(stop.get(), stream));
    CUDA_CHECK(cudaEventSynchronize(stop.get()));

    float elapsed_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start.get(), stop.get()));
    return elapsed_ms / static_cast<float>(iterations);
}

std::vector<float> cpu_reference(const SparseInputs &inputs)
{
    std::vector<float> reference(kM * kN, 0.0F);
    for (int row = 0; row < kM; ++row)
    {
        for (int column = 0; column < kN; ++column)
        {
            float accumulator = 0.0F;
            for (int reduction = 0; reduction < kK; ++reduction)
            {
                accumulator = std::fma(
                    __half2float(inputs.dense_a[row * kK + reduction]),
                    __half2float(inputs.b[column * kK + reduction]),
                    accumulator);
            }
            reference[row * kN + column] = accumulator;
        }
    }
    return reference;
}

struct Comparison
{
    size_t mismatch_count = 0;
    float max_absolute_error = 0.0F;
};

Comparison compare_exact(
    const std::vector<float> &actual,
    const std::vector<float> &reference)
{
    Comparison result;
    for (size_t index = 0; index < actual.size(); ++index)
    {
        const float error = std::abs(actual[index] - reference[index]);
        result.max_absolute_error =
            std::max(result.max_absolute_error, error);
        result.mismatch_count += static_cast<size_t>(error != 0.0F);
    }
    return result;
}

int run(const Options &options)
{
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    if (properties.major < 8)
    {
        throw std::runtime_error("FP16 sparse mma requires compute capability 8.0+");
    }

    const SparseInputs inputs = make_sparse_inputs();
    if (!validate_2_to_4(inputs.dense_a))
    {
        throw std::runtime_error("generated A does not satisfy exact 2:4 sparsity");
    }

    DeviceBuffer<cutlass::half_t> device_compressed_a(
        inputs.compressed_a.size());
    DeviceBuffer<cutlass::half_t> device_b(inputs.b.size());
    DeviceBuffer<__half> device_dense_a(inputs.dense_a.size());
    DeviceBuffer<uint32_t> device_metadata(inputs.metadata.size());
    DeviceBuffer<float> device_sparse_c(kM * kN);
    DeviceBuffer<float> device_dense_c(kM * kN);
    CudaStream stream;
    CublasHandle cublas;
    CUBLAS_CHECK(cublasSetStream(cublas.get(), stream.get()));

    CUDA_CHECK(cudaMemcpyAsync(
        device_compressed_a.get(),
        inputs.compressed_a.data(),
        inputs.compressed_a.size() * sizeof(__half),
        cudaMemcpyHostToDevice,
        stream.get()));
    CUDA_CHECK(cudaMemcpyAsync(
        device_b.get(),
        inputs.b.data(),
        inputs.b.size() * sizeof(__half),
        cudaMemcpyHostToDevice,
        stream.get()));
    CUDA_CHECK(cudaMemcpyAsync(
        device_dense_a.get(),
        inputs.dense_a.data(),
        inputs.dense_a.size() * sizeof(__half),
        cudaMemcpyHostToDevice,
        stream.get()));
    CUDA_CHECK(cudaMemcpyAsync(
        device_metadata.get(),
        inputs.metadata.data(),
        inputs.metadata.size() * sizeof(uint32_t),
        cudaMemcpyHostToDevice,
        stream.get()));

    std::cout << "2:4 Structured Sparse Tensor Core 配置\n"
              << "  GPU                 : " << properties.name << '\n'
              << "  固定指令 tile       : 16x8x32, one warp\n"
              << "  输入/累加/输出       : FP16/FP32/FP32\n"
              << "  A dense/compressed  : " << kM * kK << " / "
              << kM * kCompressedK << " elements\n"
              << "  2:4 保留位置        : {0, 1}\n"
              << "  metadata nibble     : 0x" << std::hex
              << kMetadataNibble << std::dec << '\n'
              << "  packed metadata     : 0x" << std::hex
              << kPackedMetadata << std::dec << '\n'
              << "  MMA                 : mma.sp m16n8k32 f32.f16.f16.f32\n"
              << "  范围                : 指令级 tile；未实现多 CTA/SMEM pipeline\n\n";

    std::cout << "阶段 1/3：验证 2:4、压缩 A 与 metadata 后执行 sparse MMA\n";
    const float sparse_ms = benchmark(
        [&]()
        {
            launch_sparse(
                device_compressed_a.get(),
                device_b.get(),
                device_metadata.get(),
                device_sparse_c.get(),
                stream.get());
        },
        options.warmup,
        options.iterations,
        stream.get());

    std::cout << "\n阶段 2/3：运行展开 A 的 dense cuBLAS 和 CPU reference\n";
    launch_cublas(
        cublas.get(),
        device_dense_a.get(),
        reinterpret_cast<const __half *>(device_b.get()),
        device_dense_c.get());

    std::vector<float> sparse_c(kM * kN);
    std::vector<float> dense_c(kM * kN);
    CUDA_CHECK(cudaMemcpyAsync(
        sparse_c.data(),
        device_sparse_c.get(),
        sparse_c.size() * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream.get()));
    CUDA_CHECK(cudaMemcpyAsync(
        dense_c.data(),
        device_dense_c.get(),
        dense_c.size() * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream.get()));
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));

    const std::vector<float> cpu_c = cpu_reference(inputs);
    const Comparison cublas_comparison = compare_exact(sparse_c, dense_c);
    const Comparison cpu_comparison = compare_exact(sparse_c, cpu_c);
    std::cout << "  2:4 结构合法        : yes\n"
              << "  sparse/cuBLAS mismatch: "
              << cublas_comparison.mismatch_count << '\n'
              << "  sparse/CPU mismatch : "
              << cpu_comparison.mismatch_count << '\n'
              << "  最大绝对误差        : "
              << std::max(
                     cublas_comparison.max_absolute_error,
                     cpu_comparison.max_absolute_error)
              << '\n'
              << "  C[0,0] sparse/dense : " << sparse_c.front() << " / "
              << dense_c.front() << '\n';

    std::cout << "\n阶段 3/3：指令级延迟结果\n"
              << std::fixed << std::setprecision(6)
              << "  kernel launch       : " << sparse_ms << " ms\n\n";

    if (cublas_comparison.mismatch_count != 0 ||
        cpu_comparison.mismatch_count != 0)
    {
        throw std::runtime_error("2:4 sparse MMA validation failed");
    }

    std::cout << "[SUCCESS] 2:4 structured sparse MMA 指令验证通过\n";
    return 0;
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        return run(parse_options(argc, argv));
    }
    catch (const std::exception &error)
    {
        std::cout << "[FAILED] " << error.what() << '\n';
        return 1;
    }
}
