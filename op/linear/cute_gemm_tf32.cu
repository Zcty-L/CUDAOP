/***************************************************************************************************
 * Copyright (c) 2026
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

// FP32 storage -> TF32 Tensor Core MMA -> FP32 accumulate/output validation.
//
// This deliberately uses a register-fed, one-warp m16n8 kernel.  The small
// kernel isolates FP32-to-TF32 conversion and mma.sync semantics from shared
// memory staging.  A and B are row-major [M,K] and [N,K], so C = A * B^T.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>

namespace
{

constexpr int kDefaultM = 256;
constexpr int kDefaultN = 256;
constexpr int kDefaultK = 256;
constexpr int kDefaultWarmup = 2;
constexpr int kDefaultIterations = 5;
constexpr int kTileM = 16;
constexpr int kTileN = 8;
constexpr int kTileK = 8;
constexpr uint32_t kSeedA = 0x1234abcdU;
constexpr uint32_t kSeedB = 0x9e3779b9U;

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

    const T *get() const
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
        CUBLAS_CHECK(cublasSetMathMode(handle_, CUBLAS_TF32_TENSOR_OP_MATH));
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
    int m = kDefaultM;
    int n = kDefaultN;
    int k = kDefaultK;
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
        if (name == "--m")
        {
            options.m = value;
        }
        else if (name == "--n")
        {
            options.n = value;
        }
        else if (name == "--k")
        {
            options.k = value;
        }
        else if (name == "--warmup")
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

    if (options.m % kTileM != 0 ||
        options.n % kTileN != 0 ||
        options.k % kTileK != 0)
    {
        throw std::runtime_error(
            "M/N/K must be multiples of the 16x8x8 MMA tile");
    }
    return options;
}

uint32_t mix_bits(uint32_t value)
{
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return value;
}

float make_input(size_t index, uint32_t seed)
{
    const uint32_t folded =
        static_cast<uint32_t>(index) ^
        static_cast<uint32_t>(index >> 32);
    const uint32_t bits = mix_bits(folded ^ seed);
    const int centered = static_cast<int>(bits & 0x7ffU) - 1024;
    const float coarse = static_cast<float>(centered) / 1024.0F;
    const float fine = static_cast<float>((bits >> 12) & 0xffU) / 1048576.0F;
    return coarse + fine;
}

float round_to_tf32(float value)
{
    return static_cast<float>(cutlass::tfloat32_t(value));
}

template <class ProblemShape, class CtaTiler, class TiledMma>
__global__ __launch_bounds__(32)
void cute_gemm_tf32_kernel(
    ProblemShape problem_shape,
    CtaTiler cta_tiler,
    const float *a,
    const float *b,
    float *c,
    TiledMma tiled_mma)
{
    using namespace cute;

    const auto stride_a = make_stride(get<2>(problem_shape), Int<1>{});
    const auto stride_b = make_stride(get<2>(problem_shape), Int<1>{});
    const auto stride_c = make_stride(get<1>(problem_shape), Int<1>{});
    Tensor global_a = make_tensor(
        make_gmem_ptr(a),
        select<0, 2>(problem_shape),
        stride_a);
    Tensor global_b = make_tensor(
        make_gmem_ptr(b),
        select<1, 2>(problem_shape),
        stride_b);
    Tensor global_c = make_tensor(
        make_gmem_ptr(c),
        select<0, 1>(problem_shape),
        stride_c);

    const auto coordinate = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor block_a = local_tile(
        global_a,
        cta_tiler,
        coordinate,
        Step<_1, X, _1>{});
    Tensor block_b = local_tile(
        global_b,
        cta_tiler,
        coordinate,
        Step<X, _1, _1>{});
    Tensor block_c = local_tile(
        global_c,
        cta_tiler,
        coordinate,
        Step<_1, _1, X>{});

    auto thread_mma = tiled_mma.get_thread_slice(threadIdx.x);
    Tensor thread_a = thread_mma.partition_A(block_a);
    Tensor thread_b = thread_mma.partition_B(block_b);
    Tensor thread_c = thread_mma.partition_C(block_c);
    Tensor fragment_a =
        thread_mma.partition_fragment_A(block_a(_, _, Int<0>{}));
    Tensor fragment_b =
        thread_mma.partition_fragment_B(block_b(_, _, Int<0>{}));
    Tensor accumulator = thread_mma.make_fragment_C(thread_c);
    clear(accumulator);

    const int k_tile_count = size<3>(thread_a);
    CUTE_NO_UNROLL
    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile)
    {
        // GMEM FP32 -> RMEM TF32: the destination fragment type is
        // cutlass::tfloat32_t, so CuTe converts and rounds every operand here.
        copy(thread_a(_, _, _, k_tile), fragment_a);
        copy(thread_b(_, _, _, k_tile), fragment_b);

        // m16n8k8 mma.sync performs TF32 x TF32 with FP32 accumulation.
        cute::gemm(tiled_mma, fragment_a, fragment_b, accumulator);
    }

    copy(accumulator, thread_c);
}

void launch_cute(
    const float *a,
    const float *b,
    float *c,
    int m,
    int n,
    int k,
    cudaStream_t stream)
{
    using namespace cute;

    const auto problem_shape = make_shape(m, n, k);
    const auto cta_tiler = make_shape(_16{}, _8{}, _8{});
    const auto tiled_mma = make_tiled_mma(
        SM80_16x8x8_F32TF32TF32F32_TN{},
        Layout<Shape<_1, _1, _1>>{},
        Tile<_16, _8, _8>{});
    const dim3 grid(m / kTileM, n / kTileN);
    const dim3 block(size(tiled_mma));

    cute_gemm_tf32_kernel<<<grid, block, 0, stream>>>(
        problem_shape,
        cta_tiler,
        a,
        b,
        c,
        tiled_mma);
    CUDA_CHECK(cudaGetLastError());
}

void launch_cublas(
    cublasHandle_t handle,
    const float *a,
    const float *b,
    float *c,
    int m,
    int n,
    int k)
{
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        b,
        CUDA_R_32F,
        k,
        a,
        CUDA_R_32F,
        k,
        &beta,
        c,
        CUDA_R_32F,
        n,
        CUBLAS_COMPUTE_32F_FAST_TF32,
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

struct Comparison
{
    size_t mismatch_count = 0;
    float max_absolute_error = 0.0F;
    float max_relative_error = 0.0F;
};

Comparison compare_outputs(
    const std::vector<float> &actual,
    const std::vector<float> &reference)
{
    Comparison result;
    for (size_t index = 0; index < actual.size(); ++index)
    {
        const float absolute_error =
            std::abs(actual[index] - reference[index]);
        const float relative_error = absolute_error /
            std::max(std::abs(reference[index]), 1.0e-5F);
        result.max_absolute_error =
            std::max(result.max_absolute_error, absolute_error);
        result.max_relative_error =
            std::max(result.max_relative_error, relative_error);
        if (absolute_error > 2.0e-2F && relative_error > 2.0e-3F)
        {
            ++result.mismatch_count;
        }
    }
    return result;
}

struct CpuSamples
{
    size_t count = 0;
    size_t mismatch_count = 0;
    float max_absolute_error = 0.0F;
};

CpuSamples verify_cpu_samples(
    const std::vector<float> &a,
    const std::vector<float> &b,
    const std::vector<float> &c,
    int m,
    int n,
    int k)
{
    CpuSamples result;
    const int sample_rows[] = {0, m / 3, m / 2, m - 1};
    const int sample_columns[] = {0, n / 3, n / 2, n - 1};
    for (int sample = 0; sample < 4; ++sample)
    {
        const int row = sample_rows[sample];
        const int column = sample_columns[3 - sample];
        float reference = 0.0F;
        for (int reduction = 0; reduction < k; ++reduction)
        {
            reference = std::fma(
                round_to_tf32(a[static_cast<size_t>(row) * k + reduction]),
                round_to_tf32(b[static_cast<size_t>(column) * k + reduction]),
                reference);
        }

        const float actual = c[static_cast<size_t>(row) * n + column];
        const float error = std::abs(actual - reference);
        result.max_absolute_error =
            std::max(result.max_absolute_error, error);
        if (error > 2.0e-2F)
        {
            ++result.mismatch_count;
        }
        ++result.count;
    }
    return result;
}

double tflops(int m, int n, int k, float milliseconds)
{
    return 2.0 * static_cast<double>(m) * n * k /
        (static_cast<double>(milliseconds) * 1.0e9);
}

int run(const Options &options)
{
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    if (properties.major < 8)
    {
        throw std::runtime_error("TF32 mma.sync requires compute capability 8.0+");
    }

    const size_t count_a = static_cast<size_t>(options.m) * options.k;
    const size_t count_b = static_cast<size_t>(options.n) * options.k;
    const size_t count_c = static_cast<size_t>(options.m) * options.n;
    std::vector<float> host_a(count_a);
    std::vector<float> host_b(count_b);
    for (size_t index = 0; index < count_a; ++index)
    {
        host_a[index] = make_input(index, kSeedA);
    }
    for (size_t index = 0; index < count_b; ++index)
    {
        host_b[index] = make_input(index, kSeedB);
    }

    DeviceBuffer<float> device_a(count_a);
    DeviceBuffer<float> device_b(count_b);
    DeviceBuffer<float> device_cute(count_c);
    DeviceBuffer<float> device_cublas(count_c);
    CudaStream stream;
    CublasHandle cublas;
    CUBLAS_CHECK(cublasSetStream(cublas.get(), stream.get()));
    CUDA_CHECK(cudaMemcpyAsync(
        device_a.get(),
        host_a.data(),
        count_a * sizeof(float),
        cudaMemcpyHostToDevice,
        stream.get()));
    CUDA_CHECK(cudaMemcpyAsync(
        device_b.get(),
        host_b.data(),
        count_b * sizeof(float),
        cudaMemcpyHostToDevice,
        stream.get()));

    std::cout << "TF32 Tensor Core GEMM 配置\n"
              << "  GPU                 : " << properties.name << '\n'
              << "  问题                : [" << options.m << ", "
              << options.k << "] x [" << options.n << ", "
              << options.k << "]^T\n"
              << "  存储/乘法/累加/输出 : FP32/TF32/FP32/FP32\n"
              << "  MMA                 : m16n8k8.row.col.f32.tf32.tf32.f32\n"
              << "  CTA                 : 1 warp, 16x8 output tile\n"
              << "  主循环 stages       : register-fed, no SMEM pipeline\n\n";

    std::cout << "阶段 1/3：运行 CuTe TF32 MMA 与 cuBLAS FAST_TF32\n";
    const float cute_ms = benchmark(
        [&]()
        {
            launch_cute(
                device_a.get(),
                device_b.get(),
                device_cute.get(),
                options.m,
                options.n,
                options.k,
                stream.get());
        },
        options.warmup,
        options.iterations,
        stream.get());
    const float cublas_ms = benchmark(
        [&]()
        {
            launch_cublas(
                cublas.get(),
                device_a.get(),
                device_b.get(),
                device_cublas.get(),
                options.m,
                options.n,
                options.k);
        },
        options.warmup,
        options.iterations,
        stream.get());

    std::vector<float> host_cute(count_c);
    std::vector<float> host_cublas(count_c);
    CUDA_CHECK(cudaMemcpyAsync(
        host_cute.data(),
        device_cute.get(),
        count_c * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream.get()));
    CUDA_CHECK(cudaMemcpyAsync(
        host_cublas.data(),
        device_cublas.get(),
        count_c * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream.get()));
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));

    std::cout << "\n阶段 2/3：对比 cuBLAS TF32 与 CPU TF32-rounded 样本\n";
    const Comparison comparison = compare_outputs(host_cute, host_cublas);
    const CpuSamples samples = verify_cpu_samples(
        host_a,
        host_b,
        host_cute,
        options.m,
        options.n,
        options.k);
    std::cout << "  cuBLAS mismatch     : " << comparison.mismatch_count << '\n'
              << "  最大绝对误差        : " << comparison.max_absolute_error << '\n'
              << "  最大相对误差        : " << comparison.max_relative_error << '\n'
              << "  CPU 样本            : " << samples.count << '\n'
              << "  CPU mismatch        : " << samples.mismatch_count << '\n'
              << "  CPU 最大绝对误差    : " << samples.max_absolute_error << '\n';

    std::cout << "\n阶段 3/3：性能结果\n"
              << std::fixed << std::setprecision(6)
              << "  CuTe                : " << cute_ms << " ms, "
              << tflops(options.m, options.n, options.k, cute_ms)
              << " TFLOP/s\n"
              << "  cuBLAS              : " << cublas_ms << " ms, "
              << tflops(options.m, options.n, options.k, cublas_ms)
              << " TFLOP/s\n\n";

    if (comparison.mismatch_count != 0 || samples.mismatch_count != 0)
    {
        throw std::runtime_error("TF32 validation failed");
    }

    std::cout << "[SUCCESS] TF32 Tensor Core GEMM 验证通过\n";
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
