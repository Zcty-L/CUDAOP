/***************************************************************************************************
 * Copyright (c) 2026
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

// Native signed INT4 x INT4 -> INT32 CuTe Tensor Core validation.
// A and B are row-major [M,K] and [N,K]. Two signed INT4 values are packed
// into every byte in two's-complement form; C[M,N] = A * B^T is INT32.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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
constexpr int kTileK = 32;
constexpr int kCpuSampleCount = 16;
constexpr uint32_t kSeedA = 0x31415926U;
constexpr uint32_t kSeedB = 0x27182818U;

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

#define CUDA_CHECK(expression) \
    check_cuda((expression), #expression, __FILE__, __LINE__)

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
            "M/N/K must be multiples of the 16x8x32 MMA tile");
    }

    const int64_t worst_case =
        static_cast<int64_t>(options.k) * 8 * 8;
    if (worst_case > std::numeric_limits<int32_t>::max())
    {
        throw std::runtime_error("K can overflow an INT32 accumulator");
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

int8_t make_int4(size_t index, uint32_t seed)
{
    const uint32_t folded =
        static_cast<uint32_t>(index) ^
        static_cast<uint32_t>(index >> 32);
    return static_cast<int8_t>(
        static_cast<int>(mix_bits(folded ^ seed) & 0xfU) - 8);
}

void set_packed_int4(std::vector<uint8_t> &packed, size_t index, int value)
{
    const uint8_t nibble = static_cast<uint8_t>(value) & 0xfU;
    const size_t byte_index = index / 2;
    if ((index & 1U) == 0U)
    {
        packed[byte_index] =
            static_cast<uint8_t>((packed[byte_index] & 0xf0U) | nibble);
    }
    else
    {
        packed[byte_index] = static_cast<uint8_t>(
            (packed[byte_index] & 0x0fU) | (nibble << 4));
    }
}

int unpack_int4(const std::vector<uint8_t> &packed, size_t index)
{
    const uint8_t byte = packed[index / 2];
    const int nibble = (index & 1U) == 0U
        ? static_cast<int>(byte & 0xfU)
        : static_cast<int>((byte >> 4) & 0xfU);
    return nibble >= 8 ? nibble - 16 : nibble;
}

std::vector<uint8_t> make_packed_input(size_t count, uint32_t seed)
{
    std::vector<uint8_t> packed((count + 1) / 2, 0U);
    for (size_t index = 0; index < count; ++index)
    {
        set_packed_int4(packed, index, make_int4(index, seed));
    }
    return packed;
}

template <class ProblemShape, class CtaTiler, class TiledMma>
__global__ __launch_bounds__(32)
void cute_gemm_int4_kernel(
    ProblemShape problem_shape,
    CtaTiler cta_tiler,
    const uint8_t *packed_a,
    const uint8_t *packed_b,
    int32_t *c,
    TiledMma tiled_mma)
{
    using namespace cute;

    const auto stride_a = make_stride(get<2>(problem_shape), Int<1>{});
    const auto stride_b = make_stride(get<2>(problem_shape), Int<1>{});
    const auto stride_c = make_stride(get<1>(problem_shape), Int<1>{});

    // The pointer remains byte-addressed physically, while CuTe's subbyte
    // iterator exposes one logical signed INT4 element per tensor coordinate.
    Tensor global_a = make_tensor(
        make_gmem_ptr<cutlass::int4b_t>(
            static_cast<const void *>(packed_a)),
        select<0, 2>(problem_shape),
        stride_a);
    Tensor global_b = make_tensor(
        make_gmem_ptr<cutlass::int4b_t>(
            static_cast<const void *>(packed_b)),
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
        // GMEM packed nibbles -> RMEM fragments in the lane/value order
        // required by m16n8k32. No dequantization occurs on this path.
        copy(thread_a(_, _, _, k_tile), fragment_a);
        copy(thread_b(_, _, _, k_tile), fragment_b);

        // Native signed IMMA: INT4 x INT4 products accumulate in INT32.
        cute::gemm(tiled_mma, fragment_a, fragment_b, accumulator);
    }

    copy(accumulator, thread_c);
}

void launch_cute(
    const uint8_t *packed_a,
    const uint8_t *packed_b,
    int32_t *c,
    int m,
    int n,
    int k,
    cudaStream_t stream)
{
    using namespace cute;

    const auto problem_shape = make_shape(m, n, k);
    const auto cta_tiler = make_shape(_16{}, _8{}, _32{});
    const auto tiled_mma = make_tiled_mma(
        SM80_16x8x32_S32S4S4S32_TN{},
        Layout<Shape<_1, _1, _1>>{},
        Tile<_16, _8, _32>{});
    const dim3 grid(m / kTileM, n / kTileN);
    const dim3 block(size(tiled_mma));

    cute_gemm_int4_kernel<<<grid, block, 0, stream>>>(
        problem_shape,
        cta_tiler,
        packed_a,
        packed_b,
        c,
        tiled_mma);
    CUDA_CHECK(cudaGetLastError());
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

struct SampleResult
{
    int row = 0;
    int column = 0;
    int32_t actual = 0;
    int32_t reference = 0;
};

std::vector<SampleResult> verify_cpu_samples(
    const std::vector<uint8_t> &packed_a,
    const std::vector<uint8_t> &packed_b,
    const std::vector<int32_t> &c,
    int m,
    int n,
    int k)
{
    std::vector<SampleResult> results;
    results.reserve(kCpuSampleCount);
    for (int sample = 0; sample < kCpuSampleCount; ++sample)
    {
        const int row = (sample * 97 + 3) % m;
        const int column = (sample * 53 + 7) % n;
        int32_t reference = 0;
        for (int reduction = 0; reduction < k; ++reduction)
        {
            const int value_a = unpack_int4(
                packed_a,
                static_cast<size_t>(row) * k + reduction);
            const int value_b = unpack_int4(
                packed_b,
                static_cast<size_t>(column) * k + reduction);
            reference += value_a * value_b;
        }
        results.push_back({
            row,
            column,
            c[static_cast<size_t>(row) * n + column],
            reference});
    }
    return results;
}

double tops(int m, int n, int k, float milliseconds)
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
        throw std::runtime_error("INT4 m16n8k32 mma.sync requires compute capability 8.0+");
    }

    const size_t count_a = static_cast<size_t>(options.m) * options.k;
    const size_t count_b = static_cast<size_t>(options.n) * options.k;
    const size_t count_c = static_cast<size_t>(options.m) * options.n;
    const std::vector<uint8_t> packed_a =
        make_packed_input(count_a, kSeedA);
    const std::vector<uint8_t> packed_b =
        make_packed_input(count_b, kSeedB);

    DeviceBuffer<uint8_t> device_a(packed_a.size());
    DeviceBuffer<uint8_t> device_b(packed_b.size());
    DeviceBuffer<int32_t> device_c(count_c);
    CudaStream stream;
    CUDA_CHECK(cudaMemcpyAsync(
        device_a.get(),
        packed_a.data(),
        packed_a.size(),
        cudaMemcpyHostToDevice,
        stream.get()));
    CUDA_CHECK(cudaMemcpyAsync(
        device_b.get(),
        packed_b.data(),
        packed_b.size(),
        cudaMemcpyHostToDevice,
        stream.get()));

    std::cout << "原生 INT4 Tensor Core GEMM 配置\n"
              << "  GPU                 : " << properties.name << '\n'
              << "  问题                : [" << options.m << ", "
              << options.k << "] x [" << options.n << ", "
              << options.k << "]^T\n"
              << "  输入/累加/输出       : signed INT4/signed INT32/signed INT32\n"
              << "  存储                : 2 个二补码 nibble/byte\n"
              << "  MMA                 : m16n8k32.row.col.s32.s4.s4.s32\n"
              << "  CTA                 : 1 warp, 16x8 output tile\n"
              << "  主循环 stages       : register-fed, no SMEM pipeline\n\n";

    std::cout << "阶段 1/3：执行 packed INT4 -> CuTe IMMA\n";
    const float elapsed_ms = benchmark(
        [&]()
        {
            launch_cute(
                device_a.get(),
                device_b.get(),
                device_c.get(),
                options.m,
                options.n,
                options.k,
                stream.get());
        },
        options.warmup,
        options.iterations,
        stream.get());

    std::vector<int32_t> host_c(count_c);
    CUDA_CHECK(cudaMemcpyAsync(
        host_c.data(),
        device_c.get(),
        count_c * sizeof(int32_t),
        cudaMemcpyDeviceToHost,
        stream.get()));
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));

    std::cout << "\n阶段 2/3：CPU 解包并精确验证 INT32 点积样本\n";
    const std::vector<SampleResult> samples = verify_cpu_samples(
        packed_a,
        packed_b,
        host_c,
        options.m,
        options.n,
        options.k);
    size_t mismatch_count = 0;
    int32_t max_absolute_error = 0;
    for (const SampleResult &sample : samples)
    {
        const int32_t error = std::abs(sample.actual - sample.reference);
        max_absolute_error = std::max(max_absolute_error, error);
        mismatch_count += static_cast<size_t>(error != 0);
    }
    std::cout << "  CPU 样本数          : " << samples.size() << '\n'
              << "  mismatch            : " << mismatch_count << '\n'
              << "  最大绝对误差        : " << max_absolute_error << '\n'
              << "  首个样本 (row,col)  : (" << samples.front().row << ", "
              << samples.front().column << ")\n"
              << "  首个样本 actual/ref : " << samples.front().actual << " / "
              << samples.front().reference << '\n';

    std::cout << "\n阶段 3/3：性能结果\n"
              << std::fixed << std::setprecision(6)
              << "  CuTe                : " << elapsed_ms << " ms, "
              << tops(options.m, options.n, options.k, elapsed_ms)
              << " TOPS\n\n";

    if (mismatch_count != 0)
    {
        throw std::runtime_error("native INT4 validation failed");
    }

    std::cout << "[SUCCESS] 原生 signed INT4 Tensor Core GEMM 验证通过\n";
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
