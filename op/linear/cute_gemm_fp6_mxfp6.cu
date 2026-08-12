/***************************************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

// SM120 CuTe FP6 / MXFP6 GEMM 指令验证：
//   Case A: E3M2 x E3M2 -> FP32 accumulate -> FP16 output；
//   Case B: E3M2 + UE8M0 block32 -> FP32 accumulate -> FP16 output。
//
// A/B 分别按 row-major [M,K] / [N,K] 保存，计算 C[M,N] = A * B^T。
// PTX f8f6f4 的 FP6 寄存器接口要求每个值占一个 8-bit container，仅低 6 bit
// 有效。本测试因此不使用 CUTLASS 的紧凑 6-bit GMEM iterator，而是显式保存
// 一个 E3M2 对应一个 byte，并在 MMA kernel 中按 uint8_t 原始位搬入 fragment。

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cute/algorithm/gemm.hpp>
#include <cute/tensor.hpp>

namespace
{

constexpr int kDefaultM = 8192;
constexpr int kDefaultN = 8192;
constexpr int kDefaultK = 8192;
constexpr int kDefaultWarmupIterations = 2;
constexpr int kDefaultBenchmarkIterations = 5;

constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockK = 64;
constexpr int kScaleVectorSize = 32;
constexpr float kE3M2Maximum = 28.0F;
constexpr float kAbsoluteTolerance = 5.0e-1F;
constexpr float kRelativeTolerance = 1.0e-2F;
constexpr uint32_t kSeedA = 0x1234abcdU;
constexpr uint32_t kSeedB = 0x9e3779b9U;

constexpr const char *kFp6CuteName = "CuTe SM120 FP6";
constexpr const char *kMxFp6CuteName = "CuTe SM120 MXFP6";
constexpr const char *kCublasName = "cuBLAS FP16 reference";

static_assert(
    sizeof(cutlass::float_e3m2_t) == sizeof(uint8_t),
    "E3M2 storage must use one byte per logical value");

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

template <typename T>
class DeviceBuffer
{
public:
    explicit DeviceBuffer(size_t count)
        : count_(count)
    {
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&pointer_),
            count_ * sizeof(T)));
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
    size_t count_ = 0;
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
        CUDA_CHECK(cudaEventCreate(&event_));
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

class CublasFp16Reference
{
public:
    CublasFp16Reference()
    {
        CUBLAS_CHECK(cublasCreate(&handle_));
        CUBLAS_CHECK(cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH));
    }

    ~CublasFp16Reference()
    {
        if (handle_ != nullptr)
        {
            cublasDestroy(handle_);
        }
    }

    CublasFp16Reference(const CublasFp16Reference &) = delete;
    CublasFp16Reference &operator=(const CublasFp16Reference &) = delete;

    void launch(
        const __half *a,
        const __half *b,
        __half *c,
        int m,
        int n,
        int k,
        cudaStream_t stream)
    {
        constexpr float alpha = 1.0F;
        constexpr float beta = 0.0F;
        CUBLAS_CHECK(cublasSetStream(handle_, stream));

        // cuBLAS 使用 column-major。row-major C=A*B^T 等价于
        // column-major C^T=B*A^T，因此交换 A/B 的参数位置。
        CUBLAS_CHECK(cublasGemmEx(
            handle_,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            n,
            m,
            k,
            &alpha,
            b,
            CUDA_R_16F,
            k,
            a,
            CUDA_R_16F,
            k,
            &beta,
            c,
            CUDA_R_16F,
            n,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

private:
    cublasHandle_t handle_ = nullptr;
};

struct Options
{
    int m = kDefaultM;
    int n = kDefaultN;
    int k = kDefaultK;
    int warmup_iterations = kDefaultWarmupIterations;
    int benchmark_iterations = kDefaultBenchmarkIterations;
};

int parse_positive_integer(const char *text, const char *option_name)
{
    size_t parsed_characters = 0;
    const int value = std::stoi(text, &parsed_characters);
    if (parsed_characters != std::string(text).size() || value <= 0)
    {
        throw std::runtime_error(
            std::string(option_name) + " requires a positive integer");
    }
    return value;
}

Options parse_options(int argc, char **argv)
{
    Options options;
    for (int index = 1; index < argc; ++index)
    {
        const std::string option = argv[index];
        if (index + 1 >= argc)
        {
            throw std::runtime_error(option + " requires a value");
        }

        if (option == "--m")
        {
            options.m = parse_positive_integer(argv[++index], "--m");
        }
        else if (option == "--n")
        {
            options.n = parse_positive_integer(argv[++index], "--n");
        }
        else if (option == "--k")
        {
            options.k = parse_positive_integer(argv[++index], "--k");
        }
        else if (option == "--warmup")
        {
            options.warmup_iterations =
                parse_positive_integer(argv[++index], "--warmup");
        }
        else if (option == "--iterations")
        {
            options.benchmark_iterations =
                parse_positive_integer(argv[++index], "--iterations");
        }
        else
        {
            throw std::runtime_error("unknown option: " + option);
        }
    }

    if (options.m % kBlockM != 0 ||
        options.n % kBlockN != 0 ||
        options.k % kBlockK != 0)
    {
        throw std::runtime_error(
            "M/N/K must be multiples of 128/128/64");
    }
    return options;
}

__host__ __device__ uint32_t mix_bits(uint32_t value)
{
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return value;
}

__host__ __device__ float make_fp6_input_value(
    size_t index,
    uint32_t seed)
{
    const uint32_t folded_index =
        static_cast<uint32_t>(index) ^
        static_cast<uint32_t>(index >> 32);
    const uint32_t bits = mix_bits(folded_index ^ seed);
    const int centered = static_cast<int>(bits & 0x3ffU) - 512;
    return static_cast<float>(centered) * (1.0F / 512.0F);
}

__host__ __device__ float make_mxfp6_input_value(
    size_t index,
    uint32_t seed)
{
    const size_t scale_block = index / kScaleVectorSize;
    const uint32_t amplitude_code =
        mix_bits(static_cast<uint32_t>(scale_block) ^ (seed >> 7)) & 0x3U;
    const float amplitude =
        static_cast<float>(1U << amplitude_code) * 0.5F;
    return make_fp6_input_value(index, seed) * amplitude;
}

__host__ __device__ cutlass::float_ue8m0_t make_dynamic_scale(
    float maximum_absolute_value)
{
    // UE8M0 转换采用向正无穷舍入，因此结果是不小于 absmax/28 的
    // 最小 2 的幂，保证除以 scale 后落在 E3M2 的 [-28,28] 范围。
    const float requested_scale = maximum_absolute_value > 0.0F
        ? maximum_absolute_value / kE3M2Maximum
        : 1.0F;
    return cutlass::float_ue8m0_t(requested_scale);
}

__global__ void quantize_fp6_kernel(
    cutlass::float_e3m2_t *quantized,
    __half *dequantized,
    size_t element_count,
    uint32_t seed)
{
    const size_t first =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride =
        static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = first; index < element_count; index += stride)
    {
        const cutlass::float_e3m2_t value(
            make_fp6_input_value(index, seed));
        quantized[index] = value;
        dequantized[index] = __float2half_rn(static_cast<float>(value));
    }
}

__global__ void quantize_mxfp6_kernel(
    cutlass::float_e3m2_t *quantized,
    cutlass::float_ue8m0_t *scales,
    __half *dequantized,
    size_t scale_count,
    uint32_t seed)
{
    const size_t first =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride =
        static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t scale_index = first;
         scale_index < scale_count;
         scale_index += stride)
    {
        const size_t block_start = scale_index * kScaleVectorSize;
        float maximum_absolute_value = 0.0F;
        for (int lane = 0; lane < kScaleVectorSize; ++lane)
        {
            maximum_absolute_value = fmaxf(
                maximum_absolute_value,
                fabsf(make_mxfp6_input_value(block_start + lane, seed)));
        }

        const cutlass::float_ue8m0_t encoded_scale =
            make_dynamic_scale(maximum_absolute_value);
        const float scale = static_cast<float>(encoded_scale);
        scales[scale_index] = encoded_scale;

        for (int lane = 0; lane < kScaleVectorSize; ++lane)
        {
            const size_t element_index = block_start + lane;
            const cutlass::float_e3m2_t value(
                make_mxfp6_input_value(element_index, seed) / scale);
            quantized[element_index] = value;
            dequantized[element_index] = __float2half_rn(
                static_cast<float>(value) * scale);
        }
    }
}

template <
    class ProblemShape,
    class CtaTiler,
    class AStride,
    class BStride,
    class CStride,
    class TiledMma>
__global__ __launch_bounds__(decltype(cute::size(TiledMma{}))::value)
void cute_gemm_fp6_kernel(
    ProblemShape problem_shape,
    CtaTiler cta_tiler,
    const cutlass::float_e3m2_t *a,
    AStride stride_a,
    const cutlass::float_e3m2_t *b,
    BStride stride_b,
    __half *c,
    CStride stride_c,
    TiledMma tiled_mma)
{
    using namespace cute;

    CUTE_STATIC_ASSERT_V(rank(problem_shape) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});

    // 关键点：一个 FP6 值必须占一个 byte，低六位原样进入 MMA。
    // reinterpret 为 uint8_t 后，CuTe copy 进行纯位搬运，不会把 E3M2
    // 转成整数，也不会按照 sizeof_bits<E3M2> == 6 紧凑寻址。
    Tensor global_a = make_tensor(
        make_gmem_ptr(reinterpret_cast<const uint8_t *>(a)),
        select<0, 2>(problem_shape),
        stride_a);
    Tensor global_b = make_tensor(
        make_gmem_ptr(reinterpret_cast<const uint8_t *>(b)),
        select<1, 2>(problem_shape),
        stride_b);
    Tensor global_c = make_tensor(
        make_gmem_ptr(c),
        select<0, 1>(problem_shape),
        stride_c);

    const auto cta_coordinate = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor block_a = local_tile(
        global_a,
        cta_tiler,
        cta_coordinate,
        Step<_1, X, _1>{});
    Tensor block_b = local_tile(
        global_b,
        cta_tiler,
        cta_coordinate,
        Step<X, _1, _1>{});
    Tensor block_c = local_tile(
        global_c,
        cta_tiler,
        cta_coordinate,
        Step<_1, _1, X>{});

    auto thread_mma = tiled_mma.get_thread_slice(threadIdx.x);
    Tensor thread_global_a = thread_mma.partition_A(block_a);
    Tensor thread_global_b = thread_mma.partition_B(block_b);
    Tensor thread_global_c = thread_mma.partition_C(block_c);
    Tensor fragment_a =
        thread_mma.partition_fragment_A(block_a(_, _, Int<0>{}));
    Tensor fragment_b =
        thread_mma.partition_fragment_B(block_b(_, _, Int<0>{}));
    Tensor accumulator = thread_mma.make_fragment_C(thread_global_c);
    clear(accumulator);

    const int k_tile_count = size<3>(thread_global_a);
    CUTE_NO_UNROLL
    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile)
    {
        // GMEM -> RMEM：TiledMMA 为每个线程挑选当前 64-wide K tile
        // 所需的 E3M2 原始 byte，随后普通 f8f6f4 MMA 累加到 FP32。
        copy(thread_global_a(_, _, _, k_tile), fragment_a);
        copy(thread_global_b(_, _, _, k_tile), fragment_b);
        cute::gemm(tiled_mma, fragment_a, fragment_b, accumulator);
    }

    // Epilogue：FP32 accumulator 只在最终写回时舍入为 FP16。
    CUTE_UNROLL
    for (int index = 0; index < size(accumulator); ++index)
    {
        thread_global_c(index) =
            __float2half_rn(static_cast<float>(accumulator(index)));
    }
}

// CuTe block-scaled MMA 的 scale 不是普通数据 operand。以下 helper 使用
// Atom 的 SFA/SFB thread-value layout，把二维 scale tensor 分配给线程。
template <class SFATensor, class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr auto scale_partition_layout_a(
    SFATensor &&scale_tensor,
    cute::TiledMMA<Atom, TiledThr, TiledPerm> &mma)
{
    using namespace cute;

    CUTE_STATIC_ASSERT_V(rank(scale_tensor) >= Int<2>{});
    using AtomShape = typename Atom::Shape_MNK;
    using AtomScaleLayout = typename Atom::Traits::SFALayout;

    const auto permutation_mnk = TiledPerm{};
    const auto thread_layout_vmnk = mma.get_thr_layout_vmnk();
    const auto tensor_tile = make_tile(
        get<0>(permutation_mnk),
        get<2>(permutation_mnk));
    const auto permuted_tensor = logical_divide(scale_tensor, tensor_tile);
    const auto atom_tile = make_tile(
        make_layout(size<0>(AtomShape{})),
        make_layout(size<2>(AtomShape{})));
    const auto atom_tensor = zipped_divide(permuted_tensor, atom_tile);
    const auto thread_value_tensor =
        atom_tensor.compose(AtomScaleLayout{}, _);
    const auto thread_tile = make_tile(
        _,
        make_tile(
            make_layout(size<1>(thread_layout_vmnk)),
            make_layout(size<3>(thread_layout_vmnk))));
    return zipped_divide(thread_value_tensor, thread_tile);
}

template <class SFBTensor, class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr auto scale_partition_layout_b(
    SFBTensor &&scale_tensor,
    cute::TiledMMA<Atom, TiledThr, TiledPerm> &mma)
{
    using namespace cute;

    CUTE_STATIC_ASSERT_V(rank(scale_tensor) >= Int<2>{});
    using AtomShape = typename Atom::Shape_MNK;
    using AtomScaleLayout = typename Atom::Traits::SFBLayout;

    const auto permutation_mnk = TiledPerm{};
    const auto thread_layout_vmnk = mma.get_thr_layout_vmnk();
    const auto tensor_tile = make_tile(
        get<1>(permutation_mnk),
        get<2>(permutation_mnk));
    const auto permuted_tensor = logical_divide(scale_tensor, tensor_tile);
    const auto atom_tile = make_tile(
        make_layout(size<1>(AtomShape{})),
        make_layout(size<2>(AtomShape{})));
    const auto atom_tensor = zipped_divide(permuted_tensor, atom_tile);
    const auto thread_value_tensor =
        atom_tensor.compose(AtomScaleLayout{}, _);
    const auto thread_tile = make_tile(
        _,
        make_tile(
            make_layout(size<2>(thread_layout_vmnk)),
            make_layout(size<3>(thread_layout_vmnk))));
    return zipped_divide(thread_value_tensor, thread_tile);
}

template <class SFATensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto partition_scale_a(
    SFATensor &&scale_tensor,
    ThrMma &thread_mma)
{
    using namespace cute;

    const auto thread_tensor = make_tensor(
        static_cast<SFATensor &&>(scale_tensor).data(),
        scale_partition_layout_a(scale_tensor.layout(), thread_mma));
    const auto thread_vmnk = thread_mma.thr_vmnk_;
    const auto thread_vmk = make_coord(
        get<0>(thread_vmnk),
        make_coord(get<1>(thread_vmnk), get<3>(thread_vmnk)));
    return thread_tensor(
        thread_vmk,
        make_coord(_, repeat<rank<1, 1>(thread_tensor)>(_)));
}

template <class SFBTensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto partition_scale_b(
    SFBTensor &&scale_tensor,
    ThrMma &thread_mma)
{
    using namespace cute;

    const auto thread_tensor = make_tensor(
        static_cast<SFBTensor &&>(scale_tensor).data(),
        scale_partition_layout_b(scale_tensor.layout(), thread_mma));
    const auto thread_vmnk = thread_mma.thr_vmnk_;
    const auto thread_vnk = make_coord(
        get<0>(thread_vmnk),
        make_coord(get<2>(thread_vmnk), get<3>(thread_vmnk)));
    return thread_tensor(
        thread_vnk,
        make_coord(_, repeat<rank<1, 1>(thread_tensor)>(_)));
}

template <
    class ProblemShape,
    class CtaTiler,
    class AStride,
    class BStride,
    class CStride,
    class TiledMma>
__global__ __launch_bounds__(decltype(cute::size(TiledMma{}))::value)
void cute_gemm_mxfp6_kernel(
    ProblemShape problem_shape,
    CtaTiler cta_tiler,
    const cutlass::float_e3m2_t *a,
    AStride stride_a,
    const cutlass::float_ue8m0_t *scale_a,
    const cutlass::float_e3m2_t *b,
    BStride stride_b,
    const cutlass::float_ue8m0_t *scale_b,
    __half *c,
    CStride stride_c,
    TiledMma tiled_mma)
{
    using namespace cute;

    CUTE_STATIC_ASSERT_V(rank(problem_shape) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});

    // 与普通 FP6 相同，E3M2 使用一值一 byte 的低六位原始编码。
    Tensor global_a = make_tensor(
        make_gmem_ptr(reinterpret_cast<const uint8_t *>(a)),
        select<0, 2>(problem_shape),
        stride_a);
    Tensor global_b = make_tensor(
        make_gmem_ptr(reinterpret_cast<const uint8_t *>(b)),
        select<1, 2>(problem_shape),
        stride_b);
    Tensor global_c = make_tensor(
        make_gmem_ptr(c),
        select<0, 1>(problem_shape),
        stride_c);

    const auto cta_coordinate = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor block_a = local_tile(
        global_a,
        cta_tiler,
        cta_coordinate,
        Step<_1, X, _1>{});
    Tensor block_b = local_tile(
        global_b,
        cta_tiler,
        cta_coordinate,
        Step<X, _1, _1>{});
    Tensor block_c = local_tile(
        global_c,
        cta_tiler,
        cta_coordinate,
        Step<_1, _1, X>{});

    auto thread_mma = tiled_mma.get_thread_slice(threadIdx.x);
    Tensor thread_global_a = thread_mma.partition_A(block_a);
    Tensor thread_global_b = thread_mma.partition_B(block_b);
    Tensor thread_global_c = thread_mma.partition_C(block_c);
    Tensor fragment_a =
        thread_mma.partition_fragment_A(block_a(_, _, Int<0>{}));
    Tensor fragment_b =
        thread_mma.partition_fragment_B(block_b(_, _, Int<0>{}));
    Tensor accumulator = thread_mma.make_fragment_C(thread_global_c);
    clear(accumulator);

    // 逻辑 K 维写成 (32,2)：inner-32 的 stride 为 0，表示连续
    // 32 个 E3M2 共用一个 UE8M0；outer stride 为 1，一个 K64 tile
    // 读取两个不同 scale。
    const int scale_row_stride = get<2>(problem_shape) / kScaleVectorSize;
    const auto scale_layout_a = make_layout(
        make_shape(
            Int<kBlockM>{},
            make_shape(
                Int<kScaleVectorSize>{},
                Int<kBlockK / kScaleVectorSize>{})),
        make_stride(
            scale_row_stride,
            make_stride(_0{}, _1{})));
    const auto scale_layout_b = make_layout(
        make_shape(
            Int<kBlockN>{},
            make_shape(
                Int<kScaleVectorSize>{},
                Int<kBlockK / kScaleVectorSize>{})),
        make_stride(
            scale_row_stride,
            make_stride(_0{}, _1{})));

    const int first_scale_row_a = static_cast<int>(blockIdx.x) * kBlockM;
    const int first_scale_row_b = static_cast<int>(blockIdx.y) * kBlockN;
    Tensor initial_scale_tile_a = make_tensor(
        make_gmem_ptr(scale_a + first_scale_row_a * scale_row_stride),
        scale_layout_a);
    Tensor initial_scale_tile_b = make_tensor(
        make_gmem_ptr(scale_b + first_scale_row_b * scale_row_stride),
        scale_layout_b);
    Tensor initial_scale_partition_a =
        partition_scale_a(initial_scale_tile_a, thread_mma);
    Tensor initial_scale_partition_b =
        partition_scale_b(initial_scale_tile_b, thread_mma);
    Tensor fragment_scale_a =
        make_fragment_like<cutlass::float_ue8m0_t>(
            initial_scale_partition_a);
    Tensor fragment_scale_b =
        make_fragment_like<cutlass::float_ue8m0_t>(
            initial_scale_partition_b);

    const int k_tile_count = size<3>(thread_global_a);
    CUTE_NO_UNROLL
    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile)
    {
        copy(thread_global_a(_, _, _, k_tile), fragment_a);
        copy(thread_global_b(_, _, _, k_tile), fragment_b);

        const int scale_k_offset =
            k_tile * (kBlockK / kScaleVectorSize);
        Tensor scale_tile_a = make_tensor(
            make_gmem_ptr(
                scale_a + first_scale_row_a * scale_row_stride +
                scale_k_offset),
            scale_layout_a);
        Tensor scale_tile_b = make_tensor(
            make_gmem_ptr(
                scale_b + first_scale_row_b * scale_row_stride +
                scale_k_offset),
            scale_layout_b);
        Tensor scale_partition_a =
            partition_scale_a(scale_tile_a, thread_mma);
        Tensor scale_partition_b =
            partition_scale_b(scale_tile_b, thread_mma);
        copy(scale_partition_a, fragment_scale_a);
        copy(scale_partition_b, fragment_scale_b);

        // zip_tensor 把 E3M2 数据 fragment 和 UE8M0 scale fragment
        // 绑定，真正发出 block32 MXFP6 指令并在 FP32 中累加。
        cute::gemm(
            tiled_mma,
            make_zip_tensor(fragment_a, fragment_scale_a),
            make_zip_tensor(fragment_b, fragment_scale_b),
            accumulator);
    }

    CUTE_UNROLL
    for (int index = 0; index < size(accumulator); ++index)
    {
        thread_global_c(index) =
            __float2half_rn(static_cast<float>(accumulator(index)));
    }
}

void launch_cute_fp6_gemm(
    const cutlass::float_e3m2_t *a,
    const cutlass::float_e3m2_t *b,
    __half *c,
    int m,
    int n,
    int k,
    cudaStream_t stream)
{
    using namespace cute;

    const auto problem_shape = make_shape(m, n, k);
    const auto stride_a = make_stride(k, Int<1>{});
    const auto stride_b = make_stride(k, Int<1>{});
    const auto stride_c = make_stride(n, Int<1>{});
    const auto cta_tiler = make_shape(
        Int<kBlockM>{},
        Int<kBlockN>{},
        Int<kBlockK>{});

    // SM120 普通 FP6 Atom：m16n8k32 E3M2 x E3M2，FP32 C/D。
    const auto tiled_mma = make_tiled_mma(
        SM120_16x8x32_TN<
            cutlass::float_e3m2_t,
            cutlass::float_e3m2_t,
            float>{},
        Layout<Shape<_4, _2, _1>>{},
        Tile<_128, _128, _64>{});

    const dim3 block(size(tiled_mma));
    const dim3 grid(m / kBlockM, n / kBlockN);
    const auto kernel = cute_gemm_fp6_kernel<
        decltype(problem_shape),
        decltype(cta_tiler),
        decltype(stride_a),
        decltype(stride_b),
        decltype(stride_c),
        decltype(tiled_mma)>;
    kernel<<<grid, block, 0, stream>>>(
        problem_shape,
        cta_tiler,
        a,
        stride_a,
        b,
        stride_b,
        c,
        stride_c,
        tiled_mma);
}

void launch_cute_mxfp6_gemm(
    const cutlass::float_e3m2_t *a,
    const cutlass::float_ue8m0_t *scale_a,
    const cutlass::float_e3m2_t *b,
    const cutlass::float_ue8m0_t *scale_b,
    __half *c,
    int m,
    int n,
    int k,
    cudaStream_t stream)
{
    using namespace cute;

    const auto problem_shape = make_shape(m, n, k);
    const auto stride_a = make_stride(k, Int<1>{});
    const auto stride_b = make_stride(k, Int<1>{});
    const auto stride_c = make_stride(n, Int<1>{});
    const auto cta_tiler = make_shape(
        Int<kBlockM>{},
        Int<kBlockN>{},
        Int<kBlockK>{});

    // MXFP6 Atom：E3M2 x E3M2、UE8M0、每 32 个 K 元素一个 scale，
    // m16n8k32 指令直接产生 FP32 accumulator。
    const auto mma_atom =
        SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
            cutlass::float_e3m2_t,
            cutlass::float_e3m2_t,
            float,
            cutlass::float_ue8m0_t,
            kScaleVectorSize>{};
    const auto tiled_mma = make_tiled_mma(
        mma_atom,
        Layout<Shape<_4, _2, _1>>{},
        Tile<_128, _128, _64>{});

    const dim3 block(size(tiled_mma));
    const dim3 grid(m / kBlockM, n / kBlockN);
    const auto kernel = cute_gemm_mxfp6_kernel<
        decltype(problem_shape),
        decltype(cta_tiler),
        decltype(stride_a),
        decltype(stride_b),
        decltype(stride_c),
        decltype(tiled_mma)>;
    kernel<<<grid, block, 0, stream>>>(
        problem_shape,
        cta_tiler,
        a,
        stride_a,
        scale_a,
        b,
        stride_b,
        scale_b,
        c,
        stride_c,
        tiled_mma);
}

struct DeviceComparison
{
    unsigned long long mismatch_count;
    unsigned int max_absolute_error_bits;
    unsigned int max_relative_error_bits;
};

__global__ void compare_fp16_outputs_kernel(
    const __half *actual,
    const __half *reference,
    size_t element_count,
    DeviceComparison *comparison)
{
    const size_t first =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride =
        static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = first; index < element_count; index += stride)
    {
        const float actual_value = __half2float(actual[index]);
        const float reference_value = __half2float(reference[index]);
        float absolute_error = fabsf(actual_value - reference_value);
        float relative_error = absolute_error /
            fmaxf(fabsf(reference_value), 1.0e-6F);
        if (!isfinite(absolute_error))
        {
            constexpr int positive_infinity_bits = 0x7f800000;
            absolute_error = __int_as_float(positive_infinity_bits);
            relative_error = __int_as_float(positive_infinity_bits);
        }

        atomicMax(
            &comparison->max_absolute_error_bits,
            __float_as_uint(absolute_error));
        atomicMax(
            &comparison->max_relative_error_bits,
            __float_as_uint(relative_error));
        const float tolerance = kAbsoluteTolerance +
            kRelativeTolerance * fabsf(reference_value);
        if (absolute_error > tolerance)
        {
            atomicAdd(&comparison->mismatch_count, 1ULL);
        }
    }
}

DeviceComparison compare_outputs(
    const __half *actual,
    const __half *reference,
    size_t element_count,
    int block_count,
    cudaStream_t stream)
{
    DeviceBuffer<DeviceComparison> device_comparison(1);
    CUDA_CHECK(cudaMemsetAsync(
        device_comparison.get(),
        0,
        sizeof(DeviceComparison),
        stream));
    compare_fp16_outputs_kernel<<<block_count, 256, 0, stream>>>(
        actual,
        reference,
        element_count,
        device_comparison.get());
    CUDA_CHECK(cudaGetLastError());

    DeviceComparison host_comparison{};
    CUDA_CHECK(cudaMemcpyAsync(
        &host_comparison,
        device_comparison.get(),
        sizeof(DeviceComparison),
        cudaMemcpyDeviceToHost,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return host_comparison;
}

float decode_float_bits(uint32_t bits)
{
    float value = 0.0F;
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

float copy_device_value(const __half *data, size_t index)
{
    __half value{};
    CUDA_CHECK(cudaMemcpy(
        &value,
        data + index,
        sizeof(value),
        cudaMemcpyDeviceToHost));
    return __half2float(value);
}

float host_mxfp6_scale(size_t block_start, uint32_t seed)
{
    float maximum_absolute_value = 0.0F;
    for (int lane = 0; lane < kScaleVectorSize; ++lane)
    {
        maximum_absolute_value = std::max(
            maximum_absolute_value,
            std::abs(make_mxfp6_input_value(block_start + lane, seed)));
    }
    return static_cast<float>(make_dynamic_scale(maximum_absolute_value));
}

float quantize_dequantize_host(
    size_t element_index,
    uint32_t seed,
    bool block_scaled)
{
    if (!block_scaled)
    {
        return static_cast<float>(cutlass::float_e3m2_t(
            make_fp6_input_value(element_index, seed)));
    }

    const size_t block_start =
        (element_index / kScaleVectorSize) * kScaleVectorSize;
    const float scale = host_mxfp6_scale(block_start, seed);
    const cutlass::float_e3m2_t quantized(
        make_mxfp6_input_value(element_index, seed) / scale);
    return static_cast<float>(quantized) * scale;
}

double cpu_reference_value(
    int row,
    int column,
    int k,
    bool block_scaled)
{
    double result = 0.0;
    for (int reduction = 0; reduction < k; ++reduction)
    {
        const size_t index_a =
            static_cast<size_t>(row) * k + reduction;
        const size_t index_b =
            static_cast<size_t>(column) * k + reduction;
        const float a = quantize_dequantize_host(
            index_a,
            kSeedA,
            block_scaled);
        const float b = quantize_dequantize_host(
            index_b,
            kSeedB,
            block_scaled);
        result += static_cast<double>(a) * static_cast<double>(b);
    }
    return result;
}

bool verify_cpu_samples(
    const __half *cute_output,
    const __half *cublas_output,
    int m,
    int n,
    int k,
    bool block_scaled)
{
    const std::array<std::pair<int, int>, 8> samples =
    {{
        {0, 0},
        {0, n - 1},
        {m / 4, n / 3},
        {m / 3, n / 2},
        {m / 2, n / 2},
        {m * 3 / 4, n * 2 / 3},
        {m - 1, 0},
        {m - 1, n - 1}
    }};

    std::cout << "  " << std::left << std::setw(16) << "Coordinate"
              << std::right << std::setw(20) << "CPU quantized FP64"
              << std::setw(20) << "CuTe FP16"
              << std::setw(20) << "cuBLAS FP16" << '\n';

    bool passed = true;
    for (const auto &sample : samples)
    {
        const int row = sample.first;
        const int column = sample.second;
        const size_t index = static_cast<size_t>(row) * n + column;
        const double reference = cpu_reference_value(
            row,
            column,
            k,
            block_scaled);
        const float cute_value = copy_device_value(cute_output, index);
        const float cublas_value = copy_device_value(cublas_output, index);
        const double tolerance =
            static_cast<double>(kAbsoluteTolerance) +
            static_cast<double>(kRelativeTolerance) * std::abs(reference);
        passed = passed &&
            std::abs(static_cast<double>(cute_value) - reference) <= tolerance &&
            std::abs(static_cast<double>(cublas_value) - reference) <= tolerance;

        const std::string coordinate =
            "(" + std::to_string(row) + "," +
            std::to_string(column) + ")";
        std::cout << "  " << std::left << std::setw(16) << coordinate
                  << std::right << std::setw(20) << reference
                  << std::setw(20) << cute_value
                  << std::setw(20) << cublas_value << '\n';
    }
    return passed;
}

std::pair<float, float> inspect_scale_range(
    const cutlass::float_ue8m0_t *device_scales,
    size_t scale_count)
{
    const size_t inspection_count = std::min<size_t>(scale_count, 4096);
    std::vector<cutlass::float_ue8m0_t> host_scales(inspection_count);
    CUDA_CHECK(cudaMemcpy(
        host_scales.data(),
        device_scales,
        inspection_count * sizeof(cutlass::float_ue8m0_t),
        cudaMemcpyDeviceToHost));

    float minimum = static_cast<float>(host_scales.front());
    float maximum = minimum;
    for (const auto scale : host_scales)
    {
        minimum = std::min(minimum, static_cast<float>(scale));
        maximum = std::max(maximum, static_cast<float>(scale));
    }
    return {minimum, maximum};
}

double benchmark(
    const std::function<void()> &launch,
    int warmup_iterations,
    int benchmark_iterations,
    cudaStream_t stream)
{
    for (int iteration = 0; iteration < warmup_iterations; ++iteration)
    {
        launch();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CudaEvent start;
    CudaEvent stop;
    std::vector<double> samples;
    samples.reserve(benchmark_iterations);
    for (int iteration = 0; iteration < benchmark_iterations; ++iteration)
    {
        CUDA_CHECK(cudaEventRecord(start.get(), stream));
        launch();
        CUDA_CHECK(cudaEventRecord(stop.get(), stream));
        CUDA_CHECK(cudaEventSynchronize(stop.get()));

        float elapsed_milliseconds = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(
            &elapsed_milliseconds,
            start.get(),
            stop.get()));
        samples.push_back(elapsed_milliseconds);
    }

    std::sort(samples.begin(), samples.end());
    const size_t middle = samples.size() / 2;
    if (samples.size() % 2 == 0)
    {
        return (samples[middle - 1] + samples[middle]) / 2.0;
    }
    return samples[middle];
}

double calculate_tflops(int m, int n, int k, double milliseconds)
{
    const double operations =
        2.0 * static_cast<double>(m) * n * k;
    return operations / (milliseconds * 1.0e9);
}

struct Buffers
{
    Buffers(size_t count_a, size_t count_b, size_t count_c)
        : a(count_a),
          b(count_b),
          dequantized_a(count_a),
          dequantized_b(count_b),
          cute_c(count_c),
          cublas_c(count_c)
    {
    }

    DeviceBuffer<cutlass::float_e3m2_t> a;
    DeviceBuffer<cutlass::float_e3m2_t> b;
    DeviceBuffer<__half> dequantized_a;
    DeviceBuffer<__half> dequantized_b;
    DeviceBuffer<__half> cute_c;
    DeviceBuffer<__half> cublas_c;
};

void report_comparison(
    const DeviceComparison &comparison)
{
    std::cout << std::fixed << std::setprecision(7);
    std::cout << "  " << std::left << std::setw(32)
              << "Mismatches" << comparison.mismatch_count << '\n';
    std::cout << "  " << std::left << std::setw(32)
              << "Max absolute error"
              << decode_float_bits(comparison.max_absolute_error_bits)
              << '\n';
    std::cout << "  " << std::left << std::setw(32)
              << "Max relative error"
              << decode_float_bits(comparison.max_relative_error_bits)
              << '\n';
}

void report_benchmark(
    const char *cute_name,
    double cute_milliseconds,
    double cublas_milliseconds,
    const Options &options)
{
    std::cout << "  " << std::left << std::setw(28)
              << "Implementation"
              << std::right << std::setw(16) << "Median ms"
              << std::setw(18) << "TFLOP/s" << '\n';
    std::cout << "  " << std::left << std::setw(28) << cute_name
              << std::right << std::setw(16) << cute_milliseconds
              << std::setw(18) << calculate_tflops(
                     options.m,
                     options.n,
                     options.k,
                     cute_milliseconds)
              << '\n';
    std::cout << "  " << std::left << std::setw(28) << kCublasName
              << std::right << std::setw(16) << cublas_milliseconds
              << std::setw(18) << calculate_tflops(
                     options.m,
                     options.n,
                     options.k,
                     cublas_milliseconds)
              << '\n';
}

void run_fp6_case(
    const Options &options,
    const cudaDeviceProp &properties,
    Buffers &buffers,
    CublasFp16Reference &cublas,
    cudaStream_t stream)
{
    const size_t count_a =
        static_cast<size_t>(options.m) * options.k;
    const size_t count_b =
        static_cast<size_t>(options.n) * options.k;
    const size_t count_c =
        static_cast<size_t>(options.m) * options.n;
    const int utility_block_count =
        std::max(1, properties.multiProcessorCount * 8);

    std::cout << "[Case A: ordinary FP6 setup]\n";
    quantize_fp6_kernel<<<utility_block_count, 256, 0, stream>>>(
        buffers.a.get(),
        buffers.dequantized_a.get(),
        count_a,
        kSeedA);
    quantize_fp6_kernel<<<utility_block_count, 256, 0, stream>>>(
        buffers.b.get(),
        buffers.dequantized_b.get(),
        count_b,
        kSeedB);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::cout << "  FP32 source -> one-byte E3M2 raw storage completed\n";
    std::cout << "  Same decoded FP16 A/B feed the cuBLAS reference\n\n";

    std::cout << "[Case A: correctness]\n";
    launch_cute_fp6_gemm(
        buffers.a.get(),
        buffers.b.get(),
        buffers.cute_c.get(),
        options.m,
        options.n,
        options.k,
        stream);
    CUDA_CHECK(cudaGetLastError());
    cublas.launch(
        buffers.dequantized_a.get(),
        buffers.dequantized_b.get(),
        buffers.cublas_c.get(),
        options.m,
        options.n,
        options.k,
        stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const DeviceComparison comparison = compare_outputs(
        buffers.cute_c.get(),
        buffers.cublas_c.get(),
        count_c,
        utility_block_count,
        stream);
    report_comparison(comparison);
    std::cout << '\n';

    std::cout << "[Case A: CPU quantized FP64 samples]\n";
    const bool samples_passed = verify_cpu_samples(
        buffers.cute_c.get(),
        buffers.cublas_c.get(),
        options.m,
        options.n,
        options.k,
        false);
    std::cout << '\n';
    if (comparison.mismatch_count != 0 || !samples_passed)
    {
        throw std::runtime_error("ordinary FP6 correctness failed");
    }

    std::cout << "[Case A: benchmark]\n";
    const double cute_milliseconds = benchmark(
        [&]()
        {
            launch_cute_fp6_gemm(
                buffers.a.get(),
                buffers.b.get(),
                buffers.cute_c.get(),
                options.m,
                options.n,
                options.k,
                stream);
            CUDA_CHECK(cudaGetLastError());
        },
        options.warmup_iterations,
        options.benchmark_iterations,
        stream);
    const double cublas_milliseconds = benchmark(
        [&]()
        {
            cublas.launch(
                buffers.dequantized_a.get(),
                buffers.dequantized_b.get(),
                buffers.cublas_c.get(),
                options.m,
                options.n,
                options.k,
                stream);
        },
        options.warmup_iterations,
        options.benchmark_iterations,
        stream);
    report_benchmark(
        kFp6CuteName,
        cute_milliseconds,
        cublas_milliseconds,
        options);
    std::cout << "\n[Case A key result]\n";
    std::cout << "  " << std::left << std::setw(32)
              << "MMA path" << "E3M2 x E3M2, non-block-scaled\n";
    std::cout << "  " << std::left << std::setw(32)
              << "Accumulator / output" << "FP32 / FP16\n\n";
}

void run_mxfp6_case(
    const Options &options,
    const cudaDeviceProp &properties,
    Buffers &buffers,
    CublasFp16Reference &cublas,
    cudaStream_t stream)
{
    const size_t count_a =
        static_cast<size_t>(options.m) * options.k;
    const size_t count_b =
        static_cast<size_t>(options.n) * options.k;
    const size_t count_c =
        static_cast<size_t>(options.m) * options.n;
    const size_t scale_count_a = count_a / kScaleVectorSize;
    const size_t scale_count_b = count_b / kScaleVectorSize;
    const int utility_block_count =
        std::max(1, properties.multiProcessorCount * 8);
    DeviceBuffer<cutlass::float_ue8m0_t> scale_a(scale_count_a);
    DeviceBuffer<cutlass::float_ue8m0_t> scale_b(scale_count_b);

    std::cout << "[Case B: MXFP6 setup]\n";
    quantize_mxfp6_kernel<<<utility_block_count, 256, 0, stream>>>(
        buffers.a.get(),
        scale_a.get(),
        buffers.dequantized_a.get(),
        scale_count_a,
        kSeedA);
    quantize_mxfp6_kernel<<<utility_block_count, 256, 0, stream>>>(
        buffers.b.get(),
        scale_b.get(),
        buffers.dequantized_b.get(),
        scale_count_b,
        kSeedB);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const auto scale_range_a = inspect_scale_range(
        scale_a.get(),
        scale_count_a);
    const auto scale_range_b = inspect_scale_range(
        scale_b.get(),
        scale_count_b);
    std::cout << "  Per-32-element GPU absmax quantization completed\n";
    std::cout << std::scientific << std::setprecision(7);
    std::cout << "  " << std::left << std::setw(32)
              << "Observed A scale range"
              << scale_range_a.first << " .. " << scale_range_a.second
              << '\n';
    std::cout << "  " << std::left << std::setw(32)
              << "Observed B scale range"
              << scale_range_b.first << " .. " << scale_range_b.second
              << "\n\n";

    std::cout << "[Case B: correctness]\n";
    launch_cute_mxfp6_gemm(
        buffers.a.get(),
        scale_a.get(),
        buffers.b.get(),
        scale_b.get(),
        buffers.cute_c.get(),
        options.m,
        options.n,
        options.k,
        stream);
    CUDA_CHECK(cudaGetLastError());
    cublas.launch(
        buffers.dequantized_a.get(),
        buffers.dequantized_b.get(),
        buffers.cublas_c.get(),
        options.m,
        options.n,
        options.k,
        stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const DeviceComparison comparison = compare_outputs(
        buffers.cute_c.get(),
        buffers.cublas_c.get(),
        count_c,
        utility_block_count,
        stream);
    report_comparison(comparison);
    std::cout << '\n';

    std::cout << "[Case B: CPU quantized FP64 samples]\n";
    const bool samples_passed = verify_cpu_samples(
        buffers.cute_c.get(),
        buffers.cublas_c.get(),
        options.m,
        options.n,
        options.k,
        true);
    std::cout << '\n';
    if (comparison.mismatch_count != 0 || !samples_passed)
    {
        throw std::runtime_error("MXFP6 correctness failed");
    }

    std::cout << "[Case B: benchmark]\n";
    const double cute_milliseconds = benchmark(
        [&]()
        {
            launch_cute_mxfp6_gemm(
                buffers.a.get(),
                scale_a.get(),
                buffers.b.get(),
                scale_b.get(),
                buffers.cute_c.get(),
                options.m,
                options.n,
                options.k,
                stream);
            CUDA_CHECK(cudaGetLastError());
        },
        options.warmup_iterations,
        options.benchmark_iterations,
        stream);
    const double cublas_milliseconds = benchmark(
        [&]()
        {
            cublas.launch(
                buffers.dequantized_a.get(),
                buffers.dequantized_b.get(),
                buffers.cublas_c.get(),
                options.m,
                options.n,
                options.k,
                stream);
        },
        options.warmup_iterations,
        options.benchmark_iterations,
        stream);
    report_benchmark(
        kMxFp6CuteName,
        cute_milliseconds,
        cublas_milliseconds,
        options);
    std::cout << "\n[Case B key result]\n";
    std::cout << "  " << std::left << std::setw(32)
              << "MMA path" << "E3M2 x E3M2 + UE8M0 block32\n";
    std::cout << "  " << std::left << std::setw(32)
              << "Accumulator / output" << "FP32 / FP16\n\n";
}

}  // namespace

int main(int argc, char **argv)
{
    try
    {
        const Options options = parse_options(argc, argv);

        int device = 0;
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        if (properties.major != 12 || properties.minor != 0)
        {
            throw std::runtime_error(
                "FP6/MXFP6 mma.sync test requires compute capability 12.0");
        }

        const size_t count_a =
            static_cast<size_t>(options.m) * options.k;
        const size_t count_b =
            static_cast<size_t>(options.n) * options.k;
        const size_t count_c =
            static_cast<size_t>(options.m) * options.n;
        const size_t scale_count =
            (count_a + count_b) / kScaleVectorSize;
        const size_t required_bytes =
            (count_a + count_b) * sizeof(cutlass::float_e3m2_t) +
            (count_a + count_b + 2 * count_c) * sizeof(__half) +
            scale_count * sizeof(cutlass::float_ue8m0_t);

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

        std::cout << "CuTe SM120 FP6 and MXFP6 Tensor Core validation\n\n";
        std::cout << "[Configuration]\n";
        std::cout << "  " << std::left << std::setw(32)
                  << "GPU" << properties.name << '\n';
        std::cout << "  " << std::left << std::setw(32)
                  << "Compute capability"
                  << properties.major << '.' << properties.minor << '\n';
        std::cout << "  " << std::left << std::setw(32)
                  << "A layout" << "row-major [M,K] = ["
                  << options.m << ',' << options.k << "]\n";
        std::cout << "  " << std::left << std::setw(32)
                  << "B layout" << "row-major [N,K] = ["
                  << options.n << ',' << options.k << "]\n";
        std::cout << "  " << std::left << std::setw(32)
                  << "C operation" << "FP16 C[M,N] = A * B^T\n";
        std::cout << "  " << std::left << std::setw(32)
                  << "FP6 storage" << "one E3M2 per byte, low 6 bits valid\n";
        std::cout << "  " << std::left << std::setw(32)
                  << "Case A" << "ordinary E3M2, no block scale\n";
        std::cout << "  " << std::left << std::setw(32)
                  << "Case B" << "MXFP6 E3M2 + UE8M0, block size 32\n";
        std::cout << "  " << std::left << std::setw(32)
                  << "CTA tile" << kBlockM << 'x' << kBlockN
                  << 'x' << kBlockK << '\n';
        std::cout << "  " << std::left << std::setw(32)
                  << "Thread block" << "256 threads (8 warps)\n";
        std::cout << "  " << std::left << std::setw(32)
                  << "Data path" << "E3M2/scale GMEM -> RMEM fragment\n";
        std::cout << "  " << std::left << std::setw(32)
                  << "MMA" << "m16n8k32, FP32 accumulate\n";
        std::cout << "  " << std::left << std::setw(32)
                  << "Required device memory"
                  << std::fixed << std::setprecision(2)
                  << static_cast<double>(required_bytes) /
                         (1024.0 * 1024.0)
                  << " MiB\n\n";

        constexpr size_t memory_margin = 256ULL * 1024ULL * 1024ULL;
        if (required_bytes + memory_margin > free_bytes)
        {
            throw std::runtime_error(
                "insufficient free device memory for FP6/MXFP6 validation");
        }

        CudaStream stream;
        CublasFp16Reference cublas;
        Buffers buffers(count_a, count_b, count_c);

        run_fp6_case(
            options,
            properties,
            buffers,
            cublas,
            stream.get());
        run_mxfp6_case(
            options,
            properties,
            buffers,
            cublas,
            stream.get());

        std::cout << "[SUCCESS] CuTe SM120 ordinary FP6 and MXFP6 "
                  << "E3M2 -> FP32 accumulate -> FP16 output "
                  << "validation passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cout << "[FAILED] " << error.what() << '\n';
        return 1;
    }
}
