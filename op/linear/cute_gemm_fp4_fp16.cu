/***************************************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

// SM120 CuTe NVFP4 GEMM 验证：
//   FP32 source -> FP4 E2M1 + UE4M3 block scale -> FP32 MMA accumulate -> FP16 output。
//
// A/B 都按 row-major [M,K] / [N,K] 保存，kernel 计算 C[M,N] = A * B^T。
// FP4 数据每字节打包两个 E2M1 元素；本测试使用 NVFP4 的 16-element
// block scale。为了单独验证 MMA，所有 block 先使用相同且可精确表示的
// UE4M3 scale=0.125。scale fragment 仍按硬件要求逐 block 送入 MMA。

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

#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>

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
constexpr int kScaleVectorSize = 16;

constexpr float kScaleA = 0.125F;
constexpr float kScaleB = 0.125F;
constexpr float kAbsoluteTolerance = 1.0e-2F;
constexpr float kRelativeTolerance = 2.0e-3F;
constexpr uint32_t kSeedA = 0x1234abcdU;
constexpr uint32_t kSeedB = 0x9e3779b9U;

constexpr const char *kCuteName = "CuTe SM120 NVFP4";
constexpr const char *kCublasName = "cuBLAS FP16 reference";

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

    size_t count() const
    {
        return count_;
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
        CUDA_CHECK(cudaStreamCreate(&stream_));
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

class CublasFp16Reference
{
public:
    CublasFp16Reference()
    {
        CUBLAS_CHECK(cublasCreate(&handle_));
        CUBLAS_CHECK(cublasSetMathMode(
            handle_,
            CUBLAS_TENSOR_OP_MATH));
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

        // cuBLAS 使用 column-major。row-major C=A*B^T 等价于计算
        // column-major C^T=B*A^T，因此交换 A/B 并令第一个操作数转置。
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

__host__ __device__ float make_input_value(size_t index, uint32_t seed)
{
    const uint32_t folded_index =
        static_cast<uint32_t>(index) ^
        static_cast<uint32_t>(index >> 32);
    const uint32_t bits = mix_bits(folded_index ^ seed);
    const int centered = static_cast<int>(bits & 0x1ffU) - 256;
    return static_cast<float>(centered) * (1.0F / 512.0F);
}

__global__ void initialize_packed_fp4_kernel(
    uint8_t *packed_data,
    size_t element_count,
    uint32_t seed,
    float scale)
{
    const size_t byte_count = (element_count + 1) / 2;
    const size_t first =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride =
        static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t byte_index = first;
         byte_index < byte_count;
         byte_index += stride)
    {
        const size_t first_element = byte_index * 2;
        const cutlass::float_e2m1_t first_value(
            make_input_value(first_element, seed) / scale);
        uint8_t packed = first_value.raw() & 0x0fU;
        if (first_element + 1 < element_count)
        {
            const cutlass::float_e2m1_t second_value(
                make_input_value(first_element + 1, seed) / scale);
            packed |= static_cast<uint8_t>(
                (second_value.raw() & 0x0fU) << 4);
        }
        packed_data[byte_index] = packed;
    }
}

__device__ float decode_fp4(uint8_t raw_value, float scale)
{
    cutlass::float_e2m1_t value;
    value.raw() = raw_value & 0x0fU;
    return static_cast<float>(value) * scale;
}

__global__ void dequantize_fp4_to_fp16_kernel(
    const uint8_t *packed_data,
    __half *output,
    size_t element_count,
    float scale)
{
    const size_t first =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride =
        static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = first; index < element_count; index += stride)
    {
        const uint8_t packed = packed_data[index / 2];
        const uint8_t raw_value = index % 2 == 0
            ? packed & 0x0fU
            : packed >> 4;
        output[index] = __float2half_rn(decode_fp4(raw_value, scale));
    }
}

// 以下四个 helper 根据 TiledMMA 的 thread/value layout，为每个线程构造
// scale-factor fragment。它们与 A/B fragment 具有相同的 MMA_M/N/K 分块，
// 但物理寄存器只保存每 16 个 K 元素对应的一个 UE4M3 scale。
template <class SFATensor, class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr auto scale_fragment_layout_a(
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
CUTE_HOST_DEVICE constexpr auto scale_fragment_layout_b(
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
CUTE_HOST_DEVICE constexpr auto make_scale_fragment_a(
    SFATensor &&scale_tensor,
    ThrMma &thread_mma)
{
    using namespace cute;

    using ScaleType = typename ThrMma::Atom::Traits::ValTypeSF;
    const auto thread_tensor = make_tensor(
        static_cast<SFATensor &&>(scale_tensor).data(),
        scale_fragment_layout_a(scale_tensor.layout(), thread_mma));
    const auto thread_vmnk = thread_mma.thr_vmnk_;
    const auto thread_vmk = make_coord(
        get<0>(thread_vmnk),
        make_coord(get<1>(thread_vmnk), get<3>(thread_vmnk)));
    const auto partition = thread_tensor(
        thread_vmk,
        make_coord(_, repeat<rank<1, 1>(thread_tensor)>(_)));
    return make_fragment_like<ScaleType>(partition);
}

template <class SFBTensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto make_scale_fragment_b(
    SFBTensor &&scale_tensor,
    ThrMma &thread_mma)
{
    using namespace cute;

    using ScaleType = typename ThrMma::Atom::Traits::ValTypeSF;
    const auto thread_tensor = make_tensor(
        static_cast<SFBTensor &&>(scale_tensor).data(),
        scale_fragment_layout_b(scale_tensor.layout(), thread_mma));
    const auto thread_vmnk = thread_mma.thr_vmnk_;
    const auto thread_vnk = make_coord(
        get<0>(thread_vmnk),
        make_coord(get<2>(thread_vmnk), get<3>(thread_vmnk)));
    const auto partition = thread_tensor(
        thread_vnk,
        make_coord(_, repeat<rank<1, 1>(thread_tensor)>(_)));
    return make_fragment_like<ScaleType>(partition);
}

template <
    class ProblemShape,
    class CtaTiler,
    class AStride,
    class BStride,
    class CStride,
    class TiledMma>
__global__ __launch_bounds__(decltype(cute::size(TiledMma{}))::value)
void cute_gemm_fp4_fp16_kernel(
    ProblemShape problem_shape,
    CtaTiler cta_tiler,
    const uint8_t *packed_a,
    AStride stride_a,
    const uint8_t *packed_b,
    BStride stride_b,
    __half *c,
    CStride stride_c,
    TiledMma tiled_mma,
    cutlass::float_ue4m3_t scale_a,
    cutlass::float_ue4m3_t scale_b)
{
    using namespace cute;

    CUTE_STATIC_ASSERT_V(rank(problem_shape) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});

    // packed_a/packed_b 的每个 byte 含两个 E2M1。recast_ptr<uint4_t>
    // 把 byte 地址转换成 CuTe subbyte iterator，tensor 的逻辑索引仍以
    // 单个 FP4 元素计数，不需要手工处理 nibble 地址。
    Tensor global_a = make_tensor(
        make_gmem_ptr<uint4_t>(static_cast<const void *>(packed_a)),
        select<0, 2>(problem_shape),
        stride_a);
    Tensor global_b = make_tensor(
        make_gmem_ptr<uint4_t>(static_cast<const void *>(packed_b)),
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

    // tiled_mma 把 256 个线程组织为 8 个 warp，并将 CTA 的
    // 128x128 C tile 与相应 A/B fragment 分配给各线程。
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

    // 这里只需要 scale tensor 的静态 M/K 或 N/K layout 来生成 fragment。
    // 真实 scale 值随后直接写入寄存器；所有 16-element block 使用同一个
    // scale，但 fragment 的结构仍完整覆盖每个硬件 scale slot。
    cutlass::float_ue4m3_t scale_a_storage = scale_a;
    cutlass::float_ue4m3_t scale_b_storage = scale_b;
    const auto scale_layout_a = make_layout(
        make_shape(
            Int<kBlockM>{},
            make_shape(
                Int<kScaleVectorSize>{},
                Int<kBlockK / kScaleVectorSize>{})),
        make_stride(
            Int<kBlockK / kScaleVectorSize>{},
            make_stride(_0{}, _1{})));
    const auto scale_layout_b = make_layout(
        make_shape(
            Int<kBlockN>{},
            make_shape(
                Int<kScaleVectorSize>{},
                Int<kBlockK / kScaleVectorSize>{})),
        make_stride(
            Int<kBlockK / kScaleVectorSize>{},
            make_stride(_0{}, _1{})));
    Tensor scale_a_tensor = make_tensor(
        make_rmem_ptr(&scale_a_storage),
        scale_layout_a);
    Tensor scale_b_tensor = make_tensor(
        make_rmem_ptr(&scale_b_storage),
        scale_layout_b);
    Tensor fragment_scale_a =
        make_scale_fragment_a(scale_a_tensor, thread_mma);
    Tensor fragment_scale_b =
        make_scale_fragment_b(scale_b_tensor, thread_mma);
    fill(fragment_scale_a, scale_a);
    fill(fragment_scale_b, scale_b);

    const int k_tile_count = size<3>(thread_global_a);
    CUTE_NO_UNROLL
    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile)
    {
        // GMEM -> RMEM：partition_A/B 已按 m16n8k64 MMA fragment 的
        // thread/value layout 分配元素。当前第一版刻意不引入 SMEM pipeline，
        // 先独立验证 packed FP4、scale fragment 与 MMA 指令语义。
        copy(thread_global_a(_, _, _, k_tile), fragment_a);
        copy(thread_global_b(_, _, _, k_tile), fragment_b);

        // zip_tensor 将 E2M1 data fragment 与 UE4M3 scale fragment 绑定。
        // SM120 执行 m16n8k64 block-scaled MMA，accumulator 始终为 FP32。
        cute::gemm(
            tiled_mma,
            make_zip_tensor(fragment_a, fragment_scale_a),
            make_zip_tensor(fragment_b, fragment_scale_b),
            accumulator);
    }

    // Epilogue：MMA 的 FP32 accumulator 在写回前转换为 FP16。
    CUTE_UNROLL
    for (int index = 0; index < size(accumulator); ++index)
    {
        thread_global_c(index) =
            __float2half_rn(static_cast<float>(accumulator(index)));
    }
}

void launch_cute_gemm(
    const uint8_t *packed_a,
    const uint8_t *packed_b,
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

    // NVFP4 MMA Atom：A/B 是 E2M1，scale 是 UE4M3，每 16 个 K 元素
    // 共享一个 scale，D/C accumulator 类型为 float。
    const auto mma_atom =
        SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
            cutlass::float_e2m1_t,
            cutlass::float_e2m1_t,
            float,
            cutlass::float_ue4m3_t,
            kScaleVectorSize>{};
    const auto tiled_mma = make_tiled_mma(
        mma_atom,
        Layout<Shape<_4, _2, _1>>{},
        Tile<_128, _128, _64>{});

    const dim3 block(size(tiled_mma));
    const dim3 grid(m / kBlockM, n / kBlockN);
    const auto kernel = cute_gemm_fp4_fp16_kernel<
        decltype(problem_shape),
        decltype(cta_tiler),
        decltype(stride_a),
        decltype(stride_b),
        decltype(stride_c),
        decltype(tiled_mma)>;

    kernel<<<grid, block, 0, stream>>>(
        problem_shape,
        cta_tiler,
        packed_a,
        stride_a,
        packed_b,
        stride_b,
        c,
        stride_c,
        tiled_mma,
        cutlass::float_ue4m3_t(kScaleA),
        cutlass::float_ue4m3_t(kScaleB));
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
    float absolute_tolerance,
    float relative_tolerance,
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
        const float tolerance = absolute_tolerance +
            relative_tolerance * fabsf(reference_value);
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
        kAbsoluteTolerance,
        kRelativeTolerance,
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

float quantize_dequantize_host(float value, float scale)
{
    const cutlass::float_e2m1_t quantized(value / scale);
    return static_cast<float>(quantized) * scale;
}

double cpu_reference_value(int row, int column, int k)
{
    double result = 0.0;
    for (int reduction = 0; reduction < k; ++reduction)
    {
        const size_t index_a =
            static_cast<size_t>(row) * k + reduction;
        const size_t index_b =
            static_cast<size_t>(column) * k + reduction;
        const float a = quantize_dequantize_host(
            make_input_value(index_a, kSeedA),
            kScaleA);
        const float b = quantize_dequantize_host(
            make_input_value(index_b, kSeedB),
            kScaleB);
        result += static_cast<double>(a) * static_cast<double>(b);
    }
    return result;
}

bool verify_cpu_samples(
    const __half *cute_output,
    const __half *cublas_output,
    int m,
    int n,
    int k)
{
    const std::array<std::pair<int, int>, 8> samples = {{
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
              << std::setw(22) << kCuteName
              << std::setw(24) << kCublasName << '\n';

    bool passed = true;
    for (const auto &sample : samples)
    {
        const int row = sample.first;
        const int column = sample.second;
        const size_t index = static_cast<size_t>(row) * n + column;
        const double reference = cpu_reference_value(row, column, k);
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
                  << std::setw(22) << cute_value
                  << std::setw(24) << cublas_value << '\n';
    }
    return passed;
}

struct BenchmarkCase
{
    const char *name;
    std::function<void()> launch;
};

template <size_t CaseCount>
std::array<double, CaseCount> benchmark_round_robin(
    const std::array<BenchmarkCase, CaseCount> &benchmark_cases,
    int warmup_iterations,
    int benchmark_iterations,
    cudaStream_t stream)
{
    for (int iteration = 0; iteration < warmup_iterations; ++iteration)
    {
        for (const BenchmarkCase &benchmark_case : benchmark_cases)
        {
            benchmark_case.launch();
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CudaEvent start;
    CudaEvent stop;
    std::array<std::vector<double>, CaseCount> samples;
    const int case_count = static_cast<int>(benchmark_cases.size());
    for (int iteration = 0;
         iteration < benchmark_iterations;
         ++iteration)
    {
        for (int offset = 0; offset < case_count; ++offset)
        {
            const int index = (iteration + offset) % case_count;
            CUDA_CHECK(cudaEventRecord(start.get(), stream));
            benchmark_cases[index].launch();
            CUDA_CHECK(cudaEventRecord(stop.get(), stream));
            CUDA_CHECK(cudaEventSynchronize(stop.get()));

            float elapsed_milliseconds = 0.0F;
            CUDA_CHECK(cudaEventElapsedTime(
                &elapsed_milliseconds,
                start.get(),
                stop.get()));
            samples[index].push_back(elapsed_milliseconds);
        }
    }

    std::array<double, CaseCount> medians{};
    for (size_t index = 0; index < samples.size(); ++index)
    {
        std::sort(samples[index].begin(), samples[index].end());
        const size_t middle = samples[index].size() / 2;
        if (samples[index].size() % 2 == 0)
        {
            medians[index] =
                (samples[index][middle - 1] + samples[index][middle]) /
                2.0;
        }
        else
        {
            medians[index] = samples[index][middle];
        }
    }
    return medians;
}

double calculate_tflops(int m, int n, int k, double milliseconds)
{
    const double operations =
        2.0 * static_cast<double>(m) * n * k;
    return operations / (milliseconds * 1.0e9);
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
                "NVFP4 mma.sync test requires compute capability 12.0");
        }

        const size_t count_a =
            static_cast<size_t>(options.m) * options.k;
        const size_t count_b =
            static_cast<size_t>(options.n) * options.k;
        const size_t count_c =
            static_cast<size_t>(options.m) * options.n;
        const size_t packed_count_a = (count_a + 1) / 2;
        const size_t packed_count_b = (count_b + 1) / 2;
        const size_t required_bytes =
            packed_count_a + packed_count_b +
            (count_a + count_b + 2 * count_c) * sizeof(__half);

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

        std::cout << "CuTe SM120 FP4 x FP4 -> FP32 accumulate -> FP16 output\n\n";
        std::cout << "[Configuration]\n";
        std::cout << "  " << std::left << std::setw(32)
                  << "GPU" << properties.name << '\n';
        std::cout << "  " << std::left << std::setw(32)
                  << "Compute capability"
                  << properties.major << '.' << properties.minor << '\n';
        std::cout << "  " << std::left << std::setw(32)
                  << "A layout" << "packed E2M1 row-major [M,K] = ["
                  << options.m << ',' << options.k << "]\n";
        std::cout << "  " << std::left << std::setw(32)
                  << "B layout" << "packed E2M1 row-major [N,K] = ["
                  << options.n << ',' << options.k << "]\n";
        std::cout << "  " << std::left << std::setw(32)
                  << "C operation" << "FP16 C[M,N] = A * B^T\n";
        std::cout << "  " << std::left << std::setw(32)
                  << "Quantization" << "q = E2M1(x / scale), two values/byte\n";
        std::cout << "  " << std::left << std::setw(32)
                  << "Scale format" << "NVFP4 UE4M3, block size 16\n";
        std::cout << "  " << std::left << std::setw(32)
                  << "Scale A / B" << kScaleA << " / " << kScaleB << '\n';
        std::cout << "  " << std::left << std::setw(32)
                  << "CTA tile" << kBlockM << 'x' << kBlockN
                  << 'x' << kBlockK << '\n';
        std::cout << "  " << std::left << std::setw(32)
                  << "Thread block" << "256 threads (8 warps)\n";
        std::cout << "  " << std::left << std::setw(32)
                  << "Data path" << "packed GMEM -> RMEM fragment\n";
        std::cout << "  " << std::left << std::setw(32)
                  << "MMA" << "m16n8k64 E2M1/UE4M3, FP32 accumulate\n";
        std::cout << "  " << std::left << std::setw(32)
                  << "Epilogue" << "FP32 accumulator -> FP16 output\n";
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
                "insufficient free device memory for FP4 GEMM validation");
        }

        CudaStream stream;
        CublasFp16Reference cublas;
        DeviceBuffer<uint8_t> device_packed_a(packed_count_a);
        DeviceBuffer<uint8_t> device_packed_b(packed_count_b);
        DeviceBuffer<__half> device_dequantized_a(count_a);
        DeviceBuffer<__half> device_dequantized_b(count_b);
        DeviceBuffer<__half> device_cute_c(count_c);
        DeviceBuffer<__half> device_cublas_c(count_c);
        const int utility_block_count =
            std::max(1, properties.multiProcessorCount * 8);

        std::cout << "[Setup]\n";
        initialize_packed_fp4_kernel<<<
            utility_block_count,
            256,
            0,
            stream.get()>>>(
                device_packed_a.get(),
                count_a,
                kSeedA,
                kScaleA);
        initialize_packed_fp4_kernel<<<
            utility_block_count,
            256,
            0,
            stream.get()>>>(
                device_packed_b.get(),
                count_b,
                kSeedB,
                kScaleB);
        dequantize_fp4_to_fp16_kernel<<<
            utility_block_count,
            256,
            0,
            stream.get()>>>(
                device_packed_a.get(),
                device_dequantized_a.get(),
                count_a,
                kScaleA);
        dequantize_fp4_to_fp16_kernel<<<
            utility_block_count,
            256,
            0,
            stream.get()>>>(
                device_packed_b.get(),
                device_dequantized_b.get(),
                count_b,
                kScaleB);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));
        std::cout << "  FP32 deterministic inputs quantized and packed as E2M1\n";
        std::cout << "  The same packed inputs dequantized to FP16 for cuBLAS reference\n\n";

        std::cout << "[Correctness]\n";
        launch_cute_gemm(
            device_packed_a.get(),
            device_packed_b.get(),
            device_cute_c.get(),
            options.m,
            options.n,
            options.k,
            stream.get());
        CUDA_CHECK(cudaGetLastError());
        cublas.launch(
            device_dequantized_a.get(),
            device_dequantized_b.get(),
            device_cublas_c.get(),
            options.m,
            options.n,
            options.k,
            stream.get());
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));

        const DeviceComparison comparison = compare_outputs(
            device_cute_c.get(),
            device_cublas_c.get(),
            count_c,
            utility_block_count,
            stream.get());
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
                  << "\n\n";

        std::cout << "[CPU quantized FP64 samples]\n";
        const bool samples_passed = verify_cpu_samples(
            device_cute_c.get(),
            device_cublas_c.get(),
            options.m,
            options.n,
            options.k);
        std::cout << '\n';
        if (comparison.mismatch_count != 0 || !samples_passed)
        {
            throw std::runtime_error(
                "FP4 input / FP32 accumulate / FP16 output correctness failed");
        }

        std::cout << "[Benchmark]\n";
        std::cout << "  cuBLAS timing excludes one-time FP4 dequantization and is "
                  << "an FP16 correctness/performance reference\n";
        std::cout << "  Method: rotating order; median of "
                  << options.benchmark_iterations << " samples\n\n";
        const std::array<BenchmarkCase, 2> benchmark_cases = {{
            {
                kCuteName,
                [&]()
                {
                    launch_cute_gemm(
                        device_packed_a.get(),
                        device_packed_b.get(),
                        device_cute_c.get(),
                        options.m,
                        options.n,
                        options.k,
                        stream.get());
                    CUDA_CHECK(cudaGetLastError());
                }
            },
            {
                kCublasName,
                [&]()
                {
                    cublas.launch(
                        device_dequantized_a.get(),
                        device_dequantized_b.get(),
                        device_cublas_c.get(),
                        options.m,
                        options.n,
                        options.k,
                        stream.get());
                }
            }
        }};
        const std::array<double, 2> benchmark_milliseconds =
            benchmark_round_robin(
                benchmark_cases,
                options.warmup_iterations,
                options.benchmark_iterations,
                stream.get());

        std::cout << "  " << std::left << std::setw(28)
                  << "Implementation"
                  << std::right << std::setw(16) << "Median ms"
                  << std::setw(18) << "TFLOP/s" << '\n';
        for (size_t index = 0; index < benchmark_cases.size(); ++index)
        {
            const double tflops = calculate_tflops(
                options.m,
                options.n,
                options.k,
                benchmark_milliseconds[index]);
            std::cout << "  " << std::left << std::setw(28)
                      << benchmark_cases[index].name
                      << std::right << std::setw(16)
                      << benchmark_milliseconds[index]
                      << std::setw(18) << tflops << '\n';
        }

        std::cout << "\n[Key result]\n";
        std::cout << "  " << std::left << std::setw(32)
                  << "MMA path"
                  << "FP4 E2M1 x E2M1 + UE4M3 block scale\n";
        std::cout << "  " << std::left << std::setw(32)
                  << "Accumulator / output" << "FP32 / FP16\n\n";
        std::cout << "[SUCCESS] CuTe SM120 FP4 x FP4 -> FP32 accumulate "
                  << "-> FP16 output validation passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cout << "[FAILED] " << error.what() << '\n';
        return 1;
    }
}
