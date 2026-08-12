/***************************************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

// SM120 CuTe 动态 NVFP4 GEMM 验证：
//   FP32 source -> per-row/per-K16 UE4M3 scale + packed E2M1
//   -> FP32 Tensor Core accumulate -> FP16 output。
//
// A/B 分别以 row-major [M,K] / [N,K] 解释，计算 C[M,N] = A * B^T。
// 每行连续 16 个 K 元素独立计算 absmax，并将 absmax / max(E2M1)
// 舍入为可表示的 UE4M3 scale；量化和后续解码都使用舍入后的实际 scale。

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
constexpr int kScaleVectorSize = 16;
constexpr int kScalesPerKTile = kBlockK / kScaleVectorSize;

constexpr float kFp4Maximum = 6.0F;
constexpr float kMinimumUe4m3Scale = 1.0F / 512.0F;
constexpr float kAbsoluteTolerance = 1.0e-2F;
constexpr float kRelativeTolerance = 2.0e-3F;
constexpr uint32_t kSeedA = 0x1234abcdU;
constexpr uint32_t kSeedB = 0x9e3779b9U;

constexpr const char *kCuteName = "CuTe SM120 dynamic NVFP4";
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

        // row-major C=A*B^T 等价于 column-major C^T=B*A^T。
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

    // 每个 K16 block 使用独立幅值，使动态 scale 覆盖多个 UE4M3 编码。
    const uint32_t block_bits = mix_bits(
        static_cast<uint32_t>(index / kScaleVectorSize) ^
        (seed >> 3));
    const float amplitude =
        0.5F * static_cast<float>(1U << (block_bits & 0x3U));
    return static_cast<float>(centered) * (amplitude / 256.0F);
}

__host__ __device__ cutlass::float_ue4m3_t make_dynamic_scale(
    float absolute_maximum)
{
    const float ideal_scale = fmaxf(
        absolute_maximum / kFp4Maximum,
        kMinimumUe4m3Scale);
    return cutlass::float_ue4m3_t(ideal_scale);
}

__global__ void quantize_dynamic_nvfp4_kernel(
    uint8_t *packed_data,
    cutlass::float_ue4m3_t *scales,
    int row_count,
    int k,
    uint32_t seed)
{
    const int blocks_per_row = k / kScaleVectorSize;
    const size_t scale_count =
        static_cast<size_t>(row_count) * blocks_per_row;
    const size_t first =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride =
        static_cast<size_t>(gridDim.x) * blockDim.x;

    for (size_t scale_index = first;
         scale_index < scale_count;
         scale_index += stride)
    {
        const int row = static_cast<int>(scale_index / blocks_per_row);
        const int k_block = static_cast<int>(scale_index % blocks_per_row);
        const size_t first_element =
            static_cast<size_t>(row) * k +
            static_cast<size_t>(k_block) * kScaleVectorSize;

        float absolute_maximum = 0.0F;
        for (int offset = 0; offset < kScaleVectorSize; ++offset)
        {
            absolute_maximum = fmaxf(
                absolute_maximum,
                fabsf(make_input_value(first_element + offset, seed)));
        }

        const cutlass::float_ue4m3_t encoded_scale =
            make_dynamic_scale(absolute_maximum);
        const float decoded_scale = static_cast<float>(encoded_scale);
        scales[scale_index] = encoded_scale;

        // 两个 E2M1 nibble 共用一个 byte；每个 K16 block 正好写 8 byte。
        const size_t first_byte = first_element / 2;
        for (int pair = 0; pair < kScaleVectorSize / 2; ++pair)
        {
            const size_t even_index = first_element + pair * 2;
            const cutlass::float_e2m1_t even_value(
                make_input_value(even_index, seed) / decoded_scale);
            const cutlass::float_e2m1_t odd_value(
                make_input_value(even_index + 1, seed) / decoded_scale);
            packed_data[first_byte + pair] = static_cast<uint8_t>(
                (even_value.raw() & 0x0fU) |
                ((odd_value.raw() & 0x0fU) << 4));
        }
    }
}

__device__ float decode_fp4(uint8_t raw_value, float scale)
{
    cutlass::float_e2m1_t value;
    value.raw() = raw_value & 0x0fU;
    return static_cast<float>(value) * scale;
}

__global__ void dequantize_dynamic_nvfp4_kernel(
    const uint8_t *packed_data,
    const cutlass::float_ue4m3_t *scales,
    __half *output,
    int row_count,
    int k)
{
    const size_t element_count = static_cast<size_t>(row_count) * k;
    const int blocks_per_row = k / kScaleVectorSize;
    const size_t first =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride =
        static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = first; index < element_count; index += stride)
    {
        const int row = static_cast<int>(index / k);
        const int column = static_cast<int>(index % k);
        const size_t scale_index =
            static_cast<size_t>(row) * blocks_per_row +
            column / kScaleVectorSize;
        const float scale = static_cast<float>(scales[scale_index]);
        const uint8_t packed = packed_data[index / 2];
        const uint8_t raw_value = (column & 1) == 0
            ? packed & 0x0fU
            : packed >> 4;
        output[index] = __float2half_rn(decode_fp4(raw_value, scale));
    }
}

// 以下 helper 完整复用 SM120 MMA Atom 的 SFA/SFB thread/value layout。
// 输入 identity tensor 后，partition 中的每个坐标就是该 scale register
// 对应的 CTA 内 row 与 K16 block；kernel 据此读取真实 GMEM scale。
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
void cute_gemm_nvfp4_dynamic_kernel(
    ProblemShape problem_shape,
    CtaTiler cta_tiler,
    const uint8_t *packed_a,
    AStride stride_a,
    const cutlass::float_ue4m3_t *scales_a,
    const uint8_t *packed_b,
    BStride stride_b,
    const cutlass::float_ue4m3_t *scales_b,
    __half *c,
    CStride stride_c,
    TiledMma tiled_mma)
{
    using namespace cute;

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

    // scale 的逻辑 K shape 是 (16,4)，其中 K16 内 stride=0、block stride=1。
    // identity tensor 经 SFALayout/SFBLayout 分区后显式给出每个寄存器 slot
    // 的 (row, (K16-inner, K16-block)) 坐标。
    const auto scale_layout_a = make_layout(
        make_shape(
            Int<kBlockM>{},
            make_shape(Int<kScaleVectorSize>{}, Int<kScalesPerKTile>{})),
        make_stride(
            Int<kScalesPerKTile>{},
            make_stride(_0{}, _1{})));
    const auto scale_layout_b = make_layout(
        make_shape(
            Int<kBlockN>{},
            make_shape(Int<kScaleVectorSize>{}, Int<kScalesPerKTile>{})),
        make_stride(
            Int<kScalesPerKTile>{},
            make_stride(_0{}, _1{})));
    Tensor scale_coordinates_a = make_identity_tensor(shape(scale_layout_a));
    Tensor scale_coordinates_b = make_identity_tensor(shape(scale_layout_b));
    Tensor thread_scale_coordinates_a =
        partition_scale_a(scale_coordinates_a, thread_mma);
    Tensor thread_scale_coordinates_b =
        partition_scale_b(scale_coordinates_b, thread_mma);

    // identity tensor 的 stride 是 ScaledBasis，只用于读坐标；实际 scale
    // fragment 必须从普通数值 layout 构造，才能得到紧凑的寄存器 layout。
    cutlass::float_ue4m3_t scale_storage_a{};
    cutlass::float_ue4m3_t scale_storage_b{};
    Tensor scale_values_a = make_tensor(
        make_rmem_ptr(&scale_storage_a),
        scale_layout_a);
    Tensor scale_values_b = make_tensor(
        make_rmem_ptr(&scale_storage_b),
        scale_layout_b);
    Tensor thread_scale_values_a =
        partition_scale_a(scale_values_a, thread_mma);
    Tensor thread_scale_values_b =
        partition_scale_b(scale_values_b, thread_mma);
    Tensor fragment_scale_a =
        make_fragment_like<cutlass::float_ue4m3_t>(
            thread_scale_values_a);
    Tensor fragment_scale_b =
        make_fragment_like<cutlass::float_ue4m3_t>(
            thread_scale_values_b);

    const int scales_per_row = get<2>(problem_shape) / kScaleVectorSize;
    const int k_tile_count = size<3>(thread_global_a);
    CUTE_NO_UNROLL
    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile)
    {
        // GMEM -> RMEM：FP4 data fragment 仍沿 MMA A/B layout 搬运。
        copy(thread_global_a(_, _, _, k_tile), fragment_a);
        copy(thread_global_b(_, _, _, k_tile), fragment_b);

        // GMEM scale -> RMEM：坐标的第二层 K block 取值为 0..3，
        // k_tile*4 将它映射到全局每行连续的 K/16 scale 数组。
        CUTE_UNROLL
        for (int index = 0; index < size(fragment_scale_a); ++index)
        {
            const auto coordinate = thread_scale_coordinates_a(index);
            const int local_row = static_cast<int>(get<0>(coordinate));
            const int local_k_block = static_cast<int>(
                get<1>(get<1>(coordinate)));
            const int global_row = blockIdx.x * kBlockM + local_row;
            const int global_k_block =
                k_tile * kScalesPerKTile + local_k_block;
            fragment_scale_a(index) = scales_a[
                static_cast<size_t>(global_row) * scales_per_row +
                global_k_block];
        }
        CUTE_UNROLL
        for (int index = 0; index < size(fragment_scale_b); ++index)
        {
            const auto coordinate = thread_scale_coordinates_b(index);
            const int local_row = static_cast<int>(get<0>(coordinate));
            const int local_k_block = static_cast<int>(
                get<1>(get<1>(coordinate)));
            const int global_row = blockIdx.y * kBlockN + local_row;
            const int global_k_block =
                k_tile * kScalesPerKTile + local_k_block;
            fragment_scale_b(index) = scales_b[
                static_cast<size_t>(global_row) * scales_per_row +
                global_k_block];
        }

        // zip(data, scale) 触发 SM120 m16n8k64 NVFP4 block-scaled MMA。
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

void launch_cute_gemm(
    const uint8_t *packed_a,
    const cutlass::float_ue4m3_t *scales_a,
    const uint8_t *packed_b,
    const cutlass::float_ue4m3_t *scales_b,
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
    const auto kernel = cute_gemm_nvfp4_dynamic_kernel<
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
        scales_a,
        packed_b,
        stride_b,
        scales_b,
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

struct ScaleStatistics
{
    float minimum = 0.0F;
    float maximum = 0.0F;
    int distinct_encoding_count = 0;
};

ScaleStatistics calculate_scale_statistics(
    const cutlass::float_ue4m3_t *device_scales,
    size_t scale_count)
{
    std::vector<cutlass::float_ue4m3_t> host_scales(scale_count);
    CUDA_CHECK(cudaMemcpy(
        host_scales.data(),
        device_scales,
        scale_count * sizeof(cutlass::float_ue4m3_t),
        cudaMemcpyDeviceToHost));

    std::array<bool, 256> encodings{};
    ScaleStatistics statistics;
    statistics.minimum = static_cast<float>(host_scales.front());
    statistics.maximum = statistics.minimum;
    for (const cutlass::float_ue4m3_t scale : host_scales)
    {
        const float value = static_cast<float>(scale);
        statistics.minimum = std::min(statistics.minimum, value);
        statistics.maximum = std::max(statistics.maximum, value);
        encodings[scale.raw()] = true;
    }
    statistics.distinct_encoding_count = static_cast<int>(
        std::count(encodings.begin(), encodings.end(), true));
    return statistics;
}

bool verify_quantization_samples(
    const uint8_t *device_packed,
    const cutlass::float_ue4m3_t *device_scales,
    int row_count,
    int k,
    uint32_t seed,
    const char *operand_name)
{
    const int blocks_per_row = k / kScaleVectorSize;
    const std::array<std::pair<int, int>, 6> samples = {{
        {0, 0},
        {0, blocks_per_row - 1},
        {row_count / 3, blocks_per_row / 3},
        {row_count / 2, blocks_per_row / 2},
        {row_count - 1, 0},
        {row_count - 1, blocks_per_row - 1}
    }};

    std::cout << "  " << operand_name << " scale/index samples\n";
    std::cout << "    " << std::left << std::setw(18) << "(row,K16 block)"
              << std::right << std::setw(16) << "Expected scale"
              << std::setw(16) << "Stored scale"
              << std::setw(14) << "Packed match" << '\n';

    bool passed = true;
    for (const auto &sample : samples)
    {
        const int row = sample.first;
        const int k_block = sample.second;
        const size_t scale_index =
            static_cast<size_t>(row) * blocks_per_row + k_block;
        const size_t first_element =
            static_cast<size_t>(row) * k +
            static_cast<size_t>(k_block) * kScaleVectorSize;
        cutlass::float_ue4m3_t stored_scale;
        std::array<uint8_t, kScaleVectorSize / 2> packed{};
        CUDA_CHECK(cudaMemcpy(
            &stored_scale,
            device_scales + scale_index,
            sizeof(stored_scale),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            packed.data(),
            device_packed + first_element / 2,
            packed.size(),
            cudaMemcpyDeviceToHost));

        float absolute_maximum = 0.0F;
        for (int offset = 0; offset < kScaleVectorSize; ++offset)
        {
            absolute_maximum = std::max(
                absolute_maximum,
                std::abs(make_input_value(first_element + offset, seed)));
        }
        const cutlass::float_ue4m3_t expected_scale =
            make_dynamic_scale(absolute_maximum);
        bool packed_matches = expected_scale.raw() == stored_scale.raw();
        const float decoded_scale = static_cast<float>(stored_scale);
        for (int offset = 0; offset < kScaleVectorSize; ++offset)
        {
            const cutlass::float_e2m1_t expected_value(
                make_input_value(first_element + offset, seed) /
                decoded_scale);
            const uint8_t byte = packed[offset / 2];
            const uint8_t raw_value = (offset & 1) == 0
                ? byte & 0x0fU
                : byte >> 4;
            packed_matches = packed_matches &&
                raw_value == (expected_value.raw() & 0x0fU);
        }
        passed = passed && packed_matches;

        const std::string coordinate =
            "(" + std::to_string(row) + "," +
            std::to_string(k_block) + ")";
        std::cout << "    " << std::left << std::setw(18) << coordinate
                  << std::right << std::setw(16)
                  << static_cast<float>(expected_scale)
                  << std::setw(16) << decoded_scale
                  << std::setw(14) << (packed_matches ? "yes" : "no")
                  << '\n';
    }
    return passed;
}

struct HostQuantizedRow
{
    std::vector<uint8_t> packed;
    std::vector<cutlass::float_ue4m3_t> scales;
};

HostQuantizedRow copy_quantized_row(
    const uint8_t *device_packed,
    const cutlass::float_ue4m3_t *device_scales,
    int row,
    int k)
{
    HostQuantizedRow result;
    result.packed.resize(static_cast<size_t>(k) / 2);
    result.scales.resize(static_cast<size_t>(k) / kScaleVectorSize);
    CUDA_CHECK(cudaMemcpy(
        result.packed.data(),
        device_packed + static_cast<size_t>(row) * k / 2,
        result.packed.size(),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        result.scales.data(),
        device_scales +
            static_cast<size_t>(row) * k / kScaleVectorSize,
        result.scales.size() * sizeof(cutlass::float_ue4m3_t),
        cudaMemcpyDeviceToHost));
    return result;
}

float decode_host_value(const HostQuantizedRow &row, int k_index)
{
    const uint8_t byte = row.packed[k_index / 2];
    const uint8_t raw_value = (k_index & 1) == 0
        ? byte & 0x0fU
        : byte >> 4;
    cutlass::float_e2m1_t value;
    value.raw() = raw_value;
    return static_cast<float>(value) *
        static_cast<float>(row.scales[k_index / kScaleVectorSize]);
}

double cpu_reference_value(
    const HostQuantizedRow &row_a,
    const HostQuantizedRow &row_b,
    int k)
{
    double result = 0.0;
    for (int reduction = 0; reduction < k; ++reduction)
    {
        result += static_cast<double>(decode_host_value(row_a, reduction)) *
            static_cast<double>(decode_host_value(row_b, reduction));
    }
    return result;
}

bool verify_cpu_samples(
    const uint8_t *packed_a,
    const cutlass::float_ue4m3_t *scales_a,
    const uint8_t *packed_b,
    const cutlass::float_ue4m3_t *scales_b,
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
              << std::right << std::setw(20) << "CPU decoded FP64"
              << std::setw(24) << kCuteName
              << std::setw(24) << kCublasName << '\n';

    bool passed = true;
    for (const auto &sample : samples)
    {
        const int row = sample.first;
        const int column = sample.second;
        const HostQuantizedRow row_a =
            copy_quantized_row(packed_a, scales_a, row, k);
        const HostQuantizedRow row_b =
            copy_quantized_row(packed_b, scales_b, column, k);
        const double reference = cpu_reference_value(row_a, row_b, k);
        const size_t index = static_cast<size_t>(row) * n + column;
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
                  << std::setw(24) << cute_value
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
    for (int iteration = 0; iteration < benchmark_iterations; ++iteration)
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
        medians[index] = samples[index].size() % 2 == 0
            ? (samples[index][middle - 1] + samples[index][middle]) / 2.0
            : samples[index][middle];
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
                "dynamic NVFP4 mma.sync test requires compute capability 12.0");
        }

        const size_t count_a =
            static_cast<size_t>(options.m) * options.k;
        const size_t count_b =
            static_cast<size_t>(options.n) * options.k;
        const size_t count_c =
            static_cast<size_t>(options.m) * options.n;
        const size_t packed_count_a = count_a / 2;
        const size_t packed_count_b = count_b / 2;
        const size_t scale_count_a = count_a / kScaleVectorSize;
        const size_t scale_count_b = count_b / kScaleVectorSize;
        const size_t required_bytes =
            packed_count_a + packed_count_b +
            (scale_count_a + scale_count_b) *
                sizeof(cutlass::float_ue4m3_t) +
            (count_a + count_b + 2 * count_c) * sizeof(__half);

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

        std::cout << "CuTe SM120 dynamic NVFP4 GEMM validation\n\n";
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
                  << "Quantization"
                  << "per-row/per-K16 absmax -> UE4M3 scale\n";
        std::cout << "  " << std::left << std::setw(32)
                  << "Scale formula" << "UE4M3(max(abs(x))/6), block size 16\n";
        std::cout << "  " << std::left << std::setw(32)
                  << "CTA / threads" << "128x128x64 / 256\n";
        std::cout << "  " << std::left << std::setw(32)
                  << "Data path" << "packed FP4 GMEM + scale GMEM -> RMEM\n";
        std::cout << "  " << std::left << std::setw(32)
                  << "MMA" << "m16n8k64 E2M1/UE4M3, FP32 accumulate\n";
        std::cout << "  " << std::left << std::setw(32)
                  << "Output" << "FP16 C[M,N] = A * B^T\n";
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
                "insufficient free device memory for dynamic NVFP4 test");
        }

        CudaStream stream;
        CublasFp16Reference cublas;
        DeviceBuffer<uint8_t> device_packed_a(packed_count_a);
        DeviceBuffer<uint8_t> device_packed_b(packed_count_b);
        DeviceBuffer<cutlass::float_ue4m3_t> device_scales_a(scale_count_a);
        DeviceBuffer<cutlass::float_ue4m3_t> device_scales_b(scale_count_b);
        DeviceBuffer<__half> device_dequantized_a(count_a);
        DeviceBuffer<__half> device_dequantized_b(count_b);
        DeviceBuffer<__half> device_cute_c(count_c);
        DeviceBuffer<__half> device_cublas_c(count_c);
        const int utility_block_count =
            std::max(1, properties.multiProcessorCount * 8);

        std::cout << "[Setup: dynamic quantization]\n";
        quantize_dynamic_nvfp4_kernel<<<
            utility_block_count,
            256,
            0,
            stream.get()>>>(
                device_packed_a.get(),
                device_scales_a.get(),
                options.m,
                options.k,
                kSeedA);
        quantize_dynamic_nvfp4_kernel<<<
            utility_block_count,
            256,
            0,
            stream.get()>>>(
                device_packed_b.get(),
                device_scales_b.get(),
                options.n,
                options.k,
                kSeedB);
        dequantize_dynamic_nvfp4_kernel<<<
            utility_block_count,
            256,
            0,
            stream.get()>>>(
                device_packed_a.get(),
                device_scales_a.get(),
                device_dequantized_a.get(),
                options.m,
                options.k);
        dequantize_dynamic_nvfp4_kernel<<<
            utility_block_count,
            256,
            0,
            stream.get()>>>(
                device_packed_b.get(),
                device_scales_b.get(),
                device_dequantized_b.get(),
                options.n,
                options.k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));

        const ScaleStatistics statistics_a = calculate_scale_statistics(
            device_scales_a.get(), scale_count_a);
        const ScaleStatistics statistics_b = calculate_scale_statistics(
            device_scales_b.get(), scale_count_b);
        std::cout << "  " << std::left << std::setw(32)
                  << "A scale min / max"
                  << statistics_a.minimum << " / " << statistics_a.maximum
                  << '\n';
        std::cout << "  " << std::left << std::setw(32)
                  << "B scale min / max"
                  << statistics_b.minimum << " / " << statistics_b.maximum
                  << '\n';
        std::cout << "  " << std::left << std::setw(32)
                  << "A / B distinct scale encodings"
                  << statistics_a.distinct_encoding_count << " / "
                  << statistics_b.distinct_encoding_count << "\n\n";
        if (statistics_a.distinct_encoding_count <= 1 ||
            statistics_b.distinct_encoding_count <= 1)
        {
            throw std::runtime_error("dynamic scale generation did not vary");
        }

        std::cout << "[Scale and packed-layout verification]\n";
        const bool quantization_a_passed = verify_quantization_samples(
            device_packed_a.get(),
            device_scales_a.get(),
            options.m,
            options.k,
            kSeedA,
            "A");
        const bool quantization_b_passed = verify_quantization_samples(
            device_packed_b.get(),
            device_scales_b.get(),
            options.n,
            options.k,
            kSeedB,
            "B");
        std::cout << '\n';
        if (!quantization_a_passed || !quantization_b_passed)
        {
            throw std::runtime_error("dynamic NVFP4 quantization layout failed");
        }

        std::cout << "[Correctness]\n";
        launch_cute_gemm(
            device_packed_a.get(),
            device_scales_a.get(),
            device_packed_b.get(),
            device_scales_b.get(),
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

        std::cout << "[CPU packed-data/scale FP64 samples]\n";
        const bool cpu_samples_passed = verify_cpu_samples(
            device_packed_a.get(),
            device_scales_a.get(),
            device_packed_b.get(),
            device_scales_b.get(),
            device_cute_c.get(),
            device_cublas_c.get(),
            options.m,
            options.n,
            options.k);
        std::cout << '\n';
        if (comparison.mismatch_count != 0 || !cpu_samples_passed)
        {
            throw std::runtime_error(
                "dynamic NVFP4 GEMM correctness validation failed");
        }

        std::cout << "[Benchmark]\n";
        std::cout << "  Quantization/dequantization excluded; cuBLAS uses the same "
                  << "decoded FP16 inputs and FP32 accumulation\n";
        std::cout << "  Method: rotating order; median of "
                  << options.benchmark_iterations << " samples\n\n";
        const std::array<BenchmarkCase, 2> benchmark_cases = {{
            {
                kCuteName,
                [&]()
                {
                    launch_cute_gemm(
                        device_packed_a.get(),
                        device_scales_a.get(),
                        device_packed_b.get(),
                        device_scales_b.get(),
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

        std::cout << "  " << std::left << std::setw(32)
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
            std::cout << "  " << std::left << std::setw(32)
                      << benchmark_cases[index].name
                      << std::right << std::setw(16)
                      << benchmark_milliseconds[index]
                      << std::setw(18) << tflops << '\n';
        }

        std::cout << "\n[Key result]\n";
        std::cout << "  " << std::left << std::setw(32)
                  << "Quantization"
                  << "independent UE4M3 scale for every row/K16 block\n";
        std::cout << "  " << std::left << std::setw(32)
                  << "MMA path"
                  << "E2M1 x E2M1 + UE4M3 scale_vec::4X\n";
        std::cout << "  " << std::left << std::setw(32)
                  << "Accumulator / output" << "FP32 / FP16\n\n";
        std::cout << "[SUCCESS] CuTe SM120 dynamic NVFP4 block16 validation passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cout << "[FAILED] " << error.what() << '\n';
        return 1;
    }
}
