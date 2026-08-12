/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

// CuTe W4A16 GEMM：A 是 FP16 activation，B 是按 [N,K] row-major 打包的 signed INT4
// weight。每个 B row 的每 64 个权重共享一个动态对称 scale。主 GEMM kernel 在
// GMEM -> SMEM 路径中融合 INT4 解包和反量化，随后执行
// FP16 x FP16 -> FP32 accumulate -> FP16 output。

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
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

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
constexpr int kGroupSize = 64;
constexpr int kPipelineStages = 3;
constexpr int kThreads = 128;

constexpr float kAbsoluteTolerance = 6.25e-2F;
constexpr float kRelativeTolerance = 1.0e-2F;
constexpr uint32_t kSeedA = 0x1234abcdU;
constexpr uint32_t kSeedB = 0x9e3779b9U;

constexpr const char *kCuteName = "CuTe fused W4A16";
constexpr const char *kCublasName = "cuBLAS decoded FP16";

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

class CublasHandle
{
public:
    CublasHandle()
    {
        CUBLAS_CHECK(cublasCreate(&handle_));
    }

    ~CublasHandle()
    {
        if (handle_ != nullptr)
        {
            cublasDestroy(handle_);
        }
    }

    CublasHandle(const CublasHandle &) = delete;
    CublasHandle &operator=(const CublasHandle &) = delete;

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
    for (int argument = 1; argument < argc; argument += 2)
    {
        if (argument + 1 >= argc)
        {
            throw std::runtime_error(
                std::string("missing value for option ") + argv[argument]);
        }

        const std::string option = argv[argument];
        const int value = parse_positive_integer(
            argv[argument + 1],
            argv[argument]);
        if (option == "--m")
        {
            options.m = value;
        }
        else if (option == "--n")
        {
            options.n = value;
        }
        else if (option == "--k")
        {
            options.k = value;
        }
        else if (option == "--warmup")
        {
            options.warmup_iterations = value;
        }
        else if (option == "--iterations")
        {
            options.benchmark_iterations = value;
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
            "M/N/K must be exact multiples of the 128x128x64 CTA tile");
    }
    if (options.k < (kPipelineStages - 1) * kBlockK)
    {
        throw std::runtime_error("K must contain at least two CTA K tiles");
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

__global__ void initialize_activation_kernel(
    cute::half_t *activation,
    size_t element_count)
{
    const size_t first =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride =
        static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = first; index < element_count; index += stride)
    {
        activation[index] = cute::half_t(make_input_value(index, kSeedA));
    }
}

// 一个线程处理 B 的一个 64-element group。先从确定性 FP32 source 统计 absmax，
// 将 scale 舍入为 FP16，然后用实际存储的 FP16 scale 完成 signed INT4 对称量化。
// -8 保留不用，因此量化范围为 [-7, 7]，zero point 固定为 0。
__global__ void quantize_weight_groups_kernel(
    uint8_t *packed_weight,
    cute::half_t *weight_scales,
    int n,
    int k)
{
    const int groups_per_row = k / kGroupSize;
    const size_t group_count = static_cast<size_t>(n) * groups_per_row;
    const size_t first =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride =
        static_cast<size_t>(gridDim.x) * blockDim.x;

    for (size_t group_index = first;
         group_index < group_count;
         group_index += stride)
    {
        const int row = static_cast<int>(group_index / groups_per_row);
        const int group = static_cast<int>(group_index % groups_per_row);
        const int group_k = group * kGroupSize;

        float absolute_maximum = 0.0F;
        for (int offset = 0; offset < kGroupSize; ++offset)
        {
            const size_t weight_index =
                static_cast<size_t>(row) * k + group_k + offset;
            absolute_maximum = fmaxf(
                absolute_maximum,
                fabsf(make_input_value(weight_index, kSeedB)));
        }

        const float candidate_scale =
            absolute_maximum > 0.0F ? absolute_maximum / 7.0F : 1.0F;
        const cute::half_t stored_scale(candidate_scale);
        weight_scales[group_index] = stored_scale;
        const float quantization_scale = static_cast<float>(stored_scale);

        const size_t packed_group_start =
            (static_cast<size_t>(row) * k + group_k) / 2;
        for (int pair = 0; pair < kGroupSize / 2; ++pair)
        {
            uint8_t packed = 0;
            for (int nibble_index = 0; nibble_index < 2; ++nibble_index)
            {
                const int offset = pair * 2 + nibble_index;
                const size_t weight_index =
                    static_cast<size_t>(row) * k + group_k + offset;
                int quantized = __float2int_rn(
                    make_input_value(weight_index, kSeedB) /
                    quantization_scale);
                quantized = max(-7, min(7, quantized));
                const uint8_t nibble =
                    static_cast<uint8_t>(quantized) & 0x0fU;
                packed |= static_cast<uint8_t>(
                    nibble << (nibble_index * 4));
            }
            packed_weight[packed_group_start + pair] = packed;
        }
    }
}

__device__ int decode_signed_int4(uint8_t packed, int k_index)
{
    const int nibble =
        (packed >> ((k_index & 1) * 4)) & 0x0fU;
    return nibble >= 8 ? nibble - 16 : nibble;
}

// 仅为 cuBLAS reference 完整展开 B。被测 CuTe kernel 从不读取此缓冲。
__global__ void decode_weight_reference_kernel(
    const uint8_t *packed_weight,
    const cute::half_t *weight_scales,
    cute::half_t *decoded_weight,
    int n,
    int k)
{
    const size_t element_count = static_cast<size_t>(n) * k;
    const int groups_per_row = k / kGroupSize;
    const size_t first =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride =
        static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = first; index < element_count; index += stride)
    {
        const int row = static_cast<int>(index / k);
        const int k_index = static_cast<int>(index % k);
        const uint8_t packed = packed_weight[index / 2];
        const int quantized = decode_signed_int4(packed, k_index);
        const float scale = static_cast<float>(
            weight_scales[
                static_cast<size_t>(row) * groups_per_row +
                k_index / kGroupSize]);
        decoded_weight[index] = cute::half_t(quantized * scale);
    }
}

template <
    class ElementA,
    class ElementB,
    class SmemLayoutA,
    class SmemLayoutB>
struct SharedStorage
{
    cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> a;
    cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> b;
};

template <class SharedTensorB>
__device__ void unpack_weight_tile_to_smem(
    const uint8_t *packed_weight,
    const cute::half_t *weight_scales,
    SharedTensorB shared_b,
    int block_n,
    int k_tile,
    int pipe,
    int k)
{
    const int packed_row_stride = k / 2;
    const int groups_per_row = k / kGroupSize;

    // 128 个线程协作产生 128x64 个 FP16 SMEM 元素。每个线程解包 64 个
    // nibble。这里无法直接用 cp.async 完成类型转换，因此 packed GMEM 读取、
    // two's-complement 解码、乘 scale 和 FP16 写 SMEM 全部融合在 GEMM kernel 内。
    for (int linear = threadIdx.x;
         linear < kBlockN * kBlockK;
         linear += blockDim.x)
    {
        const int local_n = linear / kBlockK;
        const int local_k = linear % kBlockK;
        const int global_n = block_n * kBlockN + local_n;
        const int global_k = k_tile * kBlockK + local_k;
        const size_t packed_index =
            static_cast<size_t>(global_n) * packed_row_stride +
            global_k / 2;
        const int quantized = decode_signed_int4(
            packed_weight[packed_index],
            global_k);
        const float scale = static_cast<float>(
            weight_scales[
                static_cast<size_t>(global_n) * groups_per_row +
                global_k / kGroupSize]);
        shared_b(local_n, local_k, pipe) =
            cute::half_t(static_cast<float>(quantized) * scale);
    }
}

template <
    class ProblemShape,
    class CtaTiler,
    class AStride,
    class ASmemLayout,
    class TiledCopyA,
    class S2RAtomA,
    class BSmemLayout,
    class S2RAtomB,
    class CStride,
    class TiledMma>
__global__ __launch_bounds__(decltype(cute::size(TiledMma{}))::value)
void cute_gemm_w4a16_kernel(
    ProblemShape problem_shape,
    CtaTiler cta_tiler,
    const cute::half_t *activation,
    AStride stride_a,
    ASmemLayout smem_layout_a,
    TiledCopyA copy_a,
    S2RAtomA smem_to_register_a,
    const uint8_t *packed_weight,
    const cute::half_t *weight_scales,
    BSmemLayout smem_layout_b,
    S2RAtomB smem_to_register_b,
    cute::half_t *output,
    CStride stride_c,
    TiledMma tiled_mma)
{
    using namespace cute;

    CUTE_STATIC_ASSERT_V(rank(problem_shape) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size(copy_a) == size(tiled_mma));

    Tensor global_a = make_tensor(
        make_gmem_ptr(activation),
        select<0, 2>(problem_shape),
        stride_a);
    Tensor global_c = make_tensor(
        make_gmem_ptr(output),
        select<0, 1>(problem_shape),
        stride_c);
    const auto cta_coordinate = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor block_a = local_tile(
        global_a,
        cta_tiler,
        cta_coordinate,
        Step<_1, X, _1>{});
    Tensor block_c = local_tile(
        global_c,
        cta_tiler,
        cta_coordinate,
        Step<_1, _1, X>{});

    extern __shared__ char shared_memory[];
    using Storage = SharedStorage<
        half_t,
        half_t,
        ASmemLayout,
        BSmemLayout>;
    Storage &storage = *reinterpret_cast<Storage *>(shared_memory);
    Tensor shared_a = make_tensor(
        make_smem_ptr(storage.a.begin()),
        smem_layout_a);
    Tensor shared_b = make_tensor(
        make_smem_ptr(storage.b.begin()),
        smem_layout_b);

    // A 的每条 cp.async 搬 8 个 FP16（16B），B 则保持 packed INT4，直到
    // 本 CTA 在 G2S 阶段调用 unpack_weight_tile_to_smem 才解包和反量化。
    ThrCopy thread_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor thread_global_a = thread_copy_a.partition_S(block_a);
    Tensor thread_shared_a = thread_copy_a.partition_D(shared_a);

    const int pipe_count = size<3>(thread_shared_a);
    int remaining_k_tiles = size<3>(thread_global_a);
    int next_k_tile = 0;
    CUTE_UNROLL
    for (int pipe = 0; pipe < pipe_count - 1; ++pipe)
    {
        copy(
            copy_a,
            thread_global_a(_, _, _, next_k_tile),
            thread_shared_a(_, _, _, pipe));
        unpack_weight_tile_to_smem(
            packed_weight,
            weight_scales,
            shared_b,
            static_cast<int>(blockIdx.y),
            next_k_tile,
            pipe,
            get<2>(problem_shape));
        cp_async_fence();
        --remaining_k_tiles;
        if (remaining_k_tiles > 0)
        {
            ++next_k_tile;
        }
    }

    // SM80 m16n8k16 Atom 的输入 fragment 为 FP16，累加 fragment 为 FP32。
    // 这里的 B fragment 来源始终是 kernel 内生成的反量化 SMEM tile。
    ThrMMA thread_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor thread_global_c = thread_mma.partition_C(block_c);
    Tensor register_a =
        thread_mma.partition_fragment_A(shared_a(_, _, 0));
    Tensor register_b =
        thread_mma.partition_fragment_B(shared_b(_, _, 0));
    Tensor accumulator = thread_mma.make_fragment_C(thread_global_c);
    clear(accumulator);

    TiledCopy tiled_smem_to_register_a =
        make_tiled_copy_A(smem_to_register_a, tiled_mma);
    ThrCopy thread_smem_to_register_a =
        tiled_smem_to_register_a.get_slice(threadIdx.x);
    Tensor thread_smem_a =
        thread_smem_to_register_a.partition_S(shared_a);
    Tensor thread_register_a =
        thread_smem_to_register_a.retile_D(register_a);

    TiledCopy tiled_smem_to_register_b =
        make_tiled_copy_B(smem_to_register_b, tiled_mma);
    ThrCopy thread_smem_to_register_b =
        tiled_smem_to_register_b.get_slice(threadIdx.x);
    Tensor thread_smem_b =
        thread_smem_to_register_b.partition_S(shared_b);
    Tensor thread_register_b =
        thread_smem_to_register_b.retile_D(register_b);

    int read_pipe = 0;
    int write_pipe = pipe_count - 1;
    Tensor current_smem_a = thread_smem_a(_, _, _, read_pipe);
    Tensor current_smem_b = thread_smem_b(_, _, _, read_pipe);
    const int register_k_blocks = size<2>(register_a);

    if (register_k_blocks > 1)
    {
        cp_async_wait<kPipelineStages - 2>();
        __syncthreads();
        copy(
            smem_to_register_a,
            current_smem_a(_, _, Int<0>{}),
            thread_register_a(_, _, Int<0>{}));
        copy(
            smem_to_register_b,
            current_smem_b(_, _, Int<0>{}),
            thread_register_b(_, _, Int<0>{}));
    }

    // 三 stage 环形流水。A 的 cp.async 可与当前 tile MMA 重叠；B 的 nibble
    // expansion 是同步算术，但写入独立的下一 stage，并由消费前的
    // cp_async_wait + __syncthreads__ 一并建立跨 warp 可见性。
    CUTE_NO_UNROLL
    while (remaining_k_tiles > -(pipe_count - 1))
    {
        CUTE_UNROLL
        for (int k_block = 0;
             k_block < register_k_blocks;
             ++k_block)
        {
            if (k_block == register_k_blocks - 1)
            {
                current_smem_a = thread_smem_a(_, _, _, read_pipe);
                current_smem_b = thread_smem_b(_, _, _, read_pipe);
                cp_async_wait<kPipelineStages - 2>();
                __syncthreads();
            }

            const auto next_register_k_block =
                (k_block + Int<1>{}) % register_k_blocks;
            copy(
                smem_to_register_a,
                current_smem_a(_, _, next_register_k_block),
                thread_register_a(_, _, next_register_k_block));
            copy(
                smem_to_register_b,
                current_smem_b(_, _, next_register_k_block),
                thread_register_b(_, _, next_register_k_block));

            if (k_block == 0)
            {
                copy(
                    copy_a,
                    thread_global_a(_, _, _, next_k_tile),
                    thread_shared_a(_, _, _, write_pipe));
                unpack_weight_tile_to_smem(
                    packed_weight,
                    weight_scales,
                    shared_b,
                    static_cast<int>(blockIdx.y),
                    next_k_tile,
                    write_pipe,
                    get<2>(problem_shape));
                cp_async_fence();
                --remaining_k_tiles;
                if (remaining_k_tiles > 0)
                {
                    ++next_k_tile;
                }

                write_pipe = read_pipe;
                read_pipe =
                    read_pipe == pipe_count - 1 ? 0 : read_pipe + 1;
            }

            gemm(
                tiled_mma,
                register_a(_, _, k_block),
                register_b(_, _, k_block),
                accumulator);
        }
    }

    // Epilogue 只在所有 K tile 已于 FP32 accumulator 中完成累加后，执行一次
    // FP32 -> FP16 转换。输出语义与常见 W4A16 LLM linear 层一致。
    CUTE_UNROLL
    for (int index = 0; index < size(accumulator); ++index)
    {
        thread_global_c(index) =
            half_t(static_cast<float>(accumulator(index)));
    }
}

void launch_cute_gemm(
    const cute::half_t *activation,
    const uint8_t *packed_weight,
    const cute::half_t *weight_scales,
    cute::half_t *output,
    int m,
    int n,
    int k,
    cudaStream_t stream)
{
    using namespace cute;

    const auto problem_shape = make_shape(m, n, k);
    const auto stride_a = make_stride(k, Int<1>{});
    const auto stride_c = make_stride(n, Int<1>{});
    const auto cta_tiler = make_shape(
        Int<kBlockM>{},
        Int<kBlockN>{},
        Int<kBlockK>{});

    // FP16 的 Swizzle<3,3,3> 以连续 8 half（16B）为 vector base，匹配
    // cp.async 和 ldmatrix.x4，同时通过 XOR 映射降低 SMEM bank conflict。
    const auto swizzle_atom = composition(
        Swizzle<3, 3, 3>{},
        Layout<
            Shape<_8, Shape<_8, _8>>,
            Stride<_8, Stride<_1, _64>>>{});
    const auto smem_layout_a = tile_to_shape(
        swizzle_atom,
        make_shape(
            Int<kBlockM>{},
            Int<kBlockK>{},
            Int<kPipelineStages>{}));
    const auto smem_layout_b = tile_to_shape(
        swizzle_atom,
        make_shape(
            Int<kBlockN>{},
            Int<kBlockK>{},
            Int<kPipelineStages>{}));

    const auto copy_a = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _8>>{});
    const auto tiled_mma = make_tiled_mma(
        SM80_16x8x16_F32F16F16F32_TN{},
        Layout<Shape<_2, _2>>{},
        Tile<_32, _32, _16>{});
    const auto smem_to_register_a =
        Copy_Atom<SM75_U32x4_LDSM_N, half_t>{};
    const auto smem_to_register_b =
        Copy_Atom<SM75_U32x4_LDSM_N, half_t>{};

    using Storage = SharedStorage<
        half_t,
        half_t,
        decltype(smem_layout_a),
        decltype(smem_layout_b)>;
    const int shared_memory_bytes = static_cast<int>(sizeof(Storage));
    const dim3 block(size(tiled_mma));
    const dim3 grid(m / kBlockM, n / kBlockN);

    const auto kernel = cute_gemm_w4a16_kernel<
        decltype(problem_shape),
        decltype(cta_tiler),
        decltype(stride_a),
        decltype(smem_layout_a),
        decltype(copy_a),
        decltype(smem_to_register_a),
        decltype(smem_layout_b),
        decltype(smem_to_register_b),
        decltype(stride_c),
        decltype(tiled_mma)>;
    CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_memory_bytes));
    CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100));

    kernel<<<grid, block, shared_memory_bytes, stream>>>(
        problem_shape,
        cta_tiler,
        activation,
        stride_a,
        smem_layout_a,
        copy_a,
        smem_to_register_a,
        packed_weight,
        weight_scales,
        smem_layout_b,
        smem_to_register_b,
        output,
        stride_c,
        tiled_mma);
}

void launch_cublas_gemm(
    cublasHandle_t handle,
    const cute::half_t *activation,
    const cute::half_t *decoded_weight,
    cute::half_t *output,
    int m,
    int n,
    int k)
{
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;

    // row-major C[M,N] 等价于 column-major C^T[N,M]，所以计算
    // C^T = B * A^T。输入/输出 FP16，compute type 明确为 FP32。
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        decoded_weight,
        CUDA_R_16F,
        k,
        activation,
        CUDA_R_16F,
        k,
        &beta,
        output,
        CUDA_R_16F,
        n,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

struct DeviceComparison
{
    unsigned long long mismatch_count;
    unsigned int max_absolute_error_bits;
    unsigned int max_relative_error_bits;
};

__global__ void compare_outputs_kernel(
    const cute::half_t *actual,
    const cute::half_t *reference,
    size_t element_count,
    DeviceComparison *comparison)
{
    const size_t first =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride =
        static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = first; index < element_count; index += stride)
    {
        const float actual_value = static_cast<float>(actual[index]);
        const float reference_value = static_cast<float>(reference[index]);
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
    const cute::half_t *actual,
    const cute::half_t *reference,
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
    compare_outputs_kernel<<<block_count, 256, 0, stream>>>(
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

float copy_device_half(const cute::half_t *data, size_t index)
{
    cute::half_t value;
    CUDA_CHECK(cudaMemcpy(
        &value,
        data + index,
        sizeof(value),
        cudaMemcpyDeviceToHost));
    return static_cast<float>(value);
}

bool verify_cpu_samples(
    const cute::half_t *cute_output,
    const cute::half_t *cublas_output,
    const uint8_t *packed_weight,
    const cute::half_t *weight_scales,
    int m,
    int n,
    int k)
{
    const std::array<std::array<int, 2>, 8> samples = {{
        {{0, 0}},
        {{0, n - 1}},
        {{m / 4, n / 3}},
        {{m / 3, n / 2}},
        {{m / 2, n / 2}},
        {{m * 3 / 4, n * 2 / 3}},
        {{m - 1, 0}},
        {{m - 1, n - 1}}
    }};
    const int groups_per_row = k / kGroupSize;
    std::vector<uint8_t> host_packed(static_cast<size_t>(k) / 2);
    std::vector<cute::half_t> host_scales(groups_per_row);

    std::cout << "  " << std::left << std::setw(16) << "Coordinate"
              << std::right << std::setw(18) << "CPU FP64"
              << std::setw(20) << kCuteName
              << std::setw(24) << kCublasName << '\n';

    bool passed = true;
    for (const auto &sample : samples)
    {
        const int row = sample[0];
        const int column = sample[1];
        CUDA_CHECK(cudaMemcpy(
            host_packed.data(),
            packed_weight + static_cast<size_t>(column) * k / 2,
            host_packed.size(),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            host_scales.data(),
            weight_scales + static_cast<size_t>(column) * groups_per_row,
            host_scales.size() * sizeof(cute::half_t),
            cudaMemcpyDeviceToHost));

        double cpu_reference = 0.0;
        for (int reduction = 0; reduction < k; ++reduction)
        {
            const float activation = static_cast<float>(cute::half_t(
                make_input_value(
                    static_cast<size_t>(row) * k + reduction,
                    kSeedA)));
            const uint8_t packed = host_packed[reduction / 2];
            const int nibble =
                (packed >> ((reduction & 1) * 4)) & 0x0fU;
            const int quantized = nibble >= 8 ? nibble - 16 : nibble;
            const float scale = static_cast<float>(
                host_scales[reduction / kGroupSize]);
            const float decoded_weight = static_cast<float>(cute::half_t(
                static_cast<float>(quantized) * scale));
            cpu_reference +=
                static_cast<double>(activation) * decoded_weight;
        }

        const size_t output_index =
            static_cast<size_t>(row) * n + column;
        const float cute_value = copy_device_half(cute_output, output_index);
        const float cublas_value =
            copy_device_half(cublas_output, output_index);
        const double tolerance =
            static_cast<double>(kAbsoluteTolerance) +
            static_cast<double>(kRelativeTolerance) *
                std::abs(cpu_reference);
        passed = passed &&
            std::abs(static_cast<double>(cute_value) - cpu_reference) <=
                tolerance &&
            std::abs(static_cast<double>(cublas_value) - cpu_reference) <=
                tolerance;

        const std::string coordinate =
            "(" + std::to_string(row) + "," +
            std::to_string(column) + ")";
        std::cout << "  " << std::left << std::setw(16) << coordinate
                  << std::right << std::setw(18) << cpu_reference
                  << std::setw(20) << cute_value
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
    for (int iteration = 0;
         iteration < benchmark_iterations;
         ++iteration)
    {
        for (size_t offset = 0; offset < CaseCount; ++offset)
        {
            const size_t index =
                (static_cast<size_t>(iteration) + offset) % CaseCount;
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
    for (size_t index = 0; index < CaseCount; ++index)
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
        if (properties.major < 8)
        {
            throw std::runtime_error(
                "W4A16 Tensor Core GEMM requires compute capability 8.0+");
        }

        const size_t count_a =
            static_cast<size_t>(options.m) * options.k;
        const size_t count_b =
            static_cast<size_t>(options.n) * options.k;
        const size_t count_packed_b = count_b / 2;
        const size_t count_scales =
            static_cast<size_t>(options.n) * options.k / kGroupSize;
        const size_t count_c =
            static_cast<size_t>(options.m) * options.n;
        const size_t required_bytes =
            count_a * sizeof(cute::half_t) +
            count_packed_b * sizeof(uint8_t) +
            count_scales * sizeof(cute::half_t) +
            count_b * sizeof(cute::half_t) +
            2 * count_c * sizeof(cute::half_t);

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

        std::cout << "CuTe W4A16 fused dequantization GEMM vs cuBLAS\n\n";
        std::cout << "[Configuration]\n";
        std::cout << "  " << std::left << std::setw(31)
                  << "GPU" << properties.name << '\n';
        std::cout << "  " << std::left << std::setw(31)
                  << "Compute capability"
                  << properties.major << '.' << properties.minor << '\n';
        std::cout << "  " << std::left << std::setw(31)
                  << "A activation" << "FP16 row-major [M,K] = ["
                  << options.m << ',' << options.k << "]\n";
        std::cout << "  " << std::left << std::setw(31)
                  << "B weight" << "packed signed INT4 row-major [N,K] = ["
                  << options.n << ',' << options.k << "]\n";
        std::cout << "  " << std::left << std::setw(31)
                  << "Quantization" << "symmetric per-row/per-group, group=64, "
                  << "range=[-7,7]\n";
        std::cout << "  " << std::left << std::setw(31)
                  << "Scale type" << "FP16\n";
        std::cout << "  " << std::left << std::setw(31)
                  << "C operation" << "FP16 C[M,N] = A * dequant(B)^T\n";
        std::cout << "  " << std::left << std::setw(31)
                  << "CTA tile / threads" << kBlockM << 'x' << kBlockN
                  << 'x' << kBlockK << " / " << kThreads << '\n';
        std::cout << "  " << std::left << std::setw(31)
                  << "Pipeline stages" << kPipelineStages << '\n';
        std::cout << "  " << std::left << std::setw(31)
                  << "A G2S" << "16B cp.async\n";
        std::cout << "  " << std::left << std::setw(31)
                  << "B G2S" << "packed load + fused INT4 decode/scale to FP16 SMEM\n";
        std::cout << "  " << std::left << std::setw(31)
                  << "SMEM / S2R" << "Swizzle<3,3,3> / ldmatrix.x4\n";
        std::cout << "  " << std::left << std::setw(31)
                  << "MMA" << "m16n8k16 FP16 x FP16 -> FP32\n";
        std::cout << "  " << std::left << std::setw(31)
                  << "Epilogue" << "FP32 -> FP16\n";
        std::cout << "  " << std::left << std::setw(31)
                  << "Required device memory"
                  << std::fixed << std::setprecision(2)
                  << static_cast<double>(required_bytes) /
                         (1024.0 * 1024.0)
                  << " MiB\n\n";

        constexpr size_t memory_margin = 256ULL * 1024ULL * 1024ULL;
        if (required_bytes + memory_margin > free_bytes)
        {
            throw std::runtime_error(
                "insufficient free device memory for W4A16 validation");
        }

        CudaStream stream;
        CublasHandle cublas;
        CUBLAS_CHECK(cublasSetStream(cublas.get(), stream.get()));
        CUBLAS_CHECK(cublasSetMathMode(
            cublas.get(),
            CUBLAS_DEFAULT_MATH));

        DeviceBuffer<cute::half_t> device_a(count_a);
        DeviceBuffer<uint8_t> device_packed_b(count_packed_b);
        DeviceBuffer<cute::half_t> device_scales(count_scales);
        DeviceBuffer<cute::half_t> device_decoded_b(count_b);
        DeviceBuffer<cute::half_t> device_cute_c(count_c);
        DeviceBuffer<cute::half_t> device_cublas_c(count_c);
        const int utility_block_count =
            std::max(1, properties.multiProcessorCount * 8);

        std::cout << "[Setup]\n";
        initialize_activation_kernel<<<
            utility_block_count,
            256,
            0,
            stream.get()>>>(
                device_a.get(),
                device_a.count());
        quantize_weight_groups_kernel<<<
            utility_block_count,
            256,
            0,
            stream.get()>>>(
                device_packed_b.get(),
                device_scales.get(),
                options.n,
                options.k);
        decode_weight_reference_kernel<<<
            utility_block_count,
            256,
            0,
            stream.get()>>>(
                device_packed_b.get(),
                device_scales.get(),
                device_decoded_b.get(),
                options.n,
                options.k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));
        std::cout << "  Deterministic FP16 activations initialized\n";
        std::cout << "  FP32 source weights dynamically quantized to packed INT4\n";
        std::cout << "  Decoded FP16 B created only for cuBLAS reference\n\n";

        std::cout << "[Correctness]\n";
        launch_cute_gemm(
            device_a.get(),
            device_packed_b.get(),
            device_scales.get(),
            device_cute_c.get(),
            options.m,
            options.n,
            options.k,
            stream.get());
        CUDA_CHECK(cudaGetLastError());
        launch_cublas_gemm(
            cublas.get(),
            device_a.get(),
            device_decoded_b.get(),
            device_cublas_c.get(),
            options.m,
            options.n,
            options.k);
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));

        const DeviceComparison comparison = compare_outputs(
            device_cute_c.get(),
            device_cublas_c.get(),
            count_c,
            utility_block_count,
            stream.get());
        std::cout << "  " << std::left << std::setw(31)
                  << "Mismatch count" << comparison.mismatch_count << '\n';
        std::cout << "  " << std::left << std::setw(31)
                  << "Max absolute error"
                  << decode_float_bits(comparison.max_absolute_error_bits)
                  << '\n';
        std::cout << "  " << std::left << std::setw(31)
                  << "Max relative error"
                  << decode_float_bits(comparison.max_relative_error_bits)
                  << "\n\n";

        std::cout << "[CPU FP64 samples]\n";
        std::cout << std::fixed << std::setprecision(7);
        const bool cpu_samples_passed = verify_cpu_samples(
            device_cute_c.get(),
            device_cublas_c.get(),
            device_packed_b.get(),
            device_scales.get(),
            options.m,
            options.n,
            options.k);
        std::cout << '\n';
        if (comparison.mismatch_count != 0 || !cpu_samples_passed)
        {
            throw std::runtime_error("W4A16 GEMM correctness failed");
        }

        std::cout << "[Benchmark]\n";
        std::cout << "  The cuBLAS timing excludes one-time reference decode; "
                  << "CuTe includes fused per-CTA INT4 decode.\n";
        std::cout << "  Method: rotating order; median of "
                  << options.benchmark_iterations << " samples\n\n";
        const std::array<BenchmarkCase, 2> benchmark_cases = {{
            {
                kCuteName,
                [&]()
                {
                    launch_cute_gemm(
                        device_a.get(),
                        device_packed_b.get(),
                        device_scales.get(),
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
                    launch_cublas_gemm(
                        cublas.get(),
                        device_a.get(),
                        device_decoded_b.get(),
                        device_cublas_c.get(),
                        options.m,
                        options.n,
                        options.k);
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
                  << std::right << std::setw(14) << "Median ms"
                  << std::setw(18) << "Effective TFLOP/s" << '\n';
        for (size_t index = 0; index < benchmark_cases.size(); ++index)
        {
            std::cout << "  " << std::left << std::setw(28)
                      << benchmark_cases[index].name
                      << std::right << std::setw(14)
                      << benchmark_milliseconds[index]
                      << std::setw(18)
                      << calculate_tflops(
                             options.m,
                             options.n,
                             options.k,
                             benchmark_milliseconds[index])
                      << '\n';
        }

        std::cout << "\n[Key result]\n";
        std::cout << "  " << std::left << std::setw(31)
                  << "Packed weight storage"
                  << static_cast<double>(count_packed_b) /
                         (1024.0 * 1024.0)
                  << " MiB\n";
        std::cout << "  " << std::left << std::setw(31)
                  << "Decoded FP16 storage avoided"
                  << static_cast<double>(count_b * sizeof(cute::half_t)) /
                         (1024.0 * 1024.0)
                  << " MiB in tested CuTe path\n\n";
        std::cout << "[SUCCESS] W4A16 packed INT4 weight-only, fused "
                  << "dequantization, FP32 accumulation, FP16 output and "
                  << "cuBLAS validation passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cout << "[FAILED] " << error.what() << '\n';
        return 1;
    }
}
