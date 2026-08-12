/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

// FP16 输入、FP32 累加的 CuTe GEMM。主循环参考 CUTLASS
// examples/cute/tutorial/sgemm_sm80.cu，并加入与 FP32 测试相同的
// CUTLASS-style Block Swizzle8 映射，用于验证 CTA 发射顺序对访存局部性的影响。

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
constexpr int kPipelineStages = 3;
constexpr int kBlockSwizzleSize = 8;

constexpr const char *kIdentityName = "FP16-TC-Identity";
constexpr const char *kBlockSwizzleName = "FP16-TC-BlockSwizzle8";
constexpr const char *kCublasName = "cuBLAS FP16/FP32";

constexpr float kAbsoluteTolerance = 5.0e-2F;
constexpr float kRelativeTolerance = 2.0e-3F;
constexpr uint32_t kSeedA = 0x1234abcdU;
constexpr uint32_t kSeedB = 0x9e3779b9U;

enum class BlockRasterMode
{
    kIdentity,
    kSwizzle8
};

struct BlockTileCoordinate
{
    int m;
    int n;
};

// Block Swizzle 只改变物理 blockIdx 到逻辑 GEMM tile 的映射，不改变
// CTA 内的 SMEM layout、MMA 或累加顺序。完整 8x8 tile 区域采用
// CUTLASS Blackwell tile scheduler 的 AlongM 映射，M/N 尾部保持 identity。
__host__ __device__ BlockTileCoordinate map_block_to_gemm_tile(
    int physical_m,
    int physical_n,
    int tile_count_m,
    int tile_count_n,
    BlockRasterMode raster_mode)
{
    if (raster_mode == BlockRasterMode::kIdentity)
    {
        return {physical_m, physical_n};
    }

    const int swizzled_tile_count_m =
        tile_count_m / kBlockSwizzleSize * kBlockSwizzleSize;
    const int swizzled_tile_count_n =
        tile_count_n / kBlockSwizzleSize * kBlockSwizzleSize;
    if (physical_m >= swizzled_tile_count_m ||
        physical_n >= swizzled_tile_count_n)
    {
        return {physical_m, physical_n};
    }

    const int m_group_count = tile_count_m / kBlockSwizzleSize;
    const int m_group = physical_m / kBlockSwizzleSize;
    const int m_in_group = physical_m % kBlockSwizzleSize;
    const int n_group = physical_n / kBlockSwizzleSize;
    const int n_in_group = physical_n % kBlockSwizzleSize;
    return {
        m_group + m_group_count * n_in_group,
        m_in_group + kBlockSwizzleSize * n_group};
}

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

struct BlockSwizzleMappingValidation
{
    size_t tile_count = 0;
    size_t unique_tile_count = 0;
    size_t duplicate_count = 0;
    size_t out_of_range_count = 0;
    size_t relocated_tile_count = 0;
};

BlockSwizzleMappingValidation validate_block_swizzle_mapping(
    int tile_count_m,
    int tile_count_n)
{
    BlockSwizzleMappingValidation result;
    result.tile_count =
        static_cast<size_t>(tile_count_m) * tile_count_n;
    std::vector<uint32_t> visits(result.tile_count, 0U);

    for (int physical_n = 0; physical_n < tile_count_n; ++physical_n)
    {
        for (int physical_m = 0; physical_m < tile_count_m; ++physical_m)
        {
            const BlockTileCoordinate logical_tile = map_block_to_gemm_tile(
                physical_m,
                physical_n,
                tile_count_m,
                tile_count_n,
                BlockRasterMode::kSwizzle8);
            if (logical_tile.m < 0 || logical_tile.m >= tile_count_m ||
                logical_tile.n < 0 || logical_tile.n >= tile_count_n)
            {
                ++result.out_of_range_count;
                continue;
            }

            if (logical_tile.m != physical_m ||
                logical_tile.n != physical_n)
            {
                ++result.relocated_tile_count;
            }

            const size_t logical_index =
                static_cast<size_t>(logical_tile.n) * tile_count_m +
                logical_tile.m;
            if (visits[logical_index] == 0U)
            {
                ++result.unique_tile_count;
            }
            else
            {
                ++result.duplicate_count;
            }
            ++visits[logical_index];
        }
    }
    return result;
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
    // 1/512 是二进制精确值，转换为 FP16 后 CPU reference 可以精确复现输入。
    return static_cast<float>(centered) * (1.0F / 512.0F);
}

__global__ void initialize_input_kernel(
    cute::half_t *data,
    size_t element_count,
    uint32_t seed)
{
    const size_t first =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride =
        static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = first; index < element_count; index += stride)
    {
        data[index] = cute::half_t(make_input_value(index, seed));
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

template <
    class ProblemShape,
    class CtaTiler,
    class AStride,
    class ASmemLayout,
    class TiledCopyA,
    class S2RAtomA,
    class BStride,
    class BSmemLayout,
    class TiledCopyB,
    class S2RAtomB,
    class CStride,
    class TiledMma>
__global__ __launch_bounds__(decltype(cute::size(TiledMma{}))::value)
void cute_gemm_fp16_fp32_kernel(
    ProblemShape problem_shape,
    CtaTiler cta_tiler,
    int tile_count_m,
    int tile_count_n,
    BlockRasterMode block_raster_mode,
    const cute::half_t *a,
    AStride stride_a,
    ASmemLayout smem_layout_a,
    TiledCopyA copy_a,
    S2RAtomA smem_to_register_a,
    const cute::half_t *b,
    BStride stride_b,
    BSmemLayout smem_layout_b,
    TiledCopyB copy_b,
    S2RAtomB smem_to_register_b,
    float *c,
    CStride stride_c,
    TiledMma tiled_mma)
{
    using namespace cute;

    CUTE_STATIC_ASSERT_V(rank(problem_shape) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size(copy_a) == size(tiled_mma));
    CUTE_STATIC_ASSERT_V(size(copy_b) == size(tiled_mma));

    // make_tensor 创建 row-major A[M,K]、B[N,K]、C[M,N] 视图。
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

    // Identity 与 Swizzle8 的唯一差别在这里。后面的 G2S、SMEM、S2R、
    // Tensor Core MMA 和 epilogue 使用完全相同的指令路径。
    const BlockTileCoordinate block_tile = map_block_to_gemm_tile(
        static_cast<int>(blockIdx.x),
        static_cast<int>(blockIdx.y),
        tile_count_m,
        tile_count_n,
        block_raster_mode);
    const auto cta_coordinate = make_coord(block_tile.m, block_tile.n, _);
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

    // 每个线程每次搬运连续 8 个 FP16（16B）。partition_S/D 把完整
    // [128,64] tile 分给 128 个线程，copy 会生成 cp.async G2S。
    ThrCopy thread_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor thread_global_a = thread_copy_a.partition_S(block_a);
    Tensor thread_shared_a = thread_copy_a.partition_D(shared_a);
    ThrCopy thread_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor thread_global_b = thread_copy_b.partition_S(block_b);
    Tensor thread_shared_b = thread_copy_b.partition_D(shared_b);

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
        copy(
            copy_b,
            thread_global_b(_, _, _, next_k_tile),
            thread_shared_b(_, _, _, pipe));
        cp_async_fence();
        --remaining_k_tiles;
        if (remaining_k_tiles > 0)
        {
            ++next_k_tile;
        }
    }

    // SM80_16x8x16_F32F16F16F32_TN 令 A/B fragment 为 FP16，
    // make_fragment_C 得到 FP32 寄存器累加器。
    ThrMMA thread_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor thread_global_c = thread_mma.partition_C(block_c);
    Tensor register_a =
        thread_mma.partition_fragment_A(shared_a(_, _, 0));
    Tensor register_b =
        thread_mma.partition_fragment_B(shared_b(_, _, 0));
    Tensor accumulator = thread_mma.make_fragment_C(thread_global_c);
    clear(accumulator);

    // LDSM.x4 依据 Tensor Core lane layout，从 swizzled SMEM 直接装载
    // MMA 所需的寄存器 fragment。
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

    // 三层流水：cp.async 负责 GMEM->SMEM，LDSM 负责 SMEM->寄存器，
    // gemm 发出 mma.sync，并始终在 FP32 accumulator 中完成 K 维累加。
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
                copy(
                    copy_b,
                    thread_global_b(_, _, _, next_k_tile),
                    thread_shared_b(_, _, _, write_pipe));
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

    // 输出保持 FP32，不在 epilogue 中降精度。
    copy(accumulator, thread_global_c);
}

void launch_cute_gemm(
    const cute::half_t *a,
    const cute::half_t *b,
    float *c,
    int m,
    int n,
    int k,
    cudaStream_t stream,
    BlockRasterMode block_raster_mode)
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

    // FP16 元素占 2B，Swizzle<3,3,3> 保留 8 个元素（16B）作为
    // vector base，并将 K 高位 XOR 到 bank 位。这是经典的 Tensor Core
    // 16B G2S + LDSM 布局，既匹配 cp.async，也避免 LDSM bank conflict。
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

    // 128 个线程按 16x8 排列，每线程每条 cp.async 搬运 8 个 half=16B。
    const auto copy_a = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _8>>{});
    const auto copy_b = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _8>>{});

    // 指令语义为 D(FP32) = A(FP16) * B(FP16) + C(FP32)。2x2 个
    // MMA atom 构成 128-thread tiled MMA，随后在 CTA tile 内重复铺满 128x128。
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
    const int tile_count_m = size(ceil_div(m, Int<kBlockM>{}));
    const int tile_count_n = size(ceil_div(n, Int<kBlockN>{}));
    const dim3 block(size(tiled_mma));
    const dim3 grid(tile_count_m, tile_count_n);

    const auto kernel = cute_gemm_fp16_fp32_kernel<
        decltype(problem_shape),
        decltype(cta_tiler),
        decltype(stride_a),
        decltype(smem_layout_a),
        decltype(copy_a),
        decltype(smem_to_register_a),
        decltype(stride_b),
        decltype(smem_layout_b),
        decltype(copy_b),
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
        tile_count_m,
        tile_count_n,
        block_raster_mode,
        a,
        stride_a,
        smem_layout_a,
        copy_a,
        smem_to_register_a,
        b,
        stride_b,
        smem_layout_b,
        copy_b,
        smem_to_register_b,
        c,
        stride_c,
        tiled_mma);
}

void launch_cublas_gemm(
    cublasHandle_t handle,
    const cute::half_t *a,
    const cute::half_t *b,
    float *c,
    int m,
    int n,
    int k)
{
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;

    // row-major C[M,N] 的内存等价于 column-major C^T[N,M]，因此令
    // cuBLAS 计算 C^T = B * A^T。A/B 数据类型为 CUDA_R_16F，
    // computeType 和 C 数据类型均为 FP32。
    CUBLAS_CHECK(cublasGemmEx(
        handle,
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
        CUDA_R_32F,
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
    const float *actual,
    const float *reference,
    size_t element_count,
    float absolute_tolerance,
    float relative_tolerance,
    bool compare_bits,
    DeviceComparison *comparison)
{
    const size_t first =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride =
        static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = first; index < element_count; index += stride)
    {
        const float actual_value = actual[index];
        const float reference_value = reference[index];
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
        const bool mismatch = compare_bits
            ? __float_as_uint(actual_value) !=
                __float_as_uint(reference_value)
            : absolute_error > tolerance;
        if (mismatch)
        {
            atomicAdd(&comparison->mismatch_count, 1ULL);
        }
    }
}

DeviceComparison compare_outputs(
    const float *actual,
    const float *reference,
    size_t element_count,
    int block_count,
    cudaStream_t stream,
    float absolute_tolerance = kAbsoluteTolerance,
    float relative_tolerance = kRelativeTolerance,
    bool compare_bits = false)
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
        absolute_tolerance,
        relative_tolerance,
        compare_bits,
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

float copy_device_value(const float *data, size_t index)
{
    float value = 0.0F;
    CUDA_CHECK(cudaMemcpy(
        &value,
        data + index,
        sizeof(float),
        cudaMemcpyDeviceToHost));
    return value;
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
        result +=
            static_cast<double>(make_input_value(index_a, kSeedA)) *
            static_cast<double>(make_input_value(index_b, kSeedB));
    }
    return result;
}

bool verify_cpu_samples(
    const float *identity_output,
    const float *block_swizzle_output,
    const float *cublas_output,
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
              << std::right << std::setw(16) << "CPU FP64"
              << std::setw(20) << kIdentityName
              << std::setw(26) << kBlockSwizzleName
              << std::setw(22) << kCublasName << '\n';

    bool passed = true;
    for (const auto &sample : samples)
    {
        const int row = sample.first;
        const int column = sample.second;
        const size_t index = static_cast<size_t>(row) * n + column;
        const float identity_value =
            copy_device_value(identity_output, index);
        const float block_swizzle_value =
            copy_device_value(block_swizzle_output, index);
        const float cublas_value = copy_device_value(cublas_output, index);
        const double reference = cpu_reference_value(row, column, k);
        const double tolerance =
            static_cast<double>(kAbsoluteTolerance) +
            static_cast<double>(kRelativeTolerance) * std::abs(reference);
        passed = passed &&
            std::abs(static_cast<double>(identity_value) - reference) <=
                tolerance &&
            std::abs(static_cast<double>(block_swizzle_value) - reference) <=
                tolerance &&
            std::abs(static_cast<double>(cublas_value) - reference) <=
                tolerance;

        const std::string coordinate =
            "(" + std::to_string(row) + "," +
            std::to_string(column) + ")";
        std::cout << "  " << std::left << std::setw(16) << coordinate
                  << std::right << std::setw(16) << reference
                  << std::setw(20) << identity_value
                  << std::setw(26) << block_swizzle_value
                  << std::setw(22) << cublas_value << '\n';
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
        if (properties.major < 8)
        {
            throw std::runtime_error(
                "FP16 Tensor Core GEMM requires compute capability 8.0+");
        }

        const size_t count_a =
            static_cast<size_t>(options.m) * options.k;
        const size_t count_b =
            static_cast<size_t>(options.n) * options.k;
        const size_t count_c =
            static_cast<size_t>(options.m) * options.n;
        const size_t required_bytes =
            (count_a + count_b) * sizeof(cute::half_t) +
            3 * count_c * sizeof(float);

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

        std::cout << "CuTe FP16 input / FP32 accumulate GEMM Block Swizzle "
                  << "test vs cuBLAS\n\n";
        std::cout << "[Configuration]\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "GPU" << properties.name << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "Compute capability"
                  << properties.major << '.' << properties.minor << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "A layout" << "FP16 row-major [M,K] = ["
                  << options.m << ',' << options.k << "]\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "B layout" << "FP16 row-major [N,K] = ["
                  << options.n << ',' << options.k << "]\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "C operation" << "FP32 C[M,N] = A * B^T\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "CTA tile" << kBlockM << 'x' << kBlockN
                  << 'x' << kBlockK << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "Thread block" << "128 threads\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "Pipeline stages" << kPipelineStages << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "G2S" << "16B cp.async per instruction\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "SMEM layout" << "Swizzle<3,3,3>\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "S2R" << "ldmatrix.x4\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "MMA" << "m16n8k16 FP16 inputs / FP32 accumulate\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "Block raster"
                  << "Identity / CUTLASS-style 2D Swizzle8\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "Required device memory"
                  << std::fixed << std::setprecision(2)
                  << static_cast<double>(required_bytes) /
                         (1024.0 * 1024.0)
                  << " MiB\n\n";

        constexpr size_t memory_margin = 256ULL * 1024ULL * 1024ULL;
        if (required_bytes + memory_margin > free_bytes)
        {
            throw std::runtime_error(
                "insufficient free device memory for A, B and three C buffers");
        }

        CudaStream stream;
        CublasHandle cublas;
        CUBLAS_CHECK(cublasSetStream(cublas.get(), stream.get()));
        CUBLAS_CHECK(cublasSetMathMode(
            cublas.get(),
            CUBLAS_DEFAULT_MATH));

        std::cout << "[Setup]\n";
        DeviceBuffer<cute::half_t> device_a(count_a);
        DeviceBuffer<cute::half_t> device_b(count_b);
        DeviceBuffer<float> device_identity_c(count_c);
        DeviceBuffer<float> device_block_swizzle_c(count_c);
        DeviceBuffer<float> device_cublas_c(count_c);
        const int utility_block_count =
            std::max(1, properties.multiProcessorCount * 8);
        initialize_input_kernel<<<
            utility_block_count,
            256,
            0,
            stream.get()>>>(
                device_a.get(),
                device_a.count(),
                kSeedA);
        initialize_input_kernel<<<
            utility_block_count,
            256,
            0,
            stream.get()>>>(
                device_b.get(),
                device_b.count(),
                kSeedB);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));
        std::cout << "  Deterministic FP16 inputs initialized on GPU\n\n";

        const int tile_count_m = options.m / kBlockM;
        const int tile_count_n = options.n / kBlockN;
        const BlockSwizzleMappingValidation mapping_validation =
            validate_block_swizzle_mapping(tile_count_m, tile_count_n);
        const bool mapping_passed =
            mapping_validation.unique_tile_count ==
                mapping_validation.tile_count &&
            mapping_validation.duplicate_count == 0 &&
            mapping_validation.out_of_range_count == 0;

        std::cout << "[Block Swizzle mapping]\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "Physical tile grid" << tile_count_m << 'x'
                  << tile_count_n << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "Swizzled main region"
                  << tile_count_m / kBlockSwizzleSize * kBlockSwizzleSize
                  << 'x'
                  << tile_count_n / kBlockSwizzleSize * kBlockSwizzleSize
                  << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "Unique logical tiles"
                  << mapping_validation.unique_tile_count << " / "
                  << mapping_validation.tile_count << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "Relocated tiles"
                  << mapping_validation.relocated_tile_count << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "Duplicate / out-of-range"
                  << mapping_validation.duplicate_count << " / "
                  << mapping_validation.out_of_range_count << "\n\n";
        if (!mapping_passed)
        {
            throw std::runtime_error(
                "Block Swizzle mapping is not a bijection");
        }

        std::cout << "[Correctness]\n";
        launch_cute_gemm(
            device_a.get(),
            device_b.get(),
            device_identity_c.get(),
            options.m,
            options.n,
            options.k,
            stream.get(),
            BlockRasterMode::kIdentity);
        launch_cute_gemm(
            device_a.get(),
            device_b.get(),
            device_block_swizzle_c.get(),
            options.m,
            options.n,
            options.k,
            stream.get(),
            BlockRasterMode::kSwizzle8);
        CUDA_CHECK(cudaGetLastError());
        launch_cublas_gemm(
            cublas.get(),
            device_a.get(),
            device_b.get(),
            device_cublas_c.get(),
            options.m,
            options.n,
            options.k);
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));

        const DeviceComparison identity_vs_cublas = compare_outputs(
            device_identity_c.get(),
            device_cublas_c.get(),
            count_c,
            utility_block_count,
            stream.get());
        const DeviceComparison block_swizzle_vs_cublas = compare_outputs(
            device_block_swizzle_c.get(),
            device_cublas_c.get(),
            count_c,
            utility_block_count,
            stream.get());
        const DeviceComparison block_swizzle_vs_identity = compare_outputs(
            device_block_swizzle_c.get(),
            device_identity_c.get(),
            count_c,
            utility_block_count,
            stream.get(),
            0.0F,
            0.0F,
            true);

        std::cout << "  " << std::left << std::setw(28) << "Comparison"
                  << std::right << std::setw(16) << "Mismatches"
                  << std::setw(18) << "Max abs error"
                  << std::setw(18) << "Max rel error" << '\n';
        const std::array<std::pair<const char *, DeviceComparison>, 2>
            comparisons = {{
                {"Identity vs cuBLAS", identity_vs_cublas},
                {"Swizzle8 vs cuBLAS", block_swizzle_vs_cublas}
            }};
        bool correctness_passed = true;
        for (const auto &comparison : comparisons)
        {
            std::cout << "  " << std::left << std::setw(28)
                      << comparison.first
                      << std::right << std::setw(16)
                      << comparison.second.mismatch_count
                      << std::setw(18)
                      << decode_float_bits(
                             comparison.second.max_absolute_error_bits)
                      << std::setw(18)
                      << decode_float_bits(
                             comparison.second.max_relative_error_bits)
                      << '\n';
            correctness_passed = correctness_passed &&
                comparison.second.mismatch_count == 0;
        }
        std::cout << "  " << std::left << std::setw(28)
                  << "Swizzle8 vs Identity"
                  << std::right << std::setw(16)
                  << block_swizzle_vs_identity.mismatch_count
                  << " bitwise mismatches\n\n";
        correctness_passed = correctness_passed &&
            block_swizzle_vs_identity.mismatch_count == 0;

        std::cout << "[CPU FP64 samples]\n";
        std::cout << std::fixed << std::setprecision(7);
        const bool cpu_samples_passed = verify_cpu_samples(
            device_identity_c.get(),
            device_block_swizzle_c.get(),
            device_cublas_c.get(),
            options.m,
            options.n,
            options.k);
        std::cout << '\n';
        if (!correctness_passed || !cpu_samples_passed)
        {
            throw std::runtime_error(
                "FP16 input / FP32 accumulate GEMM correctness failed");
        }

        std::cout << "[Benchmark]\n";
        std::cout << "  Method: rotating order; median of "
                  << options.benchmark_iterations << " samples\n\n";
        const std::array<BenchmarkCase, 3> benchmark_cases = {{
            {
                kIdentityName,
                [&]()
                {
                    launch_cute_gemm(
                        device_a.get(),
                        device_b.get(),
                        device_identity_c.get(),
                        options.m,
                        options.n,
                        options.k,
                        stream.get(),
                        BlockRasterMode::kIdentity);
                    CUDA_CHECK(cudaGetLastError());
                }
            },
            {
                kBlockSwizzleName,
                [&]()
                {
                    launch_cute_gemm(
                        device_a.get(),
                        device_b.get(),
                        device_block_swizzle_c.get(),
                        options.m,
                        options.n,
                        options.k,
                        stream.get(),
                        BlockRasterMode::kSwizzle8);
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
                        device_b.get(),
                        device_cublas_c.get(),
                        options.m,
                        options.n,
                        options.k);
                }
            }
        }};
        const std::array<double, 3> benchmark_milliseconds =
            benchmark_round_robin(
                benchmark_cases,
                options.warmup_iterations,
                options.benchmark_iterations,
                stream.get());

        std::cout << "  " << std::left << std::setw(28)
                  << "Implementation"
                  << std::right << std::setw(14) << "Median ms"
                  << std::setw(16) << "TFLOP/s"
                  << std::setw(18) << "vs cuBLAS" << '\n';
        const double cublas_tflops = calculate_tflops(
            options.m,
            options.n,
            options.k,
            benchmark_milliseconds[2]);
        for (size_t index = 0; index < benchmark_cases.size(); ++index)
        {
            const double tflops = calculate_tflops(
                options.m,
                options.n,
                options.k,
                benchmark_milliseconds[index]);
            std::cout << "  " << std::left << std::setw(28)
                      << benchmark_cases[index].name
                      << std::right << std::setw(14)
                      << benchmark_milliseconds[index]
                      << std::setw(16) << tflops
                      << std::setw(17) << tflops / cublas_tflops * 100.0
                      << "%\n";
        }

        const double block_swizzle_speedup =
            benchmark_milliseconds[0] / benchmark_milliseconds[1];
        std::cout << "\n[Key result]\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "Block Swizzle speedup"
                  << std::fixed << std::setprecision(3)
                  << block_swizzle_speedup << "x vs Identity\n\n";
        std::cout << "[SUCCESS] FP16 inputs, FP32 accumulation, "
                  << "Swizzle<3,3,3>, Block Swizzle8 and cuBLAS "
                  << "validation passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cout << "[FAILED] " << error.what() << '\n';
        return 1;
    }
}
