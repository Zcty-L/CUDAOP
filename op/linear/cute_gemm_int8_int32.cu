/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

// INT8 输入、INT32 累加的 CuTe GEMM。测试显式覆盖
// FP32 -> INT8 tensor-wide 对称量化、量化域 Tensor Core IMMA、
// 原始 INT32 精确校验、INT32 -> FP32 反量化，以及 Block Swizzle8 CTA 映射。

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
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cute/algorithm/cooperative_gemm.hpp>

namespace
{

constexpr int kDefaultM = 8192;
constexpr int kDefaultN = 8192;
constexpr int kDefaultK = 8192;
constexpr int kDefaultWarmupIterations = 2;
constexpr int kDefaultBenchmarkIterations = 5;

constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockK = 128;
constexpr int kPipelineStages = 3;
constexpr int kBlockSwizzleSize = 8;

constexpr const char *kIdentityName = "INT8/INT32-Identity";
constexpr const char *kBlockSwizzleName = "INT8/INT32-BlockSwizzle8";
constexpr const char *kCublasName = "cuBLASLt INT8/INT32";

// 生成值位于 [-0.5, 0.5)，使用对称 per-tensor 量化：
// q = clamp(round(x / scale), -127, 127)，zero point 固定为 0。
// INT8 的 -128 不参与量化，使正负量化范围保持对称。
constexpr float kInputAbsoluteMaximum = 0.5F;
constexpr float kQuantizedAbsoluteTarget = 127.0F;
constexpr float kScaleA =
    kInputAbsoluteMaximum / kQuantizedAbsoluteTarget;
constexpr float kScaleB =
    kInputAbsoluteMaximum / kQuantizedAbsoluteTarget;
constexpr float kDequantScale = kScaleA * kScaleB;

// K=8192 时最坏累加绝对值为 8192 * 127^2 = 132,128,768，
// 仍处于 INT32 范围。本测试比较原始 INT32，CuTe 与 cuBLASLt 应完全相同。
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

class CublasLtGemm
{
public:
    CublasLtGemm(int m, int n, int k)
        : workspace_(kWorkspaceBytes)
    {
        CUBLAS_CHECK(cublasLtCreate(&handle_));
        CUBLAS_CHECK(cublasLtMatmulDescCreate(
            &operation_,
            CUBLAS_COMPUTE_32I,
            CUDA_R_32I));

        // row-major C[M,N] 的内存等价于 column-major C^T[N,M]。
        // 令 cuBLASLt 计算 C^T = B * A^T，可同时满足 INT8 kernel 要求的
        // TN 指令布局：第一个 operand 转置，第二个 operand 不转置。
        constexpr cublasOperation_t transpose = CUBLAS_OP_T;
        constexpr cublasOperation_t no_transpose = CUBLAS_OP_N;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            operation_,
            CUBLASLT_MATMUL_DESC_TRANSA,
            &transpose,
            sizeof(transpose)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            operation_,
            CUBLASLT_MATMUL_DESC_TRANSB,
            &no_transpose,
            sizeof(no_transpose)));

        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
            &layout_a_,
            CUDA_R_8I,
            k,
            n,
            k));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
            &layout_b_,
            CUDA_R_8I,
            k,
            m,
            k));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
            &layout_c_,
            CUDA_R_32I,
            n,
            m,
            n));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
            &layout_d_,
            CUDA_R_32I,
            n,
            m,
            n));

        CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference_));
        const size_t workspace_bytes = kWorkspaceBytes;
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
            preference_,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_bytes,
            sizeof(workspace_bytes)));

        constexpr int requested_algorithm_count = 8;
        std::array<cublasLtMatmulHeuristicResult_t,
                   requested_algorithm_count> results{};
        int returned_algorithm_count = 0;
        CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
            handle_,
            operation_,
            layout_a_,
            layout_b_,
            layout_c_,
            layout_d_,
            preference_,
            requested_algorithm_count,
            results.data(),
            &returned_algorithm_count));
        if (returned_algorithm_count == 0)
        {
            throw std::runtime_error(
                "cuBLASLt found no INT8 GEMM algorithm");
        }
        algorithm_ = results[0].algo;
    }

    ~CublasLtGemm()
    {
        if (preference_ != nullptr)
        {
            cublasLtMatmulPreferenceDestroy(preference_);
        }
        if (layout_d_ != nullptr)
        {
            cublasLtMatrixLayoutDestroy(layout_d_);
        }
        if (layout_c_ != nullptr)
        {
            cublasLtMatrixLayoutDestroy(layout_c_);
        }
        if (layout_b_ != nullptr)
        {
            cublasLtMatrixLayoutDestroy(layout_b_);
        }
        if (layout_a_ != nullptr)
        {
            cublasLtMatrixLayoutDestroy(layout_a_);
        }
        if (operation_ != nullptr)
        {
            cublasLtMatmulDescDestroy(operation_);
        }
        if (handle_ != nullptr)
        {
            cublasLtDestroy(handle_);
        }
    }

    CublasLtGemm(const CublasLtGemm &) = delete;
    CublasLtGemm &operator=(const CublasLtGemm &) = delete;

    void launch(
        const int8_t *a,
        const int8_t *b,
        int32_t *c,
        cudaStream_t stream)
    {
        constexpr int32_t alpha = 1;
        constexpr int32_t beta = 0;
        CUBLAS_CHECK(cublasLtMatmul(
            handle_,
            operation_,
            &alpha,
            b,
            layout_a_,
            a,
            layout_b_,
            &beta,
            c,
            layout_c_,
            c,
            layout_d_,
            &algorithm_,
            workspace_.get(),
            workspace_.count(),
            stream));
    }

private:
    static constexpr size_t kWorkspaceBytes = 32ULL * 1024ULL * 1024ULL;

    cublasLtHandle_t handle_ = nullptr;
    cublasLtMatmulDesc_t operation_ = nullptr;
    cublasLtMatrixLayout_t layout_a_ = nullptr;
    cublasLtMatrixLayout_t layout_b_ = nullptr;
    cublasLtMatrixLayout_t layout_c_ = nullptr;
    cublasLtMatrixLayout_t layout_d_ = nullptr;
    cublasLtMatmulPreference_t preference_ = nullptr;
    cublasLtMatmulAlgo_t algorithm_{};
    DeviceBuffer<uint8_t> workspace_;
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
            "M/N/K must be exact multiples of the 128x128x128 CTA tile");
    }
    if (options.k < (kPipelineStages - 1) * kBlockK)
    {
        throw std::runtime_error("K must contain at least two CTA K tiles");
    }
    constexpr int64_t maximum_product = 127LL * 127LL;
    if (static_cast<int64_t>(options.k) * maximum_product > INT32_MAX)
    {
        throw std::runtime_error(
            "K is too large for worst-case non-saturating INT32 accumulation");
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
    // 1/512 是二进制精确值，便于 CPU reference 复现量化前输入。
    return static_cast<float>(centered) * (1.0F / 512.0F);
}

__host__ __device__ int8_t quantize_input(float value, float scale)
{
#if defined(__CUDA_ARCH__)
    int quantized = __float2int_rn(value / scale);
#else
    int quantized = static_cast<int>(std::nearbyint(value / scale));
#endif
    quantized = quantized < -127 ? -127 : quantized;
    quantized = quantized > 127 ? 127 : quantized;
    return static_cast<int8_t>(quantized);
}

__global__ void initialize_input_kernel(
    int8_t *data,
    size_t element_count,
    uint32_t seed,
    float scale)
{
    const size_t first =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride =
        static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = first; index < element_count; index += stride)
    {
        // tensor-wide 对称量化：q = INT8(x / scale)。这里的数据生成器
        // 已知 amax，因此无需额外启动 GPU reduction 求 amax。
        data[index] = quantize_input(
            make_input_value(index, seed),
            scale);
    }
}

__global__ void dequantize_output_kernel(
    const int32_t *input,
    float *output,
    size_t element_count,
    float scale)
{
    const size_t first =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride =
        static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = first; index < element_count; index += stride)
    {
        // 反量化只恢复实数语义，不改变作为权威正确性结果的 INT32 C。
        output[index] = static_cast<float>(input[index]) * scale;
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
    class BStride,
    class BSmemLayout,
    class TiledCopyB,
    class CStride,
    class TiledMma>
__global__ __launch_bounds__(decltype(cute::size(TiledMma{}))::value)
void cute_gemm_int8_int32_kernel(
    ProblemShape problem_shape,
    CtaTiler cta_tiler,
    int tile_count_m,
    int tile_count_n,
    BlockRasterMode block_raster_mode,
    const int8_t *a,
    AStride stride_a,
    ASmemLayout smem_layout_a,
    TiledCopyA copy_a,
    const int8_t *b,
    BStride stride_b,
    BSmemLayout smem_layout_b,
    TiledCopyB copy_b,
    int32_t *c,
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
        int8_t,
        int8_t,
        ASmemLayout,
        BSmemLayout>;
    Storage &storage = *reinterpret_cast<Storage *>(shared_memory);
    Tensor shared_a = make_tensor(
        make_smem_ptr(storage.a.begin()),
        smem_layout_a);
    Tensor shared_b = make_tensor(
        make_smem_ptr(storage.b.begin()),
        smem_layout_b);

    // 每个线程每次搬运连续 16 个 INT8（16B）。partition_S/D 把完整
    // [128,128] tile 分给 128 个线程，copy 会生成 cp.async G2S。
    ThrCopy thread_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor thread_global_a = thread_copy_a.partition_S(block_a);
    Tensor thread_shared_a = thread_copy_a.partition_D(shared_a);
    ThrCopy thread_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor thread_global_b = thread_copy_b.partition_S(block_b);
    Tensor thread_shared_b = thread_copy_b.partition_D(shared_b);

    // SM80_16x8x32_S32S8S8S32_TN 令 A/B fragment 为 INT8，
    // make_fragment_C 得到 INT32 寄存器累加器。
    ThrMMA thread_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor thread_global_c = thread_mma.partition_C(block_c);
    Tensor accumulator = thread_mma.make_fragment_C(thread_global_c);
    clear(accumulator);

    // 三阶段环形流水的 prologue：先把两个 [128,128] 输入 tile 分别
    // 发往 stage 0/1。每个 cp_async_fence() 提交一个独立 group，使后续
    // wait_group<1> 可以只等待最老 group，同时保留下一 group 在途。
    const int k_tile_count = size<3>(thread_global_a);
    CUTE_UNROLL
    for (int stage = 0; stage < kPipelineStages - 1; ++stage)
    {
        copy(
            copy_a,
            thread_global_a(_, _, _, stage),
            thread_shared_a(_, _, _, stage));
        copy(
            copy_b,
            thread_global_b(_, _, _, stage),
            thread_shared_b(_, _, _, stage));
        cp_async_fence();
    }

    int next_k_tile = kPipelineStages - 1;
    int read_stage = 0;
    int outstanding_groups = kPipelineStages - 1;

    // Steady state：wait_group<1> 只保证最老的 read stage 已就绪；随后
    // 立即向空闲 write stage 发出下一 tile，再计算 read stage。因此下一
    // tile 的 cp.async G2S 可以与当前 tile 的 S2R/IMMA 跨 stage 重叠。
    CUTE_NO_UNROLL
    for (int computed_k_tile = 0;
         computed_k_tile < k_tile_count;
         ++computed_k_tile)
    {
        if (outstanding_groups > kPipelineStages - 2)
        {
            cp_async_wait<kPipelineStages - 2>();
        }
        else
        {
            // Tail 中只剩一个 group 时，wait_group<1> 可能立即返回；必须
            // 改用 wait_group<0>，确保最后一个 read stage 已真正完成。
            cp_async_wait<0>();
        }
        --outstanding_groups;
        __syncthreads();

        if (next_k_tile < k_tile_count)
        {
            const int write_stage = next_k_tile % kPipelineStages;
            copy(
                copy_a,
                thread_global_a(_, _, _, next_k_tile),
                thread_shared_a(_, _, _, write_stage));
            copy(
                copy_b,
                thread_global_b(_, _, _, next_k_tile),
                thread_shared_b(_, _, _, write_stage));
            cp_async_fence();
            ++outstanding_groups;
            ++next_k_tile;
        }

        cute::detail::cooperative_gemm_no_predication(
            static_cast<uint32_t>(threadIdx.x),
            thread_mma,
            shared_a(_, _, read_stage),
            shared_b(_, _, read_stage),
            accumulator,
            identity{},
            identity{},
            AutoVectorizingCopyWithAssumedAlignment<128>{},
            AutoVectorizingCopyWithAssumedAlignment<128>{});

        read_stage = read_stage == kPipelineStages - 1
            ? 0
            : read_stage + 1;
    }

    // 保留量化域的 INT32 accumulator 作为主输出。这样可以无容差地
    // 验证 IMMA 本身；FP32 反量化由独立 epilogue kernel 完成。
    CUTE_UNROLL
    for (int index = 0; index < size(accumulator); ++index)
    {
        thread_global_c(index) = accumulator(index);
    }
}

void launch_cute_gemm(
    const int8_t *a,
    const int8_t *b,
    int32_t *c,
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

    // INT8 元素占 1B。K tile 每行正好 128B，Swizzle<3,4,3>
    // 保留 16 个元素（16B）作为 vector base，并把 K 高位 XOR 到
    // shared-memory bank 位，匹配 128B swizzle 周期。
    const auto swizzle_atom = composition(
        Swizzle<3, 4, 3>{},
        Layout<
            Shape<_8, Shape<_16, _8>>,
            Stride<_16, Stride<_1, _128>>>{});
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

    // 128 个线程按 16x8 排列，每线程每条 cp.async 搬运 16 个 INT8=16B。
    // thread K=8 与 value K=16 合成 K=128，copy atom 的逻辑覆盖为
    // [16,128]，随后沿 M 维重复 8 次完整覆盖 [128,128]。
    const auto copy_a = make_tiled_copy(
        Copy_Atom<
            SM80_CP_ASYNC_CACHEALWAYS<uint128_t>,
            int8_t>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _16>>{});
    const auto copy_b = make_tiled_copy(
        Copy_Atom<
            SM80_CP_ASYNC_CACHEALWAYS<uint128_t>,
            int8_t>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _16>>{});

    // 指令语义为 D(INT32) = A(INT8) * B(INT8) + C(INT32)。2x2 个
    // m16n8k32 atom 构成 128-thread tiled MMA，再铺满 128x128 CTA tile。
    const auto tiled_mma = make_tiled_mma(
        SM80_16x8x32_S32S8S8S32_TN{},
        Layout<Shape<_2, _2>>{},
        Tile<_32, _32, _32>{});
    using Storage = SharedStorage<
        int8_t,
        int8_t,
        decltype(smem_layout_a),
        decltype(smem_layout_b)>;
    const int shared_memory_bytes = static_cast<int>(sizeof(Storage));
    const int tile_count_m = size(ceil_div(m, Int<kBlockM>{}));
    const int tile_count_n = size(ceil_div(n, Int<kBlockN>{}));
    const dim3 block(size(tiled_mma));
    const dim3 grid(tile_count_m, tile_count_n);

    const auto kernel = cute_gemm_int8_int32_kernel<
        decltype(problem_shape),
        decltype(cta_tiler),
        decltype(stride_a),
        decltype(smem_layout_a),
        decltype(copy_a),
        decltype(stride_b),
        decltype(smem_layout_b),
        decltype(copy_b),
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
        b,
        stride_b,
        smem_layout_b,
        copy_b,
        c,
        stride_c,
        tiled_mma);
}

struct DeviceComparison
{
    unsigned long long mismatch_count;
    unsigned long long max_absolute_error;
};

__global__ void compare_outputs_kernel(
    const int32_t *actual,
    const int32_t *reference,
    size_t element_count,
    DeviceComparison *comparison)
{
    const size_t first =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride =
        static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = first; index < element_count; index += stride)
    {
        const int64_t difference =
            static_cast<int64_t>(actual[index]) - reference[index];
        const unsigned long long absolute_error =
            static_cast<unsigned long long>(
                difference < 0 ? -difference : difference);
        atomicMax(&comparison->max_absolute_error, absolute_error);
        if (difference != 0)
        {
            atomicAdd(&comparison->mismatch_count, 1ULL);
        }
    }
}

DeviceComparison compare_outputs(
    const int32_t *actual,
    const int32_t *reference,
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

template <typename T>
T copy_device_value(const T *data, size_t index)
{
    T value{};
    CUDA_CHECK(cudaMemcpy(
        &value,
        data + index,
        sizeof(T),
        cudaMemcpyDeviceToHost));
    return value;
}

int32_t cpu_reference_value(int row, int column, int k)
{
    int32_t result = 0;
    for (int reduction = 0; reduction < k; ++reduction)
    {
        const size_t index_a =
            static_cast<size_t>(row) * k + reduction;
        const size_t index_b =
            static_cast<size_t>(column) * k + reduction;
        const int8_t quantized_a = quantize_input(
            make_input_value(index_a, kSeedA),
            kScaleA);
        const int8_t quantized_b = quantize_input(
            make_input_value(index_b, kSeedB),
            kScaleB);
        result += static_cast<int32_t>(quantized_a) *
            static_cast<int32_t>(quantized_b);
    }
    return result;
}

bool verify_cpu_samples(
    const int32_t *identity_output,
    const int32_t *block_swizzle_output,
    const int32_t *cublas_output,
    const float *dequantized_output,
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
              << std::right << std::setw(16) << "CPU INT32"
              << std::setw(20) << kIdentityName
              << std::setw(26) << kBlockSwizzleName
              << std::setw(22) << kCublasName
              << std::setw(20) << "Dequant FP32" << '\n';

    bool passed = true;
    for (const auto &sample : samples)
    {
        const int row = sample.first;
        const int column = sample.second;
        const size_t index = static_cast<size_t>(row) * n + column;
        const int32_t identity_value =
            copy_device_value(identity_output, index);
        const int32_t block_swizzle_value =
            copy_device_value(block_swizzle_output, index);
        const int32_t cublas_value = copy_device_value(cublas_output, index);
        const float dequantized_value =
            copy_device_value(dequantized_output, index);
        const int32_t reference = cpu_reference_value(row, column, k);
        const float expected_dequantized =
            static_cast<float>(reference) * kDequantScale;
        passed = passed &&
            identity_value == reference &&
            block_swizzle_value == reference &&
            cublas_value == reference &&
            dequantized_value == expected_dequantized;

        const std::string coordinate =
            "(" + std::to_string(row) + "," +
            std::to_string(column) + ")";
        std::cout << "  " << std::left << std::setw(16) << coordinate
                  << std::right << std::setw(20) << reference
                  << std::setw(20) << identity_value
                  << std::setw(26) << block_swizzle_value
                  << std::setw(22) << cublas_value
                  << std::setw(20) << dequantized_value << '\n';
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

double calculate_tops(int m, int n, int k, double milliseconds)
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
                "INT8 Tensor Core GEMM requires compute capability 8.0+");
        }

        const size_t count_a =
            static_cast<size_t>(options.m) * options.k;
        const size_t count_b =
            static_cast<size_t>(options.n) * options.k;
        const size_t count_c =
            static_cast<size_t>(options.m) * options.n;
        const size_t required_bytes =
            (count_a + count_b) * sizeof(int8_t) +
            3 * count_c * sizeof(int32_t) +
            count_c * sizeof(float) +
            32ULL * 1024ULL * 1024ULL;

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

        std::cout << "CuTe INT8 input / INT32 accumulate GEMM "
                  << "quantization and Block Swizzle test vs cuBLASLt\n\n";
        std::cout << "[Configuration]\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "GPU" << properties.name << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "Compute capability"
                  << properties.major << '.' << properties.minor << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "A layout" << "INT8 row-major [M,K] = ["
                  << options.m << ',' << options.k << "]\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "B layout" << "INT8 row-major [N,K] = ["
                  << options.n << ',' << options.k << "]\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "C operation" << "INT32 C[M,N] = A * B^T\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "Quantization"
                  << "q = clamp(round(x / scale), -127, 127), zp=0\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "Scale A / B" << kScaleA << " / " << kScaleB << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "Quantized amax target" << kQuantizedAbsoluteTarget
                  << " (INT32 accumulation headroom)\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "Dequantization"
                  << "FP32(C_int32) * scale_A * scale_B\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "CTA tile" << kBlockM << 'x' << kBlockN
                  << 'x' << kBlockK << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "Thread block" << "128 threads\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "SMEM stage buffers" << kPipelineStages
                  << " (ring pipeline)\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "Pipeline schedule"
                  << "prefetch 2; steady wait_group<1>; tail wait_group<0>\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "G2S" << "16B cp.async per instruction\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "SMEM layout" << "Swizzle<3,4,3> (128B period)\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "S2R" << "CuTe MMA-fragment vectorized copy\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "MMA" << "m16n8k32 INT8 inputs / INT32 accumulate\n";
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
        CublasLtGemm cublas(options.m, options.n, options.k);

        std::cout << "[Setup]\n";
        DeviceBuffer<int8_t> device_a(count_a);
        DeviceBuffer<int8_t> device_b(count_b);
        DeviceBuffer<int32_t> device_identity_c(count_c);
        DeviceBuffer<int32_t> device_block_swizzle_c(count_c);
        DeviceBuffer<int32_t> device_cublas_c(count_c);
        DeviceBuffer<float> device_dequantized_c(count_c);
        const int utility_block_count =
            std::max(1, properties.multiProcessorCount * 8);
        initialize_input_kernel<<<
            utility_block_count,
            256,
            0,
            stream.get()>>>(
                device_a.get(),
                device_a.count(),
                kSeedA,
                kScaleA);
        initialize_input_kernel<<<
            utility_block_count,
            256,
            0,
            stream.get()>>>(
                device_b.get(),
                device_b.count(),
                kSeedB,
                kScaleB);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));
        std::cout << "  Deterministic FP32 source values quantized to INT8 "
                  << "on GPU\n\n";

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
        std::cout << std::fixed << std::setprecision(7);
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
        cublas.launch(
            device_a.get(),
            device_b.get(),
            device_cublas_c.get(),
            stream.get());
        dequantize_output_kernel<<<
            utility_block_count,
            256,
            0,
            stream.get()>>>(
                device_identity_c.get(),
                device_dequantized_c.get(),
                count_c,
                kDequantScale);
        CUDA_CHECK(cudaGetLastError());
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
            stream.get());

        std::cout << "  " << std::left << std::setw(28) << "Comparison"
                  << std::right << std::setw(16) << "Mismatches"
                  << std::setw(22) << "Max INT32 abs error" << '\n';
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
                      << std::setw(22)
                      << comparison.second.max_absolute_error
                      << '\n';
            correctness_passed = correctness_passed &&
                comparison.second.mismatch_count == 0;
        }
        std::cout << "  " << std::left << std::setw(28)
                  << "Swizzle8 vs Identity"
                  << std::right << std::setw(16)
                  << block_swizzle_vs_identity.mismatch_count
                  << std::setw(22)
                  << block_swizzle_vs_identity.max_absolute_error
                  << '\n' << '\n';
        correctness_passed = correctness_passed &&
            block_swizzle_vs_identity.mismatch_count == 0;

        std::cout << "[CPU INT32 and FP32 dequantization samples]\n";
        std::cout << std::fixed << std::setprecision(7);
        const bool cpu_samples_passed = verify_cpu_samples(
            device_identity_c.get(),
            device_block_swizzle_c.get(),
            device_cublas_c.get(),
            device_dequantized_c.get(),
            options.m,
            options.n,
            options.k);
        std::cout << '\n';
        if (!correctness_passed || !cpu_samples_passed)
        {
            throw std::runtime_error(
                "INT8 input / INT32 accumulate GEMM correctness failed");
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
                    cublas.launch(
                        device_a.get(),
                        device_b.get(),
                        device_cublas_c.get(),
                        stream.get());
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
                  << std::setw(16) << "TOP/s"
                  << std::setw(18) << "vs cuBLAS" << '\n';
        const double cublas_tops = calculate_tops(
            options.m,
            options.n,
            options.k,
            benchmark_milliseconds[2]);
        for (size_t index = 0; index < benchmark_cases.size(); ++index)
        {
            const double tops = calculate_tops(
                options.m,
                options.n,
                options.k,
                benchmark_milliseconds[index]);
            std::cout << "  " << std::left << std::setw(28)
                      << benchmark_cases[index].name
                      << std::right << std::setw(14)
                      << benchmark_milliseconds[index]
                      << std::setw(16) << tops
                      << std::setw(17) << tops / cublas_tops * 100.0
                      << "%\n";
        }

        const double block_swizzle_speedup =
            benchmark_milliseconds[0] / benchmark_milliseconds[1];
        std::cout << "\n[Key result]\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "Block Swizzle speedup"
                  << std::fixed << std::setprecision(3)
                  << block_swizzle_speedup << "x vs Identity\n\n";
        std::cout << "[SUCCESS] INT8 quantization, INT32 accumulation, "
                  << "FP32 dequantization, Swizzle<3,4,3>, Block Swizzle8 "
                  << "and cuBLASLt "
                  << "validation passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cout << "[FAILED] " << error.what() << '\n';
        return 1;
    }
}
