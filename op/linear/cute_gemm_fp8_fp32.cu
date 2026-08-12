/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

// CuTe FP8 E4M3 GEMM correctness and performance validation:
//   FP32 source -> tensor-wide symmetric E4M3 quantization
//   E4M3 x E4M3 -> FP32 Tensor Core accumulation
//   FP32 accumulator -> scale_A * scale_B dequantization -> FP16 output
// The cuBLASLt reference consumes the same quantized FP8 buffers, performs
// FP32 accumulation, applies the same dequantization scale, and writes FP16.

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

#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cute/algorithm/cooperative_gemm.hpp>
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
constexpr int kBlockK = 128;
constexpr int kPipelineStages = 3;

constexpr const char *kCuteName = "CuTe FP8/FP32/FP16";
constexpr const char *kCublasName = "cuBLASLt FP8/FP32/FP16";

// Deterministic source values lie in [-0.5, 0.5). The tensor-wide scale maps
// that range near [-16, 16], well inside E4M3's finite range. Quantization is
// q = E4M3(x / scale), while dequantization after MMA is
// C = FP16(acc_fp32 * scale_A * scale_B).
constexpr float kInputAbsoluteMaximum = 0.5F;
constexpr float kQuantizedAbsoluteTarget = 16.0F;
constexpr float kScaleA =
    kInputAbsoluteMaximum / kQuantizedAbsoluteTarget;
constexpr float kScaleB =
    kInputAbsoluteMaximum / kQuantizedAbsoluteTarget;
constexpr float kDequantScale = kScaleA * kScaleB;

// Different legal FP32 Tensor Core reduction orders can differ slightly before
// the final FP16 conversion. These tolerances are intentionally tighter than a
// typical FP8 inference tolerance and primarily cover one FP16 output ULP.
constexpr float kAbsoluteTolerance = 2.0e-2F;
constexpr float kRelativeTolerance = 2.0e-3F;
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
        "cuBLASLt error: status=" + std::to_string(status) +
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

class CublasLtFp8Reference
{
public:
    CublasLtFp8Reference(int m, int n, int k)
        : workspace_(kWorkspaceBytes)
    {
        CUBLAS_CHECK(cublasLtCreate(&handle_));
        CUBLAS_CHECK(cublasLtMatmulDescCreate(
            &operation_,
            CUBLAS_COMPUTE_32F,
            CUDA_R_32F));

        // Row-major C[M,N] is column-major C^T[N,M]. The reference therefore
        // computes C^T = B * A^T, consuming the exact FP8 A/B buffers used by
        // the CuTe kernel.
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
            CUDA_R_8F_E4M3,
            k,
            n,
            k));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
            &layout_b_,
            CUDA_R_8F_E4M3,
            k,
            m,
            k));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
            &layout_c_,
            CUDA_R_16F,
            n,
            m,
            n));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
            &layout_d_,
            CUDA_R_16F,
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
                "cuBLASLt found no FP8/FP32/FP16 GEMM algorithm");
        }
        algorithm_ = results[0].algo;
    }

    ~CublasLtFp8Reference()
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

    CublasLtFp8Reference(const CublasLtFp8Reference &) = delete;
    CublasLtFp8Reference &operator=(const CublasLtFp8Reference &) = delete;

    void launch(
        const cute::float_e4m3_t *a,
        const cute::float_e4m3_t *b,
        __half *c,
        cudaStream_t stream)
    {
        // alpha performs dequantization after FP32 accumulation. beta=0 makes
        // the input C value irrelevant; D is converted to FP16 by cuBLASLt.
        constexpr float alpha = kDequantScale;
        constexpr float beta = 0.0F;
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
            "M/N/K must be exact multiples of the 128x128x128 CTA tile");
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

__global__ void initialize_input_kernel(
    cute::float_e4m3_t *data,
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
        // Tensor-wide symmetric quantization. The deterministic generator has
        // a known amax, so this test does not need a separate reduction kernel.
        data[index] = cute::float_e4m3_t(
            make_input_value(index, seed) / scale);
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
void cute_gemm_fp8_fp32_kernel(
    ProblemShape problem_shape,
    CtaTiler cta_tiler,
    const cute::float_e4m3_t *a,
    AStride stride_a,
    ASmemLayout smem_layout_a,
    TiledCopyA copy_a,
    const cute::float_e4m3_t *b,
    BStride stride_b,
    BSmemLayout smem_layout_b,
    TiledCopyB copy_b,
    __half *c,
    CStride stride_c,
    TiledMma tiled_mma,
    float dequant_scale)
{
    using namespace cute;

    CUTE_STATIC_ASSERT_V(rank(problem_shape) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size(copy_a) == size(tiled_mma));
    CUTE_STATIC_ASSERT_V(size(copy_b) == size(tiled_mma));

    // GMEM views are row-major A[M,K], B[N,K], and C[M,N]. The operation is
    // C = A * B^T, matching the common inference weight layout [N,K].
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

    extern __shared__ char shared_memory[];
    using Storage = SharedStorage<
        float_e4m3_t,
        float_e4m3_t,
        ASmemLayout,
        BSmemLayout>;
    Storage &storage = *reinterpret_cast<Storage *>(shared_memory);
    Tensor shared_a = make_tensor(
        make_smem_ptr(storage.a.begin()),
        smem_layout_a);
    Tensor shared_b = make_tensor(
        make_smem_ptr(storage.b.begin()),
        smem_layout_b);

    // G2S: each copy instruction moves 16 contiguous FP8 elements (16 bytes)
    // with cp.async. partition_S/D assigns the complete CTA tiles to 128 threads.
    ThrCopy thread_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor thread_global_a = thread_copy_a.partition_S(block_a);
    Tensor thread_shared_a = thread_copy_a.partition_D(shared_a);
    ThrCopy thread_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor thread_global_b = thread_copy_b.partition_S(block_b);
    Tensor thread_shared_b = thread_copy_b.partition_D(shared_b);

    // This atom is the defining property of this validation: E4M3 operands and
    // FP32 C/D registers for mma.sync.aligned.m16n8k32. The output tensor type
    // does not change the accumulator fragment type selected by the MMA atom.
    ThrMMA thread_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor thread_global_c = thread_mma.partition_C(block_c);
    Tensor accumulator = thread_mma.make_fragment_C(thread_global_c);
    clear(accumulator);

    // Three-stage ring pipeline prologue. Two cp.async groups are submitted
    // before MMA begins, leaving one stage available for the next K tile.
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
            cp_async_wait<0>();
        }
        --outstanding_groups;
        __syncthreads();

        // G2S for the next K tile overlaps the S2R plus MMA work of the current
        // read stage. write_stage cannot alias read_stage in this ring schedule.
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

        // S2R and Tensor Core MMA: CuTe partitions swizzled SMEM into the
        // fragments required by the m16n8k32 FP8/FP32 instruction.
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

    // Epilogue: the FP32 accumulator is still in the quantized domain. Apply
    // scale_A * scale_B in FP32, then round once to the requested FP16 output.
    CUTE_UNROLL
    for (int index = 0; index < size(accumulator); ++index)
    {
        thread_global_c(index) = __float2half_rn(
            static_cast<float>(accumulator(index)) * dequant_scale);
    }
}

void launch_cute_gemm(
    const cute::float_e4m3_t *a,
    const cute::float_e4m3_t *b,
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

    // FP8 rows are 128 bytes wide per K tile. Swizzle<3,4,3> preserves each
    // 16-byte vector and XORs higher address bits into SMEM bank bits over a
    // 128-byte period, matching both cp.async and MMA fragment access.
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

    // 128 threads each move a 16-byte vector. The logical copy covers a
    // [16,128] region and repeats along M/N to fill the 128x128 input tile.
    const auto copy_a = make_tiled_copy(
        Copy_Atom<
            SM80_CP_ASYNC_CACHEALWAYS<uint128_t>,
            float_e4m3_t>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _16>>{});
    const auto copy_b = make_tiled_copy(
        Copy_Atom<
            SM80_CP_ASYNC_CACHEALWAYS<uint128_t>,
            float_e4m3_t>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _16>>{});

    // Four warps arrange 2x2 m16n8k32 atoms. The atom explicitly emits
    // mma.sync ... f32.e4m3.e4m3.f32, so all K tiles accumulate into FP32.
    const auto tiled_mma = make_tiled_mma(
        SM89_16x8x32_F32E4M3E4M3F32_TN{},
        Layout<Shape<_2, _2>>{},
        Tile<_32, _32, _32>{});

    using Storage = SharedStorage<
        float_e4m3_t,
        float_e4m3_t,
        decltype(smem_layout_a),
        decltype(smem_layout_b)>;
    const int shared_memory_bytes = static_cast<int>(sizeof(Storage));
    const dim3 block(size(tiled_mma));
    const dim3 grid(m / kBlockM, n / kBlockN);

    const auto kernel = cute_gemm_fp8_fp32_kernel<
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
        tiled_mma,
        kDequantScale);
}

struct DeviceComparison
{
    unsigned long long mismatch_count;
    unsigned int max_absolute_error_bits;
    unsigned int max_relative_error_bits;
};

__global__ void compare_outputs_kernel(
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

double cpu_reference_value(int row, int column, int k)
{
    double quantized_dot = 0.0;
    for (int reduction = 0; reduction < k; ++reduction)
    {
        const size_t index_a =
            static_cast<size_t>(row) * k + reduction;
        const size_t index_b =
            static_cast<size_t>(column) * k + reduction;
        const cute::float_e4m3_t quantized_a(
            make_input_value(index_a, kSeedA) / kScaleA);
        const cute::float_e4m3_t quantized_b(
            make_input_value(index_b, kSeedB) / kScaleB);
        quantized_dot +=
            static_cast<double>(static_cast<float>(quantized_a)) *
            static_cast<double>(static_cast<float>(quantized_b));
    }
    return quantized_dot * static_cast<double>(kDequantScale);
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
              << std::right << std::setw(22) << "CPU quantized FP64"
              << std::setw(24) << kCuteName
              << std::setw(28) << kCublasName << '\n';

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
                  << std::right << std::setw(22) << reference
                  << std::setw(24) << cute_value
                  << std::setw(28) << cublas_value << '\n';
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
        if (properties.major * 10 + properties.minor < 89)
        {
            throw std::runtime_error(
                "FP8 Tensor Core GEMM requires compute capability 8.9+");
        }

        const size_t count_a =
            static_cast<size_t>(options.m) * options.k;
        const size_t count_b =
            static_cast<size_t>(options.n) * options.k;
        const size_t count_c =
            static_cast<size_t>(options.m) * options.n;
        const size_t required_bytes =
            (count_a + count_b) * sizeof(cute::float_e4m3_t) +
            2 * count_c * sizeof(__half) +
            32ULL * 1024ULL * 1024ULL;

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

        std::cout << "CuTe FP8 E4M3 input / FP32 accumulate / FP16 output "
                  << "GEMM test vs cuBLASLt\n\n";
        std::cout << "[Configuration]\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "GPU" << properties.name << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "Compute capability"
                  << properties.major << '.' << properties.minor << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "A layout" << "FP8 E4M3 row-major [M,K] = ["
                  << options.m << ',' << options.k << "]\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "B layout" << "FP8 E4M3 row-major [N,K] = ["
                  << options.n << ',' << options.k << "]\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "C operation" << "FP16 C[M,N] = A * B^T\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "Quantization" << "q = E4M3(x / tensor_scale)\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "Scale A / B" << kScaleA << " / " << kScaleB << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "Quantized amax target" << kQuantizedAbsoluteTarget
                  << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "Accumulator" << "FP32 Tensor Core registers\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "Epilogue"
                  << "FP16(acc_fp32 * scale_A * scale_B)\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "CTA tile" << kBlockM << 'x' << kBlockN
                  << 'x' << kBlockK << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "Thread block" << "128 threads\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "SMEM stage buffers" << kPipelineStages
                  << " (ring pipeline)\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "G2S" << "16B cp.async per instruction\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "SMEM layout" << "Swizzle<3,4,3> (128B period)\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "MMA" << "m16n8k32 E4M3 inputs / FP32 accumulate\n";
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
                "insufficient free device memory for FP8 inputs and FP16 outputs");
        }

        CudaStream stream;
        CublasLtFp8Reference cublas(options.m, options.n, options.k);
        DeviceBuffer<cute::float_e4m3_t> device_a(count_a);
        DeviceBuffer<cute::float_e4m3_t> device_b(count_b);
        DeviceBuffer<__half> device_cute_c(count_c);
        DeviceBuffer<__half> device_cublas_c(count_c);
        const int utility_block_count =
            std::max(1, properties.multiProcessorCount * 8);

        std::cout << "[Stage 1: Quantization]\n";
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
        std::cout << "  Deterministic FP32 source values quantized to E4M3 "
                  << "on GPU\n\n";

        std::cout << "[Stage 2: Tensor Core GEMM]\n";
        launch_cute_gemm(
            device_a.get(),
            device_b.get(),
            device_cute_c.get(),
            options.m,
            options.n,
            options.k,
            stream.get());
        CUDA_CHECK(cudaGetLastError());
        cublas.launch(
            device_a.get(),
            device_b.get(),
            device_cublas_c.get(),
            stream.get());
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));
        std::cout << "  CuTe and cuBLASLt completed with FP32 accumulation "
                  << "and FP16 output\n\n";

        std::cout << "[Stage 3: Correctness]\n";
        const DeviceComparison comparison = compare_outputs(
            device_cute_c.get(),
            device_cublas_c.get(),
            count_c,
            utility_block_count,
            stream.get());
        std::cout << std::fixed << std::setprecision(7);
        std::cout << "  " << std::left << std::setw(30)
                  << "CuTe vs cuBLASLt mismatches"
                  << comparison.mismatch_count << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "Maximum absolute error"
                  << decode_float_bits(comparison.max_absolute_error_bits)
                  << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "Maximum relative error"
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
                "FP8/FP32/FP16 GEMM correctness validation failed");
        }

        std::cout << "[Stage 4: Benchmark]\n";
        std::cout << "  Method: rotating order; median of "
                  << options.benchmark_iterations << " samples\n\n";
        const std::array<BenchmarkCase, 2> benchmark_cases = {{
            {
                kCuteName,
                [&]()
                {
                    launch_cute_gemm(
                        device_a.get(),
                        device_b.get(),
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
                        device_a.get(),
                        device_b.get(),
                        device_cublas_c.get(),
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

        std::cout << "  " << std::left << std::setw(30)
                  << "Implementation"
                  << std::right << std::setw(14) << "Median ms"
                  << std::setw(16) << "TFLOP/s"
                  << std::setw(18) << "vs cuBLASLt" << '\n';
        const double cublas_tflops = calculate_tflops(
            options.m,
            options.n,
            options.k,
            benchmark_milliseconds[1]);
        for (size_t index = 0; index < benchmark_cases.size(); ++index)
        {
            const double tflops = calculate_tflops(
                options.m,
                options.n,
                options.k,
                benchmark_milliseconds[index]);
            std::cout << "  " << std::left << std::setw(30)
                      << benchmark_cases[index].name
                      << std::right << std::setw(14)
                      << benchmark_milliseconds[index]
                      << std::setw(16) << tflops
                      << std::setw(17) << tflops / cublas_tflops * 100.0
                      << "%\n";
        }

        std::cout << "\n[Key result]\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "CuTe/cuBLASLt throughput"
                  << calculate_tflops(
                         options.m,
                         options.n,
                         options.k,
                         benchmark_milliseconds[0]) /
                         cublas_tflops * 100.0
                  << "%\n\n";
        std::cout << "[SUCCESS] FP8 E4M3 quantization, FP32 Tensor Core "
                  << "accumulation, FP16 epilogue, CPU samples, and "
                  << "cuBLASLt validation passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cout << "[FAILED] " << error.what() << '\n';
        return 1;
    }
}
