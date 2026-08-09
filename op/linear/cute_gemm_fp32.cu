/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************************************/

// CuTe GEMM 主循环基于 CUTLASS examples/cute/tutorial/sgemm_sm80.cu 整理，
// 并针对 row-major A[M,K]、B[N,K]、C[M,N] 接口及验证流程进行了改写。

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>

namespace
{

constexpr int kDefaultM = 8192;
constexpr int kDefaultN = 8192;
constexpr int kDefaultK = 8192;
constexpr int kDefaultWarmupIterations = 2;
constexpr int kDefaultBenchmarkIterations = 5;

constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockK = 8;
constexpr int kPipelineStages = 3;
constexpr const char *kTmaGemmName = "TMA-SW32";

constexpr float kAbsoluteTolerance = 5.0e-2F;
constexpr float kRelativeTolerance = 2.0e-3F;
constexpr uint32_t kSeedA = 0x1234abcdU;
constexpr uint32_t kSeedB = 0x9e3779b9U;

enum class SharedLayoutMode
{
    kPadding129,
    kSwizzle223,
    kSwizzle323,
    kVector16Swizzle123
};

template <SharedLayoutMode Mode>
constexpr const char *shared_layout_name()
{
    if constexpr (Mode == SharedLayoutMode::kPadding129)
    {
        return "Padding-129";
    }
    else if constexpr (Mode == SharedLayoutMode::kSwizzle223)
    {
        return "Swizzle<2,2,3>";
    }
    else if constexpr (Mode == SharedLayoutMode::kSwizzle323)
    {
        return "Swizzle<3,2,3>";
    }
    else
    {
        return "Vec16-Swizzle<1,2,3>";
    }
}

template <SharedLayoutMode Mode>
constexpr auto make_smem_layout_atom()
{
    using namespace cute;

    // CuTe 的 Swizzle 参数作用于“元素 offset”，不是字节地址。FP32 的
    // MBase=2 保留 4 个元素，恰好对应一段 16B；这与 FP16 常用的
    // MBase=3（8 个 half，同样是 16B）在字节粒度上等价。
    if constexpr (Mode == SharedLayoutMode::kPadding129)
    {
        // Padding 基线：K 每前进一列，物理地址跨过 129 个 float。
        return make_layout(
            make_shape(Int<kBlockM>{}, Int<kBlockK>{}),
            make_stride(Int<1>{}, Int<kBlockM + 1>{}));
    }
    else if constexpr (Mode == SharedLayoutMode::kSwizzle223)
    {
        // Swizzle<2,2,3>：16x8 atom 共 128 个 float。对当前固定 M、
        // K=0..7 的 8-lane 搬运，高位 K bit XOR 到 bank bit [2,4)，
        // 未参与 XOR 的最低 K bit 自身位于 bank bit 4，八个 bank 均不同。
        return composition(
            Swizzle<2, 2, 3>{},
            Layout<
                Shape<_16, _8>,
                Stride<_1, _16>>{});
    }
    else if constexpr (Mode == SharedLayoutMode::kSwizzle323)
    {
        // Swizzle<3,2,3>：32x8 atom 共 256 个 float。K 的三个 bit
        // 全部 XOR 到 FP32 bank bit [2,5)，使同一 8-lane 子组访问
        // 八个不同 bank；该配置也与 CUTLASS 的 32-bit SM80 atom 一致。
        return composition(
            Swizzle<3, 2, 3>{},
            Layout<
                Shape<_32, _8>,
                Stride<_1, _32>>{});
    }
    else
    {
        // 16B G2S 布局：K 被拆成 (K-inner=4, K-outer=2)，使每组
        // 4 个 FP32 在 GMEM 和 SMEM 中都物理连续。M 每前进一行跨过
        // 4 个 float，Swizzle<1,2,3> 再用 K-outer bit XOR bank bit 2。
        // 该 nested atom 来自 CUTLASS Ampere FP32 128-bit copy 的布局形式。
        return composition(
            Swizzle<1, 2, 3>{},
            Layout<
                Shape<_8, Shape<_4, _2>>,
                Stride<_4, Stride<_1, _32>>>{});
    }
}

template <SharedLayoutMode Mode>
constexpr auto make_gmem_to_smem_copy()
{
    using namespace cute;

    if constexpr (Mode == SharedLayoutMode::kVector16Swizzle123)
    {
        // 256 个线程覆盖 [M=128,K-group=2]，每线程沿 K 搬运 4 个
        // 连续 FP32。uint128_t 使一条 cp.async 的传输宽度成为 16B。
        return make_tiled_copy(
            Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, float>{},
            Layout<Shape<_128, _2>, Stride<_2, _1>>{},
            Layout<Shape<_1, _4>>{});
    }
    else
    {
        // 标量对照：每个 Copy_Atom 发出一条 4B cp.async。
        return make_tiled_copy(
            Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<float>, float>{},
            Layout<Shape<_32, _8>, Stride<_8, _1>>{},
            Layout<Shape<_1, _1>>{});
    }
}

template <SharedLayoutMode Mode>
constexpr auto make_fp32_tiled_mma()
{
    using namespace cute;

    if constexpr (Mode == SharedLayoutMode::kVector16Swizzle123)
    {
        // Vec16 的 SMEM 每行占用 4-bank 对齐的向量，bank group 只有 8 个。
        // 令一个 warp 仅覆盖 8 个 M 线程、其余 lane 沿 N 展开，使 A/B
        // 的重复地址形成合法 broadcast，避免 16x16 映射中的跨行冲突。
        return make_tiled_mma(
            UniversalFMA<float, float, float>{},
            Layout<Shape<_8, _32, _1>>{});
    }
    else
    {
        return make_tiled_mma(
            UniversalFMA<float, float, float>{},
            Layout<Shape<_16, _16, _1>>{});
    }
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
            "M/N/K must be exact multiples of the 128x128x8 CTA tile");
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
    float *data,
    size_t element_count,
    uint32_t seed)
{
    const size_t first =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride =
        static_cast<size_t>(gridDim.x) * blockDim.x;

    for (size_t index = first; index < element_count; index += stride)
    {
        data[index] = make_input_value(index, seed);
    }
}

// CuTe 用静态 Layout 描述每一级流水的共享内存形状。A、B 各自包含
// kPipelineStages 个 K tile，供 cp.async 和计算并行推进。
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

// TMA 目标地址必须满足硬件 swizzle 的对齐要求。A/B 分别按 128B 对齐，
// 每个 pipeline stage 配一个 64-bit mbarrier，用于跟踪 A/B 两笔 TMA
// 是否完成；普通 FMA consumer 使用 CTA barrier 保证 stage 可安全复用。
template <
    class ElementA,
    class ElementB,
    class SmemLayoutA,
    class SmemLayoutB>
struct TmaSharedStorage
{
    alignas(128) cute::ArrayEngine<
        ElementA,
        cute::cosize_v<SmemLayoutA>> a;
    alignas(128) cute::ArrayEngine<
        ElementB,
        cute::cosize_v<SmemLayoutB>> b;
    alignas(16) uint64_t tma_barrier[kPipelineStages];
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
__global__ __launch_bounds__(decltype(size(TiledMma{}))::value)
void cute_gemm_kernel(
    ProblemShape problem_shape,
    CtaTiler cta_tiler,
    const float *a,
    AStride stride_a,
    ASmemLayout smem_layout_a,
    TiledCopyA copy_a,
    S2RAtomA smem_to_register_a,
    const float *b,
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

    // make_tensor 只创建轻量级的“指针 + shape + stride”视图，不搬运数据。
    // A[M,K]、B[N,K]、C[M,N] 都是 row-major，所以最后一维 stride 为 1。
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

    // local_tile 根据 blockIdx 取出本 CTA 负责的逻辑 tile：
    // A[128,8,k_tile]、B[128,8,k_tile] 和 C[128,128]。
    const auto cta_coordinate =
        make_coord(blockIdx.x, blockIdx.y, _);
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
        float,
        float,
        ASmemLayout,
        BSmemLayout>;
    Storage &storage = *reinterpret_cast<Storage *>(shared_memory);

    // 共享内存张量包含 PIPE 维，表示三段流水中的三个独立缓冲区。
    Tensor shared_a = make_tensor(
        make_smem_ptr(storage.a.begin()),
        smem_layout_a);
    Tensor shared_b = make_tensor(
        make_smem_ptr(storage.b.begin()),
        smem_layout_b);

    // TiledCopy 把一个 tile 的搬运工作分配给 256 个线程。由于 A/B 在
    // global memory 中 K 连续，线程布局让相邻线程沿 K 维读取连续地址。
    ThrCopy thread_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor thread_global_a = thread_copy_a.partition_S(block_a);
    Tensor thread_shared_a = thread_copy_a.partition_D(shared_a);

    ThrCopy thread_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor thread_global_b = thread_copy_b.partition_S(block_b);
    Tensor thread_shared_b = thread_copy_b.partition_D(shared_b);

    const int pipe_count = size<3>(thread_shared_a);
    int remaining_k_tiles = size<3>(thread_global_a);
    int next_k_tile = 0;

    // 预填前两个 pipe。下面的 cute::copy 使用 SM80 cp.async Copy_Atom，
    // 直接完成 GMEM -> SMEM 异步搬运，不占用中间数据寄存器。
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

        // cp_async_fence 提交当前线程发出的异步 copy group。
        cp_async_fence();
        --remaining_k_tiles;
        if (remaining_k_tiles > 0)
        {
            ++next_k_tile;
        }
    }

    // TiledMMA 定义每个线程负责哪些 A/B 元素以及哪些 C 累加器。
    ThrMMA thread_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor thread_global_c = thread_mma.partition_C(block_c);

    // A/B fragment 位于寄存器，C fragment 是每线程持有的 FP32 累加器。
    Tensor register_a =
        thread_mma.partition_fragment_A(shared_a(_, _, 0));
    Tensor register_b =
        thread_mma.partition_fragment_B(shared_b(_, _, 0));
    Tensor accumulator = thread_mma.make_fragment_C(thread_global_c);
    clear(accumulator);

    // make_tiled_copy_A/B 根据 MMA 的线程映射重新组织 SMEM -> 寄存器搬运，
    // 保证每个线程拿到执行自己那部分 FMA 所需的 A/B fragment。
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

    // 等待第一个共享内存 tile 到达，然后把第一个 K fragment 从
    // SMEM -> 寄存器，作为寄存器级流水的预取数据。
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

    // 主循环同时推进三件事：
    //   1. GMEM -> SMEM：cp.async 写入 write_pipe；
    //   2. SMEM -> 寄存器：预取下一个 register K block；
    //   3. 寄存器 -> FP32 累加器：UniversalFMA 执行当前 K block。
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

                // 在切换 read pipe 前，等待相应 cp.async group 完成；CTA
                // 同步保证所有线程都能安全读取刚填充的共享内存 tile。
                cp_async_wait<kPipelineStages - 2>();
                __syncthreads();
            }

            const auto next_register_k_block =
                (k_block + Int<1>{}) % register_k_blocks;

            // 该 copy 完成下一小段 SMEM -> 寄存器搬运，与当前 FMA 交叠。
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
                // 该 copy 完成下一整个 K tile 的 GMEM -> SMEM 搬运。
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

            // CuTe gemm 将当前 A/B 寄存器 fragment 展开为逐元素 FP32 FMA，
            // 结果持续累加在 accumulator 中，不在 K 循环内写回内存。
            gemm(
                tiled_mma,
                register_a(_, _, k_block),
                register_b(_, _, k_block),
                accumulator);
        }
    }

    // K 归约结束后，CuTe 按 partition_C 的坐标把每线程寄存器累加器写回
    // row-major C[M,N]。本测试固定 alpha=1、beta=0，因此无需读取旧 C。
    copy(accumulator, thread_global_c);
}

template <
    class ProblemShape,
    class CtaTiler,
    class TmaCopyA,
    class ASmemLayout,
    class TmaCopyB,
    class BSmemLayout,
    class CStride,
    class TiledMma>
__global__ __launch_bounds__(decltype(cute::size(TiledMma{}))::value)
void cute_tma_gemm_kernel(
    ProblemShape problem_shape,
    CtaTiler cta_tiler,
    CUTE_GRID_CONSTANT TmaCopyA const tma_copy_a,
    ASmemLayout smem_layout_a,
    CUTE_GRID_CONSTANT TmaCopyB const tma_copy_b,
    BSmemLayout smem_layout_b,
    float *c,
    CStride stride_c,
    TiledMma tiled_mma)
{
    using namespace cute;

    CUTE_STATIC_ASSERT_V(rank(problem_shape) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});
    static_assert(is_static<ASmemLayout>::value);
    static_assert(is_static<BSmemLayout>::value);

    // TMA descriptor 已在 host 端保存了 GMEM 基址、shape、byte stride、
    // box shape 与 SMEM swizzle。get_tma_tensor 在 device 端生成的是坐标
    // tensor；访问它得到 TMA 坐标，不会执行普通的逐元素 GMEM load。
    Tensor global_a = tma_copy_a.get_tma_tensor(
        select<0, 2>(problem_shape));
    Tensor global_b = tma_copy_b.get_tma_tensor(
        select<1, 2>(problem_shape));
    Tensor global_c = make_tensor(
        make_gmem_ptr(c),
        select<0, 1>(problem_shape),
        stride_c);

    const auto cta_coordinate =
        make_coord(blockIdx.x, blockIdx.y, _);
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
    using Storage = TmaSharedStorage<
        float,
        float,
        ASmemLayout,
        BSmemLayout>;
    Storage &storage = *reinterpret_cast<Storage *>(shared_memory);
    Tensor shared_a = make_tensor(
        make_smem_ptr(storage.a.begin()),
        smem_layout_a);
    Tensor shared_b = make_tensor(
        make_smem_ptr(storage.b.begin()),
        smem_layout_b);

    // 非 multicast 时 CTA rank 固定为 0。TMA slice 的 partition_S/D
    // 根据 descriptor 的 box 与 swizzle，把 GMEM/SMEM 都组织成
    // [TMA-values, rest]；group_modes 再把 rest 合并为 k_tile 或 PIPE。
    // 与 cp.async TiledCopy 不同，这里没有 256 份 thread partition；
    // 后续只需一个 elected thread 对整个 128x8 tile 发一次 copy。
    auto cta_tma_a = tma_copy_a.get_slice(Int<0>{});
    Tensor tma_global_a_partitioned = cta_tma_a.partition_S(block_a);
    Tensor tma_shared_a_partitioned = cta_tma_a.partition_D(shared_a);
    Tensor tma_global_a = group_modes<
        1,
        rank(tma_global_a_partitioned)>(tma_global_a_partitioned);
    Tensor tma_shared_a = group_modes<
        1,
        rank(tma_shared_a_partitioned)>(tma_shared_a_partitioned);

    auto cta_tma_b = tma_copy_b.get_slice(Int<0>{});
    Tensor tma_global_b_partitioned = cta_tma_b.partition_S(block_b);
    Tensor tma_shared_b_partitioned = cta_tma_b.partition_D(shared_b);
    Tensor tma_global_b = group_modes<
        1,
        rank(tma_global_b_partitioned)>(tma_global_b_partitioned);
    Tensor tma_shared_b = group_modes<
        1,
        rank(tma_shared_b_partitioned)>(tma_shared_b_partitioned);

    constexpr int tma_transaction_bytes =
        sizeof(make_tensor_like(tensor<0>(tma_shared_a))) +
        sizeof(make_tensor_like(tensor<0>(tma_shared_b)));
    const int k_tile_count = size<1>(tma_global_a);
    const int initial_stage_count =
        k_tile_count < kPipelineStages ?
        k_tile_count : kPipelineStages;

    using ProducerBarrier = cutlass::arch::ClusterTransactionBarrier;

    // producer mbarrier 的 init(1) 表示每轮只有一个软件 producer arrive。
    // 被选中的 producer 声明 transaction bytes；所有 consumer 等待该
    // barrier phase 翻转后，才从相应 stage 读取 A/B。
    if (threadIdx.x < kPipelineStages)
    {
        ProducerBarrier::init(
            &storage.tma_barrier[threadIdx.x],
            1);
    }
    // 将普通 shared proxy 中的 barrier init 发布给 TMA 使用的 async
    // proxy；随后 CTA 同步，保证 producer 发首批 TMA 前初始化已可见。
    cutlass::arch::fence_barrier_init();
    __syncthreads();

    // 先预填最多三个 stage。copy(tma.with(barrier), ...) 会生成
    // cp.async.bulk.tensor GMEM -> SMEM，而不是普通 LDG + STS。
    const bool elected_producer =
        (threadIdx.x / 32 == 0) && elect_one_sync();
    if (elected_producer)
    {
        for (int stage = 0;
             stage < initial_stage_count;
             ++stage)
        {
            // arrive.expect_tx 同时声明 A+B 的总字节数；只有两笔 TMA
            // 都完成并回减 transaction count 后，producer phase 才翻转。
            ProducerBarrier::arrive_and_expect_tx(
                &storage.tma_barrier[stage],
                tma_transaction_bytes);
            copy(
                tma_copy_a.with(storage.tma_barrier[stage]),
                tma_global_a(_, stage),
                tma_shared_a(_, stage));
            copy(
                tma_copy_b.with(storage.tma_barrier[stage]),
                tma_global_b(_, stage),
                tma_shared_b(_, stage));
        }
    }

    // TMA 路径的计算仍沿用 256-thread UniversalFMA；与 cp.async 路径
    // 不同的部分集中在 G2S 发射和 mbarrier 同步。下面建立相同类型的
    // SMEM -> register 映射以及每线程持有的 C accumulator。
    ThrMMA thread_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor thread_global_c = thread_mma.partition_C(block_c);
    Tensor register_a =
        thread_mma.partition_fragment_A(shared_a(_, _, Int<0>{}));
    Tensor register_b =
        thread_mma.partition_fragment_B(shared_b(_, _, Int<0>{}));
    Tensor accumulator = thread_mma.make_fragment_C(thread_global_c);
    clear(accumulator);

    const auto smem_to_register_a =
        Copy_Atom<AutoVectorizingCopy, float>{};
    const auto smem_to_register_b =
        Copy_Atom<AutoVectorizingCopy, float>{};
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

    const int register_k_blocks = size<2>(register_a);

    CUTE_NO_UNROLL
    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile)
    {
        const int stage = k_tile % kPipelineStages;
        // mbarrier.try_wait.parity 等待 phase 相对给定值发生翻转。同一个
        // stage 每绕流水线一圈复用一次，因此等待的旧 phase 也交替 0/1。
        const int phase = (k_tile / kPipelineStages) & 1;
        ProducerBarrier::wait(
            &storage.tma_barrier[stage],
            phase);

        Tensor current_smem_a = thread_smem_a(_, _, _, stage);
        Tensor current_smem_b = thread_smem_b(_, _, _, stage);
        CUTE_UNROLL
        for (int k_block = 0;
             k_block < register_k_blocks;
             ++k_block)
        {
            copy(
                smem_to_register_a,
                current_smem_a(_, _, k_block),
                thread_register_a(_, _, k_block));
            copy(
                smem_to_register_b,
                current_smem_b(_, _, k_block),
                thread_register_b(_, _, k_block));
            gemm(
                tiled_mma,
                register_a(_, _, k_block),
                register_b(_, _, k_block),
                accumulator);
        }

        // 当前计算不是异步 WGMMA，因此用 CTA barrier 等待 256 个普通
        // FMA consumer 全部读完，再让 TMA 覆写刚释放的 stage。
        __syncthreads();
        const int next_k_tile = k_tile + kPipelineStages;
        if (elected_producer && next_k_tile < k_tile_count)
        {
            ProducerBarrier::arrive_and_expect_tx(
                &storage.tma_barrier[stage],
                tma_transaction_bytes);
            copy(
                tma_copy_a.with(storage.tma_barrier[stage]),
                tma_global_a(_, next_k_tile),
                tma_shared_a(_, stage));
            copy(
                tma_copy_b.with(storage.tma_barrier[stage]),
                tma_global_b(_, next_k_tile),
                tma_shared_b(_, stage));
        }
    }

    copy(accumulator, thread_global_c);
}

template <SharedLayoutMode Mode>
void launch_cute_gemm(
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

    // A/B/C 都是 row-major：A(m,k) 与 B(n,k) 的 K stride 为 1，
    // C(m,n) 的 N stride 为 1。
    const auto stride_a = make_stride(k, Int<1>{});
    const auto stride_b = make_stride(k, Int<1>{});
    const auto stride_c = make_stride(n, Int<1>{});

    const auto cta_tiler = make_shape(
        Int<kBlockM>{},
        Int<kBlockN>{},
        Int<kBlockK>{});

    // Mode 在编译期选择 padding 或 XOR swizzle atom。A/B 的 CTA tile
    // 形状相同，所以共用相同 atom 类型，但拥有互相独立的共享内存区域。
    const auto smem_atom_a = make_smem_layout_atom<Mode>();
    const auto smem_atom_b = make_smem_layout_atom<Mode>();

    // tile_to_shape 将 atom 平铺到完整 128x8 tile，并增加第三维 PIPE。
    // CuTe 保留 composed swizzle，因此 copy 和 MMA partition 都使用逻辑
    // (m,k) 坐标，不需要手工计算 XOR 后的共享内存地址。
    const auto smem_layout_a = tile_to_shape(
        smem_atom_a,
        make_shape(
            Int<kBlockM>{},
            Int<kBlockK>{},
            Int<kPipelineStages>{}));
    const auto smem_layout_b = tile_to_shape(
        smem_atom_b,
        make_shape(
            Int<kBlockN>{},
            Int<kBlockK>{},
            Int<kPipelineStages>{}));

    // Mode 同时选择与 SMEM 物理连续方向匹配的 G2S TiledCopy：前三个
    // 布局使用 4B 标量 cp.async，nested K-contiguous 布局使用 16B。
    const auto copy_a = make_gmem_to_smem_copy<Mode>();
    const auto copy_b = make_gmem_to_smem_copy<Mode>();

    // UniversalFMA<float> 明确使用完整 FP32 CUDA Core FMA，而不是把输入
    // 截断为 TF32。两种线程布局都由 256 个单线程 FMA atom 组成；Vec16
    // 使用 8x32 以匹配其 K-contiguous SMEM 的 bank-group 数量。
    const auto tiled_mma = make_fp32_tiled_mma<Mode>();
    const auto smem_to_register_a =
        Copy_Atom<AutoVectorizingCopy, float>{};
    const auto smem_to_register_b =
        Copy_Atom<AutoVectorizingCopy, float>{};

    using Storage = SharedStorage<
        float,
        float,
        decltype(smem_layout_a),
        decltype(smem_layout_b)>;
    const int shared_memory_bytes = static_cast<int>(sizeof(Storage));

    const dim3 block(size(tiled_mma));
    const dim3 grid(
        size(ceil_div(m, Int<kBlockM>{})),
        size(ceil_div(n, Int<kBlockN>{})));

    cute_gemm_kernel<<<grid, block, shared_memory_bytes, stream>>>(
        problem_shape,
        cta_tiler,
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

template <
    class ProblemShape,
    class CtaTiler,
    class TmaCopyA,
    class ASmemLayout,
    class TmaCopyB,
    class BSmemLayout,
    class CStride,
    class TiledMma>
struct TmaGemmPlan
{
    ProblemShape problem_shape;
    CtaTiler cta_tiler;
    TmaCopyA tma_copy_a;
    ASmemLayout smem_layout_a;
    TmaCopyB tma_copy_b;
    BSmemLayout smem_layout_b;
    CStride stride_c;
    TiledMma tiled_mma;
};

auto make_cute_tma_gemm_plan(
    const float *a,
    const float *b,
    int m,
    int n,
    int k)
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

    // K=8 个 FP32 恰好是 32B，因此选用 TMA 可编码的 SW32 atom。
    // 其底层是以 16B 为 base 的 Swizzle<1,4,3>；CuTe 的 typed layout
    // 将 bit-level 物理布局还原为逻辑 [M,K] 坐标供后续 S2R 使用。
    const auto smem_layout_a = tile_to_shape(
        GMMA::Layout_K_SW32_Atom<float>{},
        make_shape(
            Int<kBlockM>{},
            Int<kBlockK>{},
            Int<kPipelineStages>{}));
    const auto smem_layout_b = tile_to_shape(
        GMMA::Layout_K_SW32_Atom<float>{},
        make_shape(
            Int<kBlockN>{},
            Int<kBlockK>{},
            Int<kPipelineStages>{}));

    // make_tma_copy 在 host 端调用 CUDA Driver API 编码 tensor map：
    //   1. 从 global tensor 提取 GMEM 基址、shape 与 byte stride；
    //   2. 从 stage-0 SMEM layout 推导 128x8 box 和 SW32 模式；
    //   3. 保存坐标到 GMEM 地址、逻辑值到 swizzled SMEM 地址的映射。
    // descriptor 随 kernel 参数传入，device 端每次只提供 tile 坐标与
    // mbarrier 地址，不再由 256 个线程分别计算和发出 load/store。
    Tensor global_a = make_tensor(
        make_gmem_ptr(a),
        select<0, 2>(problem_shape),
        stride_a);
    Tensor global_b = make_tensor(
        make_gmem_ptr(b),
        select<1, 2>(problem_shape),
        stride_b);
    const auto tma_copy_a = make_tma_copy(
        SM90_TMA_LOAD{},
        global_a,
        smem_layout_a(_, _, Int<0>{}),
        select<0, 2>(cta_tiler),
        Int<1>{});
    const auto tma_copy_b = make_tma_copy(
        SM90_TMA_LOAD{},
        global_b,
        smem_layout_b(_, _, Int<0>{}),
        select<1, 2>(cta_tiler),
        Int<1>{});

    // 保持 256 个 UniversalFMA<float> 线程以及与 Vec16 相同的 8x32
    // thread layout，使主要实现差异集中在 G2S 与相应的同步协议。
    const auto tiled_mma = make_tiled_mma(
        UniversalFMA<float, float, float>{},
        Layout<Shape<_8, _32, _1>>{});

    using Plan = TmaGemmPlan<
        decltype(problem_shape),
        decltype(cta_tiler),
        decltype(tma_copy_a),
        decltype(smem_layout_a),
        decltype(tma_copy_b),
        decltype(smem_layout_b),
        decltype(stride_c),
        decltype(tiled_mma)>;
    return Plan{
        problem_shape,
        cta_tiler,
        tma_copy_a,
        smem_layout_a,
        tma_copy_b,
        smem_layout_b,
        stride_c,
        tiled_mma};
}

template <
    class ProblemShape,
    class CtaTiler,
    class TmaCopyA,
    class ASmemLayout,
    class TmaCopyB,
    class BSmemLayout,
    class CStride,
    class TiledMma>
void launch_cute_tma_gemm(
    const TmaGemmPlan<
        ProblemShape,
        CtaTiler,
        TmaCopyA,
        ASmemLayout,
        TmaCopyB,
        BSmemLayout,
        CStride,
        TiledMma> &plan,
    float *c,
    cudaStream_t stream)
{
    using namespace cute;

    using Storage = TmaSharedStorage<
        float,
        float,
        ASmemLayout,
        BSmemLayout>;
    const int shared_memory_bytes = static_cast<int>(sizeof(Storage));
    const dim3 block(size(plan.tiled_mma));
    const dim3 grid(
        size(ceil_div(get<0>(plan.problem_shape), Int<kBlockM>{})),
        size(ceil_div(get<1>(plan.problem_shape), Int<kBlockN>{})));

    cute_tma_gemm_kernel<<<grid, block, shared_memory_bytes, stream>>>(
        plan.problem_shape,
        plan.cta_tiler,
        plan.tma_copy_a,
        plan.smem_layout_a,
        plan.tma_copy_b,
        plan.smem_layout_b,
        c,
        plan.stride_c,
        plan.tiled_mma);
}

void launch_cublas_gemm(
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

    // cuBLAS 使用 column-major。row-major C[M,N] 的同一段内存可解释为
    // column-major C^T[N,M]，因此计算 C^T = B * A^T：
    //   B row-major [N,K] -> column-major view [K,N]，使用 OP_T；
    //   A row-major [M,K] -> column-major view [K,M]，使用 OP_N。
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
        CUBLAS_COMPUTE_32F_PEDANTIC,
        CUBLAS_GEMM_DEFAULT));
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

        const float tolerance =
            absolute_tolerance +
            relative_tolerance * fabsf(reference_value);
        if (absolute_error > tolerance)
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

bool verify_cpu_samples(
    const float *padding_output,
    const float *swizzle_223_output,
    const float *swizzle_323_output,
    const float *vector_16_output,
    const float *tma_output,
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

    std::cout << "  " << std::left << std::setw(14) << "Coordinate"
              << std::right << std::setw(15) << "CPU FP64"
              << std::setw(15) << "Padding-129"
              << std::setw(15) << "Swizzle223"
              << std::setw(15) << "Swizzle323"
              << std::setw(18) << "Vec16Swizzle123"
              << std::setw(15) << kTmaGemmName
              << std::setw(15) << "cuBLAS FP32"
              << '\n';

    bool passed = true;
    for (const auto &sample : samples)
    {
        const int row = sample.first;
        const int column = sample.second;
        const size_t index = static_cast<size_t>(row) * n + column;

        const float padding_value =
            copy_device_value(padding_output, index);
        const float swizzle_223_value =
            copy_device_value(swizzle_223_output, index);
        const float swizzle_323_value =
            copy_device_value(swizzle_323_output, index);
        const float vector_16_value =
            copy_device_value(vector_16_output, index);
        const float tma_value =
            copy_device_value(tma_output, index);
        const float cublas_value =
            copy_device_value(cublas_output, index);

        const double reference = cpu_reference_value(row, column, k);
        const double padding_error =
            std::abs(static_cast<double>(padding_value) - reference);
        const double swizzle_223_error =
            std::abs(static_cast<double>(swizzle_223_value) - reference);
        const double swizzle_323_error =
            std::abs(static_cast<double>(swizzle_323_value) - reference);
        const double vector_16_error =
            std::abs(static_cast<double>(vector_16_value) - reference);
        const double tma_error =
            std::abs(static_cast<double>(tma_value) - reference);
        const double cublas_error =
            std::abs(static_cast<double>(cublas_value) - reference);
        const double tolerance =
            static_cast<double>(kAbsoluteTolerance) +
            static_cast<double>(kRelativeTolerance) * std::abs(reference);
        passed = passed &&
            padding_error <= tolerance &&
            swizzle_223_error <= tolerance &&
            swizzle_323_error <= tolerance &&
            vector_16_error <= tolerance &&
            tma_error <= tolerance &&
            cublas_error <= tolerance;

        const std::string coordinate =
            "(" + std::to_string(row) + "," +
            std::to_string(column) + ")";
        std::cout << "  " << std::left << std::setw(14) << coordinate
                  << std::right << std::setw(15) << std::setprecision(7)
                  << reference
                  << std::setw(15) << padding_value
                  << std::setw(15) << swizzle_223_value
                  << std::setw(15) << swizzle_323_value
                  << std::setw(18) << vector_16_value
                  << std::setw(15) << tma_value
                  << std::setw(15) << cublas_value
                  << '\n';
    }

    return passed;
}

struct ComparisonResult
{
    const char *name;
    DeviceComparison comparison;
};

struct BenchmarkResult
{
    const char *name;
    double milliseconds;
    double tflops;
};

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
    for (int iteration = 0;
         iteration < warmup_iterations;
         ++iteration)
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

    // 每轮轮换起始实现，避免固定的执行顺序让功耗、温度或动态频率系统性
    // 偏向某一种 shared layout。每次只计量一个 kernel，最终取中位数。
    for (int iteration = 0;
         iteration < benchmark_iterations;
         ++iteration)
    {
        for (int offset = 0;
             offset < case_count;
             ++offset)
        {
            const int index =
                (iteration + offset) % case_count;
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

float decode_float_bits(uint32_t bits)
{
    float value = 0.0F;
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&value, &bits, sizeof(value));
    return value;
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
        if (properties.major < 9)
        {
            throw std::runtime_error(
                "CuTe TMA GEMM requires compute capability 9.0+");
        }

        const size_t count_a =
            static_cast<size_t>(options.m) * options.k;
        const size_t count_b =
            static_cast<size_t>(options.n) * options.k;
        const size_t count_c =
            static_cast<size_t>(options.m) * options.n;
        const size_t required_bytes =
            (count_a + count_b + 6 * count_c) * sizeof(float);

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

        std::cout << "CuTe FP32 GEMM G2S/shared-layout test vs cuBLAS\n\n";
        std::cout << "[Configuration]\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "GPU" << properties.name << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "Compute capability"
                  << properties.major << '.' << properties.minor << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "A layout" << "row-major [M,K] = ["
                  << options.m << ',' << options.k << "]\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "B layout" << "row-major [N,K] = ["
                  << options.n << ',' << options.k << "]\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "C operation" << "C[M,N] = A * B^T\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "CTA tile" << kBlockM << 'x' << kBlockN
                  << 'x' << kBlockK << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "Pipeline stages" << kPipelineStages << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "Shared layouts"
                  << "Padding-129 / Swizzle<2,2,3> / Swizzle<3,2,3> / "
                  << "Vec16-Swizzle<1,2,3> / TMA-SW32\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "G2S copy widths"
                  << "4B scalar (first 3) / 16B cp.async (Vec16) / "
                  << "4096B/operand per TMA tile\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "Arithmetic" << "FP32 CUDA Core FMA / no TF32\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "cuBLAS compute mode" << "CUBLAS_COMPUTE_32F_PEDANTIC\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "Required device memory"
                  << std::fixed << std::setprecision(2)
                  << static_cast<double>(required_bytes) /
                         (1024.0 * 1024.0)
                  << " MiB\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "Free device memory"
                  << static_cast<double>(free_bytes) /
                         (1024.0 * 1024.0)
                  << " MiB\n\n";

        constexpr size_t memory_margin = 256ULL * 1024ULL * 1024ULL;
        if (required_bytes + memory_margin > free_bytes)
        {
            throw std::runtime_error(
                "insufficient free device memory for A, B and six C buffers");
        }

        CudaStream stream;
        CublasHandle cublas;
        CUBLAS_CHECK(cublasSetStream(cublas.get(), stream.get()));
        CUBLAS_CHECK(cublasSetMathMode(
            cublas.get(),
            CUBLAS_PEDANTIC_MATH));

        std::cout << "[Setup]\n";
        DeviceBuffer<float> device_a(count_a);
        DeviceBuffer<float> device_b(count_b);
        DeviceBuffer<float> device_padding_c(count_c);
        DeviceBuffer<float> device_swizzle_223_c(count_c);
        DeviceBuffer<float> device_swizzle_323_c(count_c);
        DeviceBuffer<float> device_vector_16_c(count_c);
        DeviceBuffer<float> device_tma_c(count_c);
        DeviceBuffer<float> device_cublas_c(count_c);
        // Tensor-map 编码是 host setup，不属于 kernel 时间；plan 在所有
        // correctness/benchmark launch 之间复用同一对 TMA descriptor。
        const auto tma_gemm_plan = make_cute_tma_gemm_plan(
            device_a.get(),
            device_b.get(),
            options.m,
            options.n,
            options.k);

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
        std::cout << "  Deterministic FP32 inputs initialized on GPU\n";
        std::cout << "  TMA tensor maps encoded once on host\n\n";

        std::cout << "[Correctness]\n";
        std::cout << std::defaultfloat << std::setprecision(6);
        launch_cute_gemm<SharedLayoutMode::kPadding129>(
            device_a.get(),
            device_b.get(),
            device_padding_c.get(),
            options.m,
            options.n,
            options.k,
            stream.get());
        launch_cute_gemm<SharedLayoutMode::kSwizzle223>(
            device_a.get(),
            device_b.get(),
            device_swizzle_223_c.get(),
            options.m,
            options.n,
            options.k,
            stream.get());
        launch_cute_gemm<SharedLayoutMode::kSwizzle323>(
            device_a.get(),
            device_b.get(),
            device_swizzle_323_c.get(),
            options.m,
            options.n,
            options.k,
            stream.get());
        launch_cute_gemm<SharedLayoutMode::kVector16Swizzle123>(
            device_a.get(),
            device_b.get(),
            device_vector_16_c.get(),
            options.m,
            options.n,
            options.k,
            stream.get());
        launch_cute_tma_gemm(
            tma_gemm_plan,
            device_tma_c.get(),
            stream.get());
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

        const std::array<ComparisonResult, 5> comparisons = {{
            {
                shared_layout_name<SharedLayoutMode::kPadding129>(),
                compare_outputs(
                    device_padding_c.get(),
                    device_cublas_c.get(),
                    count_c,
                    utility_block_count,
                    stream.get())
            },
            {
                shared_layout_name<SharedLayoutMode::kSwizzle223>(),
                compare_outputs(
                    device_swizzle_223_c.get(),
                    device_cublas_c.get(),
                    count_c,
                    utility_block_count,
                    stream.get())
            },
            {
                shared_layout_name<SharedLayoutMode::kSwizzle323>(),
                compare_outputs(
                    device_swizzle_323_c.get(),
                    device_cublas_c.get(),
                    count_c,
                    utility_block_count,
                    stream.get())
            },
            {
                shared_layout_name<
                    SharedLayoutMode::kVector16Swizzle123>(),
                compare_outputs(
                    device_vector_16_c.get(),
                    device_cublas_c.get(),
                    count_c,
                    utility_block_count,
                    stream.get())
            },
            {
                kTmaGemmName,
                compare_outputs(
                    device_tma_c.get(),
                    device_cublas_c.get(),
                    count_c,
                    utility_block_count,
                    stream.get())
            }
        }};

        std::cout << "  " << std::left << std::setw(30)
                  << "Compared elements per layout" << count_c << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "Tolerance" << "atol=" << kAbsoluteTolerance
                  << ", rtol=" << kRelativeTolerance << "\n\n";
        std::cout << "  " << std::left << std::setw(24) << "Shared layout"
                  << std::right << std::setw(16) << "Mismatches"
                  << std::setw(18) << "Max abs error"
                  << std::setw(18) << "Max rel error" << '\n';

        bool all_layouts_passed = true;
        for (const ComparisonResult &result : comparisons)
        {
            const float max_absolute_error = decode_float_bits(
                result.comparison.max_absolute_error_bits);
            const float max_relative_error = decode_float_bits(
                result.comparison.max_relative_error_bits);
            std::cout << "  " << std::left << std::setw(24) << result.name
                      << std::right << std::setw(16)
                      << result.comparison.mismatch_count
                      << std::setw(18) << max_absolute_error
                      << std::setw(18) << max_relative_error
                      << '\n';
            all_layouts_passed = all_layouts_passed &&
                result.comparison.mismatch_count == 0;
        }
        std::cout << '\n';

        std::cout << "[CPU FP64 samples]\n";
        std::cout << std::fixed << std::setprecision(7);
        const bool cpu_samples_passed = verify_cpu_samples(
            device_padding_c.get(),
            device_swizzle_223_c.get(),
            device_swizzle_323_c.get(),
            device_vector_16_c.get(),
            device_tma_c.get(),
            device_cublas_c.get(),
            options.m,
            options.n,
            options.k);
        std::cout << '\n';

        if (!all_layouts_passed || !cpu_samples_passed)
        {
            throw std::runtime_error("FP32 GEMM correctness validation failed");
        }

        std::cout << "[Benchmark]\n";
        std::cout << "  Method: rotate the starting implementation each round; "
                  << "report the median of "
                  << options.benchmark_iterations << " samples\n\n";
        const std::array<BenchmarkCase, 6> benchmark_cases = {{
            {
                shared_layout_name<SharedLayoutMode::kPadding129>(),
                [&]()
                {
                    launch_cute_gemm<SharedLayoutMode::kPadding129>(
                        device_a.get(),
                        device_b.get(),
                        device_padding_c.get(),
                        options.m,
                        options.n,
                        options.k,
                        stream.get());
                    CUDA_CHECK(cudaGetLastError());
                }
            },
            {
                shared_layout_name<SharedLayoutMode::kSwizzle223>(),
                [&]()
                {
                    launch_cute_gemm<SharedLayoutMode::kSwizzle223>(
                        device_a.get(),
                        device_b.get(),
                        device_swizzle_223_c.get(),
                        options.m,
                        options.n,
                        options.k,
                        stream.get());
                    CUDA_CHECK(cudaGetLastError());
                }
            },
            {
                shared_layout_name<SharedLayoutMode::kSwizzle323>(),
                [&]()
                {
                    launch_cute_gemm<SharedLayoutMode::kSwizzle323>(
                        device_a.get(),
                        device_b.get(),
                        device_swizzle_323_c.get(),
                        options.m,
                        options.n,
                        options.k,
                        stream.get());
                    CUDA_CHECK(cudaGetLastError());
                }
            },
            {
                shared_layout_name<
                    SharedLayoutMode::kVector16Swizzle123>(),
                [&]()
                {
                    launch_cute_gemm<
                        SharedLayoutMode::kVector16Swizzle123>(
                            device_a.get(),
                            device_b.get(),
                            device_vector_16_c.get(),
                            options.m,
                            options.n,
                            options.k,
                            stream.get());
                    CUDA_CHECK(cudaGetLastError());
                }
            },
            {
                kTmaGemmName,
                [&]()
                {
                    launch_cute_tma_gemm(
                        tma_gemm_plan,
                        device_tma_c.get(),
                        stream.get());
                    CUDA_CHECK(cudaGetLastError());
                }
            },
            {
                "cuBLAS FP32",
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

        const std::array<double, 6> benchmark_milliseconds =
            benchmark_round_robin(
                benchmark_cases,
                options.warmup_iterations,
                options.benchmark_iterations,
                stream.get());

        const double cublas_tflops = calculate_tflops(
            options.m,
            options.n,
            options.k,
            benchmark_milliseconds[5]);

        std::array<BenchmarkResult, 6> benchmark_results{};
        for (size_t index = 0; index < benchmark_results.size(); ++index)
        {
            benchmark_results[index] = {
                benchmark_cases[index].name,
                benchmark_milliseconds[index],
                calculate_tflops(
                    options.m,
                    options.n,
                    options.k,
                    benchmark_milliseconds[index])};
        }

        std::cout << "  " << std::left << std::setw(24) << "Implementation"
                  << std::right << std::setw(16) << "Latency (ms)"
                  << std::setw(16) << "TFLOP/s"
                  << std::setw(16) << "% cuBLAS" << '\n';
        std::cout << std::fixed << std::setprecision(3);
        for (const BenchmarkResult &result : benchmark_results)
        {
            std::cout << "  " << std::left << std::setw(24) << result.name
                      << std::right << std::setprecision(3)
                      << std::setw(16)
                      << result.milliseconds
                      << std::setw(16) << result.tflops
                      << std::setw(15) << std::setprecision(2)
                      << result.tflops / cublas_tflops * 100.0
                      << "%\n";
        }
        std::cout << '\n';

        std::cout << "[SUCCESS] cp.async/TMA CuTe FP32 GEMMs passed "
                  << "correctness and cuBLAS comparison\n";
    }
    catch (const std::exception &error)
    {
        std::cerr << "[ERROR] " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
