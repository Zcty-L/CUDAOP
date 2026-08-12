#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace
{

constexpr int kBlockThreads = 128;
constexpr int kBlockCount = 4096;
constexpr int kWordsPerVector = 4;
constexpr int kWarmupIterations = 5;
constexpr int kBenchmarkIterations = 50;
constexpr size_t kVectorBytes = sizeof(uint4);
constexpr size_t kVectorCount =
    static_cast<size_t>(kBlockCount) * kBlockThreads;
constexpr size_t kTensorBytes = kVectorCount * kVectorBytes;

static_assert(kVectorBytes == 16, "cp.async test requires a 16-byte vector");

void check_cuda(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

uint32_t make_input_word(size_t index)
{
    uint32_t value = static_cast<uint32_t>(index);
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return value;
}

// PTX ISA 参考：
// https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async
//
// 指令名称：cp.async.ca.shared.global
// 用途：直接把 global memory 数据异步搬到 shared memory，避免先通过
//      通用寄存器执行 load，再执行 store 到 shared memory。
//
// 这个 wrapper 固定使用 cp-size=16。`.ca` 是 cache at all levels 提示；
// 对 cp.async 而言，16B 是最适合向量化搬运、也能直接映射到 CuTe
// uint128_t copy atom 的粒度。global/shared 两端在本测试中都显式保持 16B
// 对齐，避免编译器或硬件需要拆分事务。
__device__ __forceinline__ void cp_async_ca_shared_global_16(
    void *shared_destination,
    const void *global_source)
{
    // CUDA C++ 的 shared_destination 是 generic address。PTX 的
    // cp.async.shared.global 第一个操作数却要求 shared state-space 地址，
    // 即 32-bit shared-memory offset，不能直接传 generic 64-bit 指针。
    // __cvta_generic_to_shared 完成 generic -> shared 的地址空间转换；
    // 随后的 uint32_t 截取符合 PTX shared 地址操作数的 `r` 约束。
    const uint32_t shared_address =
        static_cast<uint32_t>(__cvta_generic_to_shared(shared_destination));

    // `r` 传入 32-bit shared offset，`l` 传入 64-bit global address。
    // memory clobber 阻止编译器把周围内存访问错误地越过这条异步搬运。
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :
        : "r"(shared_address), "l"(global_source)
        : "memory");
}

// PTX ISA 参考：
// https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-commit-group
//
// 指令名称：cp.async.commit_group
// 用途：把当前线程此前尚未提交的 cp.async 操作组成并提交一个 group。
// commit 只划定异步操作组，并不等待数据已经到达 shared memory。
__device__ __forceinline__ void cp_async_commit_group()
{
    asm volatile("cp.async.commit_group;\n" : : : "memory");
}

// PTX ISA 参考：
// https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-wait-group
//
// 指令名称：cp.async.wait_group 0
// 用途：等待当前线程提交的所有先前 cp.async group 完成。立即数 0 的
// 含义是允许仍未完成的 prior group 数为零，而不是“等待第 0 个 group”。
// wait_group 只跟踪发出这些异步操作的线程；它不是 CTA barrier，不能取代
// __syncthreads() 来向其他线程发布 shared-memory 数据。
__device__ __forceinline__ void cp_async_wait_group_0()
{
    asm volatile("cp.async.wait_group 0;\n" : : : "memory");
}

__global__ void cp_async_copy_kernel(
    const uint4 *__restrict__ input,
    uint4 *__restrict__ output)
{
    // uint4 的大小和对齐均为 16B。数组基地址以及每个元素的地址因此都满足
    // cp.async 16B 事务的 shared-memory 对齐条件。
    __shared__ __align__(16) uint4 shared_tile[kBlockThreads];

    const int local_vector = static_cast<int>(threadIdx.x);
    const size_t global_vector =
        static_cast<size_t>(blockIdx.x) * blockDim.x + local_vector;

    // 线程协作方式：一个 128-thread CTA 共同搬运 128 个 uint4；每个线程
    // 发出一条 16B cp.async，处理互不重叠的一段。CUDA device allocation
    // 至少满足 256B 对齐，uint4 索引又保持 16B 步长，所以 global_source
    // 同样始终 16B 对齐。
    cp_async_ca_shared_global_16(
        &shared_tile[local_vector],
        &input[global_vector]);

    // 每线程把自己刚发出的 cp.async 放入一个 group。真实 GEMM 通常会在
    // commit 后计算上一 stage，再用 wait_group<N> 控制流水线中尚未完成的
    // group 数；这个最小正确性测试不跨 stage 重叠，因此直接等待到 0。
    cp_async_commit_group();
    cp_async_wait_group_0();

    // wait_group 保证“本线程发起”的异步写入已完成，但不同线程的完成和
    // shared-memory 可见性仍需要 CTA barrier 汇合。barrier 之后，所有线程
    // 都可以安全读取整个 shared_tile；此处每线程只写回自己的 uint4。
    __syncthreads();
    output[global_vector] = shared_tile[local_vector];
}

void launch_copy(const uint4 *input, uint4 *output)
{
    cp_async_copy_kernel<<<kBlockCount, kBlockThreads>>>(input, output);
    check_cuda(cudaGetLastError(), "cp_async_copy_kernel launch");
}

bool vectors_equal(const uint4 &lhs, const uint4 &rhs)
{
    return lhs.x == rhs.x &&
           lhs.y == rhs.y &&
           lhs.z == rhs.z &&
           lhs.w == rhs.w;
}

} // namespace

int main()
{
    uint4 *device_input = nullptr;
    uint4 *device_output = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    try
    {
        int device = 0;
        cudaDeviceProp properties{};
        check_cuda(cudaGetDevice(&device), "cudaGetDevice");
        check_cuda(
            cudaGetDeviceProperties(&properties, device),
            "cudaGetDeviceProperties");

        if (properties.major < 8)
        {
            throw std::runtime_error(
                "cp.async.shared.global requires compute capability 8.0+");
        }

        std::cout << "\n[配置]\n"
                  << "  GPU                 : " << properties.name << '\n'
                  << "  指令                : cp.async.ca.shared.global (16B)\n"
                  << "  block 数            : " << kBlockCount << '\n'
                  << "  每 block 线程数     : " << kBlockThreads << '\n'
                  << "  每线程搬运          : " << kVectorBytes << " B\n"
                  << "  总数据量            : " << kTensorBytes << " B\n"
                  << "  warmup / iterations : " << kWarmupIterations
                  << " / " << kBenchmarkIterations << '\n';

        std::cout << "\n[阶段 1] 构造并上传 16B 对齐的输入\n";
        std::vector<uint4> host_input(kVectorCount);
        std::vector<uint4> host_output(kVectorCount);
        for (size_t vector_index = 0;
             vector_index < kVectorCount;
             ++vector_index)
        {
            const size_t word_index = vector_index * kWordsPerVector;
            host_input[vector_index] = make_uint4(
                make_input_word(word_index),
                make_input_word(word_index + 1),
                make_input_word(word_index + 2),
                make_input_word(word_index + 3));
        }

        check_cuda(
            cudaMalloc(reinterpret_cast<void **>(&device_input), kTensorBytes),
            "cudaMalloc input");
        check_cuda(
            cudaMalloc(reinterpret_cast<void **>(&device_output), kTensorBytes),
            "cudaMalloc output");
        check_cuda(
            cudaMemcpy(
                device_input,
                host_input.data(),
                kTensorBytes,
                cudaMemcpyHostToDevice),
            "cudaMemcpy input");

        std::cout << "\n[阶段 2] 执行 GMEM -> SMEM -> GMEM 搬运\n";
        for (int iteration = 0;
             iteration < kWarmupIterations;
             ++iteration)
        {
            launch_copy(device_input, device_output);
        }
        check_cuda(cudaDeviceSynchronize(), "warmup synchronize");

        check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
        check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");
        check_cuda(cudaEventRecord(start), "cudaEventRecord start");
        for (int iteration = 0;
             iteration < kBenchmarkIterations;
             ++iteration)
        {
            launch_copy(device_input, device_output);
        }
        check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
        check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

        float elapsed_ms = 0.0F;
        check_cuda(
            cudaEventElapsedTime(&elapsed_ms, start, stop),
            "cudaEventElapsedTime");
        check_cuda(
            cudaMemcpy(
                host_output.data(),
                device_output,
                kTensorBytes,
                cudaMemcpyDeviceToHost),
            "cudaMemcpy output");

        std::cout << "\n[阶段 3] 逐个 uint4 验证结果\n";
        size_t mismatch_count = 0;
        size_t first_mismatch = kVectorCount;
        for (size_t index = 0; index < kVectorCount; ++index)
        {
            if (!vectors_equal(host_input[index], host_output[index]))
            {
                ++mismatch_count;
                first_mismatch = std::min(first_mismatch, index);
            }
        }

        const double average_ms =
            static_cast<double>(elapsed_ms) / kBenchmarkIterations;
        const double logical_bandwidth_gb_s =
            static_cast<double>(2 * kTensorBytes) /
            (average_ms * 1.0e6);

        std::cout << "\n[结果]\n"
                  << "  mismatch 数         : " << mismatch_count << '\n'
                  << std::fixed << std::setprecision(6)
                  << "  平均 kernel 时间    : " << average_ms << " ms\n"
                  << "  逻辑读写带宽        : " << logical_bandwidth_gb_s
                  << " GB/s\n";

        if (mismatch_count != 0)
        {
            const uint4 expected = host_input[first_mismatch];
            const uint4 actual = host_output[first_mismatch];
            std::cout << "  首个错误 vector     : " << first_mismatch << '\n'
                      << "  expected            : "
                      << expected.x << ", " << expected.y << ", "
                      << expected.z << ", " << expected.w << '\n'
                      << "  actual              : "
                      << actual.x << ", " << actual.y << ", "
                      << actual.z << ", " << actual.w << '\n';
            throw std::runtime_error("cp.async copy verification failed");
        }

        check_cuda(cudaEventDestroy(start), "cudaEventDestroy start");
        start = nullptr;
        check_cuda(cudaEventDestroy(stop), "cudaEventDestroy stop");
        stop = nullptr;
        check_cuda(cudaFree(device_input), "cudaFree input");
        device_input = nullptr;
        check_cuda(cudaFree(device_output), "cudaFree output");
        device_output = nullptr;

        // CuTe 映射关系：
        // Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, T> 对应这里的
        // cp.async.ca.shared.global ... 16；CuTe cp_async_fence() 对应
        // commit_group，cp_async_wait<0>() 对应 wait_group 0。
        // feature/linear 的 FP16/FP8/INT8/W4A16 pipeline 使用这条路径完成
        // 16B GMEM -> SMEM 搬运。cute_gemm_fp4_fp16.cu 当前则是 packed FP4
        // GMEM -> RMEM，不经过该 cp.async pipeline，不能把本文件误解为其
        // 当前实现的原样拆解。
        std::cout << "\n[SUCCESS] cp.async 16B 搬运与同步语义验证通过\n";
        return EXIT_SUCCESS;
    }
    catch (const std::exception &error)
    {
        if (start != nullptr)
        {
            cudaEventDestroy(start);
        }
        if (stop != nullptr)
        {
            cudaEventDestroy(stop);
        }
        if (device_input != nullptr)
        {
            cudaFree(device_input);
        }
        if (device_output != nullptr)
        {
            cudaFree(device_output);
        }

        std::cout << "\n[FAILED] " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
