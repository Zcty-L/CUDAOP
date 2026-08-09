#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "ptx_utils.cuh"

namespace
{

constexpr int kBlockThreads = 128;
constexpr int kTileElements = 128;
constexpr int kBlockCount = 4096;
constexpr int kWarmupIterations = 10;
constexpr int kBenchmarkIterations = 100;
constexpr size_t kTileBytes = static_cast<size_t>(kTileElements) * sizeof(uint32_t);
constexpr size_t kElementCount = static_cast<size_t>(kBlockCount) * kTileElements;
constexpr size_t kTensorBytes = kElementCount * sizeof(uint32_t);

static_assert(
    kTileBytes % 16 == 0,
    "TMA tile size must preserve 16-byte alignment");
static_assert(
    kTileElements <= 256,
    "A Tensor Map box dimension cannot exceed 256 elements");

void check_cuda(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void check_driver(CUresult status, const char *operation)
{
    if (status == CUDA_SUCCESS)
    {
        return;
    }

    const char *error_name = nullptr;
    const char *error_message = nullptr;
    cuGetErrorName(status, &error_name);
    cuGetErrorString(status, &error_message);

    throw std::runtime_error(
        std::string(operation) + ": " +
        (error_name == nullptr ? "unknown" : error_name) + " (" +
        (error_message == nullptr ? "no description" : error_message) +
        ")");
}

uint32_t make_input_value(size_t index)
{
    uint32_t value = static_cast<uint32_t>(index);
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return value;
}

// __grid_constant__ 表示整个 grid 只读这份 kernel 参数对象，不为每个线程生成可写的局部副本；TMA 指令使用该 CUtensorMap 参数在参数空间中的地址。
__global__ void tma_copy_kernel(
    __grid_constant__ const CUtensorMap tensor_map,
    uint32_t *output,
    uint64_t *completion_cycles)
{
    __shared__ __align__(128) uint32_t shared_tile[kTileElements];

    // mbarrier 在共享内存中占用 8 字节。TMA 完成搬运后会更新这个
    // barrier 的 transaction byte count，线程通过它判断数据是否可用。
    __shared__ __align__(8) uint64_t transaction_barrier;

    // 一个 block 只需要一个线程初始化 barrier 和发起 TMA。TMA 是一次bulk copy，不要求所有线程分别执行 load 指令。
    if (threadIdx.x == 0)
    {
        // 参数 1 是当前 barrier phase 的 expected arrival count，表示这个
        // phase 只等待一次线程到达。后面的 arrive_expect_tx 由 thread 0
        // 执行一次，正好消费这个 arrival。
        //
        // 注意：1 不是 TMA 事务数，也不是拷贝字节数。事务字节数由
        // mbarrier_arrive_expect_tx(kTileBytes) 另行登记。
        ptx::mbarrier_init(&transaction_barrier, 1);

        // mbarrier.init 通过普通 shared-memory proxy 写入 barrier；TMA
        // 使用 async proxy 访问它。该 fence 让初始化结果对 async proxy
        // 可见，必须发生在发起 TMA 之前。
        ptx::fence_proxy_async_shared_cta();
    }

    // 确保 barrier 初始化阶段结束后，block 中的线程再进入搬运阶段。
    __syncthreads();

    if (threadIdx.x == 0)
    {
        // Tensor Map 是 rank-1，因此 coordinate 的单位是 uint32_t 元素，不是字节。
        // block b 从第 b * kTileElements 个元素开始读取。
        const int32_t coordinate = static_cast<int32_t>(blockIdx.x * kTileElements);
        const uint64_t start_cycles = ptx::read_clock64();

        // arrive_expect_tx 同时完成两件事：
        // 1. thread 0 到达 barrier，使 pending arrival count 从 1 变为 0；
        // 2. 登记本 phase 仍需由异步事务完成 kTileBytes 字节。
        // 只有 arrival count 和 transaction byte count 都归零，当前barrier phase 才算完成。
        ptx::mbarrier_arrive_expect_tx(&transaction_barrier, static_cast<uint32_t>(kTileBytes));

        // 发起异步 TMA：硬件从 tensor_map 读取基地址、元素类型、tensor
        // 尺寸和 tile 尺寸，再结合 coordinate 计算本 block 的全局内存
        // 范围，将 128 个 uint32_t 搬到 shared_tile。传入同一个 barrier
        // 后，TMA 每完成数据传输就会归还对应的 transaction bytes。
        ptx::tma_load_1d(shared_tile, &tensor_map, coordinate, &transaction_barrier);

        // 初始化后的第一个 phase 使用 parity 0。
        // 等待成功意味着本 phase 的 arrival 与 kTileBytes 异步事务均已完成，TMA 写入的数据可用。
        ptx::mbarrier_wait_parity(&transaction_barrier, 0);

        // 这里只统计单个 block 从登记事务到等待完成的 device cycles，
        // 用于观察 TMA 完成延迟，不等同于整个 kernel 的执行时间。
        completion_cycles[blockIdx.x] = ptx::read_clock64() - start_cycles;
    }

    // 先由 thread 0 等待 TMA，再用 CTA barrier 将“数据已经可读”的状态
    // 发布给其余线程，避免其他线程提前读取 shared_tile。
    __syncthreads();

    // kBlockThreads == kTileElements，因此每个线程负责写回一个元素。
    const size_t output_index = static_cast<size_t>(blockIdx.x) * kTileElements + threadIdx.x;
    output[output_index] = shared_tile[threadIdx.x];

    // 所有线程读完共享内存之后，才允许 thread 0 销毁 barrier。
    __syncthreads();
    if (threadIdx.x == 0)
    {
        ptx::mbarrier_invalidate(&transaction_barrier);
    }
}

void launch_tma_copy(
    const CUtensorMap &tensor_map,
    uint32_t *output,
    uint64_t *completion_cycles)
{
    tma_copy_kernel<<<kBlockCount, kBlockThreads>>>(
        tensor_map,
        output,
        completion_cycles);
    check_cuda(cudaGetLastError(), "tma_copy_kernel launch");
}

CUtensorMap make_tensor_map(uint32_t *device_input)
{
    // Tensor Map 是 CUDA Driver 在 host 端编码出的不透明描述符。它把 global-memory tensor 的以下元数据打包起来：
    //
    //   基地址 + 数据类型 + rank + 各维尺寸/stride + 单次搬运 box
    //   + interleave/swizzle + L2 promotion + 越界填充策略

    // cuTensorMapEncodeTiled 要求输出描述符的 host 地址按 64 字节对齐。
    alignas(64) CUtensorMap tensor_map{};

    // rank = 1，整个 global tensor 含 kElementCount 个 uint32_t 元素。
    const cuuint64_t global_dimensions[] = {
        static_cast<cuuint64_t>(kElementCount)
    };

    // rank-1 不会使用 global stride，因为不存在更高维度；但当前 Driver
    // 仍要求 global_strides 参数指向有效的 host 内存，所以传入占位数组。
    const cuuint64_t global_strides[] = {0};

    // box_dimensions 定义一条 TMA 指令搬运的 tile：这里每次沿唯一维度
    // 读取 kTileElements 个元素，即 kTileBytes 字节。
    const cuuint32_t box_dimensions[] = {
        static_cast<cuuint32_t>(kTileElements)
    };

    // element stride 为 1 表示连续遍历元素。当前 interleave 为 NONE 时，
    // CUDA 对 dimension 0 的 element stride 实际会忽略它。
    const cuuint32_t element_strides[] = {1};

    check_driver(
        cuTensorMapEncodeTiled(
            &tensor_map,
            // 描述符中的元素类型决定 coordinate 和 box 的元素粒度。
            CU_TENSOR_MAP_DATA_TYPE_UINT32,
            // 一维 tensor，TMA PTX 因此使用 tensor.1d 形式。
            1,
            // 被描述的 device global-memory 区域起始地址。
            device_input,
            global_dimensions,
            global_strides,
            box_dimensions,
            element_strides,
            // 不使用通道交错布局。
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            // 本测试验证基础 copy，共享内存布局不做 swizzle。
            CU_TENSOR_MAP_SWIZZLE_NONE,
            // 不额外提示 TMA 将数据提升到指定大小的 L2 cache line。
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            // 不请求浮点 NaN 越界填充；本测试坐标本身不会越界。
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
        "cuTensorMapEncodeTiled");

    // 对 block b 而言，kernel 传入 coordinate = b * kTileElements。
    // 结合 box_dimensions 后，硬件读取的逻辑范围为：
    // [coordinate, coordinate + kTileElements)，共 kTileBytes 字节。
    return tensor_map;
}

void verify_output(
    const std::vector<uint32_t> &expected,
    const std::vector<uint32_t> &actual)
{
    size_t mismatch_count = 0;
    size_t first_mismatch = 0;

    for (size_t index = 0; index < expected.size(); ++index)
    {
        if (actual[index] != expected[index])
        {
            if (mismatch_count == 0)
            {
                first_mismatch = index;
            }
            ++mismatch_count;
        }
    }

    if (mismatch_count != 0)
    {
        throw std::runtime_error(
            "TMA copy verification failed: mismatches=" +
            std::to_string(mismatch_count) +
            ", first_index=" + std::to_string(first_mismatch) +
            ", expected=" + std::to_string(expected[first_mismatch]) +
            ", actual=" + std::to_string(actual[first_mismatch]));
    }
}

struct CycleStatistics
{
    uint64_t minimum;
    uint64_t median;
    uint64_t maximum;
    double average;
};

CycleStatistics calculate_cycle_statistics(std::vector<uint64_t> cycles)
{
    std::sort(cycles.begin(), cycles.end());
    const long double sum = std::accumulate(
        cycles.begin(),
        cycles.end(),
        static_cast<long double>(0));

    return {
        cycles.front(),
        cycles[cycles.size() / 2],
        cycles.back(),
        static_cast<double>(sum / cycles.size())
    };
}

}  // namespace

int main()
{
    uint32_t *device_input = nullptr;
    uint32_t *device_output = nullptr;
    uint64_t *device_cycles = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;

    try
    {
        int device = 0;
        cudaDeviceProp properties{};
        check_cuda(cudaGetDevice(&device), "cudaGetDevice");
        check_cuda(
            cudaGetDeviceProperties(&properties, device),
            "cudaGetDeviceProperties");

        std::cout << "TMA 1D global-to-shared copy test\n\n";
        std::cout << "[Configuration]\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "GPU" << properties.name << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "Compute capability"
                  << properties.major << '.' << properties.minor << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "Blocks" << kBlockCount << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "Threads per block" << kBlockThreads << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "Tile bytes" << kTileBytes << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "Tensor bytes" << kTensorBytes << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "Benchmark iterations" << kBenchmarkIterations
                  << "\n\n";

        if (properties.major < 9)
        {
            throw std::runtime_error(
                "TMA requires compute capability 9.0 or newer");
        }

        std::vector<uint32_t> host_input(kElementCount);
        std::vector<uint32_t> host_output(kElementCount);
        std::vector<uint64_t> host_cycles(kBlockCount);
        for (size_t index = 0; index < host_input.size(); ++index)
        {
            host_input[index] = make_input_value(index);
        }

        std::cout << "[Setup]\n";
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void **>(&device_input),
                kTensorBytes),
            "cudaMalloc(device_input)");
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void **>(&device_output),
                kTensorBytes),
            "cudaMalloc(device_output)");
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void **>(&device_cycles),
                kBlockCount * sizeof(uint64_t)),
            "cudaMalloc(device_cycles)");
        check_cuda(
            cudaMemcpy(
                device_input,
                host_input.data(),
                kTensorBytes,
                cudaMemcpyHostToDevice),
            "cudaMemcpy(input host-to-device)");

        const CUtensorMap tensor_map = make_tensor_map(device_input);
        check_cuda(
            cudaEventCreate(&start_event),
            "cudaEventCreate(start_event)");
        check_cuda(
            cudaEventCreate(&stop_event),
            "cudaEventCreate(stop_event)");
        std::cout << "  Tensor Map encoded successfully\n\n";

        std::cout << "[Correctness]\n";
        launch_tma_copy(tensor_map, device_output, device_cycles);
        check_cuda(
            cudaDeviceSynchronize(),
            "correctness synchronization");
        check_cuda(
            cudaMemcpy(
                host_output.data(),
                device_output,
                kTensorBytes,
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(output device-to-host)");
        check_cuda(
            cudaMemcpy(
                host_cycles.data(),
                device_cycles,
                kBlockCount * sizeof(uint64_t),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(cycles device-to-host)");
        verify_output(host_input, host_output);
        std::cout << "  All " << kElementCount
                  << " elements match\n\n";

        const CycleStatistics cycle_statistics =
            calculate_cycle_statistics(host_cycles);

        std::cout << "[Benchmark]\n";
        for (int iteration = 0;
             iteration < kWarmupIterations;
             ++iteration)
        {
            launch_tma_copy(tensor_map, device_output, device_cycles);
        }
        check_cuda(
            cudaDeviceSynchronize(),
            "warmup synchronization");

        check_cuda(
            cudaEventRecord(start_event),
            "cudaEventRecord(start_event)");
        for (int iteration = 0;
             iteration < kBenchmarkIterations;
             ++iteration)
        {
            launch_tma_copy(tensor_map, device_output, device_cycles);
        }
        check_cuda(
            cudaEventRecord(stop_event),
            "cudaEventRecord(stop_event)");
        check_cuda(
            cudaEventSynchronize(stop_event),
            "cudaEventSynchronize(stop_event)");

        float elapsed_ms = 0.0F;
        check_cuda(
            cudaEventElapsedTime(
                &elapsed_ms,
                start_event,
                stop_event),
            "cudaEventElapsedTime");

        const double average_microseconds =
            static_cast<double>(elapsed_ms) * 1000.0 /
            kBenchmarkIterations;
        const double effective_read_gbps =
            static_cast<double>(kTensorBytes) *
            kBenchmarkIterations /
            (static_cast<double>(elapsed_ms) * 1.0e6);

        std::cout << "  " << std::left << std::setw(30)
                  << "Completion cycles min"
                  << cycle_statistics.minimum << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "Completion cycles median"
                  << cycle_statistics.median << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "Completion cycles average"
                  << std::fixed << std::setprecision(2)
                  << cycle_statistics.average << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "Completion cycles max"
                  << cycle_statistics.maximum << '\n';
        std::cout << "  " << std::left << std::setw(30)
                  << "Average kernel latency"
                  << average_microseconds << " us\n";
        std::cout << "  " << std::left << std::setw(30)
                  << "Effective TMA read payload"
                  << effective_read_gbps << " GB/s\n\n";

        std::cout << "[SUCCESS] TMA copy instruction completed and "
                  << "passed verification\n";

        check_cuda(cudaEventDestroy(start_event), "cudaEventDestroy(start)");
        start_event = nullptr;
        check_cuda(cudaEventDestroy(stop_event), "cudaEventDestroy(stop)");
        stop_event = nullptr;
        check_cuda(cudaFree(device_cycles), "cudaFree(device_cycles)");
        device_cycles = nullptr;
        check_cuda(cudaFree(device_output), "cudaFree(device_output)");
        device_output = nullptr;
        check_cuda(cudaFree(device_input), "cudaFree(device_input)");
        device_input = nullptr;
    }
    catch (const std::exception &error)
    {
        if (start_event != nullptr)
        {
            cudaEventDestroy(start_event);
        }
        if (stop_event != nullptr)
        {
            cudaEventDestroy(stop_event);
        }
        if (device_cycles != nullptr)
        {
            cudaFree(device_cycles);
        }
        if (device_output != nullptr)
        {
            cudaFree(device_output);
        }
        if (device_input != nullptr)
        {
            cudaFree(device_input);
        }

        std::cerr << "[ERROR] " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
