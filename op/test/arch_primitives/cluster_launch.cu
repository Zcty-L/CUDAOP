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

constexpr int kRequestedClusterBlocks = 4;
constexpr int kClusterCount = 2;
constexpr int kBlockThreads = 32;
constexpr int kWarmupIterations = 20;
constexpr int kBenchmarkIterations = 200;

struct ClusterInfo
{
    uint32_t block_id;
    uint32_t cluster_id;
    uint32_t cluster_cta_id;
    uint32_t cluster_cta_rank;
    uint32_t cluster_cta_count;
};

void check_cuda(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

// PTX ISA 参考：
// https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-clusterid
//
// 指令名称：mov.u32 from %clusterid.x
// 用途：读取当前 CTA 所属 cluster 在 grid 中的 x 维 cluster 坐标。
// 它不是普通 blockIdx；显式 cluster shape 为 4 x 1 x 1 时，每四个 CTA
// 共享同一个 cluster_id.x。
__device__ __forceinline__ uint32_t cluster_id_x_raw()
{
    uint32_t value = 0;
    asm volatile(
        "mov.u32 %0, %%clusterid.x;\n"
        : "=r"(value));
    return value;
}

// PTX ISA 参考：
// https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-cluster-ctaid
//
// 指令名称：mov.u32 from %cluster_ctaid.x
// 用途：读取 CTA 在本 cluster 的 x 维局部坐标，范围是
// [0, %cluster_nctaid.x)。对于一维 cluster，它与 flattened rank 相同。
__device__ __forceinline__ uint32_t cluster_cta_id_x_raw()
{
    uint32_t value = 0;
    asm volatile(
        "mov.u32 %0, %%cluster_ctaid.x;\n"
        : "=r"(value));
    return value;
}

// PTX ISA 参考：
// https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-cluster-ctarank
//
// 指令名称：mov.u32 from %cluster_ctarank
// 用途：读取 CTA 在 cluster 内按 x、y、z 展平后的 rank。DSM 的 mapa
// 指令以这个 flattened rank 选择远端 CTA。
__device__ __forceinline__ uint32_t cluster_cta_rank_raw()
{
    uint32_t value = 0;
    asm volatile(
        "mov.u32 %0, %%cluster_ctarank;\n"
        : "=r"(value));
    return value;
}

// PTX ISA 参考：
// https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-cluster-nctaid
//
// 指令名称：mov.u32 from %cluster_nctaid.x
// 用途：读取显式启动时配置的 cluster x 维 CTA 数。本例应恒为 4。
__device__ __forceinline__ uint32_t cluster_cta_count_x_raw()
{
    uint32_t value = 0;
    asm volatile(
        "mov.u32 %0, %%cluster_nctaid.x;\n"
        : "=r"(value));
    return value;
}

// PTX ISA 参考：
// https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-barrier-cluster
//
// 指令名称：barrier.cluster.arrive.release / barrier.cluster.wait.acquire
// 用途：所有 CTA 先以 release 语义到达 cluster barrier，再以 acquire
// 语义等待同一 cluster 的 CTA 全部到达。arrive 与 wait 分离，允许在两者
// 之间插入与 cluster 同步无依赖的工作；本最小示例直接连续执行。
__device__ __forceinline__ void cluster_sync_raw()
{
    asm volatile(
        "barrier.cluster.arrive.release;\n"
        :
        :
        : "memory");
    asm volatile(
        "barrier.cluster.wait.acquire;\n"
        :
        :
        : "memory");
}

__global__ void cluster_launch_kernel(ClusterInfo *output)
{
    ClusterInfo info{};
    if (threadIdx.x == 0)
    {
        info.block_id = blockIdx.x;
        info.cluster_id = cluster_id_x_raw();
        info.cluster_cta_id = cluster_cta_id_x_raw();
        info.cluster_cta_rank = cluster_cta_rank_raw();
        info.cluster_cta_count = cluster_cta_count_x_raw();
    }

    // 所有线程都执行 cluster barrier。即使只有 thread 0 读取特殊寄存器，
    // cluster 同步仍是 CTA 级操作，不能只放在 threadIdx.x == 0 分支中。
    cluster_sync_raw();

    if (threadIdx.x == 0)
    {
        output[blockIdx.x] = info;
    }
}

cudaLaunchConfig_t make_launch_config(
    int cluster_blocks,
    cudaLaunchAttribute &cluster_attribute)
{
    cluster_attribute = {};
    cluster_attribute.id = cudaLaunchAttributeClusterDimension;
    cluster_attribute.val.clusterDim.x = cluster_blocks;
    cluster_attribute.val.clusterDim.y = 1;
    cluster_attribute.val.clusterDim.z = 1;

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(cluster_blocks * kClusterCount, 1, 1);
    config.blockDim = dim3(kBlockThreads, 1, 1);
    config.dynamicSmemBytes = 0;
    config.stream = nullptr;
    config.attrs = &cluster_attribute;
    config.numAttrs = 1;
    return config;
}

void launch_cluster(
    const cudaLaunchConfig_t &config,
    ClusterInfo *output)
{
    check_cuda(
        cudaLaunchKernelEx(
            &config,
            cluster_launch_kernel,
            output),
        "cudaLaunchKernelEx(cluster_launch_kernel)");
}

}  // namespace

int main()
{
    ClusterInfo *device_output = nullptr;
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

        std::cout << "Explicit thread block cluster launch test\n\n";
        std::cout << "[Configuration]\n";
        std::cout << "  " << std::left << std::setw(28)
                  << "GPU" << properties.name << '\n';
        std::cout << "  " << std::left << std::setw(28)
                  << "Compute capability"
                  << properties.major << '.' << properties.minor << '\n';

        if (properties.major < 9)
        {
            throw std::runtime_error(
                "thread block clusters require compute capability 9.0 or newer");
        }

        cudaLaunchConfig_t occupancy_config{};
        occupancy_config.gridDim = dim3(kRequestedClusterBlocks, 1, 1);
        occupancy_config.blockDim = dim3(kBlockThreads, 1, 1);
        int maximum_cluster_blocks = 0;
        check_cuda(
            cudaOccupancyMaxPotentialClusterSize(
                &maximum_cluster_blocks,
                cluster_launch_kernel,
                &occupancy_config),
            "cudaOccupancyMaxPotentialClusterSize");

        if (maximum_cluster_blocks < kRequestedClusterBlocks)
        {
            throw std::runtime_error(
                "requested cluster size " +
                std::to_string(kRequestedClusterBlocks) +
                " exceeds the portable maximum " +
                std::to_string(maximum_cluster_blocks));
        }

        cudaLaunchAttribute cluster_attribute{};
        const cudaLaunchConfig_t config = make_launch_config(
            kRequestedClusterBlocks,
            cluster_attribute);
        const int grid_blocks =
            kRequestedClusterBlocks * kClusterCount;

        std::cout << "  " << std::left << std::setw(28)
                  << "Maximum cluster blocks"
                  << maximum_cluster_blocks << '\n';
        std::cout << "  " << std::left << std::setw(28)
                  << "Selected cluster shape"
                  << kRequestedClusterBlocks << " x 1 x 1\n";
        std::cout << "  " << std::left << std::setw(28)
                  << "Grid blocks" << grid_blocks << "\n\n";

        check_cuda(
            cudaMalloc(
                reinterpret_cast<void **>(&device_output),
                grid_blocks * sizeof(ClusterInfo)),
            "cudaMalloc(device_output)");
        check_cuda(
            cudaEventCreate(&start_event),
            "cudaEventCreate(start_event)");
        check_cuda(
            cudaEventCreate(&stop_event),
            "cudaEventCreate(stop_event)");

        std::cout << "[Correctness]\n";
        launch_cluster(config, device_output);
        check_cuda(
            cudaDeviceSynchronize(),
            "correctness synchronization");

        std::vector<ClusterInfo> host_output(grid_blocks);
        check_cuda(
            cudaMemcpy(
                host_output.data(),
                device_output,
                grid_blocks * sizeof(ClusterInfo),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(output device-to-host)");

        for (int block = 0; block < grid_blocks; ++block)
        {
            const uint32_t expected_cluster =
                block / kRequestedClusterBlocks;
            const uint32_t expected_rank =
                block % kRequestedClusterBlocks;
            const ClusterInfo &actual = host_output[block];

            if (actual.block_id != static_cast<uint32_t>(block) ||
                actual.cluster_id != expected_cluster ||
                actual.cluster_cta_id != expected_rank ||
                actual.cluster_cta_rank != expected_rank ||
                actual.cluster_cta_count != kRequestedClusterBlocks)
            {
                throw std::runtime_error(
                    "cluster register verification failed for block " +
                    std::to_string(block));
            }

            std::cout << "  block=" << std::setw(2) << block
                      << "  cluster=" << actual.cluster_id
                      << "  cta_id=" << actual.cluster_cta_id
                      << "  rank=" << actual.cluster_cta_rank
                      << "  cluster_blocks="
                      << actual.cluster_cta_count << '\n';
        }

        std::cout << "\n[Benchmark]\n";
        for (int iteration = 0;
             iteration < kWarmupIterations;
             ++iteration)
        {
            launch_cluster(config, device_output);
        }
        check_cuda(cudaDeviceSynchronize(), "warmup synchronization");

        check_cuda(
            cudaEventRecord(start_event),
            "cudaEventRecord(start_event)");
        for (int iteration = 0;
             iteration < kBenchmarkIterations;
             ++iteration)
        {
            launch_cluster(config, device_output);
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

        std::cout << "  " << std::left << std::setw(28)
                  << "Average cluster kernel"
                  << std::fixed << std::setprecision(2)
                  << average_microseconds << " us\n\n";
        std::cout << "[SUCCESS] Cluster launch, special registers, and "
                  << "cluster barrier passed verification\n";

        check_cuda(cudaEventDestroy(start_event), "cudaEventDestroy(start)");
        start_event = nullptr;
        check_cuda(cudaEventDestroy(stop_event), "cudaEventDestroy(stop)");
        stop_event = nullptr;
        check_cuda(cudaFree(device_output), "cudaFree(device_output)");
        device_output = nullptr;
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
        if (device_output != nullptr)
        {
            cudaFree(device_output);
        }

        std::cerr << "[ERROR] " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
