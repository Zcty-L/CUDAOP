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

constexpr int kClusterBlocks = 4;
constexpr int kClusterCount = 2;
constexpr int kBlockThreads = 32;
constexpr int kWarmupIterations = 20;
constexpr int kBenchmarkIterations = 200;

struct DsmResult
{
    uint32_t cluster_id;
    uint32_t block_rank;
    uint32_t remote_sum;
};

void check_cuda(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

__device__ __forceinline__ uint32_t shared_address_raw(const void *pointer)
{
    // mapa.shared::cluster.u32 接收 shared state-space 的 32-bit offset，
    // 不能直接接收 CUDA C++ generic shared pointer。
    return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

// PTX ISA 参考：
// https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-clusterid
//
// 指令名称：mov.u32 from %clusterid.x
// 用途：标识当前 CTA 所属的 grid cluster，本例用于构造不同 cluster 的值。
__device__ __forceinline__ uint32_t cluster_id_x_raw()
{
    uint32_t value = 0;
    asm volatile(
        "mov.u32 %0, %%clusterid.x;\n"
        : "=r"(value));
    return value;
}

// PTX ISA 参考：
// https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-cluster-ctarank
//
// 指令名称：mov.u32 from %cluster_ctarank
// 用途：读取当前 CTA 在 cluster 内的 flattened rank；mapa 的目标 CTA
// 操作数也使用同一 rank 空间。
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
// 用途：读取 cluster x 维 CTA 数，作为遍历全部远端 DSM segment 的上界。
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
// 用途：第一次同步发布每个 CTA 写入的 dsm_value，第二次同步保证任何 CTA
// 退出前都不再有其他 CTA 访问它的 shared-memory segment。
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

// PTX ISA 参考：
// https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-mapa
//
// 指令名称：mapa.shared::cluster.u32
// 用途：把当前 CTA 中某个 shared offset 映射为 target_rank CTA 的 DSM
// 地址。映射只改变所属 CTA segment，变量在该 segment 内的 offset 不变。
__device__ __forceinline__ uint32_t map_shared_rank_raw(
    const void *local_shared_pointer,
    uint32_t target_rank)
{
    const uint32_t local_address = shared_address_raw(local_shared_pointer);
    uint32_t remote_address = 0;
    asm volatile(
        "mapa.shared::cluster.u32 %0, %1, %2;\n"
        : "=r"(remote_address)
        : "r"(local_address), "r"(target_rank)
        : "memory");
    return remote_address;
}

// PTX ISA 参考：
// https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-ld
//
// 指令名称：ld.shared::cluster.b32
// 用途：从 mapa 生成的 cluster shared 地址读取 32-bit 数据。若地址属于
// 另一 CTA，该访问会通过 distributed shared memory 到达远端 segment。
__device__ __forceinline__ uint32_t load_shared_cluster_u32_raw(
    uint32_t cluster_shared_address)
{
    uint32_t value = 0;
    asm volatile(
        "ld.shared::cluster.b32 %0, [%1];\n"
        : "=r"(value)
        : "r"(cluster_shared_address)
        : "memory");
    return value;
}

__global__ void cluster_dsm_kernel(DsmResult *output)
{
    __shared__ uint32_t dsm_value;

    const uint32_t cluster_id = cluster_id_x_raw();
    const uint32_t block_rank = cluster_cta_rank_raw();
    const uint32_t cluster_blocks = cluster_cta_count_x_raw();

    if (threadIdx.x == 0)
    {
        dsm_value = cluster_id * 100 + block_rank + 1;
    }

    cluster_sync_raw();

    if (threadIdx.x == 0)
    {
        uint32_t remote_sum = 0;
        for (uint32_t target_rank = 0;
             target_rank < cluster_blocks;
            ++target_rank)
        {
            const uint32_t remote_address =
                map_shared_rank_raw(&dsm_value, target_rank);
            const uint32_t remote_value =
                load_shared_cluster_u32_raw(remote_address);
            remote_sum += remote_value;
        }

        output[blockIdx.x] = {
            cluster_id,
            block_rank,
            remote_sum
        };
    }

    // 只要还有 peer CTA 可能访问本 CTA 的 dsm_value，本 CTA 就不能退出。
    cluster_sync_raw();
}

cudaLaunchConfig_t make_launch_config(
    cudaLaunchAttribute &cluster_attribute)
{
    cluster_attribute = {};
    cluster_attribute.id = cudaLaunchAttributeClusterDimension;
    cluster_attribute.val.clusterDim.x = kClusterBlocks;
    cluster_attribute.val.clusterDim.y = 1;
    cluster_attribute.val.clusterDim.z = 1;

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(kClusterBlocks * kClusterCount, 1, 1);
    config.blockDim = dim3(kBlockThreads, 1, 1);
    config.dynamicSmemBytes = 0;
    config.stream = nullptr;
    config.attrs = &cluster_attribute;
    config.numAttrs = 1;
    return config;
}

void launch_dsm(
    const cudaLaunchConfig_t &config,
    DsmResult *output)
{
    check_cuda(
        cudaLaunchKernelEx(
            &config,
            cluster_dsm_kernel,
            output),
        "cudaLaunchKernelEx(cluster_dsm_kernel)");
}

}  // namespace

int main()
{
    DsmResult *device_output = nullptr;
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

        std::cout << "Distributed shared memory test\n\n";
        std::cout << "[Configuration]\n";
        std::cout << "  " << std::left << std::setw(28)
                  << "GPU" << properties.name << '\n';
        std::cout << "  " << std::left << std::setw(28)
                  << "Compute capability"
                  << properties.major << '.' << properties.minor << '\n';
        std::cout << "  " << std::left << std::setw(28)
                  << "Cluster shape" << kClusterBlocks << " x 1 x 1\n";
        std::cout << "  " << std::left << std::setw(28)
                  << "Cluster count" << kClusterCount << "\n\n";

        if (properties.major < 9)
        {
            throw std::runtime_error(
                "DSM requires compute capability 9.0 or newer");
        }

        cudaLaunchConfig_t occupancy_config{};
        occupancy_config.gridDim = dim3(kClusterBlocks, 1, 1);
        occupancy_config.blockDim = dim3(kBlockThreads, 1, 1);
        int maximum_cluster_blocks = 0;
        check_cuda(
            cudaOccupancyMaxPotentialClusterSize(
                &maximum_cluster_blocks,
                cluster_dsm_kernel,
                &occupancy_config),
            "cudaOccupancyMaxPotentialClusterSize");
        if (maximum_cluster_blocks < kClusterBlocks)
        {
            throw std::runtime_error(
                "DSM test requires four blocks per cluster, maximum is " +
                std::to_string(maximum_cluster_blocks));
        }

        cudaLaunchAttribute cluster_attribute{};
        const cudaLaunchConfig_t config =
            make_launch_config(cluster_attribute);
        constexpr int grid_blocks = kClusterBlocks * kClusterCount;

        check_cuda(
            cudaMalloc(
                reinterpret_cast<void **>(&device_output),
                grid_blocks * sizeof(DsmResult)),
            "cudaMalloc(device_output)");
        check_cuda(
            cudaEventCreate(&start_event),
            "cudaEventCreate(start_event)");
        check_cuda(
            cudaEventCreate(&stop_event),
            "cudaEventCreate(stop_event)");

        std::cout << "[Correctness]\n";
        launch_dsm(config, device_output);
        check_cuda(
            cudaDeviceSynchronize(),
            "correctness synchronization");

        std::vector<DsmResult> host_output(grid_blocks);
        check_cuda(
            cudaMemcpy(
                host_output.data(),
                device_output,
                grid_blocks * sizeof(DsmResult),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(output device-to-host)");

        for (int block = 0; block < grid_blocks; ++block)
        {
            const uint32_t expected_cluster = block / kClusterBlocks;
            const uint32_t expected_rank = block % kClusterBlocks;
            const uint32_t expected_sum =
                kClusterBlocks * expected_cluster * 100 +
                kClusterBlocks * (kClusterBlocks + 1) / 2;
            const DsmResult &actual = host_output[block];

            if (actual.cluster_id != expected_cluster ||
                actual.block_rank != expected_rank ||
                actual.remote_sum != expected_sum)
            {
                throw std::runtime_error(
                    "DSM verification failed for block " +
                    std::to_string(block) +
                    ", expected_sum=" + std::to_string(expected_sum) +
                    ", actual_sum=" +
                    std::to_string(actual.remote_sum));
            }

            std::cout << "  block=" << block
                      << "  cluster=" << actual.cluster_id
                      << "  rank=" << actual.block_rank
                      << "  remote_sum=" << actual.remote_sum << '\n';
        }

        std::cout << "\n[Benchmark]\n";
        for (int iteration = 0;
             iteration < kWarmupIterations;
             ++iteration)
        {
            launch_dsm(config, device_output);
        }
        check_cuda(cudaDeviceSynchronize(), "warmup synchronization");

        check_cuda(
            cudaEventRecord(start_event),
            "cudaEventRecord(start_event)");
        for (int iteration = 0;
             iteration < kBenchmarkIterations;
             ++iteration)
        {
            launch_dsm(config, device_output);
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
                  << "Average DSM kernel"
                  << std::fixed << std::setprecision(2)
                  << average_microseconds << " us\n\n";
        std::cout << "[SUCCESS] DSM remote address mapping and remote "
                  << "shared-memory loads passed verification\n";

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
