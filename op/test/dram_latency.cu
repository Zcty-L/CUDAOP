#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "ptx_utils.cuh"

namespace
{

constexpr int kWarpSize = 32;
constexpr int kLoadIterations = 10;
constexpr uint32_t kStrideBytes = 1024;
constexpr size_t kMinimumFlushBytes = size_t{128} << 20;
constexpr int kFlushBlockThreads = 256;

static_assert(
    kStrideBytes >= kWarpSize * sizeof(uint32_t) &&
    kStrideBytes % sizeof(uint32_t) == 0,
    "invalid DRAM pointer-chase stride");

void check_cuda(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

__global__ void flush_l2_kernel(
    const uint32_t *input,
    uint32_t *checksum,
    size_t element_count)
{
    const size_t thread =
        blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride =
        static_cast<size_t>(gridDim.x) * blockDim.x;
    uint32_t sum = 0;

    for (size_t i = thread; i < element_count; i += stride)
    {
        uint32_t value = 0;
        ptx::ldg32_cg(value, input + i);
        sum += value;
    }

    if (sum != 0)
    {
        atomicAdd(checksum, sum);
    }
}

__global__ __launch_bounds__(kWarpSize, 1)
void dram_latency_kernel(
    const uint32_t *pointer_chain,
    uint32_t *valid,
    uint64_t *clocks)
{
    const char *load_pointer =
        reinterpret_cast<const char *>(
            pointer_chain + threadIdx.x);
    uint32_t value = 0;

    // Populate the TLB without warming any line used in the timed region.
    ptx::ldg32_cg(value, load_pointer);
    load_pointer += value;

    __syncthreads();
    const uint64_t start = ptx::read_clock64();

    #pragma unroll
    for (int i = 0; i < kLoadIterations; ++i)
    {
        // Pointer chasing prevents independent loads from hiding latency.
        ptx::ldg32_cg(value, load_pointer);
        load_pointer += value;
    }

    const uint64_t stop = ptx::read_clock64();

    clocks[threadIdx.x] = stop - start;
    valid[threadIdx.x] = value == kStrideBytes;
}

}  // namespace

int main()
{
    uint32_t *device_pointer_chain = nullptr;
    uint32_t *device_valid = nullptr;
    uint64_t *device_clocks = nullptr;
    uint32_t *device_flush = nullptr;
    uint32_t *device_checksum = nullptr;

    try
    {
        int device = 0;
        int l2_cache_bytes = 0;
        cudaDeviceProp properties{};
        check_cuda(cudaGetDevice(&device), "cudaGetDevice");
        check_cuda(
            cudaGetDeviceProperties(&properties, device),
            "cudaGetDeviceProperties");
        check_cuda(
            cudaDeviceGetAttribute(
                &l2_cache_bytes,
                cudaDevAttrL2CacheSize,
                device),
            "cudaDeviceGetAttribute(cudaDevAttrL2CacheSize)");

        constexpr size_t pointer_chain_bytes =
            (kLoadIterations + 1) * kStrideBytes;
        constexpr size_t pointer_chain_elements =
            pointer_chain_bytes / sizeof(uint32_t);
        const size_t flush_bytes = std::max(
            kMinimumFlushBytes,
            static_cast<size_t>(l2_cache_bytes) * 2);
        const size_t flush_elements =
            flush_bytes / sizeof(uint32_t);
        const int flush_blocks =
            properties.multiProcessorCount * 8;

        std::cout << "DRAM latency test\n\n";
        std::cout << "[Configuration]\n";
        std::cout << "  " << std::left << std::setw(24)
                  << "GPU" << properties.name << '\n';
        std::cout << "  " << std::left << std::setw(24)
                  << "Compute capability"
                  << properties.major << '.' << properties.minor << '\n';
        std::cout << "  " << std::left << std::setw(24)
                  << "Dependent loads" << kLoadIterations << '\n';
        std::cout << "  " << std::left << std::setw(24)
                  << "Address stride" << kStrideBytes << " bytes\n";
        std::cout << "  " << std::left << std::setw(24)
                  << "L2 cache" << l2_cache_bytes << " bytes\n";
        std::cout << "  " << std::left << std::setw(24)
                  << "L2 flush workspace" << flush_bytes << " bytes\n\n";

        std::cout << "[Stage 1/4] Allocating device buffers\n";
        std::vector<uint32_t> host_pointer_chain(
            pointer_chain_elements,
            kStrideBytes);
        check_cuda(
            cudaMalloc(
                &device_pointer_chain,
                pointer_chain_bytes),
            "cudaMalloc(device_pointer_chain)");
        check_cuda(
            cudaMalloc(
                &device_valid,
                kWarpSize * sizeof(uint32_t)),
            "cudaMalloc(device_valid)");
        check_cuda(
            cudaMalloc(
                &device_clocks,
                kWarpSize * sizeof(uint64_t)),
            "cudaMalloc(device_clocks)");
        check_cuda(
            cudaMalloc(&device_flush, flush_bytes),
            "cudaMalloc(device_flush)");
        check_cuda(
            cudaMalloc(&device_checksum, sizeof(uint32_t)),
            "cudaMalloc(device_checksum)");
        check_cuda(
            cudaMemcpy(
                device_pointer_chain,
                host_pointer_chain.data(),
                pointer_chain_bytes,
                cudaMemcpyHostToDevice),
            "cudaMemcpy(device_pointer_chain)");
        check_cuda(
            cudaMemset(device_flush, 0, flush_bytes),
            "cudaMemset(device_flush)");
        check_cuda(
            cudaMemset(device_checksum, 0, sizeof(uint32_t)),
            "cudaMemset(device_checksum)");

        std::cout << "\n[Stage 2/4] Warming instruction cache and TLB\n";
        dram_latency_kernel<<<1, kWarpSize>>>(
            device_pointer_chain,
            device_valid,
            device_clocks);
        check_cuda(cudaGetLastError(), "warmup kernel launch");
        check_cuda(cudaDeviceSynchronize(), "warmup synchronization");

        std::cout << "\n[Stage 3/4] Flushing L2 cache\n";
        flush_l2_kernel<<<flush_blocks, kFlushBlockThreads>>>(
            device_flush,
            device_checksum,
            flush_elements);
        check_cuda(cudaGetLastError(), "L2 flush kernel launch");
        check_cuda(cudaDeviceSynchronize(), "L2 flush synchronization");

        std::cout << "\n[Stage 4/4] Measuring cold global loads\n";
        dram_latency_kernel<<<1, kWarpSize>>>(
            device_pointer_chain,
            device_valid,
            device_clocks);
        check_cuda(cudaGetLastError(), "benchmark kernel launch");
        check_cuda(cudaDeviceSynchronize(), "benchmark synchronization");

        std::array<uint32_t, kWarpSize> host_valid{};
        std::array<uint64_t, kWarpSize> host_clocks{};
        check_cuda(
            cudaMemcpy(
                host_valid.data(),
                device_valid,
                host_valid.size() * sizeof(uint32_t),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(host_valid)");
        check_cuda(
            cudaMemcpy(
                host_clocks.data(),
                device_clocks,
                host_clocks.size() * sizeof(uint64_t),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(host_clocks)");

        const bool values_valid = std::all_of(
            host_valid.begin(),
            host_valid.end(),
            [](uint32_t value)
            {
                return value == 1;
            });
        const bool clocks_valid = std::all_of(
            host_clocks.begin(),
            host_clocks.end(),
            [](uint64_t value)
            {
                return value > 0;
            });
        if (!values_valid || !clocks_valid)
        {
            throw std::runtime_error(
                "DRAM pointer-chain validation failed");
        }

        const auto [minimum, maximum] = std::minmax_element(
            host_clocks.begin(),
            host_clocks.end());
        uint64_t total = 0;
        for (uint64_t clocks : host_clocks)
        {
            total += clocks;
        }

        const double minimum_latency =
            static_cast<double>(*minimum) / kLoadIterations;
        const double average_latency =
            static_cast<double>(total) /
            (kWarpSize * kLoadIterations);
        const double maximum_latency =
            static_cast<double>(*maximum) / kLoadIterations;

        std::cout << "\n[Result]\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  " << std::left << std::setw(24)
                  << "Minimum latency" << minimum_latency
                  << " cycles/load\n";
        std::cout << "  " << std::left << std::setw(24)
                  << "Average latency" << average_latency
                  << " cycles/load\n";
        std::cout << "  " << std::left << std::setw(24)
                  << "Maximum latency" << maximum_latency
                  << " cycles/load\n";
        std::cout << "\n[SUCCESS] DRAM latency test passed\n";

        check_cuda(
            cudaFree(device_checksum),
            "cudaFree(device_checksum)");
        device_checksum = nullptr;
        check_cuda(
            cudaFree(device_flush),
            "cudaFree(device_flush)");
        device_flush = nullptr;
        check_cuda(
            cudaFree(device_clocks),
            "cudaFree(device_clocks)");
        device_clocks = nullptr;
        check_cuda(
            cudaFree(device_valid),
            "cudaFree(device_valid)");
        device_valid = nullptr;
        check_cuda(
            cudaFree(device_pointer_chain),
            "cudaFree(device_pointer_chain)");
        device_pointer_chain = nullptr;
    }
    catch (const std::exception &error)
    {
        if (device_checksum != nullptr)
        {
            cudaFree(device_checksum);
        }
        if (device_flush != nullptr)
        {
            cudaFree(device_flush);
        }
        if (device_clocks != nullptr)
        {
            cudaFree(device_clocks);
        }
        if (device_valid != nullptr)
        {
            cudaFree(device_valid);
        }
        if (device_pointer_chain != nullptr)
        {
            cudaFree(device_pointer_chain);
        }
        std::cout << "\n[FAILED] " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
