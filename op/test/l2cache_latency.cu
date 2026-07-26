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
constexpr int kWarmupIterations = 100;
constexpr int kLoadIterations = 10;
constexpr uint32_t kStrideBytes = 128;

static_assert(
    kStrideBytes >= kWarpSize * sizeof(uint32_t) &&
    kStrideBytes % sizeof(uint32_t) == 0,
    "invalid L2 pointer-chase stride");

void check_cuda(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

__global__ __launch_bounds__(kWarpSize, 1)
void l2cache_latency_kernel(
    const uint32_t *pointer_chain,
    uint32_t *valid,
    uint64_t *clocks)
{
    const char *load_pointer =
        reinterpret_cast<const char *>(
            pointer_chain + threadIdx.x);
    uint32_t value = 0;

    // Populate the TLB before entering the timed region.
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

        std::cout << "L2 cache latency test\n\n";
        std::cout << "[Configuration]\n";
        std::cout << "  " << std::left << std::setw(24)
                  << "GPU" << properties.name << '\n';
        std::cout << "  " << std::left << std::setw(24)
                  << "Compute capability"
                  << properties.major << '.' << properties.minor << '\n';
        std::cout << "  " << std::left << std::setw(24)
                  << "L2 cache" << l2_cache_bytes << " bytes\n";
        std::cout << "  " << std::left << std::setw(24)
                  << "Dependent loads" << kLoadIterations << '\n';
        std::cout << "  " << std::left << std::setw(24)
                  << "Cache-line stride" << kStrideBytes << " bytes\n";
        std::cout << "  " << std::left << std::setw(24)
                  << "Warmup launches" << kWarmupIterations << '\n';
        std::cout << "  " << std::left << std::setw(24)
                  << "Pointer-chain size"
                  << pointer_chain_bytes << " bytes\n\n";

        std::cout << "[Stage 1/3] Allocating device buffers\n";
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
            cudaMemcpy(
                device_pointer_chain,
                host_pointer_chain.data(),
                pointer_chain_bytes,
                cudaMemcpyHostToDevice),
            "cudaMemcpy(device_pointer_chain)");

        std::cout
            << "\n[Stage 2/3] Warming instruction cache, TLB, and L2\n";
        for (int i = 0; i < kWarmupIterations; ++i)
        {
            l2cache_latency_kernel<<<1, kWarpSize>>>(
                device_pointer_chain,
                device_valid,
                device_clocks);
        }
        check_cuda(cudaGetLastError(), "warmup kernel launch");
        check_cuda(cudaDeviceSynchronize(), "warmup synchronization");

        std::cout << "\n[Stage 3/3] Measuring warm L2 loads\n";
        l2cache_latency_kernel<<<1, kWarpSize>>>(
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
                "L2 pointer-chain validation failed");
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
        std::cout << "\n[SUCCESS] L2 cache latency test passed\n";

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
