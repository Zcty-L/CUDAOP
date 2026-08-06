#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "ptx_utils.cuh"

namespace
{

constexpr int kThreads = 16;
constexpr int kWarmupIterations = 100;
constexpr int kLoadIterations = 50;

void check_cuda(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

__global__ __launch_bounds__(kThreads, 1)
void smem_latency_kernel(uint32_t *valid, uint64_t *clocks)
{
    __shared__ uint32_t smem[kThreads];

    const uint32_t initial_address =
        ptx::smem_u32addr(&smem[threadIdx.x]);
    smem[threadIdx.x] = initial_address;
    __syncthreads();

    uint32_t smem_address = initial_address;
    const uint64_t start = ptx::read_clock64();

    #pragma unroll
    for (int i = 0; i < kLoadIterations; ++i)
    {
        // Pointer chasing makes every shared-memory load depend on the
        // preceding load, preventing instruction-level latency hiding.
        ptx::lds32(smem_address, smem_address);
    }

    const uint64_t stop = ptx::read_clock64();

    clocks[threadIdx.x] = stop - start;
    valid[threadIdx.x] = smem_address == initial_address;
}

}  // namespace

int main()
{
    uint32_t *device_valid = nullptr;
    uint64_t *device_clocks = nullptr;

    try
    {
        int device = 0;
        cudaDeviceProp properties{};
        check_cuda(cudaGetDevice(&device), "cudaGetDevice");
        check_cuda(
            cudaGetDeviceProperties(&properties, device),
            "cudaGetDeviceProperties");

        std::cout << "Shared memory latency test\n\n";
        std::cout << "[Configuration]\n";
        std::cout << "  " << std::left << std::setw(22)
                  << "GPU" << properties.name << '\n';
        std::cout << "  " << std::left << std::setw(22)
                  << "Compute capability"
                  << properties.major << '.' << properties.minor << '\n';
        std::cout << "  " << std::left << std::setw(22)
                  << "Threads per block" << kThreads << '\n';
        std::cout << "  " << std::left << std::setw(22)
                  << "Warmup launches" << kWarmupIterations << '\n';
        std::cout << "  " << std::left << std::setw(22)
                  << "Dependent loads" << kLoadIterations << "\n\n";

        std::cout << "[Stage 1/3] Allocating device buffers\n";
        check_cuda(
            cudaMalloc(&device_valid, kThreads * sizeof(uint32_t)),
            "cudaMalloc(device_valid)");
        check_cuda(
            cudaMalloc(&device_clocks, kThreads * sizeof(uint64_t)),
            "cudaMalloc(device_clocks)");

        std::cout << "\n[Stage 2/3] Warming instruction caches\n";
        for (int i = 0; i < kWarmupIterations; ++i)
        {
            smem_latency_kernel<<<1, kThreads>>>(
                device_valid,
                device_clocks);
        }
        check_cuda(cudaGetLastError(), "warmup kernel launch");
        check_cuda(cudaDeviceSynchronize(), "warmup synchronization");

        std::cout << "\n[Stage 3/3] Measuring dependent shared loads\n";
        smem_latency_kernel<<<1, kThreads>>>(
            device_valid,
            device_clocks);
        check_cuda(cudaGetLastError(), "benchmark kernel launch");
        check_cuda(cudaDeviceSynchronize(), "benchmark synchronization");

        std::array<uint32_t, kThreads> host_valid{};
        std::array<uint64_t, kThreads> host_clocks{};
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

        const bool addresses_valid = std::all_of(
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
        if (!addresses_valid || !clocks_valid)
        {
            throw std::runtime_error(
                "shared-memory pointer chain validation failed");
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
            (kThreads * kLoadIterations);
        const double maximum_latency =
            static_cast<double>(*maximum) / kLoadIterations;

        std::cout << "\n[Result]\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  " << std::left << std::setw(22)
                  << "Minimum latency" << minimum_latency
                  << " cycles/load\n";
        std::cout << "  " << std::left << std::setw(22)
                  << "Average latency" << average_latency
                  << " cycles/load\n";
        std::cout << "  " << std::left << std::setw(22)
                  << "Maximum latency" << maximum_latency
                  << " cycles/load\n";
        std::cout << "\n[SUCCESS] Shared memory latency test passed\n";

        check_cuda(cudaFree(device_clocks), "cudaFree(device_clocks)");
        device_clocks = nullptr;
        check_cuda(cudaFree(device_valid), "cudaFree(device_valid)");
        device_valid = nullptr;
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
        std::cout << "\n[FAILED] " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
