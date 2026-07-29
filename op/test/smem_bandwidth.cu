#include <algorithm>
#include <array>
#include <cmath>
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

constexpr int kBlockThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarps = kBlockThreads / kWarpSize;
constexpr int kWarmupIterations = 100;
constexpr int kStoreIterations = 512;

void check_cuda(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

__global__ __launch_bounds__(kBlockThreads, 1)
void smem_bandwidth_kernel(
    uint32_t *valid,
    uint64_t *clock_start,
    uint64_t *clock_stop)
{
    __shared__ int4 smem[kBlockThreads + kStoreIterations];

    const uint32_t thread = threadIdx.x;
    const uint32_t lane = thread % kWarpSize;
    const uint32_t warp = thread / kWarpSize;
    const uint32_t smem_address =
        ptx::smem_u32addr(smem + thread);
    const int4 registers = make_int4(
        thread,
        thread + 1,
        thread + 2,
        thread + 3);

    __syncthreads();
    const uint64_t start = ptx::read_clock64();

    #pragma unroll
    for (int i = 0; i < kStoreIterations; ++i)
    {
        ptx::sts128(
            registers.x,
            registers.y,
            registers.z,
            registers.w,
            smem_address + i * static_cast<uint32_t>(sizeof(int4))
        );
    }

    const uint64_t stop = ptx::read_clock64();
    __syncthreads();

    uint32_t loaded = 0;
    ptx::lds32(loaded, smem_address);
    valid[thread] =
        static_cast<int32_t>(loaded) >= 0;

    if (lane == 0)
    {
        clock_start[warp] = start;
        clock_stop[warp] = stop;
    }
}

}  // namespace

int main()
{
    uint32_t *device_valid = nullptr;
    uint64_t *device_clock_start = nullptr;
    uint64_t *device_clock_stop = nullptr;

    try
    {
        int device = 0;
        int clock_rate_khz = 0;
        cudaDeviceProp properties{};
        check_cuda(cudaGetDevice(&device), "cudaGetDevice");
        check_cuda(
            cudaGetDeviceProperties(&properties, device),
            "cudaGetDeviceProperties");
        check_cuda(
            cudaDeviceGetAttribute(
                &clock_rate_khz,
                cudaDevAttrClockRate,
                device),
            "cudaDeviceGetAttribute(cudaDevAttrClockRate)");

        constexpr uint64_t accessed_bytes =
            static_cast<uint64_t>(kBlockThreads) *
            kStoreIterations *
            sizeof(int4);
        constexpr uint64_t shared_memory_bytes =
            static_cast<uint64_t>(
                kBlockThreads + kStoreIterations) *
            sizeof(int4);

        std::cout << "Shared memory bandwidth test\n\n";
        std::cout << "[Configuration]\n";
        std::cout << "  " << std::left << std::setw(24)
                  << "GPU" << properties.name << '\n';
        std::cout << "  " << std::left << std::setw(24)
                  << "Compute capability"
                  << properties.major << '.' << properties.minor << '\n';
        std::cout << "  " << std::left << std::setw(24)
                  << "SM count"
                  << properties.multiProcessorCount << '\n';
        std::cout << "  " << std::left << std::setw(24)
                  << "Threads per block" << kBlockThreads << '\n';
        std::cout << "  " << std::left << std::setw(24)
                  << "Warmup launches" << kWarmupIterations << '\n';
        std::cout << "  " << std::left << std::setw(24)
                  << "Stores per thread" << kStoreIterations << '\n';
        std::cout << "  " << std::left << std::setw(24)
                  << "Bytes per store" << sizeof(int4) << '\n';
        std::cout << "  " << std::left << std::setw(24)
                  << "Shared memory" << shared_memory_bytes
                  << " bytes\n\n";

        std::cout << "[Stage 1/3] Allocating device buffers\n";
        check_cuda(
            cudaMalloc(
                &device_valid,
                kBlockThreads * sizeof(uint32_t)),
            "cudaMalloc(device_valid)");
        check_cuda(
            cudaMalloc(
                &device_clock_start,
                kWarps * sizeof(uint64_t)),
            "cudaMalloc(device_clock_start)");
        check_cuda(
            cudaMalloc(
                &device_clock_stop,
                kWarps * sizeof(uint64_t)),
            "cudaMalloc(device_clock_stop)");

        std::cout << "\n[Stage 2/3] Warming instruction caches\n";
        for (int i = 0; i < kWarmupIterations; ++i)
        {
            smem_bandwidth_kernel<<<1, kBlockThreads>>>(
                device_valid,
                device_clock_start,
                device_clock_stop);
        }
        check_cuda(cudaGetLastError(), "warmup kernel launch");
        check_cuda(cudaDeviceSynchronize(), "warmup synchronization");

        std::cout << "\n[Stage 3/3] Measuring shared store throughput\n";
        smem_bandwidth_kernel<<<1, kBlockThreads>>>(
            device_valid,
            device_clock_start,
            device_clock_stop);
        check_cuda(cudaGetLastError(), "benchmark kernel launch");
        check_cuda(cudaDeviceSynchronize(), "benchmark synchronization");

        std::array<uint32_t, kBlockThreads> host_valid{};
        std::array<uint64_t, kWarps> host_clock_start{};
        std::array<uint64_t, kWarps> host_clock_stop{};
        check_cuda(
            cudaMemcpy(
                host_valid.data(),
                device_valid,
                host_valid.size() * sizeof(uint32_t),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(host_valid)");
        check_cuda(
            cudaMemcpy(
                host_clock_start.data(),
                device_clock_start,
                host_clock_start.size() * sizeof(uint64_t),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(host_clock_start)");
        check_cuda(
            cudaMemcpy(
                host_clock_stop.data(),
                device_clock_stop,
                host_clock_stop.size() * sizeof(uint64_t),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(host_clock_stop)");

        const bool stores_valid = std::all_of(
            host_valid.begin(),
            host_valid.end(),
            [](uint32_t value)
            {
                return value == 1;
            });
        bool clocks_valid = true;
        for (int i = 0; i < kWarps; ++i)
        {
            clocks_valid &=
                host_clock_stop[i] > host_clock_start[i];
        }
        if (!stores_valid || !clocks_valid)
        {
            throw std::runtime_error(
                "shared-memory bandwidth validation failed");
        }

        const uint64_t start = *std::min_element(
            host_clock_start.begin(),
            host_clock_start.end());
        const uint64_t stop = *std::max_element(
            host_clock_stop.begin(),
            host_clock_stop.end());
        const uint64_t duration = stop - start;
        const double measured_bytes_per_cycle =
            static_cast<double>(accessed_bytes) / duration;
        const double inferred_bytes_per_cycle =
            std::ceil(measured_bytes_per_cycle / 32.0) * 32.0;
        const double clock_mhz =
            static_cast<double>(clock_rate_khz) / 1000.0;
        const double estimated_chip_bandwidth =
            properties.multiProcessorCount *
            inferred_bytes_per_cycle *
            clock_mhz /
            1000.0;

        std::cout << "\n[Result]\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  " << std::left << std::setw(32)
                  << "Shared memory accessed"
                  << accessed_bytes << " bytes\n";
        std::cout << "  " << std::left << std::setw(32)
                  << "Duration" << duration << " cycles\n";
        std::cout << "  " << std::left << std::setw(32)
                  << "Measured bandwidth per SM"
                  << measured_bytes_per_cycle
                  << " bytes/cycle\n";
        std::cout << "  " << std::left << std::setw(32)
                  << "Inferred peak per SM"
                  << inferred_bytes_per_cycle
                  << " bytes/cycle\n";
        std::cout << "  " << std::left << std::setw(32)
                  << "Reported SM clock"
                  << clock_mhz << " MHz\n";
        std::cout << "  " << std::left << std::setw(32)
                  << "Estimated chip peak"
                  << estimated_chip_bandwidth << " GB/s\n";
        std::cout
            << "\n[SUCCESS] Shared memory bandwidth test passed\n";

        check_cuda(
            cudaFree(device_clock_stop),
            "cudaFree(device_clock_stop)");
        device_clock_stop = nullptr;
        check_cuda(
            cudaFree(device_clock_start),
            "cudaFree(device_clock_start)");
        device_clock_start = nullptr;
        check_cuda(
            cudaFree(device_valid),
            "cudaFree(device_valid)");
        device_valid = nullptr;
    }
    catch (const std::exception &error)
    {
        if (device_clock_stop != nullptr)
        {
            cudaFree(device_clock_stop);
        }
        if (device_clock_start != nullptr)
        {
            cudaFree(device_clock_start);
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
