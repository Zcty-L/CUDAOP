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

constexpr int kBlockThreads = 128;
constexpr int kVectorLoads = 4;
constexpr int kLoadRepeats = 64;
constexpr int kWorkingSetSegments = 4;
constexpr int kBlocksPerSm = 64;
constexpr int kWarmupIterations = 20;
constexpr int kBenchmarkIterations = 100;
constexpr size_t kWorkingSetBytes =
    static_cast<size_t>(kBlockThreads) *
    kVectorLoads *
    kWorkingSetSegments *
    sizeof(uint4);

static_assert(
    (kWorkingSetSegments & (kWorkingSetSegments - 1)) == 0,
    "L1 working-set segment count must be a power of two");

void check_cuda(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <int BlockThreads, int VectorLoads, int LoadRepeats>
__global__ void l1cache_bandwidth_kernel(
    const uint4 *input,
    uint32_t *sink,
    uint32_t segment_mask)
{
    uint4 registers[VectorLoads];
    uint32_t sum = 0;

    #pragma unroll 1
    for (int repeat = 0; repeat < LoadRepeats; ++repeat)
    {
        const size_t segment =
            static_cast<uint32_t>(repeat) & segment_mask;
        const uint4 *load_pointer =
            input +
            segment * BlockThreads * VectorLoads +
            threadIdx.x;

        #pragma unroll
        for (int i = 0; i < VectorLoads; ++i)
        {
            ptx::ldg128_nc(
                registers[i],
                load_pointer + BlockThreads * i);
        }

        #pragma unroll
        for (int i = 0; i < VectorLoads; ++i)
        {
            sum += registers[i].x;
            sum += registers[i].y;
            sum += registers[i].z;
            sum += registers[i].w;
        }
    }

    if (sum != 0)
    {
        *sink = sum;
    }
}

}  // namespace

int main()
{
    uint4 *device_input = nullptr;
    uint32_t *device_sink = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

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

        const int grid_blocks =
            properties.multiProcessorCount * kBlocksPerSm;
        const size_t bytes_per_block =
            static_cast<size_t>(kBlockThreads) *
            kVectorLoads *
            kLoadRepeats *
            sizeof(uint4);
        const size_t bytes_per_launch =
            bytes_per_block * grid_blocks;
        const size_t benchmark_bytes =
            bytes_per_launch * kBenchmarkIterations;

        std::cout << "L1 cache bandwidth test\n\n";
        std::cout << "[Configuration]\n";
        std::cout << "  " << std::left << std::setw(24)
                  << "GPU" << properties.name << '\n';
        std::cout << "  " << std::left << std::setw(24)
                  << "Compute capability"
                  << properties.major << '.' << properties.minor << '\n';
        std::cout << "  " << std::left << std::setw(24)
                  << "SM count" << properties.multiProcessorCount << '\n';
        std::cout << "  " << std::left << std::setw(24)
                  << "Reported SM clock"
                  << static_cast<double>(clock_rate_khz) / 1000.0
                  << " MHz\n";
        std::cout << "  " << std::left << std::setw(24)
                  << "L1 working set" << kWorkingSetBytes << " bytes\n";
        std::cout << "  " << std::left << std::setw(24)
                  << "Working-set segments"
                  << kWorkingSetSegments << '\n';
        std::cout << "  " << std::left << std::setw(24)
                  << "Block threads" << kBlockThreads << '\n';
        std::cout << "  " << std::left << std::setw(24)
                  << "Independent loads" << kVectorLoads << '\n';
        std::cout << "  " << std::left << std::setw(24)
                  << "Vector width" << sizeof(uint4) << " bytes\n";
        std::cout << "  " << std::left << std::setw(24)
                  << "Load repeats" << kLoadRepeats << '\n';
        std::cout << "  " << std::left << std::setw(24)
                  << "Grid blocks" << grid_blocks << '\n';
        std::cout << "  " << std::left << std::setw(24)
                  << "Traffic per launch"
                  << bytes_per_launch << " bytes\n";
        std::cout << "  " << std::left << std::setw(24)
                  << "Warmup launches" << kWarmupIterations << '\n';
        std::cout << "  " << std::left << std::setw(24)
                  << "Benchmark launches"
                  << kBenchmarkIterations << "\n\n";

        std::cout << "[Stage 1/3] Allocating device buffers\n";
        check_cuda(
            cudaMalloc(&device_input, kWorkingSetBytes),
            "cudaMalloc(device_input)");
        check_cuda(
            cudaMalloc(&device_sink, sizeof(uint32_t)),
            "cudaMalloc(device_sink)");
        check_cuda(
            cudaMemset(device_input, 0, kWorkingSetBytes),
            "cudaMemset(device_input)");
        check_cuda(
            cudaMemset(device_sink, 0, sizeof(uint32_t)),
            "cudaMemset(device_sink)");
        check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
        check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

        std::cout << "\n[Stage 2/3] Warming per-SM L1 caches\n";
        for (int i = 0; i < kWarmupIterations; ++i)
        {
            l1cache_bandwidth_kernel<
                kBlockThreads,
                kVectorLoads,
                kLoadRepeats>
                <<<grid_blocks, kBlockThreads>>>(
                    device_input,
                    device_sink,
                    kWorkingSetSegments - 1);
        }
        check_cuda(cudaGetLastError(), "warmup kernel launch");
        check_cuda(cudaDeviceSynchronize(), "warmup synchronization");

        std::cout << "\n[Stage 3/3] Measuring aggregate L1 throughput\n";
        check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
        for (int i = 0; i < kBenchmarkIterations; ++i)
        {
            l1cache_bandwidth_kernel<
                kBlockThreads,
                kVectorLoads,
                kLoadRepeats>
                <<<grid_blocks, kBlockThreads>>>(
                    device_input,
                    device_sink,
                    kWorkingSetSegments - 1);
        }
        check_cuda(cudaGetLastError(), "benchmark kernel launch");
        check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
        check_cuda(
            cudaEventSynchronize(stop),
            "cudaEventSynchronize(stop)");

        float elapsed_ms = 0.0F;
        check_cuda(
            cudaEventElapsedTime(&elapsed_ms, start, stop),
            "cudaEventElapsedTime");

        uint32_t host_sink = 1;
        check_cuda(
            cudaMemcpy(
                &host_sink,
                device_sink,
                sizeof(uint32_t),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(host_sink)");
        if (host_sink != 0)
        {
            throw std::runtime_error(
                "L1 bandwidth data validation failed");
        }

        const double elapsed_seconds =
            static_cast<double>(elapsed_ms) / 1000.0;
        const double bandwidth_gbps =
            static_cast<double>(benchmark_bytes) /
            elapsed_seconds /
            1.0e9;
        const double clock_ghz =
            static_cast<double>(clock_rate_khz) / 1.0e6;
        const double bytes_per_cycle_per_sm =
            bandwidth_gbps /
            (properties.multiProcessorCount * clock_ghz);
        if (!std::isfinite(bandwidth_gbps) ||
            !std::isfinite(bytes_per_cycle_per_sm) ||
            bandwidth_gbps <= 0.0 ||
            bytes_per_cycle_per_sm <= 0.0)
        {
            throw std::runtime_error(
                "invalid L1 bandwidth result");
        }

        std::cout << "\n[Result]\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  " << std::left << std::setw(28)
                  << "Elapsed time" << elapsed_ms << " ms\n";
        std::cout << "  " << std::left << std::setw(28)
                  << "Aggregate L1 bandwidth"
                  << bandwidth_gbps << " GB/s\n";
        std::cout << "  " << std::left << std::setw(28)
                  << "Bandwidth per SM"
                  << bytes_per_cycle_per_sm
                  << " bytes/cycle\n";
        std::cout << "\n[SUCCESS] L1 cache bandwidth test passed\n";

        check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
        stop = nullptr;
        check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
        start = nullptr;
        check_cuda(cudaFree(device_sink), "cudaFree(device_sink)");
        device_sink = nullptr;
        check_cuda(cudaFree(device_input), "cudaFree(device_input)");
        device_input = nullptr;
    }
    catch (const std::exception &error)
    {
        if (stop != nullptr)
        {
            cudaEventDestroy(stop);
        }
        if (start != nullptr)
        {
            cudaEventDestroy(start);
        }
        if (device_sink != nullptr)
        {
            cudaFree(device_sink);
        }
        if (device_input != nullptr)
        {
            cudaFree(device_input);
        }
        std::cout << "\n[FAILED] " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
