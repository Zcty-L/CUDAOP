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

constexpr size_t kDataSizeBytes = size_t{2} << 20;
constexpr int kLoadCount = (1 << 20) * 512;
constexpr int kWarmupIterations = 200;
constexpr int kBenchmarkIterations = 200;
constexpr int kBlockThreads = 128;
constexpr int kLoadUnroll = 16;
constexpr int kDataElements =
    kDataSizeBytes / sizeof(uint32_t);
constexpr int kGridBlocks =
    kLoadCount / kLoadUnroll / kBlockThreads;

static_assert(
    kDataElements >= kLoadUnroll * kBlockThreads &&
    kDataElements % (kLoadUnroll * kBlockThreads) == 0,
    "invalid L2 bandwidth configuration");

void check_cuda(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <int BlockThreads, int LoadUnroll, int DataElements>
__global__ void l2cache_bandwidth_kernel(
    const uint32_t *input,
    uint32_t *sink)
{
    const size_t block_offset =
        static_cast<size_t>(BlockThreads) *
        LoadUnroll *
        blockIdx.x;
    const size_t offset =
        (block_offset + threadIdx.x) % DataElements;
    const uint32_t *load_pointer = input + offset;
    uint32_t registers[LoadUnroll];

    #pragma unroll
    for (int i = 0; i < LoadUnroll; ++i)
    {
        ptx::ldg32_cg(
            registers[i],
            load_pointer + BlockThreads * i);
    }

    uint32_t sum = 0;
    #pragma unroll
    for (int i = 0; i < LoadUnroll; ++i)
    {
        sum += registers[i];
    }

    if (sum != 0)
    {
        *sink = sum;
    }
}

}  // namespace

int main()
{
    uint32_t *device_input = nullptr;
    uint32_t *device_sink = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

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

        if (kDataSizeBytes >= static_cast<size_t>(l2_cache_bytes))
        {
            throw std::runtime_error(
                "L2 working set must be smaller than the L2 cache");
        }

        constexpr size_t bytes_per_launch =
            static_cast<size_t>(kLoadCount) * sizeof(uint32_t);
        constexpr size_t benchmark_bytes =
            bytes_per_launch * kBenchmarkIterations;

        std::cout << "L2 cache bandwidth test\n\n";
        std::cout << "[Configuration]\n";
        std::cout << "  " << std::left << std::setw(24)
                  << "GPU" << properties.name << '\n';
        std::cout << "  " << std::left << std::setw(24)
                  << "Compute capability"
                  << properties.major << '.' << properties.minor << '\n';
        std::cout << "  " << std::left << std::setw(24)
                  << "L2 cache" << l2_cache_bytes << " bytes\n";
        std::cout << "  " << std::left << std::setw(24)
                  << "Working set" << kDataSizeBytes << " bytes\n";
        std::cout << "  " << std::left << std::setw(24)
                  << "Block threads" << kBlockThreads << '\n';
        std::cout << "  " << std::left << std::setw(24)
                  << "Loads per thread" << kLoadUnroll << '\n';
        std::cout << "  " << std::left << std::setw(24)
                  << "Grid blocks" << kGridBlocks << '\n';
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
            cudaMalloc(&device_input, kDataSizeBytes),
            "cudaMalloc(device_input)");
        check_cuda(
            cudaMalloc(&device_sink, sizeof(uint32_t)),
            "cudaMalloc(device_sink)");
        check_cuda(
            cudaMemset(device_input, 0, kDataSizeBytes),
            "cudaMemset(device_input)");
        check_cuda(
            cudaMemset(device_sink, 0, sizeof(uint32_t)),
            "cudaMemset(device_sink)");
        check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
        check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

        std::cout << "\n[Stage 2/3] Warming L2 cache\n";
        for (int i = 0; i < kWarmupIterations; ++i)
        {
            l2cache_bandwidth_kernel<
                kBlockThreads,
                kLoadUnroll,
                kDataElements>
                <<<kGridBlocks, kBlockThreads>>>(
                    device_input,
                    device_sink);
        }
        check_cuda(cudaGetLastError(), "warmup kernel launch");
        check_cuda(cudaDeviceSynchronize(), "warmup synchronization");

        std::cout << "\n[Stage 3/3] Measuring L2 read throughput\n";
        check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
        for (int i = 0; i < kBenchmarkIterations; ++i)
        {
            l2cache_bandwidth_kernel<
                kBlockThreads,
                kLoadUnroll,
                kDataElements>
                <<<kGridBlocks, kBlockThreads>>>(
                    device_input,
                    device_sink);
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
                "L2 bandwidth data validation failed");
        }

        const double elapsed_seconds =
            static_cast<double>(elapsed_ms) / 1000.0;
        const double bandwidth_gbps =
            static_cast<double>(benchmark_bytes) /
            elapsed_seconds /
            1.0e9;
        if (!std::isfinite(bandwidth_gbps) ||
            bandwidth_gbps <= 0.0)
        {
            throw std::runtime_error(
                "invalid L2 bandwidth result");
        }

        std::cout << "\n[Result]\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  " << std::left << std::setw(24)
                  << "Elapsed time" << elapsed_ms << " ms\n";
        std::cout << "  " << std::left << std::setw(24)
                  << "Aggregate L2 bandwidth"
                  << bandwidth_gbps << " GB/s\n";
        std::cout << "\n[SUCCESS] L2 cache bandwidth test passed\n";

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
