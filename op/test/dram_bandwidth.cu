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

constexpr size_t kMemoryOffsetBytes = size_t{16} << 20;
constexpr int kBenchmarkIterations = 100;
constexpr int kBlockThreads = 128;
constexpr int kVectorUnroll = 1;
constexpr size_t kMinimumSizeBytes = size_t{4} << 20;
constexpr size_t kMaximumSizeBytes = size_t{1} << 30;

static_assert(
    kMemoryOffsetBytes % sizeof(uint4) == 0,
    "invalid DRAM benchmark memory offset");

void check_cuda(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <int BlockThreads, int VectorUnroll>
__global__ void read_kernel(const void *input, void *sink)
{
    const size_t index =
        static_cast<size_t>(blockIdx.x) *
        BlockThreads *
        VectorUnroll +
        threadIdx.x;

    const uint4 *load_pointer =
        reinterpret_cast<const uint4 *>(input) + index;
    uint4 registers[VectorUnroll];

    #pragma unroll
    for (int i = 0; i < VectorUnroll; ++i)
    {
        ptx::ldg128_cs(
            registers[i].x,
            registers[i].y,
            registers[i].z,
            registers[i].w,
            load_pointer + i * BlockThreads);
    }

    #pragma unroll
    for (int i = 0; i < VectorUnroll; ++i)
    {
        if (registers[i].x != 0)
        {
            ptx::stg128_cs(
                registers[i].x,
                registers[i].y,
                registers[i].z,
                registers[i].w,
                reinterpret_cast<uint4 *>(sink) + i);
        }
    }
}

template <int BlockThreads, int VectorUnroll>
__global__ void write_kernel(void *output)
{
    const size_t index =
        static_cast<size_t>(blockIdx.x) *
        BlockThreads *
        VectorUnroll +
        threadIdx.x;

    uint4 *store_pointer =
        reinterpret_cast<uint4 *>(output) + index;
    const uint4 registers = make_uint4(0, 0, 0, 0);

    #pragma unroll
    for (int i = 0; i < VectorUnroll; ++i)
    {
        ptx::stg128_cs(
            registers.x,
            registers.y,
            registers.z,
            registers.w,
            store_pointer + i * BlockThreads);
    }
}

template <int BlockThreads, int VectorUnroll>
__global__ void copy_kernel(const void *input, void *output)
{
    const size_t index =
        static_cast<size_t>(blockIdx.x) *
        BlockThreads *
        VectorUnroll +
        threadIdx.x;

    const uint4 *load_pointer =
        reinterpret_cast<const uint4 *>(input) + index;
    uint4 *store_pointer =
        reinterpret_cast<uint4 *>(output) + index;
    uint4 registers[VectorUnroll];

    #pragma unroll
    for (int i = 0; i < VectorUnroll; ++i)
    {
        ptx::ldg128_cs(
            registers[i].x,
            registers[i].y,
            registers[i].z,
            registers[i].w,
            load_pointer + i * BlockThreads);
    }

    #pragma unroll
    for (int i = 0; i < VectorUnroll; ++i)
    {
        ptx::stg128_cs(
            registers[i].x,
            registers[i].y,
            registers[i].z,
            registers[i].w,
            store_pointer + i * BlockThreads);
    }
}

struct BandwidthResult
{
    double read_gbps;
    double write_gbps;
    double copy_gbps;
};

double calculate_gbps(size_t bytes, float elapsed_ms)
{
    const double elapsed_seconds =
        static_cast<double>(elapsed_ms) / 1000.0;
    return static_cast<double>(bytes) /
        elapsed_seconds /
        1.0e9;
}

BandwidthResult benchmark(size_t size_bytes)
{
    const size_t vector_count = size_bytes / sizeof(uint4);
    const size_t grid_size =
        vector_count /
        (kBlockThreads * kVectorUnroll);
    const size_t workspace_bytes =
        size_bytes +
        kMemoryOffsetBytes * (kBenchmarkIterations - 1);
    const size_t total_transfer_bytes =
        size_bytes * kBenchmarkIterations;

    char *workspace = nullptr;
    uint4 *sink = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    check_cuda(
        cudaMalloc(&workspace, workspace_bytes),
        "cudaMalloc(workspace)");
    check_cuda(cudaMalloc(&sink, sizeof(uint4)), "cudaMalloc(sink)");
    check_cuda(
        cudaMemset(workspace, 0, workspace_bytes),
        "cudaMemset(workspace)");
    check_cuda(cudaMemset(sink, 0, sizeof(uint4)), "cudaMemset(sink)");
    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    read_kernel<kBlockThreads, kVectorUnroll>
        <<<grid_size, kBlockThreads>>>(workspace, sink);
    write_kernel<kBlockThreads, kVectorUnroll>
        <<<grid_size, kBlockThreads>>>(workspace);
    copy_kernel<kBlockThreads, kVectorUnroll>
        <<<grid_size / 2, kBlockThreads>>>(
            workspace,
            workspace + size_bytes / 2);
    check_cuda(cudaGetLastError(), "warmup kernel launch");
    check_cuda(cudaDeviceSynchronize(), "warmup synchronization");

    float elapsed_ms = 0.0F;

    check_cuda(cudaEventRecord(start), "cudaEventRecord(read start)");
    for (int i = kBenchmarkIterations - 1; i >= 0; --i)
    {
        read_kernel<kBlockThreads, kVectorUnroll>
            <<<grid_size, kBlockThreads>>>(
                workspace + i * kMemoryOffsetBytes,
                sink);
    }
    check_cuda(cudaGetLastError(), "read kernel launch");
    check_cuda(cudaEventRecord(stop), "cudaEventRecord(read stop)");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(read)");
    check_cuda(
        cudaEventElapsedTime(&elapsed_ms, start, stop),
        "cudaEventElapsedTime(read)");
    const double read_gbps =
        calculate_gbps(total_transfer_bytes, elapsed_ms);

    check_cuda(cudaEventRecord(start), "cudaEventRecord(write start)");
    for (int i = kBenchmarkIterations - 1; i >= 0; --i)
    {
        write_kernel<kBlockThreads, kVectorUnroll>
            <<<grid_size, kBlockThreads>>>(
                workspace + i * kMemoryOffsetBytes);
    }
    check_cuda(cudaGetLastError(), "write kernel launch");
    check_cuda(cudaEventRecord(stop), "cudaEventRecord(write stop)");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(write)");
    check_cuda(
        cudaEventElapsedTime(&elapsed_ms, start, stop),
        "cudaEventElapsedTime(write)");
    const double write_gbps =
        calculate_gbps(total_transfer_bytes, elapsed_ms);

    check_cuda(cudaEventRecord(start), "cudaEventRecord(copy start)");
    for (int i = kBenchmarkIterations - 1; i >= 0; --i)
    {
        char *input =
            workspace + i * kMemoryOffsetBytes;
        copy_kernel<kBlockThreads, kVectorUnroll>
            <<<grid_size / 2, kBlockThreads>>>(
                input,
                input + size_bytes / 2);
    }
    check_cuda(cudaGetLastError(), "copy kernel launch");
    check_cuda(cudaEventRecord(stop), "cudaEventRecord(copy stop)");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(copy)");
    check_cuda(
        cudaEventElapsedTime(&elapsed_ms, start, stop),
        "cudaEventElapsedTime(copy)");
    const double copy_gbps =
        calculate_gbps(total_transfer_bytes, elapsed_ms);

    if (!std::isfinite(read_gbps) ||
        !std::isfinite(write_gbps) ||
        !std::isfinite(copy_gbps) ||
        read_gbps <= 0.0 ||
        write_gbps <= 0.0 ||
        copy_gbps <= 0.0)
    {
        throw std::runtime_error("invalid DRAM bandwidth result");
    }

    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
    check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    check_cuda(cudaFree(sink), "cudaFree(sink)");
    check_cuda(cudaFree(workspace), "cudaFree(workspace)");

    return {read_gbps, write_gbps, copy_gbps};
}

}  // namespace

int main()
{
    try
    {
        int device = 0;
        cudaDeviceProp properties{};
        check_cuda(cudaGetDevice(&device), "cudaGetDevice");
        check_cuda(
            cudaGetDeviceProperties(&properties, device),
            "cudaGetDeviceProperties");

        std::cout << "DRAM bandwidth test\n\n";
        std::cout << "[Configuration]\n";
        std::cout << "  " << std::left << std::setw(24)
                  << "GPU" << properties.name << '\n';
        std::cout << "  " << std::left << std::setw(24)
                  << "Compute capability"
                  << properties.major << '.' << properties.minor << '\n';
        std::cout << "  " << std::left << std::setw(24)
                  << "Block threads" << kBlockThreads << '\n';
        std::cout << "  " << std::left << std::setw(24)
                  << "Vector width" << sizeof(uint4) << " bytes\n";
        std::cout << "  " << std::left << std::setw(24)
                  << "Benchmark iterations" << kBenchmarkIterations << '\n';
        std::cout << "  " << std::left << std::setw(24)
                  << "Iteration offset"
                  << kMemoryOffsetBytes / (size_t{1} << 20)
                  << " MiB\n";
        std::cout
            << "  Copy bandwidth counts read and write traffic together.\n";

        std::cout << "\n[Stage 1/2] Running size sweep\n\n";
        std::cout << std::right
                  << std::setw(12) << "Size (MiB)"
                  << std::setw(16) << "Read (GB/s)"
                  << std::setw(16) << "Write (GB/s)"
                  << std::setw(16) << "Copy (GB/s)"
                  << '\n';

        BandwidthResult largest_working_set{};
        for (
            size_t size_bytes = kMinimumSizeBytes;
            size_bytes <= kMaximumSizeBytes;
            size_bytes *= 2)
        {
            const BandwidthResult result = benchmark(size_bytes);
            largest_working_set = result;

            std::cout << std::fixed << std::setprecision(2)
                      << std::setw(12)
                      << size_bytes / (size_t{1} << 20)
                      << std::setw(16) << result.read_gbps
                      << std::setw(16) << result.write_gbps
                      << std::setw(16) << result.copy_gbps
                      << '\n';
        }

        std::cout
            << "\n[Stage 2/2] Reporting 1 GiB working-set bandwidth\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  " << std::left << std::setw(24)
                  << "DRAM-scale read"
                  << largest_working_set.read_gbps << " GB/s\n";
        std::cout << "  " << std::left << std::setw(24)
                  << "DRAM-scale write"
                  << largest_working_set.write_gbps << " GB/s\n";
        std::cout << "  " << std::left << std::setw(24)
                  << "DRAM-scale copy"
                  << largest_working_set.copy_gbps << " GB/s\n";
        std::cout << "\n[SUCCESS] DRAM bandwidth test passed\n";
    }
    catch (const std::exception &error)
    {
        std::cout << "\n[FAILED] " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
