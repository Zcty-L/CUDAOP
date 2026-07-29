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

constexpr int kBlockThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarps = kBlockThreads / kWarpSize;
constexpr int kSharedBytesPerWarp = 2048;
constexpr int kStoredValuesPerThread = 4;
constexpr int kKernelVariants = 3;

// Profile the isolated store with:
// ncu --metrics \
// smsp__inst_executed_op_shared_st.sum,\
// l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,\
// l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
// --kernel-name \
// 'regex:smem_sts128(_pad(16|64))?_transactions_kernel' \
// ./build/smem_sts128_transactions

void check_cuda(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

__host__ __device__ float fragment_value(
    uint32_t thread,
    uint32_t row,
    uint32_t column)
{
    uint32_t value =
        thread * 0x9e3779b9U +
        row * 0x85ebca6bU +
        column * 0xc2b2ae35U +
        0x27d4eb2fU;
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;

    return static_cast<float>((value & 0xffffU) + 1U);
}

#define CUDAOP_FRAGMENT_ROW(thread, row) \
    { \
        fragment_value((thread), (row), 0), \
        fragment_value((thread), (row), 1), \
        fragment_value((thread), (row), 2), \
        fragment_value((thread), (row), 3), \
        fragment_value((thread), (row), 4), \
        fragment_value((thread), (row), 5), \
        fragment_value((thread), (row), 6), \
        fragment_value((thread), (row), 7) \
    }

__global__ __launch_bounds__(kBlockThreads, 1)
void smem_sts128_transactions_kernel(float *output)
{
    __shared__ __align__(16) char smem[
        kWarps * kSharedBytesPerWarp];

    const uint32_t thread = threadIdx.x;
    const uint32_t lane_id = thread % kWarpSize;
    const uint32_t warp_id = thread / kWarpSize;
    const uint32_t mma_tid_x = (lane_id / 2) % 8;
    const uint32_t mma_tid_y =
        (lane_id / 16) * 2 + (lane_id % 2);

    float C_frag[8][8] = {
        CUDAOP_FRAGMENT_ROW(thread, 0),
        CUDAOP_FRAGMENT_ROW(thread, 1),
        CUDAOP_FRAGMENT_ROW(thread, 2),
        CUDAOP_FRAGMENT_ROW(thread, 3),
        CUDAOP_FRAGMENT_ROW(thread, 4),
        CUDAOP_FRAGMENT_ROW(thread, 5),
        CUDAOP_FRAGMENT_ROW(thread, 6),
        CUDAOP_FRAGMENT_ROW(thread, 7)
    };

    const uint32_t C_sts_addr = ptx::smem_u32addr(
        reinterpret_cast<float4 *>(
            smem + warp_id * kSharedBytesPerWarp) +
        mma_tid_y * 4 * 8 +
        mma_tid_x);

    // This is the isolated i=0, j=0, p=0 instance of sgemm.cu:501.
    // 一个 warp 有 8 个 transactions。以四分之一 warp 来看，
    // 0、2、4、6 与 1、3、5、7 之间存在 2-way bank conflict，
    // 因此一共有 2 x 4 = 8 个 transactions。
    ptx::sts128(
        C_frag[0][0],
        C_frag[0][1],
        C_frag[0][2],
        C_frag[0][3],
        C_sts_addr);

    __syncthreads();

    float value0;
    float value1;
    float value2;
    float value3;
    ptx::lds128(
        value0,
        value1,
        value2,
        value3,
        C_sts_addr);

    reinterpret_cast<float4 *>(output)[thread] =
        make_float4(value0, value1, value2, value3);
}

template <uint32_t kPaddingBytes>
__device__ __forceinline__ void run_padded_sts128(
    float *output,
    char *smem)
{
    static_assert(
        kPaddingBytes % sizeof(float4) == 0,
        "padding must preserve float4 alignment");

    const uint32_t thread = threadIdx.x;
    const uint32_t lane_id = thread % kWarpSize;
    const uint32_t warp_id = thread / kWarpSize;
    const uint32_t mma_tid_x = (lane_id / 2) % 8;
    const uint32_t mma_tid_y =
        (lane_id / 16) * 2 + (lane_id % 2);

    float C_frag[8][8] = {
        CUDAOP_FRAGMENT_ROW(thread, 0),
        CUDAOP_FRAGMENT_ROW(thread, 1),
        CUDAOP_FRAGMENT_ROW(thread, 2),
        CUDAOP_FRAGMENT_ROW(thread, 3),
        CUDAOP_FRAGMENT_ROW(thread, 4),
        CUDAOP_FRAGMENT_ROW(thread, 5),
        CUDAOP_FRAGMENT_ROW(thread, 6),
        CUDAOP_FRAGMENT_ROW(thread, 7)
    };

    constexpr uint32_t row_stride_float4 =
        4 * 8 + kPaddingBytes / sizeof(float4);
    const uint32_t C_sts_addr = ptx::smem_u32addr(
        reinterpret_cast<float4 *>(
            smem + warp_id * kSharedBytesPerWarp) +
        mma_tid_y * row_stride_float4 +
        mma_tid_x);

    ptx::sts128(
        C_frag[0][0],
        C_frag[0][1],
        C_frag[0][2],
        C_frag[0][3],
        C_sts_addr);

    __syncthreads();

    float value0;
    float value1;
    float value2;
    float value3;
    ptx::lds128(
        value0,
        value1,
        value2,
        value3,
        C_sts_addr);

    reinterpret_cast<float4 *>(output)[thread] =
        make_float4(value0, value1, value2, value3);
}

__global__ __launch_bounds__(kBlockThreads, 1)
void smem_sts128_pad16_transactions_kernel(float *output)
{
    __shared__ __align__(16) char smem[
        kWarps * kSharedBytesPerWarp];

    run_padded_sts128<16>(output, smem);
}

__global__ __launch_bounds__(kBlockThreads, 1)
void smem_sts128_pad64_transactions_kernel(float *output)
{
    __shared__ __align__(16) char smem[
        kWarps * kSharedBytesPerWarp];

    run_padded_sts128<64>(output, smem);
}

#undef CUDAOP_FRAGMENT_ROW

}  // namespace

int main()
{
    float *device_output = nullptr;

    try
    {
        int device = 0;
        cudaDeviceProp properties{};
        check_cuda(cudaGetDevice(&device), "cudaGetDevice");
        check_cuda(
            cudaGetDeviceProperties(&properties, device),
            "cudaGetDeviceProperties");

        constexpr size_t values_per_kernel =
            kBlockThreads * kStoredValuesPerThread;
        constexpr size_t output_values =
            kKernelVariants * values_per_kernel;
        constexpr size_t output_bytes =
            output_values * sizeof(float);

        std::cout << "Shared memory STS.128 transaction test\n\n";
        std::cout << "[Configuration]\n";
        std::cout << "  " << std::left << std::setw(28)
                  << "GPU" << properties.name << '\n';
        std::cout << "  " << std::left << std::setw(28)
                  << "Compute capability"
                  << properties.major << '.'
                  << properties.minor << '\n';
        std::cout << "  " << std::left << std::setw(28)
                  << "Threads per block" << kBlockThreads << '\n';
        std::cout << "  " << std::left << std::setw(28)
                  << "Warps per block" << kWarps << '\n';
        std::cout << "  " << std::left << std::setw(28)
                  << "STS.128 per thread" << 1 << '\n';
        std::cout << "  " << std::left << std::setw(28)
                  << "Bytes per thread" << sizeof(float4) << '\n';
        std::cout << "  " << std::left << std::setw(28)
                  << "Expected warp requests" << kWarps << "\n\n";

        std::cout << "[Stage 1/3] Allocating output buffer\n";
        check_cuda(
            cudaMalloc(&device_output, output_bytes),
            "cudaMalloc(device_output)");

        std::cout << "\n[Stage 2/3] Launching isolated STS.128 kernels\n";
        std::cout << "  Baseline: 0-byte padding\n";
        smem_sts128_transactions_kernel<<<1, kBlockThreads>>>(
            device_output);
        check_cuda(cudaGetLastError(), "baseline kernel launch");

        std::cout << "  Padded: 16-byte padding per mma_tid_y\n";
        smem_sts128_pad16_transactions_kernel<<<1, kBlockThreads>>>(
            device_output + values_per_kernel);
        check_cuda(cudaGetLastError(), "pad-16 kernel launch");

        std::cout << "  Padded: 64-byte padding per mma_tid_y\n";
        smem_sts128_pad64_transactions_kernel<<<1, kBlockThreads>>>(
            device_output + 2 * values_per_kernel);
        check_cuda(cudaGetLastError(), "pad-64 kernel launch");
        check_cuda(
            cudaDeviceSynchronize(),
            "kernel synchronization");

        std::cout << "\n[Stage 3/3] Validating shared-memory data\n";
        std::array<float, output_values> host_output{};
        check_cuda(
            cudaMemcpy(
                host_output.data(),
                device_output,
                output_bytes,
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(host_output)");

        bool output_valid = true;
        for (int variant = 0;
             variant < kKernelVariants;
             ++variant)
        {
            for (int thread = 0; thread < kBlockThreads; ++thread)
            {
                for (int column = 0;
                     column < kStoredValuesPerThread;
                     ++column)
                {
                    const size_t index =
                        variant * values_per_kernel +
                        thread * kStoredValuesPerThread +
                        column;
                    const float expected = fragment_value(
                        thread,
                        0,
                        column);
                    output_valid &=
                        host_output[index] == expected;
                }
            }
        }
        if (!output_valid)
        {
            throw std::runtime_error(
                "shared-memory STS.128 validation failed");
        }

        std::cout << "\n[Result]\n";
        std::cout << "  " << std::left << std::setw(28)
                  << "Validated values" << output_values << '\n';
        std::cout << "  " << std::left << std::setw(28)
                  << "Kernel launches" << kKernelVariants << '\n';
        std::cout
            << "\n[SUCCESS] Shared memory STS.128 test passed\n";

        check_cuda(
            cudaFree(device_output),
            "cudaFree(device_output)");
        device_output = nullptr;
    }
    catch (const std::exception &error)
    {
        if (device_output != nullptr)
        {
            cudaFree(device_output);
        }
        std::cout << "\n[FAILED] " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
