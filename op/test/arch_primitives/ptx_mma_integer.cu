#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace
{

// 本文件将 feature/linear 中由 CuTe 封装的整数 MMA Atom 单独拆开：
//
//   op/linear/cute_gemm_int8_int32.cu
//     -> SM80_16x8x32_S32S8S8S32_TN
//     -> mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
//
//   op/linear/cute_gemm_int4_int32.cu
//     -> SM80_16x8x32_S32S4S4S32_TN
//     -> mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32
//
// CuTe GEMM 会在一个 CTA 中组合多个 MMA Atom，并负责 GMEM/SMEM/RMEM
// layout 与 thread slice；这里刻意只启动一个 warp，并由每个 lane 直接构造
// 自己的寄存器 fragment，以便观察一条 raw inline PTX 的完整输入和输出。
// 学习 wrapper 因此保留在本文件，不放入生产代码共用的 op/ptx_utils.cuh。
//
// 官方来源：
// https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-fragments-for-mma-m16n8k32
// https://docs.nvidia.com/cuda/parallel-thread-execution/#multiply-and-accumulate-instruction-mma

constexpr int kWarpSize = 32;
constexpr int kM = 16;
constexpr int kN = 8;
constexpr int kK = 32;
constexpr int kOutputElements = kM * kN;

void check_cuda(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

// 使用小的正负整数，既覆盖 signed 编码，又确保乘加结果远离 INT32 溢出。
// 坐标相关的值比全 1 输入更容易发现 fragment 次序、A/B layout 或 nibble
// 高低位放反等错误。
__host__ __device__ int32_t matrix_a_s8_value(int row, int column)
{
    return ((row * 3 + column * 5 + 1) % 11) - 5;
}

__host__ __device__ int32_t matrix_b_s8_value(int row, int column)
{
    return ((row * 7 + column * 2 + 3) % 9) - 4;
}

__host__ __device__ int32_t matrix_a_s4_value(int row, int column)
{
    return ((row * 5 + column * 3 + 2) % 15) - 7;
}

__host__ __device__ int32_t matrix_b_s4_value(int row, int column)
{
    return ((row * 2 + column * 7 + 4) % 15) - 7;
}

// 非零 C 显式验证 D=A*B+C 中的累加项，而不只是 A*B。
__host__ __device__ int32_t accumulator_value(int row, int column)
{
    return row * 13 - column * 5 - 37;
}

// S8 的每个 MMA multiplicand 寄存器是一个无类型的 .b32 容器。
// PTX 按低位到高位依次将 bit[7:0]、bit[15:8]、bit[23:16]、bit[31:24]
// 解释为四个 signed 8-bit 元素。负数以二进制补码保留其原始 byte bits；
// 不是先把四个值扩展成四个 32-bit signed integer。
__device__ __forceinline__ uint32_t pack_s8_four(const int32_t *values)
{
    uint32_t packed = 0;
#pragma unroll
    for (int index = 0; index < 4; ++index)
    {
        const uint32_t byte = static_cast<uint32_t>(
            static_cast<uint8_t>(static_cast<int8_t>(values[index])));
        packed |= byte << (index * 8);
    }
    return packed;
}

// S4 同样用 .b32 承载 packed bits，但每个寄存器包含八个元素。
// bit[3:0] 是第 0 个元素，bit[7:4] 是第 1 个元素，以此类推；因此一个
// byte 内低 nibble 在前、高 nibble 在后。value & 0xf 保留 signed S4 的
// 4-bit 二进制补码，例如 -1 -> 0xf，-7 -> 0x9。
__device__ __forceinline__ uint32_t pack_s4_eight(const int32_t *values)
{
    uint32_t packed = 0;
#pragma unroll
    for (int index = 0; index < 8; ++index)
    {
        const uint32_t nibble =
            static_cast<uint32_t>(values[index]) & 0xfU;
        packed |= nibble << (index * 4);
    }
    return packed;
}

// m16n8k32.row.col 的 S8 A fragment：每 lane 有 16 个 S8，装入 4 个
// .b32；B fragment 每 lane 有 8 个 S8，装入 2 个 .b32。
//
// group_id = lane >> 2，thread_id = lane & 3。
// A 中元素 ai 的坐标：
//   row = group_id       for i in [0,3] or [8,11]
//         group_id + 8   otherwise
//   col = 4*thread_id + (i&3)       for i < 8
//         4*thread_id + (i&3) + 16  otherwise
// B 中元素 bi 的坐标：
//   row = 4*thread_id + (i&3)       for i < 4
//         4*thread_id + (i&3) + 16  otherwise
//   col = group_id
__device__ __forceinline__ void make_s8_fragments(
    int lane,
    uint32_t (&fragment_a)[4],
    uint32_t (&fragment_b)[2])
{
    const int group_id = lane >> 2;
    const int thread_id = lane & 3;
    int32_t elements_a[16];
    int32_t elements_b[8];

#pragma unroll
    for (int index = 0; index < 16; ++index)
    {
        const bool first_or_third_group =
            index < 4 || (index >= 8 && index < 12);
        const int row = group_id + (first_or_third_group ? 0 : 8);
        const int column = thread_id * 4 + (index & 3) +
                           (index >= 8 ? 16 : 0);
        elements_a[index] = matrix_a_s8_value(row, column);
    }

#pragma unroll
    for (int index = 0; index < 8; ++index)
    {
        const int row = thread_id * 4 + (index & 3) +
                        (index >= 4 ? 16 : 0);
        elements_b[index] = matrix_b_s8_value(row, group_id);
    }

#pragma unroll
    for (int register_index = 0; register_index < 4; ++register_index)
    {
        fragment_a[register_index] =
            pack_s8_four(elements_a + register_index * 4);
    }

#pragma unroll
    for (int register_index = 0; register_index < 2; ++register_index)
    {
        fragment_b[register_index] =
            pack_s8_four(elements_b + register_index * 4);
    }
}

// 同一 shape 的 S4 fragment 更紧凑：每 lane 的 A 有 16 个 S4，使用
// 2 个 .b32；B 有 8 个 S4，只使用 1 个 .b32。
//
// A 中元素 ai 的坐标：
//   row = group_id       for i < 8，否则 group_id + 8
//   col = 8*thread_id + (i&7)
// B 中元素 bi 的坐标：
//   row = 8*thread_id + i
//   col = group_id
__device__ __forceinline__ void make_s4_fragments(
    int lane,
    uint32_t (&fragment_a)[2],
    uint32_t (&fragment_b)[1])
{
    const int group_id = lane >> 2;
    const int thread_id = lane & 3;
    int32_t elements_a[16];
    int32_t elements_b[8];

#pragma unroll
    for (int index = 0; index < 16; ++index)
    {
        const int row = group_id + (index >= 8 ? 8 : 0);
        const int column = thread_id * 8 + (index & 7);
        elements_a[index] = matrix_a_s4_value(row, column);
    }

#pragma unroll
    for (int index = 0; index < 8; ++index)
    {
        const int row = thread_id * 8 + index;
        elements_b[index] = matrix_b_s4_value(row, group_id);
    }

    fragment_a[0] = pack_s4_eight(elements_a);
    fragment_a[1] = pack_s4_eight(elements_a + 8);
    fragment_b[0] = pack_s4_eight(elements_b);
}

// S8 与 S4 指令共享同一 C/D fragment 映射。每 lane 持有四个独立的
// signed INT32 accumulator 寄存器：
//
//   c0/d0 -> [group_id,     2*thread_id    ]
//   c1/d1 -> [group_id,     2*thread_id + 1]
//   c2/d2 -> [group_id + 8, 2*thread_id    ]
//   c3/d3 -> [group_id + 8, 2*thread_id + 1]
//
// 32 lanes * 4 accumulators = 16 * 8 个输出，恰好覆盖完整 D tile。
__device__ __forceinline__ void make_accumulators(
    int lane,
    int32_t (&fragment_c)[4])
{
    const int group_id = lane >> 2;
    const int thread_id = lane & 3;

#pragma unroll
    for (int index = 0; index < 4; ++index)
    {
        const int row = group_id + (index >= 2 ? 8 : 0);
        const int column = thread_id * 2 + (index & 1);
        fragment_c[index] = accumulator_value(row, column);
    }
}

__device__ __forceinline__ void store_accumulators(
    int lane,
    const int32_t (&fragment_d)[4],
    int32_t *output)
{
    const int group_id = lane >> 2;
    const int thread_id = lane & 3;

#pragma unroll
    for (int index = 0; index < 4; ++index)
    {
        const int row = group_id + (index >= 2 ? 8 : 0);
        const int column = thread_id * 2 + (index & 1);
        output[row * kN + column] = fragment_d[index];
    }
}

__device__ __forceinline__ void mma_s8_m16n8k32(
    const uint32_t (&fragment_a)[4],
    const uint32_t (&fragment_b)[2],
    const int32_t (&fragment_c)[4],
    int32_t (&fragment_d)[4])
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // 指令名称：mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
    // 来源：文件顶部 NVIDIA PTX ISA 的 mma 官方章节。
    // 用途：一个完整 warp 合作计算 D[16,8] = A[16,32] * B[32,8] + C。
    //
    // 每 lane 的寄存器操作数：
    //   D: 4 x .s32
    //   A: 4 x .b32，每个 .b32 内含 4 个 signed S8
    //   B: 2 x .b32，每个 .b32 内含 4 个 signed S8
    //   C: 4 x .s32
    //
    // 输出约束使用 =r，C 作为独立输入传入，因此精确表达 D=A*B+C。
    // .sync 要求 warp 共同执行；.aligned 要求所有 lane 执行相同指令，
    // 不能把此 wrapper 放在 warp-divergent 分支中只让部分 lane 调用。
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=r"(fragment_d[0]),
          "=r"(fragment_d[1]),
          "=r"(fragment_d[2]),
          "=r"(fragment_d[3])
        : "r"(fragment_a[0]),
          "r"(fragment_a[1]),
          "r"(fragment_a[2]),
          "r"(fragment_a[3]),
          "r"(fragment_b[0]),
          "r"(fragment_b[1]),
          "r"(fragment_c[0]),
          "r"(fragment_c[1]),
          "r"(fragment_c[2]),
          "r"(fragment_c[3]));
#else
#pragma unroll
    for (int index = 0; index < 4; ++index)
    {
        fragment_d[index] = 0;
    }
#endif
}

__device__ __forceinline__ void mma_s4_m16n8k32(
    const uint32_t (&fragment_a)[2],
    const uint32_t (&fragment_b)[1],
    const int32_t (&fragment_c)[4],
    int32_t (&fragment_d)[4])
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // 指令名称：mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32
    // 来源：文件顶部 NVIDIA PTX ISA 的 mma 官方章节。
    // 用途和输出 shape 与 S8 版本相同，但 packed S4 将输入寄存器减半。
    //
    // 每 lane 的寄存器操作数：
    //   D: 4 x .s32
    //   A: 2 x .b32，每个 .b32 内含 8 个 signed S4
    //   B: 1 x .b32，内含 8 个 signed S4
    //   C: 4 x .s32
    //
    // PTX 是虚拟 ISA，而 SASS 是具体 GPU 的机器指令。sm_80 ptxas 通常将
    // 这条 PTX 直接生成为一条 S4 IMMA；sm_120 ptxas 则可能将它降低为两条
    // S8 IMMA。后者属于 codegen 差异，并不改变这里的 signed S4 打包语义、
    // m16n8k32 逻辑 shape 或最终 INT32 精确结果。
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5}, "
        "{%6}, "
        "{%7, %8, %9, %10};"
        : "=r"(fragment_d[0]),
          "=r"(fragment_d[1]),
          "=r"(fragment_d[2]),
          "=r"(fragment_d[3])
        : "r"(fragment_a[0]),
          "r"(fragment_a[1]),
          "r"(fragment_b[0]),
          "r"(fragment_c[0]),
          "r"(fragment_c[1]),
          "r"(fragment_c[2]),
          "r"(fragment_c[3]));
#else
#pragma unroll
    for (int index = 0; index < 4; ++index)
    {
        fragment_d[index] = 0;
    }
#endif
}

__global__ void mma_integer_kernel(int32_t *output_s8, int32_t *output_s4)
{
    // launch 固定为一个 32-thread block。这个 uniform 检查不会导致 warp
    // 内部分歧；真正执行两条 mma.sync 时全部 32 个 lane 都会参与。
    if (blockIdx.x != 0 || blockDim.x != kWarpSize)
    {
        return;
    }

    const int lane = static_cast<int>(threadIdx.x);
    uint32_t fragment_a_s8[4];
    uint32_t fragment_b_s8[2];
    uint32_t fragment_a_s4[2];
    uint32_t fragment_b_s4[1];
    int32_t fragment_c[4];
    int32_t fragment_d_s8[4];
    int32_t fragment_d_s4[4];

    make_s8_fragments(lane, fragment_a_s8, fragment_b_s8);
    make_s4_fragments(lane, fragment_a_s4, fragment_b_s4);
    make_accumulators(lane, fragment_c);

    mma_s8_m16n8k32(
        fragment_a_s8,
        fragment_b_s8,
        fragment_c,
        fragment_d_s8);
    mma_s4_m16n8k32(
        fragment_a_s4,
        fragment_b_s4,
        fragment_c,
        fragment_d_s4);

    store_accumulators(lane, fragment_d_s8, output_s8);
    store_accumulators(lane, fragment_d_s4, output_s4);
}

enum class InputKind
{
    kS8,
    kS4
};

int32_t reference_value(InputKind kind, int row, int column)
{
    int64_t result = accumulator_value(row, column);
    for (int inner = 0; inner < kK; ++inner)
    {
        const int32_t value_a = kind == InputKind::kS8
            ? matrix_a_s8_value(row, inner)
            : matrix_a_s4_value(row, inner);
        const int32_t value_b = kind == InputKind::kS8
            ? matrix_b_s8_value(inner, column)
            : matrix_b_s4_value(inner, column);
        result += static_cast<int64_t>(value_a) * value_b;
    }
    return static_cast<int32_t>(result);
}

struct VerificationResult
{
    int mismatch_count = 0;
    int64_t maximum_absolute_error = 0;
    int64_t actual_checksum = 0;
    int64_t expected_checksum = 0;
    int first_mismatch_row = -1;
    int first_mismatch_column = -1;
    int32_t first_actual = 0;
    int32_t first_expected = 0;
};

VerificationResult verify(
    const std::vector<int32_t> &actual,
    InputKind kind)
{
    VerificationResult result;
    for (int row = 0; row < kM; ++row)
    {
        for (int column = 0; column < kN; ++column)
        {
            const int index = row * kN + column;
            const int32_t expected = reference_value(kind, row, column);
            const int64_t difference =
                static_cast<int64_t>(actual[index]) - expected;
            const int64_t absolute_error =
                difference < 0 ? -difference : difference;
            if (absolute_error > result.maximum_absolute_error)
            {
                result.maximum_absolute_error = absolute_error;
            }
            result.actual_checksum += actual[index];
            result.expected_checksum += expected;

            if (actual[index] != expected)
            {
                if (result.mismatch_count == 0)
                {
                    result.first_mismatch_row = row;
                    result.first_mismatch_column = column;
                    result.first_actual = actual[index];
                    result.first_expected = expected;
                }
                ++result.mismatch_count;
            }
        }
    }
    return result;
}

void show_verification(
    const char *name,
    const VerificationResult &result)
{
    std::cout << "  " << std::left << std::setw(16) << name
              << "mismatch=" << std::setw(4) << result.mismatch_count
              << " max_abs_error=" << std::setw(4)
              << result.maximum_absolute_error
              << " checksum=" << result.actual_checksum << '\n';

    if (result.mismatch_count != 0)
    {
        std::cout << "    首个错误：row=" << result.first_mismatch_row
                  << ", column=" << result.first_mismatch_column
                  << ", actual=" << result.first_actual
                  << ", expected=" << result.first_expected << '\n';
    }
}

} // namespace

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

        if (properties.major < 8)
        {
            throw std::runtime_error(
                "mma.sync m16n8k32 integer test requires SM80 or newer");
        }

        std::cout << "整数 Tensor Core raw PTX 学习测试\n\n";
        std::cout << "配置\n";
        std::cout << "  GPU                 : " << properties.name << '\n';
        std::cout << "  Compute Capability  : "
                  << properties.major << '.' << properties.minor << '\n';
        std::cout << "  Launch              : 1 block x 32 threads（单 warp）\n";
        std::cout << "  MMA shape           : m16n8k32, A row-major, B column-major\n";
        std::cout << "  S8 registers/lane   : A=4xb32, B=2xb32, C/D=4xs32\n";
        std::cout << "  S4 registers/lane   : A=2xb32, B=1xb32, C/D=4xs32\n";
        std::cout << "  Accumulation        : signed INT32, exact comparison\n\n";

        std::cout << "阶段 1/3：分配两个 16x8 INT32 输出 tile\n";
        int32_t *device_output_s8 = nullptr;
        int32_t *device_output_s4 = nullptr;
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void **>(&device_output_s8),
                kOutputElements * sizeof(int32_t)),
            "cudaMalloc output_s8");
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void **>(&device_output_s4),
                kOutputElements * sizeof(int32_t)),
            "cudaMalloc output_s4");
        check_cuda(
            cudaMemset(
                device_output_s8,
                0xa5,
                kOutputElements * sizeof(int32_t)),
            "cudaMemset output_s8");
        check_cuda(
            cudaMemset(
                device_output_s4,
                0xa5,
                kOutputElements * sizeof(int32_t)),
            "cudaMemset output_s4");

        std::cout << "\n阶段 2/3：执行 S8 IMMA 与 packed S4 IMMA\n";
        mma_integer_kernel<<<1, kWarpSize>>>(
            device_output_s8,
            device_output_s4);
        check_cuda(cudaGetLastError(), "mma_integer_kernel launch");
        check_cuda(cudaDeviceSynchronize(), "mma_integer_kernel synchronize");

        std::cout << "\n阶段 3/3：CPU 重建全部 128 个 accumulator 并精确比较\n";
        std::vector<int32_t> output_s8(kOutputElements);
        std::vector<int32_t> output_s4(kOutputElements);
        check_cuda(
            cudaMemcpy(
                output_s8.data(),
                device_output_s8,
                kOutputElements * sizeof(int32_t),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy output_s8");
        check_cuda(
            cudaMemcpy(
                output_s4.data(),
                device_output_s4,
                kOutputElements * sizeof(int32_t),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy output_s4");

        const VerificationResult s8_result =
            verify(output_s8, InputKind::kS8);
        const VerificationResult s4_result =
            verify(output_s4, InputKind::kS4);

        std::cout << "\n关键结果\n";
        show_verification("S8 x S8 -> S32", s8_result);
        show_verification("S4 x S4 -> S32", s4_result);

        check_cuda(cudaFree(device_output_s8), "cudaFree output_s8");
        check_cuda(cudaFree(device_output_s4), "cudaFree output_s4");

        if (s8_result.mismatch_count != 0 ||
            s4_result.mismatch_count != 0)
        {
            throw std::runtime_error(
                "raw integer MMA result differs from CPU reference");
        }

        std::cout << "\n[SUCCESS] 两条整数 mma.sync PTX 的全部 INT32 输出均精确匹配\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAILED] " << error.what() << '\n';
        return 1;
    }
}
