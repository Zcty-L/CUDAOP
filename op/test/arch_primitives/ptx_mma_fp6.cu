/***************************************************************************************************
 * SM120 FP6 / MXFP6 warp-level MMA PTX 学习样例
 *
 * 本文件把 feature/linear:op/linear/cute_gemm_fp6_mxfp6.cu 使用的两个 CuTe
 * MMA Atom 拆成 raw inline PTX，分别验证：
 *
 *   1. E3M2 x E3M2 -> FP32：普通 f8f6f4 MMA；
 *   2. E3M2 x E3M2 + UE8M0 block32 -> FP32：MXFP6 block-scaled MMA。
 *
 * 这些 wrapper 刻意留在学习文件中，不放入 op/ptx_utils.cuh。原 CuTe GEMM 的
 * copy(thread_global_*, fragment_*) 是 packed E3M2/scale 的 GMEM -> RMEM 搬运，
 * 没有 SMEM stage，也没有 cp.async；make_zip_tensor(data, scale) 与 cute::gemm
 * 最终分别落到本文件展示的两条 mma.sync 指令。
 *
 * 指令来源：
 *   - CUTLASS include/cute/arch/mma_sm120.hpp：
 *       SM120_16x8x32_TN<float_e3m2_t, float_e3m2_t, float>
 *       SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
 *           float_e3m2_t, float_e3m2_t, float, float_ue8m0_t, 32>
 *   - PTX ISA：
 *       https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma
 *       https://docs.nvidia.com/cuda/parallel-thread-execution/#block-scaling-for-mma-sync
 *
 * 本样例只面向 sm_120f：两条指令都是架构族特性，不能用普通 sm_120 目标替代。
 **************************************************************************************************/

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace
{

constexpr int kWarpThreads = 32;
constexpr int kM = 16;
constexpr int kN = 8;
constexpr int kK = 32;
constexpr int kAccumulatorsPerLane = 4;
constexpr int kOutputElements = kM * kN;
constexpr int kScaleBlockElements = 32;

// E3M2 是 6-bit 浮点格式：bit 5 为符号位、bits [4:2] 为 3-bit
// exponent、bits [1:0] 为 2-bit mantissa，exponent bias 为 3。
// +1.0 因而编码为 0b00'1100 = 0x0c。
//
// f8f6f4 / mxf8f6f4 MMA 的寄存器接口不是连续 6-bit 紧凑存储；每个 FP6
// 值占一个 8-bit container。有效值位于低 6 bit，最高 2 bit 是 padding，
// 本例显式将 padding 清零。因此一个 .b32 寄存器恰好容纳四个 E3M2 值。
constexpr uint8_t kE3m2One = 0x0CU;
constexpr uint32_t kPackedE3m2Ones = 0x0C0C0C0CU;

// UE8M0 只有 8-bit exponent，没有符号和 mantissa；其数值为
// 2^(raw - 127)，因此 raw=0x7f 表示 scale=1.0。
constexpr uint8_t kUe8m0One = 0x7FU;
constexpr float kExpectedValue = 32.0F;
constexpr float kTolerance = 1.0e-6F;

static_assert(
    kWarpThreads * kAccumulatorsPerLane == kOutputElements,
    "one warp must collectively own the complete 16x8 accumulator tile");
static_assert(
    kScaleBlockElements == kK,
    "scale_vec::1X supplies one block32 scale for this K=32 instruction");
static_assert(
    (kE3m2One & 0xC0U) == 0,
    "the two FP6 container padding bits must remain zero");

void check_cuda(
    cudaError_t status,
    const char *expression,
    const char *file,
    int line)
{
    if (status == cudaSuccess)
    {
        return;
    }

    throw std::runtime_error(
        std::string("CUDA error: ") + cudaGetErrorString(status) +
        ", expression=" + expression +
        ", location=" + file + ':' + std::to_string(line));
}

#define CUDA_CHECK(expression) \
    check_cuda((expression), #expression, __FILE__, __LINE__)

template <typename T>
class DeviceBuffer
{
public:
    explicit DeviceBuffer(size_t count)
        : count_(count)
    {
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&pointer_),
            count_ * sizeof(T)));
    }

    ~DeviceBuffer()
    {
        if (pointer_ != nullptr)
        {
            cudaFree(pointer_);
        }
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    T *get()
    {
        return pointer_;
    }

private:
    T *pointer_ = nullptr;
    size_t count_ = 0;
};

struct Fp32Fragment
{
    float value0;
    float value1;
    float value2;
    float value3;
};

// 普通 FP6 Atom，对应 CuTe：
//   SM120_16x8x32_TN<float_e3m2_t, float_e3m2_t, float>
__device__ __forceinline__ Fp32Fragment mma_m16n8k32_fp6(
    uint32_t a0,
    uint32_t a1,
    uint32_t a2,
    uint32_t a3,
    uint32_t b0,
    uint32_t b1,
    Fp32Fragment accumulator)
{
    Fp32Fragment result{};

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
    // 指令逐段解释：
    //
    // mma.sync.aligned
    //   warp-level 同步 MMA，整个 warp 的 32 个 lane 必须共同执行。
    // kind::f8f6f4
    //   选择 FP8/FP6/FP4 窄精度浮点数据通路；这里 A/B 类型均为 E3M2。
    // m16n8k32.row.col
    //   warp 计算 16x8 输出 tile，K=32；A 使用 row-major，B 使用
    //   column-major。原 GEMM 的 B[N,K] row-major 作为 B^T[K,N] 使用时，
    //   正好与这里的 column-major B 操作数一致。
    // f32.e3m2.e3m2.f32
    //   依次指定 D、A、B、C 类型；乘法输入为 FP6，C/D 为 FP32。
    //
    // 每个 lane 的寄存器契约：
    //   A：4 x .b32，共 16 个 8-bit FP6 container；
    //   B：2 x .b32，共  8 个 8-bit FP6 container；
    //   C：4 x .f32；D：4 x .f32。
    // 整个 warp 的 A/B 元素数分别是 32*16=16*32 与 32*8=32*8。
    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col."
        "f32.e3m2.e3m2.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(result.value0),
          "=f"(result.value1),
          "=f"(result.value2),
          "=f"(result.value3)
        : "r"(a0),
          "r"(a1),
          "r"(a2),
          "r"(a3),
          "r"(b0),
          "r"(b1),
          "f"(accumulator.value0),
          "f"(accumulator.value1),
          "f"(accumulator.value2),
          "f"(accumulator.value3));
#endif

    return result;
}

// MXFP6 Atom，对应 CuTe：
//   SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
//       float_e3m2_t, float_e3m2_t, float, float_ue8m0_t, 32>
__device__ __forceinline__ Fp32Fragment mma_m16n8k32_mxfp6(
    uint32_t a0,
    uint32_t a1,
    uint32_t a2,
    uint32_t a3,
    uint32_t b0,
    uint32_t b1,
    Fp32Fragment accumulator,
    uint32_t scale_a_data,
    uint32_t scale_b_data)
{
    Fp32Fragment result{};

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
    // 相比普通 FP6 指令，MXFP6 增加：
    //
    // kind::mxf8f6f4.block_scale
    //   计算 D=(A*scale_A)*(B*scale_B)+C；A/B 仍是 E3M2。
    // scale_vec::1X
    //   K=32 只使用一组 scale，即一个 UE8M0 scale 覆盖连续 32 个 K
    //   元素（block32）。这也是原 CuTe GEMM 的 kScaleVectorSize=32。
    // ue8m0
    //   指定 A/B scale 的格式。数据寄存器仍以 .b32 传入，但 1X 模式
    //   只选择其中一个 byte。
    //
    // scale operand 的 PTX 形式为：
    //   {scale-data}, {byte-id, thread-id}
    // byte-id 从 .b32 scale-data 中选择 byte；thread-id 选择 quad 中真正
    // 提供 scale 的 lane。CUTLASS SM120 wrapper 对 1X 固定使用四个 selector=0。
    // 本例让每个 lane 都携带相同的 0x7f，故被选择的 lane 为整个 tile 提供
    // scale=1；这样 block-scaled 结果应与普通 FP6 结果完全相同。
    constexpr uint16_t byte_id_a = 0;
    constexpr uint16_t thread_id_a = 0;
    constexpr uint16_t byte_id_b = 0;
    constexpr uint16_t thread_id_b = 0;

    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
        "m16n8k32.row.col.f32.e3m2.e3m2.f32.ue8m0 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13}, "
        "{%14}, {%15, %16}, {%17}, {%18, %19};\n"
        : "=f"(result.value0),
          "=f"(result.value1),
          "=f"(result.value2),
          "=f"(result.value3)
        : "r"(a0),
          "r"(a1),
          "r"(a2),
          "r"(a3),
          "r"(b0),
          "r"(b1),
          "f"(accumulator.value0),
          "f"(accumulator.value1),
          "f"(accumulator.value2),
          "f"(accumulator.value3),
          "r"(scale_a_data),
          "h"(byte_id_a),
          "h"(thread_id_a),
          "r"(scale_b_data),
          "h"(byte_id_b),
          "h"(thread_id_b));
#endif

    return result;
}

__device__ __forceinline__ void store_accumulator_tile(
    float *output,
    Fp32Fragment fragment)
{
    // m16n8 的 C/D fragment 坐标映射：每四个 lane 组成一个 quad。
    // d0/d1 对应 row=lane/4，d2/d3 对应 row=lane/4+8；列坐标为
    // (lane%4)*2+{0,1}。32*4 个寄存器正好还原完整的 16x8 tile。
    const int lane = static_cast<int>(threadIdx.x);
    const int row0 = lane >> 2;
    const int row1 = row0 + 8;
    const int column0 = (lane & 3) * 2;
    const int column1 = column0 + 1;

    output[row0 * kN + column0] = fragment.value0;
    output[row0 * kN + column1] = fragment.value1;
    output[row1 * kN + column0] = fragment.value2;
    output[row1 * kN + column1] = fragment.value3;
}

__global__ void ptx_mma_fp6_kernel(
    float *fp6_output,
    float *mxfp6_output)
{
    // 每个 8-bit container 都编码 +1.0，A/B 的所有逻辑元素因此均为 1。
    // fragment 直接在寄存器中构造，用来隔离 MMA，不掺入任何 load/copy 指令。
    const uint32_t packed_ones = kPackedE3m2Ones;
    const Fp32Fragment zero_accumulator = {0.0F, 0.0F, 0.0F, 0.0F};

    const Fp32Fragment fp6_result = mma_m16n8k32_fp6(
        packed_ones,
        packed_ones,
        packed_ones,
        packed_ones,
        packed_ones,
        packed_ones,
        zero_accumulator);

    // 1X 只读取被 selector 指定的一个 scale byte。高三个 byte 显式为 0，
    // 使学习样例能观察到真正使用的是低 byte 0x7f，而不是重复填充值。
    const uint32_t scale_one = static_cast<uint32_t>(kUe8m0One);
    const Fp32Fragment mxfp6_result = mma_m16n8k32_mxfp6(
        packed_ones,
        packed_ones,
        packed_ones,
        packed_ones,
        packed_ones,
        packed_ones,
        zero_accumulator,
        scale_one,
        scale_one);

    store_accumulator_tile(fp6_output, fp6_result);
    store_accumulator_tile(mxfp6_output, mxfp6_result);
}

float decode_e3m2(uint8_t bits)
{
    const uint8_t value_bits = bits & 0x3FU;
    const bool negative = (value_bits & 0x20U) != 0;
    const int exponent = static_cast<int>((value_bits >> 2) & 0x07U);
    const int mantissa = static_cast<int>(value_bits & 0x03U);

    float magnitude = 0.0F;
    if (exponent == 0)
    {
        magnitude = std::ldexp(static_cast<float>(mantissa) / 4.0F, -2);
    }
    else
    {
        magnitude = std::ldexp(
            1.0F + static_cast<float>(mantissa) / 4.0F,
            exponent - 3);
    }
    return negative ? -magnitude : magnitude;
}

float decode_ue8m0(uint8_t bits)
{
    if (bits == 0xFFU)
    {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return std::ldexp(1.0F, static_cast<int>(bits) - 127);
}

struct VerificationResult
{
    int mismatch_count = 0;
    float maximum_error = 0.0F;
};

VerificationResult verify_tile(
    const std::array<float, kOutputElements> &output,
    float reference)
{
    VerificationResult result{};
    for (float value : output)
    {
        const float error = std::abs(value - reference);
        result.maximum_error = std::max(result.maximum_error, error);
        result.mismatch_count += error > kTolerance ? 1 : 0;
    }
    return result;
}

} // namespace

int main()
{
    try
    {
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));

        cudaDeviceProp device_property{};
        CUDA_CHECK(cudaGetDeviceProperties(&device_property, device));
        if (device_property.major != 12 || device_property.minor != 0)
        {
            throw std::runtime_error(
                "This learning test requires an SM120 GPU and sm_120f code generation");
        }

        const float e3m2_one = decode_e3m2(kE3m2One);
        const float ue8m0_one = decode_ue8m0(kUe8m0One);
        const float fp6_reference =
            static_cast<float>(kK) * e3m2_one * e3m2_one;
        const float mxfp6_reference =
            static_cast<float>(kK) *
            (e3m2_one * ue8m0_one) *
            (e3m2_one * ue8m0_one);
        if (fp6_reference != kExpectedValue ||
            mxfp6_reference != kExpectedValue)
        {
            throw std::runtime_error(
                "the decoded all-ones inputs must produce the expected value 32");
        }

        std::cout << "\n[Configuration]\n"
                  << "  " << std::left << std::setw(30)
                  << "GPU" << device_property.name << '\n'
                  << "  " << std::left << std::setw(30)
                  << "MMA tile" << "16x8x32, one warp" << '\n'
                  << "  " << std::left << std::setw(30)
                  << "A/B / accumulator" << "E3M2 / FP32" << '\n'
                  << "  " << std::left << std::setw(30)
                  << "FP6 container" << "8 bits: low 6 data + high 2 padding" << '\n'
                  << "  " << std::left << std::setw(30)
                  << "E3M2 +1 raw" << "0x" << std::hex
                  << static_cast<int>(kE3m2One) << '\n'
                  << "  " << std::left << std::setw(30)
                  << "Packed .b32" << "0x" << kPackedE3m2Ones << '\n'
                  << "  " << std::left << std::setw(30)
                  << "MX scale / raw" << "UE8M0 1.0 / 0x"
                  << static_cast<int>(kUe8m0One) << std::dec << '\n'
                  << "  " << std::left << std::setw(30)
                  << "MX scale vector" << "1X, one scale per 32 K elements" << '\n'
                  << "  " << std::left << std::setw(30)
                  << "Data path" << "RMEM -> MMA -> RMEM; no cp.async" << "\n\n";

        DeviceBuffer<float> device_fp6_output(kOutputElements);
        DeviceBuffer<float> device_mxfp6_output(kOutputElements);

        std::cout << "[Stage 1] Execute ordinary FP6 and MXFP6 raw PTX\n";
        ptx_mma_fp6_kernel<<<1, kWarpThreads>>>(
            device_fp6_output.get(),
            device_mxfp6_output.get());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::array<float, kOutputElements> host_fp6_output{};
        std::array<float, kOutputElements> host_mxfp6_output{};
        CUDA_CHECK(cudaMemcpy(
            host_fp6_output.data(),
            device_fp6_output.get(),
            sizeof(host_fp6_output),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            host_mxfp6_output.data(),
            device_mxfp6_output.get(),
            sizeof(host_mxfp6_output),
            cudaMemcpyDeviceToHost));

        std::cout << "\n[Stage 2] Decode formats on CPU and verify all accumulators\n";
        const VerificationResult fp6_verification =
            verify_tile(host_fp6_output, fp6_reference);
        const VerificationResult mxfp6_verification =
            verify_tile(host_mxfp6_output, mxfp6_reference);

        std::cout << "  " << std::left << std::setw(30)
                  << "Decoded E3M2 / UE8M0" << e3m2_one << " / " << ue8m0_one << '\n'
                  << "  " << std::left << std::setw(30)
                  << "FP6 CPU / first GPU" << fp6_reference << " / "
                  << host_fp6_output.front() << '\n'
                  << "  " << std::left << std::setw(30)
                  << "FP6 mismatch / max error" << fp6_verification.mismatch_count
                  << " / " << fp6_verification.maximum_error << '\n'
                  << "  " << std::left << std::setw(30)
                  << "MXFP6 CPU / first GPU" << mxfp6_reference << " / "
                  << host_mxfp6_output.front() << '\n'
                  << "  " << std::left << std::setw(30)
                  << "MXFP6 mismatch / max error" << mxfp6_verification.mismatch_count
                  << " / " << mxfp6_verification.maximum_error << '\n'
                  << "  " << std::left << std::setw(30)
                  << "Verified accumulators" << kOutputElements << " + "
                  << kOutputElements << '\n';

        if (fp6_verification.mismatch_count != 0 ||
            mxfp6_verification.mismatch_count != 0)
        {
            throw std::runtime_error(
                "FP6/MXFP6 MMA output does not match the CPU reference");
        }

        std::cout << "\n[SUCCESS] SM120 FP6 and MXFP6 MMA PTX verified\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "\n[FAILED] " << error.what() << '\n';
        return 1;
    }
}
