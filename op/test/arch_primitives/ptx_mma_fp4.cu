/***************************************************************************************************
 * SM120 NVFP4 warp-level MMA PTX 学习样例
 *
 * 本文件刻意把一条 block-scaled FP4 MMA 从 CuTe GEMM 中拆出来，便于直接观察：
 *   E2M1 nibble 打包 -> UE4M3 block scale 打包 -> mma.sync -> FP32 accumulator -> FP16 输出。
 * 这里的 inline PTX 仅作为学习参考，因此不放入 op/ptx_utils.cuh。
 *
 * PTX 官方文档：
 *   https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma
 *   https://docs.nvidia.com/cuda/parallel-thread-execution/#block-scaling-for-mma-sync
 *
 * 指令来源与 CuTe 映射：
 *   1. CUTLASS include/cute/arch/mma_sm120.hpp 中
 *      SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
 *          float_e2m1_t, float_e2m1_t, float, float_ue4m3_t, 16>；
 *   2. feature/linear:op/linear/cute_gemm_fp4_fp16.cu 中的 mma_atom 正是上述 Atom；
 *   3. 原 GEMM 的 copy(thread_global_a/b, fragment_a/b) 是 GMEM -> RMEM，随后
 *      make_zip_tensor(data, scale) 和 cute::gemm 最终落到本文件隔离的 mma.sync。
 *
 * 特别说明：cute_gemm_fp4_fp16.cu 本身没有 cp.async，也没有 SMEM stage。
 * 它直接把 packed FP4 从 GMEM 搬进 MMA 寄存器 fragment。因此本文件没有可拆出的
 * cp.async，只隔离并验证实际使用的 MMA 指令。
 **************************************************************************************************/

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace
{

constexpr int kWarpThreads = 32;
constexpr int kM = 16;
constexpr int kN = 8;
constexpr int kK = 64;
constexpr int kScaleVectorCount = 4;
constexpr int kScaleBlockElements = kK / kScaleVectorCount;
constexpr int kOutputElements = kM * kN;

// E2M1 的 4-bit 编码：正数 0x0..0x7 依次表示
// {0, 0.5, 1, 1.5, 2, 3, 4, 6}，bit 3 是符号位。
// 这里 A=+1.5，B=-0.5，选用非零且带符号的值来验证 nibble 解码。
constexpr uint8_t kE2m1A = 0x3;
constexpr uint8_t kE2m1B = 0x9;

// UE4M3 无符号 scale 的指数 bias 为 7。下面四个 byte 分别编码：
// scale A = {0.5, 1, 2, 4}，scale B = {1, 2, 0.5, 4}。
// 每个 scale 覆盖连续 16 个 K 元素，四个 byte 正好覆盖 K=64。
constexpr std::array<uint8_t, kScaleVectorCount> kScaleA = {
    0x30, 0x38, 0x40, 0x48};
constexpr std::array<uint8_t, kScaleVectorCount> kScaleB = {
    0x38, 0x40, 0x30, 0x48};
constexpr uint32_t kPackedScaleA = 0x48403830U;
constexpr uint32_t kPackedScaleB = 0x48304038U;
constexpr float kInitialAccumulator = 0.25F;
constexpr float kTolerance = 1.0e-3F;

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

// 一个 .b32 寄存器装 8 个连续 E2M1 元素：element i 位于 bits [4*i+3:4*i]。
// .kind::mxf4nvf4 不需要 mxf8f6f4 路径使用的 8-bit 容器或额外 padding。
__host__ __device__ constexpr uint32_t repeat_e2m1_nibble(uint8_t value)
{
    uint32_t packed = 0;
    for (int index = 0; index < 8; ++index)
    {
        packed |= static_cast<uint32_t>(value & 0x0FU) << (index * 4);
    }
    return packed;
}

// scale-a-data / scale-b-data 都是一个 .b32 元数据寄存器。
// scale_vec::4X 使用其中全部四个 byte，所以 byte-id 必须为 0。
__host__ __device__ constexpr uint32_t pack_scale_bytes(
    uint8_t value0,
    uint8_t value1,
    uint8_t value2,
    uint8_t value3)
{
    return static_cast<uint32_t>(value0) |
        (static_cast<uint32_t>(value1) << 8) |
        (static_cast<uint32_t>(value2) << 16) |
        (static_cast<uint32_t>(value3) << 24);
}

struct Fp32Fragment
{
    float value0;
    float value1;
    float value2;
    float value3;
};

__device__ __forceinline__ Fp32Fragment mma_m16n8k64_nvfp4(
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
    // 指令逐段解释：
    //
    // mma.sync.aligned
    //   warp-level MMA；32 个线程必须共同执行相同指令。
    // kind::mxf4nvf4.block_scale
    //   执行 D=(A*scale_A)*(B*scale_B)+C；UE4M3 scale 对应 NVFP4。
    // scale_vec::4X
    //   K=64 上提供 4 个 scale，即每 16 个 K 元素一个 scale（block16）。
    // m16n8k64.row.col
    //   一个 warp 计算 16x8 输出，归约 K=64；A 为 row-major，B 为
    //   column-major。原 GEMM 保存的是 row-major B[N,K]，逻辑上的 B^T[K,N]
    //   正好以 column-major 参与这条指令。
    // f32.e2m1.e2m1.f32.ue4m3
    //   依次是 D、A、B、C、scale 的元素类型；因此乘法输入是 packed FP4，
    //   C/D 均为 FP32 accumulator。
    //
    // 每线程寄存器：
    //   D/C：4 x .f32；整个 warp 共 32*4=128 个输出，即 16*8。
    //   A：4 x .b32，每个寄存器含 8 个 E2M1，整个 warp 共 16*64 个元素。
    //   B：2 x .b32，每个寄存器含 8 个 E2M1，整个 warp 共 64*8 个元素。
    //   scale A/B：各 1 x .b32，每个含 4 个 UE4M3 byte。
    //
    // CUTLASS 将 selector 变量命名为 bid/tid，其中 bid 是 byte-id，绝不是
    // blockIdx。对于 scale_vec::4X，四个 selector 的合法值都只能为 0：
    //   byte-id=0 表示使用 scale-data 的全部四个 byte；
    //   thread-id-a=0 表示 quad 内低两个 lane 提供 A scale；
    //   thread-id-b=0 表示 quad 内 lane%4==0 提供 B scale。
    // 本例让每个 lane 都准备相同 scale-data，因此被选中的 lane 能为全部
    // 输出行/列提供相同的四段 scale。
    constexpr uint16_t byte_id_a = 0;
    constexpr uint16_t thread_id_a = 0;
    constexpr uint16_t byte_id_b = 0;
    constexpr uint16_t thread_id_b = 0;

    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
        "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
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

__global__ void ptx_mma_fp4_kernel(float *fp32_output, __half *fp16_output)
{
    // 这里没有任何 GMEM -> SMEM 指令：A/B fragment 直接以寄存器常量构造，
    // 从而只测 mma.sync 自身。每个 nibble 都相同也避免学习样例被 fragment
    // 坐标装载代码淹没；寄存器数量与真实 m16n8k64 Atom 完全一致。
    const uint32_t packed_a = repeat_e2m1_nibble(kE2m1A);
    const uint32_t packed_b = repeat_e2m1_nibble(kE2m1B);
    // std::array::operator[] 只标注为 host constexpr；kernel 直接使用同一组
    // byte 预先合成的常量，避免依赖 relaxed-constexpr 编译选项。
    const uint32_t scale_a_data = kPackedScaleA;
    const uint32_t scale_b_data = kPackedScaleB;

    const Fp32Fragment accumulator = {
        kInitialAccumulator,
        kInitialAccumulator,
        kInitialAccumulator,
        kInitialAccumulator};
    const Fp32Fragment result = mma_m16n8k64_nvfp4(
        packed_a,
        packed_a,
        packed_a,
        packed_a,
        packed_b,
        packed_b,
        accumulator,
        scale_a_data,
        scale_b_data);

    // m16n8k64 的 C/D fragment 坐标映射：每 4 个 lane 为一个 quad。
    // lane 的 d0/d1 属于 row=lane/4，d2/d3 属于 row=lane/4+8；
    // 列为 (lane%4)*2 + {0,1}。这样可把寄存器 fragment 还原为 16x8 tile。
    const int lane = static_cast<int>(threadIdx.x);
    const int row0 = lane >> 2;
    const int row1 = row0 + 8;
    const int column0 = (lane & 3) * 2;
    const int column1 = column0 + 1;

    fp32_output[row0 * kN + column0] = result.value0;
    fp32_output[row0 * kN + column1] = result.value1;
    fp32_output[row1 * kN + column0] = result.value2;
    fp32_output[row1 * kN + column1] = result.value3;

    // 这四次转换对应 cute_gemm_fp4_fp16.cu 的 epilogue：MMA 保持 FP32
    // accumulator，仅在最终写回时通过 round-to-nearest-even 转成 FP16。
    fp16_output[row0 * kN + column0] = __float2half_rn(result.value0);
    fp16_output[row0 * kN + column1] = __float2half_rn(result.value1);
    fp16_output[row1 * kN + column0] = __float2half_rn(result.value2);
    fp16_output[row1 * kN + column1] = __float2half_rn(result.value3);
}

float decode_e2m1(uint8_t bits)
{
    constexpr std::array<float, 8> positive_values = {
        0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
    const float magnitude = positive_values[bits & 0x7U];
    return (bits & 0x8U) == 0 ? magnitude : -magnitude;
}

float decode_ue4m3(uint8_t bits)
{
    const int exponent_bits = (bits >> 3) & 0x0F;
    const int mantissa_bits = bits & 0x07;

    if (bits == 0x7FU)
    {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (exponent_bits == 0)
    {
        return std::ldexp(static_cast<float>(mantissa_bits) / 8.0F, -6);
    }

    const float significand = 1.0F + static_cast<float>(mantissa_bits) / 8.0F;
    return std::ldexp(significand, exponent_bits - 7);
}

float make_cpu_reference()
{
    const float value_a = decode_e2m1(kE2m1A);
    const float value_b = decode_e2m1(kE2m1B);
    float reference = kInitialAccumulator;

    for (int block = 0; block < kScaleVectorCount; ++block)
    {
        const float scaled_a = value_a * decode_ue4m3(kScaleA[block]);
        const float scaled_b = value_b * decode_ue4m3(kScaleB[block]);
        reference += static_cast<float>(kScaleBlockElements) * scaled_a * scaled_b;
    }
    return reference;
}

void verify_outputs(
    const std::array<float, kOutputElements> &fp32_output,
    const std::array<__half, kOutputElements> &fp16_output,
    float reference)
{
    const float fp16_reference = __half2float(__float2half_rn(reference));
    int fp32_mismatch_count = 0;
    int fp16_mismatch_count = 0;
    float maximum_fp32_error = 0.0F;
    float maximum_fp16_error = 0.0F;

    for (int index = 0; index < kOutputElements; ++index)
    {
        const float fp32_error = std::abs(fp32_output[index] - reference);
        const float fp16_error =
            std::abs(__half2float(fp16_output[index]) - fp16_reference);
        maximum_fp32_error = std::max(maximum_fp32_error, fp32_error);
        maximum_fp16_error = std::max(maximum_fp16_error, fp16_error);
        fp32_mismatch_count += fp32_error > kTolerance ? 1 : 0;
        fp16_mismatch_count += fp16_error > kTolerance ? 1 : 0;
    }

    std::cout << "CPU reference (FP32)     : " << reference << '\n'
              << "GPU first FP32 output    : " << fp32_output.front() << '\n'
              << "GPU first FP16 output    : "
              << __half2float(fp16_output.front()) << '\n'
              << "FP32 mismatch / max error: " << fp32_mismatch_count
              << " / " << maximum_fp32_error << '\n'
              << "FP16 mismatch / max error: " << fp16_mismatch_count
              << " / " << maximum_fp16_error << '\n';

    if (fp32_mismatch_count != 0 || fp16_mismatch_count != 0)
    {
        throw std::runtime_error("NVFP4 MMA result does not match CPU decoding reference");
    }
}

} // namespace

int main()
{
    try
    {
        cudaDeviceProp device_property{};
        CUDA_CHECK(cudaGetDeviceProperties(&device_property, 0));
        if (device_property.major != 12 || device_property.minor != 0)
        {
            throw std::runtime_error(
                "This learning test requires an SM120 GPU and sm_120f code generation");
        }

        std::cout << "\n[Config]\n"
                  << "GPU                      : " << device_property.name << '\n'
                  << "MMA tile                 : " << kM << 'x' << kN << 'x' << kK << '\n'
                  << "Threads                  : " << kWarpThreads << " (one warp)\n"
                  << "Input / accumulator      : E2M1 x E2M1 / FP32\n"
                  << "Scale / block size       : UE4M3 / 16\n"
                  << "Scale vector             : 4X\n"
                  << "Output                    : FP32 and FP16\n"
                  << "Data path                 : RMEM -> MMA -> RMEM (no cp.async)\n"
                  << "Packed A register         : 0x" << std::hex
                  << repeat_e2m1_nibble(kE2m1A) << '\n'
                  << "Packed B register         : 0x"
                  << repeat_e2m1_nibble(kE2m1B) << '\n'
                  << "Packed scale A register   : 0x"
                  << pack_scale_bytes(kScaleA[0], kScaleA[1], kScaleA[2], kScaleA[3]) << '\n'
                  << "Packed scale B register   : 0x"
                  << pack_scale_bytes(kScaleB[0], kScaleB[1], kScaleB[2], kScaleB[3])
                  << std::dec << "\n\n[Stage 1] Execute raw PTX MMA\n";

        DeviceBuffer<float> device_fp32_output(kOutputElements);
        DeviceBuffer<__half> device_fp16_output(kOutputElements);
        ptx_mma_fp4_kernel<<<1, kWarpThreads>>>(
            device_fp32_output.get(),
            device_fp16_output.get());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::array<float, kOutputElements> host_fp32_output{};
        std::array<__half, kOutputElements> host_fp16_output{};
        CUDA_CHECK(cudaMemcpy(
            host_fp32_output.data(),
            device_fp32_output.get(),
            sizeof(host_fp32_output),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            host_fp16_output.data(),
            device_fp16_output.get(),
            sizeof(host_fp16_output),
            cudaMemcpyDeviceToHost));

        std::cout << "\n[Stage 2] Decode E2M1/UE4M3 on CPU and verify\n";
        verify_outputs(host_fp32_output, host_fp16_output, make_cpu_reference());

        std::cout << "\n[SUCCESS] SM120 NVFP4 block-scaled MMA PTX verified\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "\n[FAILED] " << error.what() << '\n';
        return 1;
    }
}
