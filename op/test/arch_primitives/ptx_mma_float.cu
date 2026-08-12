#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace
{

// 本文件把经典浮点 Tensor Core 的 warp-level mma.sync 从 CuTe 中单独拆出，
// 直接观察 PTX 操作数与每个 lane 所持有的寄存器 fragment。它对应以下
// feature/linear 学习对象：
//
//   FP16 : op/linear/cute_gemm_fp16_fp32.cu
//   BF16 : op/linear/cute_gemm_bf16_fp32.cu
//   TF32 : op/linear/cute_gemm_tf32.cu
//
// 上述 GEMM 由 CuTe 选择 MMA Atom；这里不使用 CuTe/WMMA MMA wrapper，
// 而是在一个完整 warp 中直接发出 raw inline PTX，并重建官方规定的寄存器
// fragment。为了只研究 MMA，本文件也刻意不加入 global/shared-memory 搬运。
//
// 官方来源：
// https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-fragments-for-mma-m16n8k8
// https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-fragments-for-mma-m16n8k16-floating-point
// https://docs.nvidia.com/cuda/parallel-thread-execution/#multiply-and-accumulate-instruction-mma

constexpr int kWarpSize = 32;
constexpr int kM = 16;
constexpr int kN = 8;
constexpr int kHalfBfloatK = 16;
constexpr int kTf32K = 8;
constexpr int kOutputElements = kM * kN;
constexpr float kTolerance = 1.0e-5F;

void check_cuda(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

// 使用小整数构造 A 和 B，因此 FP16、BF16、TF32 都能无误差地表示输入。
// 同时让数值依赖 row、column 和 K 坐标，比全 1 输入更容易发现 fragment
// 顺序或 row.col 方向写反的问题。
__host__ __device__ float matrix_a_value(int row, int column)
{
    return static_cast<float>(((row * 3 + column * 2) % 5) - 2);
}

__host__ __device__ float matrix_b_value(int row, int column)
{
    return static_cast<float>(((row * 2 + column * 3 + 1) % 5) - 2);
}

// 非零 C 用来验证最后一个操作数不是占位符：MMA 的语义是
// D = A * B + C，而不是只计算 A * B。0.25 和 0.0625 都是二进制精确数。
__host__ __device__ float accumulator_value(int row, int column)
{
    return static_cast<float>(row - 7) * 0.25F +
           static_cast<float>(column) * 0.0625F;
}

__device__ __forceinline__ uint32_t pack_fp16_pair(float low, float high)
{
    const uint32_t low_bits =
        static_cast<uint32_t>(__half_as_ushort(__float2half_rn(low)));
    const uint32_t high_bits =
        static_cast<uint32_t>(__half_as_ushort(__float2half_rn(high)));
    return low_bits | (high_bits << 16U);
}

__device__ __forceinline__ uint32_t pack_bf16_pair(float low, float high)
{
    const uint32_t low_bits =
        static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(low)));
    const uint32_t high_bits =
        static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(high)));
    return low_bits | (high_bits << 16U);
}

__device__ __forceinline__ uint32_t convert_to_tf32_bits(float value)
{
    uint32_t result = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // mma 的 .tf32 输入保存在 .b32 寄存器中，但调用者必须先把普通
    // FP32 转为 TF32 编码。cvt.rna 保留 8-bit exponent 和 10-bit
    // explicit mantissa，并以 round-to-nearest-away 处理截断位。
    asm volatile(
        "cvt.rna.tf32.f32 %0, %1;"
        : "=r"(result)
        : "f"(value));
#else
    result = __float_as_uint(value);
#endif
    return result;
}

// m16n8k16 的 FP16/BF16 fragment 映射（FP16 与 BF16 完全相同）：
//
//   group_id = lane >> 2, thread_id = lane & 3
//
// A 是逻辑 row-major [16, 16]，每 lane 持有 8 个 16-bit 元素，打包为
// 4 个 32-bit 寄存器：A[0]={a0,a1}, ..., A[3]={a6,a7}，低元素在低
// 16 bit。对逻辑元素 ai：
//
//   row = group_id       when i in {0,1,4,5}, otherwise group_id + 8
//   col = 2*thread_id + (i&1) + (i >= 4 ? 8 : 0)
//
// B 是逻辑 column-major [16, 8]，每 lane 持有 4 个 16-bit 元素，打包
// 为 2 个 32-bit 寄存器。对逻辑元素 bi：
//
//   row = 2*thread_id + (i&1) + (i >= 2 ? 8 : 0)
//   col = group_id
//
// 注意 row.col 描述 A/B 的逻辑布局：MMA 计算 A[M,K] * B[K,N]。
// 本例直接构造寄存器而没有从内存加载，但仍必须严格遵守这套逻辑坐标。
template <bool kUseBfloat16>
__device__ __forceinline__ void make_m16n8k16_fragments(
    int lane,
    uint32_t (&fragment_a)[4],
    uint32_t (&fragment_b)[2])
{
    const int group_id = lane >> 2;
    const int thread_id = lane & 3;
    float elements_a[8];
    float elements_b[4];

#pragma unroll
    for (int index = 0; index < 8; ++index)
    {
        const bool upper_rows =
            !((index < 2) || (index >= 4 && index < 6));
        const int row = group_id + (upper_rows ? 8 : 0);
        const int column = thread_id * 2 + (index & 1) +
                           (index >= 4 ? 8 : 0);
        elements_a[index] = matrix_a_value(row, column);
    }

#pragma unroll
    for (int index = 0; index < 4; ++index)
    {
        const int row = thread_id * 2 + (index & 1) +
                        (index >= 2 ? 8 : 0);
        elements_b[index] = matrix_b_value(row, group_id);
    }

#pragma unroll
    for (int index = 0; index < 4; ++index)
    {
        if constexpr (kUseBfloat16)
        {
            fragment_a[index] = pack_bf16_pair(
                elements_a[index * 2],
                elements_a[index * 2 + 1]);
        }
        else
        {
            fragment_a[index] = pack_fp16_pair(
                elements_a[index * 2],
                elements_a[index * 2 + 1]);
        }
    }

#pragma unroll
    for (int index = 0; index < 2; ++index)
    {
        if constexpr (kUseBfloat16)
        {
            fragment_b[index] = pack_bf16_pair(
                elements_b[index * 2],
                elements_b[index * 2 + 1]);
        }
        else
        {
            fragment_b[index] = pack_fp16_pair(
                elements_b[index * 2],
                elements_b[index * 2 + 1]);
        }
    }
}

// 三种指令的 C/D 映射相同。每 lane 持有 4 个独立 FP32 accumulator：
//
//   c0/d0 -> [group_id,     2*thread_id    ]
//   c1/d1 -> [group_id,     2*thread_id + 1]
//   c2/d2 -> [group_id + 8, 2*thread_id    ]
//   c3/d3 -> [group_id + 8, 2*thread_id + 1]
//
// 32 lanes * 4 values = 16 * 8，因此一个 warp 合作产生完整 D tile。
__device__ __forceinline__ void make_accumulators(
    int lane,
    float (&fragment_c)[4])
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
    const float (&fragment_d)[4],
    float *output)
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

__device__ __forceinline__ void mma_fp16_m16n8k16(
    const uint32_t (&fragment_a)[4],
    const uint32_t (&fragment_b)[2],
    const float (&fragment_c)[4],
    float (&fragment_d)[4])
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // 指令名称：mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    // 来源：上方 NVIDIA PTX ISA mma 官方章节。
    // 用途：一个 warp 合作完成 16x8x16 的 FP16 矩阵乘法，FP32 累加。
    //
    // 操作数寄存器数（每 lane）：
    //   D: 4 x .f32，A: 4 x .b32(共 8 个 FP16)，
    //   B: 2 x .b32(共 4 个 FP16)，C: 4 x .f32。
    //
    // .sync 要求 warp 中参与线程共同执行；.aligned 要求所有线程执行
    // 相同的 mma 指令。若 warp 内线程分歧后用不同 qualifier/操作数形式
    // 执行，PTX 将其定义为未定义行为。
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=f"(fragment_d[0]),
          "=f"(fragment_d[1]),
          "=f"(fragment_d[2]),
          "=f"(fragment_d[3])
        : "r"(fragment_a[0]),
          "r"(fragment_a[1]),
          "r"(fragment_a[2]),
          "r"(fragment_a[3]),
          "r"(fragment_b[0]),
          "r"(fragment_b[1]),
          "f"(fragment_c[0]),
          "f"(fragment_c[1]),
          "f"(fragment_c[2]),
          "f"(fragment_c[3]));
#else
#pragma unroll
    for (int index = 0; index < 4; ++index)
    {
        fragment_d[index] = 0.0F;
    }
#endif
}

__device__ __forceinline__ void mma_bf16_m16n8k16(
    const uint32_t (&fragment_a)[4],
    const uint32_t (&fragment_b)[2],
    const float (&fragment_c)[4],
    float (&fragment_d)[4])
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // 指令名称：mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
    // 来源：上方 NVIDIA PTX ISA mma 官方章节。
    // 用途：一个 warp 合作完成 16x8x16 的 BF16 矩阵乘法，FP32 累加。
    // A/B/C/D 寄存器数量和 lane 映射与上面的 FP16 版本完全相同；唯一
    // 差异是 A/B 每个 16-bit 域按 BF16 编码解释，而非 IEEE FP16。
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=f"(fragment_d[0]),
          "=f"(fragment_d[1]),
          "=f"(fragment_d[2]),
          "=f"(fragment_d[3])
        : "r"(fragment_a[0]),
          "r"(fragment_a[1]),
          "r"(fragment_a[2]),
          "r"(fragment_a[3]),
          "r"(fragment_b[0]),
          "r"(fragment_b[1]),
          "f"(fragment_c[0]),
          "f"(fragment_c[1]),
          "f"(fragment_c[2]),
          "f"(fragment_c[3]));
#else
#pragma unroll
    for (int index = 0; index < 4; ++index)
    {
        fragment_d[index] = 0.0F;
    }
#endif
}

__global__ void mma_fp16_kernel(float *output)
{
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    uint32_t fragment_a[4];
    uint32_t fragment_b[2];
    float fragment_c[4];
    float fragment_d[4];

    make_m16n8k16_fragments<false>(lane, fragment_a, fragment_b);
    make_accumulators(lane, fragment_c);
    mma_fp16_m16n8k16(fragment_a, fragment_b, fragment_c, fragment_d);
    store_accumulators(lane, fragment_d, output);
}

__global__ void mma_bf16_kernel(float *output)
{
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    uint32_t fragment_a[4];
    uint32_t fragment_b[2];
    float fragment_c[4];
    float fragment_d[4];

    make_m16n8k16_fragments<true>(lane, fragment_a, fragment_b);
    make_accumulators(lane, fragment_c);
    mma_bf16_m16n8k16(fragment_a, fragment_b, fragment_c, fragment_d);
    store_accumulators(lane, fragment_d, output);
}

// m16n8k8 TF32 使用同样的 16x8 accumulator tile，但 K 只有 8。
// TF32 每个元素占一个 .b32 寄存器，不能像 FP16/BF16 那样二合一打包。
// 每 lane 的 A fragment 有 4 个寄存器：
//
//   a0 -> [group_id,     thread_id    ]
//   a1 -> [group_id + 8, thread_id    ]
//   a2 -> [group_id,     thread_id + 4]
//   a3 -> [group_id + 8, thread_id + 4]
//
// 每 lane 的 B fragment 有 2 个寄存器：
//
//   b0 -> [thread_id,     group_id]
//   b1 -> [thread_id + 4, group_id]
//
// 所以寄存器总数是 D:4、A:4、B:2、C:4；元素个数分别也是
// D:4、A:4、B:2、C:4。
__device__ __forceinline__ void make_tf32_fragments(
    int lane,
    uint32_t (&fragment_a)[4],
    uint32_t (&fragment_b)[2])
{
    const int group_id = lane >> 2;
    const int thread_id = lane & 3;

    fragment_a[0] = convert_to_tf32_bits(matrix_a_value(group_id, thread_id));
    fragment_a[1] = convert_to_tf32_bits(matrix_a_value(group_id + 8, thread_id));
    fragment_a[2] = convert_to_tf32_bits(matrix_a_value(group_id, thread_id + 4));
    fragment_a[3] = convert_to_tf32_bits(matrix_a_value(group_id + 8, thread_id + 4));

    fragment_b[0] = convert_to_tf32_bits(matrix_b_value(thread_id, group_id));
    fragment_b[1] = convert_to_tf32_bits(matrix_b_value(thread_id + 4, group_id));
}

__device__ __forceinline__ void mma_tf32_m16n8k8(
    const uint32_t (&fragment_a)[4],
    const uint32_t (&fragment_b)[2],
    const float (&fragment_c)[4],
    float (&fragment_d)[4])
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // 指令名称：mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
    // 来源：上方 NVIDIA PTX ISA mma 官方章节。
    // 用途：一个 warp 合作完成 16x8x8 的 TF32 矩阵乘法，FP32 累加。
    // 与 FP16/BF16 一样，所有 32 个 lane 必须一致执行该指令。
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=f"(fragment_d[0]),
          "=f"(fragment_d[1]),
          "=f"(fragment_d[2]),
          "=f"(fragment_d[3])
        : "r"(fragment_a[0]),
          "r"(fragment_a[1]),
          "r"(fragment_a[2]),
          "r"(fragment_a[3]),
          "r"(fragment_b[0]),
          "r"(fragment_b[1]),
          "f"(fragment_c[0]),
          "f"(fragment_c[1]),
          "f"(fragment_c[2]),
          "f"(fragment_c[3]));
#else
#pragma unroll
    for (int index = 0; index < 4; ++index)
    {
        fragment_d[index] = 0.0F;
    }
#endif
}

__global__ void mma_tf32_kernel(float *output)
{
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    uint32_t fragment_a[4];
    uint32_t fragment_b[2];
    float fragment_c[4];
    float fragment_d[4];

    make_tf32_fragments(lane, fragment_a, fragment_b);
    make_accumulators(lane, fragment_c);
    mma_tf32_m16n8k8(fragment_a, fragment_b, fragment_c, fragment_d);
    store_accumulators(lane, fragment_d, output);
}

float reference_value(int row, int column, int k_extent)
{
    float result = accumulator_value(row, column);
    for (int inner = 0; inner < k_extent; ++inner)
    {
        result += matrix_a_value(row, inner) *
                  matrix_b_value(inner, column);
    }
    return result;
}

void verify_case(
    const std::string &name,
    const std::string &instruction,
    int k_extent,
    const std::vector<float> &actual)
{
    size_t mismatch_count = 0;
    float maximum_absolute_error = 0.0F;

    for (int row = 0; row < kM; ++row)
    {
        for (int column = 0; column < kN; ++column)
        {
            const size_t index = static_cast<size_t>(row) * kN + column;
            const float expected = reference_value(row, column, k_extent);
            const float absolute_error = std::abs(actual[index] - expected);
            maximum_absolute_error =
                std::max(maximum_absolute_error, absolute_error);
            if (absolute_error > kTolerance)
            {
                ++mismatch_count;
            }
        }
    }

    std::cout << "Case         : " << name << '\n'
              << "Instruction  : " << instruction << '\n'
              << "Shape        : M=" << kM
              << ", N=" << kN
              << ", K=" << k_extent << '\n'
              << "Mismatch     : " << mismatch_count << '\n'
              << "Max abs error: " << maximum_absolute_error << '\n'
              << "D[0,0]       : " << actual[0]
              << " (reference " << reference_value(0, 0, k_extent) << ")\n"
              << "D[15,7]      : " << actual[kOutputElements - 1]
              << " (reference " << reference_value(15, 7, k_extent) << ")\n";

    if (mismatch_count != 0)
    {
        throw std::runtime_error(name + " validation failed");
    }
}

template <typename Kernel>
void run_case(
    const std::string &name,
    const std::string &instruction,
    int k_extent,
    Kernel kernel,
    float *device_output,
    std::vector<float> &host_output)
{
    check_cuda(
        cudaMemset(device_output, 0, kOutputElements * sizeof(float)),
        "cudaMemset output");

    kernel<<<1, kWarpSize>>>(device_output);
    check_cuda(cudaGetLastError(), "MMA kernel launch");
    check_cuda(cudaDeviceSynchronize(), "MMA kernel synchronize");
    check_cuda(
        cudaMemcpy(
            host_output.data(),
            device_output,
            kOutputElements * sizeof(float),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy output to host");

    verify_case(name, instruction, k_extent, host_output);
}

} // namespace

int main()
{
    try
    {
        int device = 0;
        cudaDeviceProp properties{};
        check_cuda(cudaGetDevice(&device), "cudaGetDevice");
        check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");

        if (properties.major < 8)
        {
            throw std::runtime_error(
                "mma.sync FP16/BF16/TF32 learning test requires compute capability 8.0+");
        }

        std::cout << "PTX floating-point MMA learning test\n\n"
                  << "Configuration\n"
                  << "  Device       : " << properties.name << '\n'
                  << "  Compute      : " << properties.major << '.' << properties.minor << '\n'
                  << "  Launch       : grid=1, block=" << kWarpSize << '\n'
                  << "  Output tile  : " << kM << " x " << kN << '\n'
                  << "  Accumulator  : FP32, initialized with non-zero C\n"
                  << "  Data path    : lane-generated RMEM fragments -> MMA -> GMEM\n\n";

        float *device_output = nullptr;
        check_cuda(
            cudaMalloc(&device_output, kOutputElements * sizeof(float)),
            "cudaMalloc output");
        std::vector<float> host_output(kOutputElements);

        std::cout << "Stage 1/3: FP16 raw mma.sync\n";
        run_case(
            "FP16 x FP16 -> FP32",
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
            kHalfBfloatK,
            mma_fp16_kernel,
            device_output,
            host_output);

        std::cout << "\nStage 2/3: BF16 raw mma.sync\n";
        run_case(
            "BF16 x BF16 -> FP32",
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
            kHalfBfloatK,
            mma_bf16_kernel,
            device_output,
            host_output);

        std::cout << "\nStage 3/3: TF32 raw mma.sync\n";
        run_case(
            "TF32 x TF32 -> FP32",
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
            kTf32K,
            mma_tf32_kernel,
            device_output,
            host_output);

        check_cuda(cudaFree(device_output), "cudaFree output");

        std::cout << "\n[SUCCESS] All floating-point mma.sync cases passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cout << "[FAILED] " << error.what() << '\n';
        return 1;
    }
}
