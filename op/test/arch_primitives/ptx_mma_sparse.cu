/***************************************************************************************************
 * FP16 2:4 Structured Sparse MMA PTX 学习样例
 *
 * 本文件把 feature/linear:op/linear/cute_gemm_sparse.cu 中由
 * cutlass::arch::SparseMma 间接发出的 sparse Tensor Core 指令拆成 raw inline PTX：
 *
 *   mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32
 *
 * 学习目标：
 *   1. 从 dense A[16,32] 构造严格合法的 2:4 稀疏矩阵；
 *   2. 把每 4 个 dense 元素压缩为 2 个 FP16，并构造位置 metadata；
 *   3. 显式建立 A/B/C/D 的 warp 寄存器 fragment；
 *   4. 直接执行 mma.sp，而不调用 CUTLASS/CuTe/WMMA 的 MMA wrapper；
 *   5. 用展开后的 dense A 在 CPU 上计算完整 16x8 reference。
 *
 * 这里的 inline PTX 只用于指令学习，因此按任务约定留在测试文件中，不放入
 * op/ptx_utils.cuh。示例只隔离 MMA，不包含 GMEM -> SMEM pipeline。
 *
 * NVIDIA PTX ISA 官方资料：
 *   Sparse matrix storage:
 *   https://docs.nvidia.com/cuda/parallel-thread-execution/#sparse-matrix-storage
 *   m16n8k32 FP16/BF16 sparse fragment:
 *   https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-fragments-for-sparse-mma-m16n8k32-with-f16-and-bf16-types
 *   mma.sp / mma.sp::ordered_metadata:
 *   https://docs.nvidia.com/cuda/parallel-thread-execution/#multiply-and-accumulate-instruction-mma-sp
 **************************************************************************************************/

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace
{

constexpr int kWarpSize = 32;
constexpr int kM = 16;
constexpr int kN = 8;
constexpr int kK = 32;
constexpr int kCompressedK = kK / 2;
constexpr int kSparseGroupSize = 4;
constexpr int kRetainedPerGroup = 2;
constexpr int kOutputElements = kM * kN;

// 一个 metadata nibble 包含两个 2-bit index，LSB index 必须在前：
//
//   bits [1:0] = 0b00 -> 第一个压缩值放回 dense 位置 0
//   bits [3:2] = 0b01 -> 第二个压缩值放回 dense 位置 1
//   nibble      = 0b0100 = 0x4
//
// ordered_metadata 要求两个 index 按升序排列。合法的有序二选位置编码为
// 0x4、0x8、0x9、0xC、0xD、0xE，分别对应 {0,1}、{0,2}、{1,2}、
// {0,3}、{1,3}、{2,3}。本例每个 2:4 group 都保留 {0,1}。
constexpr uint32_t kMetadataNibble = 0x4U;
constexpr uint32_t kPackedMetadata = 0x44444444U;
constexpr uint32_t kSparseSelector = 0U;

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
    {
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&pointer_),
            count * sizeof(T)));
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
};

struct SparseInputs
{
    // dense_a 是人类容易检查的 A[M,K]；compressed_a 是 mma.sp 真正读取的
    // A[M,K/2]；B 是逻辑 B[K,N]，C 是非零的初始 accumulator。
    std::vector<__half> dense_a;
    std::vector<__half> compressed_a;
    std::vector<__half> b;
    std::vector<float> c;
    std::vector<uint32_t> metadata;
};

float retained_a_value(int row, int group, int retained_index)
{
    if (retained_index == 0)
    {
        return static_cast<float>((row + group) % 5 + 1);
    }

    return -static_cast<float>((row * 2 + group) % 5 + 1);
}

float b_value(int reduction, int column)
{
    return static_cast<float>((reduction * 3 + column * 5 + 1) % 9 - 4);
}

float c_value(int row, int column)
{
    // 0.25 可以被 FP32 精确表示，非零 C 明确验证 D=A*B+C 的累加语义。
    return static_cast<float>(row - column) * 0.25F;
}

SparseInputs make_sparse_inputs()
{
    SparseInputs inputs;
    inputs.dense_a.resize(kM * kK, __float2half(0.0F));
    inputs.compressed_a.resize(kM * kCompressedK);
    inputs.b.resize(kK * kN);
    inputs.c.resize(kM * kN);

    // m16n8k32 的 selector=0 表示每组连续四个 lane 中由 T0/T1
    // 提供 metadata；selector=1 才会选择 T2/T3。为了让数据结构一眼可查，
    // 32 个 lane 都保存同一合法值，被 selector 选中的 lane 会实际供数。
    // 一个 .b32 含 8 个 nibble，也就是 16 个 2-bit index。
    inputs.metadata.resize(kWarpSize, kPackedMetadata);

    for (int row = 0; row < kM; ++row)
    {
        for (int group = 0; group < kK / kSparseGroupSize; ++group)
        {
            const int dense_base = row * kK + group * kSparseGroupSize;
            const int compressed_base =
                row * kCompressedK + group * kRetainedPerGroup;
            const float retained0 = retained_a_value(row, group, 0);
            const float retained1 = retained_a_value(row, group, 1);

            // dense 语义为 [v0, v1, 0, 0]；压缩后只连续保存 [v0, v1]。
            // metadata 0x4 再把这两个值映射回 group 内的 dense 位置 {0,1}。
            inputs.dense_a[dense_base + 0] = __float2half(retained0);
            inputs.dense_a[dense_base + 1] = __float2half(retained1);
            inputs.dense_a[dense_base + 2] = __float2half(0.0F);
            inputs.dense_a[dense_base + 3] = __float2half(0.0F);
            inputs.compressed_a[compressed_base + 0] =
                __float2half(retained0);
            inputs.compressed_a[compressed_base + 1] =
                __float2half(retained1);
        }
    }

    for (int reduction = 0; reduction < kK; ++reduction)
    {
        for (int column = 0; column < kN; ++column)
        {
            inputs.b[reduction * kN + column] =
                __float2half(b_value(reduction, column));
        }
    }

    for (int row = 0; row < kM; ++row)
    {
        for (int column = 0; column < kN; ++column)
        {
            inputs.c[row * kN + column] = c_value(row, column);
        }
    }

    return inputs;
}

bool validate_sparse_encoding(const SparseInputs &inputs)
{
    for (int row = 0; row < kM; ++row)
    {
        for (int group = 0; group < kK / kSparseGroupSize; ++group)
        {
            const int dense_base = row * kK + group * kSparseGroupSize;
            const int compressed_base =
                row * kCompressedK + group * kRetainedPerGroup;
            int nonzero_count = 0;

            for (int offset = 0; offset < kSparseGroupSize; ++offset)
            {
                const float value =
                    __half2float(inputs.dense_a[dense_base + offset]);
                nonzero_count += static_cast<int>(value != 0.0F);
            }

            if (nonzero_count != kRetainedPerGroup ||
                __half2float(inputs.dense_a[dense_base + 0]) !=
                    __half2float(inputs.compressed_a[compressed_base + 0]) ||
                __half2float(inputs.dense_a[dense_base + 1]) !=
                    __half2float(inputs.compressed_a[compressed_base + 1]) ||
                __half2float(inputs.dense_a[dense_base + 2]) != 0.0F ||
                __half2float(inputs.dense_a[dense_base + 3]) != 0.0F)
            {
                return false;
            }
        }
    }

    return std::all_of(
        inputs.metadata.begin(),
        inputs.metadata.end(),
        [](uint32_t value)
        {
            return value == kPackedMetadata;
        });
}

__device__ __forceinline__ uint32_t pack_fp16_pair(
    __half low,
    __half high)
{
    // PTX 的 .b32 fragment 将低序 FP16 放在低 16 bit，高序 FP16 放在
    // 高 16 bit。显式位打包比依赖结构体重解释更适合作为学习样例。
    const uint32_t low_bits = static_cast<uint32_t>(__half_as_ushort(low));
    const uint32_t high_bits = static_cast<uint32_t>(__half_as_ushort(high));
    return low_bits | (high_bits << 16U);
}

// raw inline PTX wrapper：保留在本学习文件中，不调用 CUTLASS SparseMma。
__device__ __forceinline__ void mma_sp_m16n8k32_fp16_fp32(
    const uint32_t (&fragment_a)[4],
    const uint32_t (&fragment_b)[4],
    const float (&fragment_c)[4],
    uint32_t metadata,
    float (&fragment_d)[4])
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // 指令名称：
    //   mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.
    //   f32.f16.f16.f32
    // 来源：文件顶部 NVIDIA PTX ISA 官方链接。
    // 用途：一个完整 warp 合作执行 D[16,8] = A[16,32] * B[32,8] + C。
    // A 是 2:4 结构化稀疏输入，只在寄存器中携带 K/2 个非零值。
    //
    // 每个 lane 的寄存器操作数：
    //   A: 4 x .b32 = 8 个压缩 FP16；整个 warp 共 256 个 FP16，
    //      等于 16 * (32/2)。若是 dense m16n8k32，则 A 需要两倍数据。
    //   B: 4 x .b32 = 8 个 dense FP16；整个 warp 共 256 个 FP16，
    //      等于 32 * 8。
    //   C: 4 x .f32，D: 4 x .f32；32 * 4 = 16 * 8 个输出。
    //   E: 1 x .b32 metadata；F: 编译期立即数 sparse selector。
    //
    // .sync 表示参与 warp 的线程要在该指令处会合，.aligned 要求全部线程
    // 执行同一条 MMA。这里 kernel 固定以一个完整 32-thread warp 启动。
    //
    // ordered_metadata 要求每个 nibble 中的两个 index 从 LSB 开始严格
    // 递增；0x4 即 [00,01]。最后的 0x0 是 sparse selector：对
    // m16n8k32 FP16，它选择每个 quad 的 T0/T1，而 0x1 会选择 T2/T3。
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col."
        "f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, "
        "{%12, %13, %14, %15}, %16, 0x0;\n"
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
          "r"(fragment_b[2]),
          "r"(fragment_b[3]),
          "f"(fragment_c[0]),
          "f"(fragment_c[1]),
          "f"(fragment_c[2]),
          "f"(fragment_c[3]),
          "r"(metadata));
#else
#pragma unroll
    for (int index = 0; index < 4; ++index)
    {
        fragment_d[index] = 0.0F;
    }
#endif
}

__global__ __launch_bounds__(kWarpSize)
void ptx_mma_sparse_kernel(
    const __half *compressed_a,
    const __half *b,
    const float *c,
    const uint32_t *metadata,
    float *d)
{
    const int lane = static_cast<int>(threadIdx.x);
    const int group_id = lane >> 2;
    const int thread_id = lane & 3;
    __half elements_a[8];
    __half elements_b[8];
    uint32_t fragment_a[4];
    uint32_t fragment_b[4];
    float fragment_c[4];
    float fragment_d[4];

    // Sparse A fragment 的官方 m16n8k32 映射：
    //
    //   ai={0,1,4,5} -> row=group_id，其余 -> row=group_id+8
    //   ai<4           -> dense group 起点=thread_id*4
    //   ai>=4          -> dense group 起点=thread_id*4+16
    //
    // 每个 dense group 的 4 个值已压缩成 2 个，因此 compressed column
    // 为 group*2+(i&1)。a0/a1、a2/a3、a4/a5、a6/a7 分别打包为
    // 4 个 .b32 寄存器。
#pragma unroll
    for (int index = 0; index < 8; ++index)
    {
        const bool low_output_row =
            index < 2 || (index >= 4 && index < 6);
        const int row = group_id + (low_output_row ? 0 : 8);
        const int dense_group = thread_id + (index >= 4 ? 4 : 0);
        const int compressed_column =
            dense_group * kRetainedPerGroup + (index & 1);
        elements_a[index] =
            compressed_a[row * kCompressedK + compressed_column];
    }

    // B 是 dense B[K,N]。m16n8k32 的 B fragment 可视为两个连续的
    // m16n8k16 B fragment：
    //
    //   k = 2*thread_id + (i&1) + 8*(i/2), i in [0,7]
    //   n = group_id
    //
    // 每两个 FP16 按低位在前打包成一个 .b32，共 4 个寄存器。
#pragma unroll
    for (int index = 0; index < 8; ++index)
    {
        const int reduction =
            thread_id * 2 + (index & 1) + (index / 2) * 8;
        elements_b[index] = b[reduction * kN + group_id];
    }

#pragma unroll
    for (int index = 0; index < 4; ++index)
    {
        fragment_a[index] = pack_fp16_pair(
            elements_a[index * 2],
            elements_a[index * 2 + 1]);
        fragment_b[index] = pack_fp16_pair(
            elements_b[index * 2],
            elements_b[index * 2 + 1]);
    }

    // C/D fragment 坐标：
    //   c0/d0 -> [group_id,     2*thread_id]
    //   c1/d1 -> [group_id,     2*thread_id+1]
    //   c2/d2 -> [group_id+8,   2*thread_id]
    //   c3/d3 -> [group_id+8,   2*thread_id+1]
    // 4 个 FP32 C 寄存器在指令中是只读源，4 个 D 寄存器接收结果。
#pragma unroll
    for (int index = 0; index < 4; ++index)
    {
        const int row = group_id + (index >= 2 ? 8 : 0);
        const int column = thread_id * 2 + (index & 1);
        fragment_c[index] = c[row * kN + column];
    }

    mma_sp_m16n8k32_fp16_fp32(
        fragment_a,
        fragment_b,
        fragment_c,
        metadata[lane],
        fragment_d);

#pragma unroll
    for (int index = 0; index < 4; ++index)
    {
        const int row = group_id + (index >= 2 ? 8 : 0);
        const int column = thread_id * 2 + (index & 1);
        d[row * kN + column] = fragment_d[index];
    }
}

std::vector<float> cpu_reference(const SparseInputs &inputs)
{
    std::vector<float> reference(kOutputElements);
    for (int row = 0; row < kM; ++row)
    {
        for (int column = 0; column < kN; ++column)
        {
            float accumulator = inputs.c[row * kN + column];
            for (int reduction = 0; reduction < kK; ++reduction)
            {
                accumulator +=
                    __half2float(inputs.dense_a[row * kK + reduction]) *
                    __half2float(inputs.b[reduction * kN + column]);
            }
            reference[row * kN + column] = accumulator;
        }
    }
    return reference;
}

struct Comparison
{
    size_t mismatch_count = 0;
    float max_absolute_error = 0.0F;
};

Comparison compare_outputs(
    const std::vector<float> &actual,
    const std::vector<float> &reference)
{
    Comparison comparison;
    for (size_t index = 0; index < actual.size(); ++index)
    {
        const float error = std::abs(actual[index] - reference[index]);
        comparison.max_absolute_error =
            std::max(comparison.max_absolute_error, error);
        comparison.mismatch_count += static_cast<size_t>(error != 0.0F);
    }
    return comparison;
}

int run()
{
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    if (properties.major < 8)
    {
        throw std::runtime_error(
            "FP16 mma.sp m16n8k32 requires compute capability 8.0+");
    }

    std::cout << "2:4 Sparse MMA PTX 学习配置\n"
              << "  GPU                     : " << properties.name << '\n'
              << "  指令 tile               : M=16, N=8, K=32\n"
              << "  输入/累加/输出           : FP16 / FP32 / FP32\n"
              << "  warp threads            : " << kWarpSize << '\n'
              << "  dense/compressed A       : " << kM * kK << " / "
              << kM * kCompressedK << " FP16 elements\n"
              << "  每个 2:4 group 保留位置  : {0, 1}\n"
              << "  metadata nibble         : 0x" << std::hex
              << kMetadataNibble << '\n'
              << "  packed metadata         : 0x" << kPackedMetadata << '\n'
              << "  sparse selector         : 0x" << kSparseSelector
              << std::dec << " (每个 quad 的 T0/T1)\n"
              << "  PTX                     : mma.sp::ordered_metadata "
                 "m16n8k32 f32.f16.f16.f32\n\n";

    std::cout << "阶段 1/3：构造并检查 dense 2:4 A、压缩 A 和 metadata\n";
    const SparseInputs inputs = make_sparse_inputs();
    if (!validate_sparse_encoding(inputs))
    {
        throw std::runtime_error("2:4 压缩数据或 metadata 检查失败");
    }
    std::cout << "  2:4 group 数量           : "
              << kM * (kK / kSparseGroupSize) << '\n'
              << "  结构/压缩/metadata       : valid\n\n";

    DeviceBuffer<__half> device_compressed_a(inputs.compressed_a.size());
    DeviceBuffer<__half> device_b(inputs.b.size());
    DeviceBuffer<float> device_c(inputs.c.size());
    DeviceBuffer<uint32_t> device_metadata(inputs.metadata.size());
    DeviceBuffer<float> device_d(kOutputElements);

    CUDA_CHECK(cudaMemcpy(
        device_compressed_a.get(),
        inputs.compressed_a.data(),
        inputs.compressed_a.size() * sizeof(__half),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        device_b.get(),
        inputs.b.data(),
        inputs.b.size() * sizeof(__half),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        device_c.get(),
        inputs.c.data(),
        inputs.c.size() * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        device_metadata.get(),
        inputs.metadata.data(),
        inputs.metadata.size() * sizeof(uint32_t),
        cudaMemcpyHostToDevice));

    std::cout << "阶段 2/3：一个完整 warp 执行 raw mma.sp inline PTX\n";
    ptx_mma_sparse_kernel<<<1, kWarpSize>>>(
        device_compressed_a.get(),
        device_b.get(),
        device_c.get(),
        device_metadata.get(),
        device_d.get());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "  kernel launch            : complete\n\n";

    std::cout << "阶段 3/3：用展开后的 dense A 校验完整 16x8 输出\n";
    std::vector<float> actual(kOutputElements);
    CUDA_CHECK(cudaMemcpy(
        actual.data(),
        device_d.get(),
        actual.size() * sizeof(float),
        cudaMemcpyDeviceToHost));
    const std::vector<float> reference = cpu_reference(inputs);
    const Comparison comparison = compare_outputs(actual, reference);

    std::cout << "  checked elements         : " << kOutputElements << '\n'
              << "  mismatch count           : "
              << comparison.mismatch_count << '\n'
              << "  max absolute error       : "
              << comparison.max_absolute_error << '\n'
              << "  D[0,0] actual/reference  : " << actual[0] << " / "
              << reference[0] << '\n'
              << "  D[15,7] actual/reference : " << actual.back() << " / "
              << reference.back() << "\n\n";

    if (comparison.mismatch_count != 0)
    {
        throw std::runtime_error("sparse MMA 与 dense CPU reference 不一致");
    }

    std::cout << "[SUCCESS] raw 2:4 sparse mma.sp PTX 验证通过\n";
    return 0;
}

} // namespace

int main()
{
    try
    {
        return run();
    }
    catch (const std::exception &error)
    {
        std::cout << "[FAILED] " << error.what() << '\n';
        return 1;
    }
}
