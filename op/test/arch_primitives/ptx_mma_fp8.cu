// SM120 FP8 / MXFP8 warp-level MMA PTX learning example.
//
// Official PTX ISA reference:
// https://docs.nvidia.com/cuda/parallel-thread-execution/#multiply-and-accumulate-instruction-mma
// https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16832-f8f6f4
// https://docs.nvidia.com/cuda/parallel-thread-execution/#block-scaling-for-mma-sync
//
// This file deliberately keeps the inline-PTX wrappers local. It is a learning
// aid that exposes the registers hidden by CuTe, rather than a reusable PTX
// utility for op/ptx_utils.cuh.
//
// Related production-style CuTe examples on branch feature/linear:
//   op/linear/cute_gemm_fp8_fp32.cu
//   op/linear/cute_gemm_mxfp8_fp16.cu
//
// The first example uses an ordinary tensor-wide quantization scale: the MMA
// instruction consumes only packed E4M3 qA/qB values, then its epilogue applies
// scale_A * scale_B to the FP32 accumulator. There are no scale registers in
// that MMA instruction. MXFP8 instead stores one UE8M0 scale for each group of
// 32 consecutive K values, supplies those scales plus selectors to every MMA,
// and the Tensor Core evaluates the scaled operands inside the operation. The
// difference is thus an instruction/data-layout contract, not merely where a
// host-side scale variable happens to be declared.
//
// Both cases below execute one warp-wide m16n8k32 operation. Every logical A
// and B element is E4M3 1.0, and C is zero, so every one of the 16 * 8 logical
// output elements must be sum(k=0..31, 1 * 1) = 32. Keeping every result equal
// lets us validate all 128 distributed D fragment values without first hiding
// the instruction behind a separate row/column fragment-scatter implementation.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace
{

constexpr int kWarpThreads = 32;
constexpr int kM = 16;
constexpr int kN = 8;
constexpr int kK = 32;
constexpr int kAccumulatorRegistersPerLane = 4;
constexpr int kOutputElements = kM * kN;
constexpr float kTolerance = 0.0F;

// E4M3 1.0 has sign=0, biased exponent=7, mantissa=0: 0b0011'1000 = 0x38.
// E4M3 uses an 8-bit container. Therefore one b32 operand register packs four
// adjacent E4M3 values, least-significant byte first. Filling all four bytes
// with 0x38 produces four values of 1.0 regardless of fragment position.
constexpr uint8_t kE4m3One = 0x38U;
constexpr uint32_t kFourE4m3Ones = 0x38383838U;

// UE8M0 stores only an unsigned exponent. Raw 0x7f represents 2^(127-127)=1.
// All four bytes are initialized even though scale_vec::1X selects one byte;
// this makes the selector mechanism visible while keeping the test invariant
// under any valid byte-id.
constexpr uint8_t kUe8m0One = 0x7fU;
constexpr uint32_t kFourUe8m0Ones = 0x7f7f7f7fU;

static_assert(
    kWarpThreads * kAccumulatorRegistersPerLane == kOutputElements,
    "m16n8 FP32 D fragment has four values per lane");

void check_cuda(cudaError_t status, const char *operation)
{
    if (status == cudaSuccess)
    {
        return;
    }

    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(status));
}

// One lane's register contract for
// mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32:
//
//   A: 4 x b32 = 16 packed E4M3 values per lane
//   B: 2 x b32 =  8 packed E4M3 values per lane
//   C: 4 x f32 =  4 FP32 accumulator fragment values per lane
//   D: 4 x f32 =  4 FP32 destination fragment values per lane
//
// Across 32 lanes, the packed A and B fragments therefore hold 512 and 256
// 8-bit values, exactly matching 16*32 and 32*8. These are fragments, not four
// complete private matrices: all 32 lanes must execute the same `.sync.aligned`
// instruction, and the Tensor Core collectively computes D = A*B+C.
__device__ __forceinline__ void mma_fp8_e4m3_f32(
    float &d0,
    float &d1,
    float &d2,
    float &d3,
    uint32_t a0,
    uint32_t a1,
    uint32_t a2,
    uint32_t a3,
    uint32_t b0,
    uint32_t b1,
    float c0,
    float c1,
    float c2,
    float c3)
{
    // SM120 uses the extended f8/f6/f4 form with kind::f8f6f4. The type
    // suffix reads from left to right as D=f32, A=e4m3, B=e4m3, C=f32.
    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col."
        "f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

// MXFP8 keeps the same E4M3 A/B and FP32 C/D register fragments, but appends:
//
//   scale-a-data: one b32 metadata register per lane
//   {byte-id-a, thread-id-a}: two u16 selectors
//   scale-b-data: one b32 metadata register per lane
//   {byte-id-b, thread-id-b}: two u16 selectors
//
// For scale_vec::1X (the MXFP8 block32 form), byte-id selects one UE8M0 byte
// from the b32 metadata register. thread-id-a selects one lane pair in each
// four-lane quad (0 -> quad lanes 0/1, 1 -> 2/3); thread-id-b selects one lane
// in each quad (0..3). Selected lanes collectively provide scale matrices A
// and B. Here every byte in every lane is UE8M0 1.0, so {0,0} is simple and
// still exercises the real block-scaled instruction.
__device__ __forceinline__ void mma_mxfp8_e4m3_f32(
    float &d0,
    float &d1,
    float &d2,
    float &d3,
    uint32_t a0,
    uint32_t a1,
    uint32_t a2,
    uint32_t a3,
    uint32_t b0,
    uint32_t b1,
    float c0,
    float c1,
    float c2,
    float c3,
    uint32_t scale_a_data,
    uint32_t scale_b_data)
{
    // `bid` below means byte-id, not CUDA blockIdx. With K=32 and
    // scale_vec::1X, one UE8M0 scale covers the full 32-element K block.
    constexpr uint16_t bid_a = 0;
    constexpr uint16_t tid_a = 0;
    constexpr uint16_t bid_b = 0;
    constexpr uint16_t tid_b = 0;

    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
        "m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13}, "
        "{%14}, {%15, %16}, {%17}, {%18, %19};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3),
          "r"(scale_a_data), "h"(bid_a), "h"(tid_a),
          "r"(scale_b_data), "h"(bid_b), "h"(tid_b));
}

__global__ void fp8_mma_kernel(float *output)
{
    float d0;
    float d1;
    float d2;
    float d3;

    mma_fp8_e4m3_f32(
        d0,
        d1,
        d2,
        d3,
        kFourE4m3Ones,
        kFourE4m3Ones,
        kFourE4m3Ones,
        kFourE4m3Ones,
        kFourE4m3Ones,
        kFourE4m3Ones,
        0.0F,
        0.0F,
        0.0F,
        0.0F);

    const int output_base = threadIdx.x * kAccumulatorRegistersPerLane;
    output[output_base + 0] = d0;
    output[output_base + 1] = d1;
    output[output_base + 2] = d2;
    output[output_base + 3] = d3;
}

__global__ void mxfp8_mma_kernel(float *output)
{
    float d0;
    float d1;
    float d2;
    float d3;

    mma_mxfp8_e4m3_f32(
        d0,
        d1,
        d2,
        d3,
        kFourE4m3Ones,
        kFourE4m3Ones,
        kFourE4m3Ones,
        kFourE4m3Ones,
        kFourE4m3Ones,
        kFourE4m3Ones,
        0.0F,
        0.0F,
        0.0F,
        0.0F,
        kFourUe8m0Ones,
        kFourUe8m0Ones);

    const int output_base = threadIdx.x * kAccumulatorRegistersPerLane;
    output[output_base + 0] = d0;
    output[output_base + 1] = d1;
    output[output_base + 2] = d2;
    output[output_base + 3] = d3;
}

std::vector<float> make_cpu_reference(float scale_a, float scale_b)
{
    std::vector<float> reference(kOutputElements, 0.0F);
    for (int row = 0; row < kM; ++row)
    {
        for (int column = 0; column < kN; ++column)
        {
            float accumulator = 0.0F;
            for (int reduction = 0; reduction < kK; ++reduction)
            {
                const float a = 1.0F * scale_a;
                const float b = 1.0F * scale_b;
                accumulator += a * b;
            }
            reference[row * kN + column] = accumulator;
        }
    }
    return reference;
}

struct VerificationResult
{
    size_t mismatch_count = 0;
    float maximum_absolute_error = 0.0F;
    float first_actual = 0.0F;
    float first_expected = 0.0F;
};

VerificationResult verify_fragment(
    const std::vector<float> &actual,
    const std::vector<float> &reference)
{
    VerificationResult result{};
    for (size_t index = 0; index < actual.size(); ++index)
    {
        const float error = std::abs(actual[index] - reference[index]);
        result.maximum_absolute_error =
            std::max(result.maximum_absolute_error, error);
        if (error > kTolerance)
        {
            if (result.mismatch_count == 0)
            {
                result.first_actual = actual[index];
                result.first_expected = reference[index];
            }
            ++result.mismatch_count;
        }
    }
    return result;
}

void report_result(
    const char *name,
    const VerificationResult &result,
    const std::vector<float> &actual)
{
    std::cout << '[' << name << " result]\n";
    std::cout << "  First D fragment value : " << actual.front() << '\n';
    std::cout << "  Last D fragment value  : " << actual.back() << '\n';
    std::cout << "  Maximum absolute error : "
              << result.maximum_absolute_error << '\n';
    std::cout << "  Mismatch count         : "
              << result.mismatch_count << " / " << actual.size() << '\n';

    if (result.mismatch_count != 0)
    {
        throw std::runtime_error(
            std::string(name) + " verification failed: expected=" +
            std::to_string(result.first_expected) +
            ", actual=" + std::to_string(result.first_actual));
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

        if (properties.major < 12)
        {
            throw std::runtime_error(
                "FP8 f8f6f4/MXFP8 block-scaled MMA requires SM120 family");
        }

        std::cout << "[Configuration]\n";
        std::cout << "  GPU                    : " << properties.name << '\n';
        std::cout << "  Compute capability     : " << properties.major << '.'
                  << properties.minor << '\n';
        std::cout << "  Warp / MMA shape       : 32 threads / m16n8k32\n";
        std::cout << "  A / B                  : E4M3 raw 0x"
                  << std::hex << static_cast<unsigned int>(kE4m3One)
                  << std::dec << " = 1.0\n";
        std::cout << "  C / D                  : FP32 / FP32\n";
        std::cout << "  MX scale               : UE8M0 raw 0x"
                  << std::hex << static_cast<unsigned int>(kUe8m0One)
                  << std::dec << " = 1.0, block size 32\n";
        std::cout << "  CPU reference          : 32.0 for all 128 outputs\n\n";

        float *device_fp8_output = nullptr;
        float *device_mxfp8_output = nullptr;
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void **>(&device_fp8_output),
                kOutputElements * sizeof(float)),
            "cudaMalloc FP8 output");
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void **>(&device_mxfp8_output),
                kOutputElements * sizeof(float)),
            "cudaMalloc MXFP8 output");

        std::cout << "[Stage 1] Execute ordinary E4M3 Tensor Core MMA\n";
        fp8_mma_kernel<<<1, kWarpThreads>>>(device_fp8_output);
        check_cuda(cudaGetLastError(), "fp8_mma_kernel launch");

        std::cout << "[Stage 2] Execute MXFP8 UE8M0 block-scaled MMA\n";
        mxfp8_mma_kernel<<<1, kWarpThreads>>>(device_mxfp8_output);
        check_cuda(cudaGetLastError(), "mxfp8_mma_kernel launch");

        std::vector<float> fp8_output(kOutputElements);
        std::vector<float> mxfp8_output(kOutputElements);
        check_cuda(
            cudaMemcpy(
                fp8_output.data(),
                device_fp8_output,
                kOutputElements * sizeof(float),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy FP8 output");
        check_cuda(
            cudaMemcpy(
                mxfp8_output.data(),
                device_mxfp8_output,
                kOutputElements * sizeof(float),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy MXFP8 output");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        const std::vector<float> fp8_reference =
            make_cpu_reference(1.0F, 1.0F);
        const std::vector<float> mxfp8_reference =
            make_cpu_reference(1.0F, 1.0F);

        std::cout << "\n[Stage 3] Compare every distributed D register "
                  << "with CPU reference\n";
        report_result(
            "ordinary FP8",
            verify_fragment(fp8_output, fp8_reference),
            fp8_output);
        std::cout << '\n';
        report_result(
            "MXFP8 block scale",
            verify_fragment(mxfp8_output, mxfp8_reference),
            mxfp8_output);

        check_cuda(cudaFree(device_fp8_output), "cudaFree FP8 output");
        check_cuda(cudaFree(device_mxfp8_output), "cudaFree MXFP8 output");

        std::cout << "\n[SUCCESS] Raw FP8 and MXFP8 PTX MMA instructions "
                  << "matched the CPU reference\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cout << "[FAILED] " << error.what() << '\n';
        return 1;
    }
}
