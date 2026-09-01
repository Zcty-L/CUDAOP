#pragma once

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <type_traits>
#include <vector>

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"

namespace cudaop::grouped_gemm
{

template <bool Transpose>
using Layout = std::conditional_t<
    Transpose,
    cutlass::layout::ColumnMajor,
    cutlass::layout::RowMajor>;

using Element = cutlass::bfloat16_t;

struct K32ShapeConfig
{
    using ThreadblockShape =
        cutlass::gemm::GemmShape<128, 128, 32>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
    static constexpr int kStages = 4;
};

struct K16ShapeConfig
{
    using ThreadblockShape =
        cutlass::gemm::GemmShape<128, 128, 16>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
    static constexpr int kStages = 4;
};

using Epilogue = cutlass::epilogue::thread::LinearCombination<
    Element,
    128 / cutlass::sizeof_bits<Element>::value,
    float,
    float>;

template <typename ShapeConfig, bool TransposeA, bool TransposeB>
using Kernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    Element,
    Layout<TransposeA>,
    cutlass::ComplexTransform::kNone,
    8,
    Element,
    Layout<TransposeB>,
    cutlass::ComplexTransform::kNone,
    8,
    Element,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    typename ShapeConfig::ThreadblockShape,
    typename ShapeConfig::WarpShape,
    typename ShapeConfig::InstructionShape,
    Epilogue,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    ShapeConfig::kStages>::GemmKernel;

template <typename ShapeConfig, bool TransposeA, bool TransposeB>
using Operator = cutlass::gemm::device::GemmGrouped<
    Kernel<ShapeConfig, TransposeA, TransposeB>>;

inline void check_cuda(
    cudaError_t status,
    const char* operation)
{
    TORCH_CHECK(
        status == cudaSuccess,
        operation,
        " 失败：",
        cudaGetErrorString(status));
}

template <typename T>
torch::Tensor copy_metadata(
    const std::vector<T>& values,
    const torch::Device& device)
{
    const int64_t bytes =
        static_cast<int64_t>(values.size() * sizeof(T));
    torch::Tensor output = torch::empty(
        {bytes},
        torch::TensorOptions()
            .dtype(torch::kInt8)
            .device(device));
    check_cuda(
        cudaMemcpyAsync(
            output.data_ptr(),
            values.data(),
            bytes,
            cudaMemcpyHostToDevice,
            c10::cuda::getCurrentCUDAStream()),
        "复制 Grouped GEMM 元数据");
    return output;
}

template <typename ShapeConfig, bool TransposeA, bool TransposeB>
torch::Tensor run(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor batch_sizes)
{
    static_assert(
        !(TransposeA && TransposeB),
        "A 和 B 不能同时转置");

    TORCH_CHECK(a.is_cuda(), "a 必须是 CUDA Tensor");
    TORCH_CHECK(b.is_cuda(), "b 必须是 CUDA Tensor");
    TORCH_CHECK(
        a.device() == b.device(),
        "a 和 b 必须位于同一 CUDA 设备");
    TORCH_CHECK(a.dim() == 2, "a 必须是二维 Tensor");
    TORCH_CHECK(a.is_contiguous(), "a 必须连续");
    TORCH_CHECK(b.is_contiguous(), "b 必须连续");
    TORCH_CHECK(
        a.scalar_type() == torch::kBFloat16,
        "a 必须使用 bfloat16");
    TORCH_CHECK(
        b.scalar_type() == torch::kBFloat16,
        "b 必须使用 bfloat16");
    TORCH_CHECK(
        !batch_sizes.is_cuda(),
        "batch_sizes 必须是 CPU Tensor");
    TORCH_CHECK(
        batch_sizes.scalar_type() == torch::kInt64,
        "batch_sizes 必须使用 int64");
    TORCH_CHECK(
        batch_sizes.dim() == 1 && batch_sizes.is_contiguous(),
        "batch_sizes 必须是一维连续 Tensor");

    const int64_t groups = batch_sizes.numel();
    TORCH_CHECK(groups > 0, "batch_sizes 不能为空");
    const int64_t* sizes = batch_sizes.data_ptr<int64_t>();
    int64_t total_rows = 0;
    for (int64_t group = 0; group < groups; ++group)
    {
        TORCH_CHECK(
            sizes[group] >= 0,
            "batch_sizes 不能包含负数");
        total_rows += sizes[group];
    }
    TORCH_CHECK(
        total_rows == a.size(0),
        "batch_sizes 之和必须等于 a.size(0)");

    using Gemm = Operator<ShapeConfig, TransposeA, TransposeB>;
    using LayoutA = typename Gemm::LayoutA;
    using LayoutB = typename Gemm::LayoutB;
    using LayoutC = typename Gemm::LayoutC;

    const int64_t hidden_in = a.size(1);
    int64_t hidden_out = 0;
    torch::Tensor output;
    if constexpr (TransposeA)
    {
        TORCH_CHECK(
            b.dim() == 2 && b.size(0) == total_rows,
            "转置 A 模式下 b 的形状必须是 [tokens, hidden_out]");
        hidden_out = b.size(1);
        output = torch::zeros(
            {groups, hidden_in, hidden_out},
            a.options());
    }
    else
    {
        TORCH_CHECK(
            b.dim() == 3 && b.size(0) == groups,
            "b 的第一维必须等于 group 数量");
        const int64_t contraction =
            TransposeB ? b.size(2) : b.size(1);
        TORCH_CHECK(
            contraction == hidden_in,
            "a 和 b 的收缩维度不匹配");
        hidden_out = TransposeB ? b.size(1) : b.size(2);
        output = torch::empty(
            {total_rows, hidden_out},
            a.options());
    }

    std::vector<cutlass::gemm::GemmCoord> problems;
    std::vector<int64_t> lda;
    std::vector<int64_t> ldb;
    std::vector<int64_t> ldc;
    std::vector<int64_t> ldd;
    std::vector<Element*> ptr_a;
    std::vector<Element*> ptr_b;
    std::vector<Element*> ptr_c;
    std::vector<Element*> ptr_d;
    problems.reserve(groups);
    lda.reserve(groups);
    ldb.reserve(groups);
    ldc.reserve(groups);
    ldd.reserve(groups);
    ptr_a.reserve(groups);
    ptr_b.reserve(groups);
    ptr_c.reserve(groups);
    ptr_d.reserve(groups);

    auto* a_data = reinterpret_cast<Element*>(a.data_ptr());
    auto* b_data = reinterpret_cast<Element*>(b.data_ptr());
    auto* output_data =
        reinterpret_cast<Element*>(output.data_ptr());
    int64_t offset_a = 0;
    int64_t offset_b = 0;
    int64_t offset_output = 0;

    for (int64_t group = 0; group < groups; ++group)
    {
        const int64_t rows = sizes[group];
        cutlass::gemm::GemmCoord problem;
        if constexpr (TransposeA)
        {
            problem = cutlass::gemm::GemmCoord(
                hidden_in,
                hidden_out,
                rows);
        }
        else
        {
            problem = cutlass::gemm::GemmCoord(
                rows,
                hidden_out,
                hidden_in);
        }
        const cutlass::gemm::GemmCoord layout_problem = problem;
        if constexpr (TransposeA)
        {
            if (rows == 0)
            {
                problem = cutlass::gemm::GemmCoord(0, 0, 0);
            }
        }

        problems.push_back(problem);
        lda.push_back(
            LayoutA::packed(
                {
                    layout_problem.m(),
                    layout_problem.k(),
                }).stride(0));
        ldb.push_back(
            LayoutB::packed(
                {
                    layout_problem.k(),
                    layout_problem.n(),
                }).stride(0));
        ldc.push_back(
            LayoutC::packed(
                {
                    layout_problem.m(),
                    layout_problem.n(),
                }).stride(0));
        ldd.push_back(ldc.back());
        ptr_a.push_back(a_data + offset_a);
        ptr_b.push_back(b_data + offset_b);
        ptr_c.push_back(output_data + offset_output);
        ptr_d.push_back(output_data + offset_output);

        if constexpr (TransposeA)
        {
            offset_a += rows * hidden_in;
            offset_b += rows * hidden_out;
            offset_output += hidden_in * hidden_out;
        }
        else
        {
            offset_a += rows * hidden_in;
            offset_b += b.size(1) * b.size(2);
            offset_output += rows * hidden_out;
        }
    }

    const int threadblocks = Gemm::sufficient(
        problems.data(),
        static_cast<int>(groups));
    TORCH_CHECK(
        threadblocks > 0,
        "CUTLASS Grouped GEMM 资源不足");

    torch::Tensor device_problems =
        copy_metadata(problems, a.device());
    torch::Tensor device_lda = copy_metadata(lda, a.device());
    torch::Tensor device_ldb = copy_metadata(ldb, a.device());
    torch::Tensor device_ldc = copy_metadata(ldc, a.device());
    torch::Tensor device_ldd = copy_metadata(ldd, a.device());
    torch::Tensor device_ptr_a = copy_metadata(ptr_a, a.device());
    torch::Tensor device_ptr_b = copy_metadata(ptr_b, a.device());
    torch::Tensor device_ptr_c = copy_metadata(ptr_c, a.device());
    torch::Tensor device_ptr_d = copy_metadata(ptr_d, a.device());

    typename Gemm::EpilogueOutputOp::Params epilogue(1.0f, 0.0f);
    typename Gemm::Arguments arguments(
        reinterpret_cast<cutlass::gemm::GemmCoord*>(
            device_problems.data_ptr()),
        static_cast<int>(groups),
        threadblocks,
        epilogue,
        reinterpret_cast<Element**>(device_ptr_a.data_ptr()),
        reinterpret_cast<Element**>(device_ptr_b.data_ptr()),
        reinterpret_cast<Element**>(device_ptr_c.data_ptr()),
        reinterpret_cast<Element**>(device_ptr_d.data_ptr()),
        reinterpret_cast<int64_t*>(device_lda.data_ptr()),
        reinterpret_cast<int64_t*>(device_ldb.data_ptr()),
        reinterpret_cast<int64_t*>(device_ldc.data_ptr()),
        reinterpret_cast<int64_t*>(device_ldd.data_ptr()),
        problems.data());

    Gemm gemm;
    const int64_t workspace_bytes =
        static_cast<int64_t>(gemm.get_workspace_size(arguments));
    torch::Tensor workspace = torch::empty(
        {workspace_bytes},
        torch::TensorOptions()
            .dtype(torch::kInt8)
            .device(a.device()));
    cutlass::Status status = gemm.initialize(
        arguments,
        workspace.data_ptr());
    TORCH_CHECK(
        status == cutlass::Status::kSuccess,
        "初始化 CUTLASS Grouped GEMM 失败：",
        cutlassGetStatusString(status));
    status = gemm.run(c10::cuda::getCurrentCUDAStream());
    TORCH_CHECK(
        status == cutlass::Status::kSuccess,
        "运行 CUTLASS Grouped GEMM 失败：",
        cutlassGetStatusString(status));
    return output;
}

template <typename ShapeConfig>
torch::Tensor grouped_gemm_with_config(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor batch_sizes,
    bool transpose_a,
    bool transpose_b)
{
    TORCH_CHECK(
        !(transpose_a && transpose_b),
        "A 和 B 不能同时转置");
    if (transpose_a)
    {
        return run<ShapeConfig, true, false>(a, b, batch_sizes);
    }
    if (transpose_b)
    {
        return run<ShapeConfig, false, true>(a, b, batch_sizes);
    }
    return run<ShapeConfig, false, false>(a, b, batch_sizes);
}

inline torch::Tensor grouped_gemm(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor batch_sizes,
    bool transpose_a,
    bool transpose_b)
{
    return grouped_gemm_with_config<K32ShapeConfig>(
        a,
        b,
        batch_sizes,
        transpose_a,
        transpose_b);
}

inline torch::Tensor grouped_gemm_k16(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor batch_sizes,
    bool transpose_a,
    bool transpose_b)
{
    return grouped_gemm_with_config<K16ShapeConfig>(
        a,
        b,
        batch_sizes,
        transpose_a,
        transpose_b);
}

}  // namespace cudaop::grouped_gemm
