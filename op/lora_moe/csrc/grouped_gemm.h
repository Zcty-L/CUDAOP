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

namespace lora_moe
{

template <bool Transpose>
using GroupedLayout = std::conditional_t<
    Transpose,
    cutlass::layout::ColumnMajor,
    cutlass::layout::RowMajor>;

using GroupedElement = cutlass::bfloat16_t;
using GroupedThreadblockShape =
    cutlass::gemm::GemmShape<128, 128, 32>;
using GroupedWarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using GroupedInstructionShape =
    cutlass::gemm::GemmShape<16, 8, 16>;
using GroupedEpilogue =
    cutlass::epilogue::thread::LinearCombination<
        GroupedElement,
        128 / cutlass::sizeof_bits<GroupedElement>::value,
        float,
        float>;

template <bool TransposeA, bool TransposeB>
using GroupedKernel =
    typename cutlass::gemm::kernel::DefaultGemmGrouped<
        GroupedElement,
        GroupedLayout<TransposeA>,
        cutlass::ComplexTransform::kNone,
        8,
        GroupedElement,
        GroupedLayout<TransposeB>,
        cutlass::ComplexTransform::kNone,
        8,
        GroupedElement,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        GroupedThreadblockShape,
        GroupedWarpShape,
        GroupedInstructionShape,
        GroupedEpilogue,
        cutlass::gemm::threadblock::
            GemmBatchedIdentityThreadblockSwizzle,
        4>::GemmKernel;

template <bool TransposeA, bool TransposeB>
using GroupedOperator = cutlass::gemm::device::GemmGrouped<
    GroupedKernel<TransposeA, TransposeB>>;

inline void check_grouped_cuda(
    cudaError_t status,
    const char* operation)
{
    TORCH_CHECK(
        status == cudaSuccess,
        operation,
        " failed: ",
        cudaGetErrorString(status));
}

template <typename T>
torch::Tensor copy_grouped_metadata_to_device(
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
    check_grouped_cuda(
        cudaMemcpyAsync(
            output.data_ptr(),
            values.data(),
            bytes,
            cudaMemcpyHostToDevice,
            c10::cuda::getCurrentCUDAStream()),
        "copy grouped GEMM metadata");
    return output;
}

template <bool TransposeA, bool TransposeB>
torch::Tensor run_grouped_gemm(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor batch_sizes)
{
    static_assert(
        !(TransposeA && TransposeB),
        "A and B cannot both be transposed");

    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.dim() == 2, "a must be two-dimensional");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
    TORCH_CHECK(
        a.scalar_type() == torch::kBFloat16,
        "a must use bfloat16");
    TORCH_CHECK(
        b.scalar_type() == torch::kBFloat16,
        "b must use bfloat16");
    TORCH_CHECK(
        !batch_sizes.is_cuda(),
        "batch_sizes must be a CPU tensor");
    TORCH_CHECK(
        batch_sizes.scalar_type() == torch::kInt64,
        "batch_sizes must use int64");
    TORCH_CHECK(
        batch_sizes.dim() == 1,
        "batch_sizes must be one-dimensional");
    TORCH_CHECK(
        batch_sizes.is_contiguous(),
        "batch_sizes must be contiguous");

    const int64_t num_experts = batch_sizes.numel();
    const int64_t* sizes = batch_sizes.data_ptr<int64_t>();
    int64_t total_rows = 0;
    for (int64_t expert = 0; expert < num_experts; ++expert)
    {
        TORCH_CHECK(
            sizes[expert] >= 0,
            "batch_sizes must be non-negative");
        total_rows += sizes[expert];
    }
    TORCH_CHECK(
        total_rows == a.size(0),
        "sum(batch_sizes) must equal a.size(0)");

    using Operator = GroupedOperator<TransposeA, TransposeB>;
    using LayoutA = typename Operator::LayoutA;
    using LayoutB = typename Operator::LayoutB;
    using LayoutC = typename Operator::LayoutC;

    const int64_t hidden_in = a.size(1);
    int64_t hidden_out = 0;
    torch::Tensor output;
    if constexpr (TransposeA)
    {
        TORCH_CHECK(
            b.dim() == 2 && b.size(0) == total_rows,
            "transposed-A b must have shape [tokens, hidden_out]");
        hidden_out = b.size(1);
        output = torch::zeros(
            {num_experts, hidden_in, hidden_out},
            a.options());
    }
    else
    {
        TORCH_CHECK(
            b.dim() == 3 && b.size(0) == num_experts,
            "b must have shape [num_experts, *, *]");
        const int64_t contraction =
            TransposeB ? b.size(2) : b.size(1);
        TORCH_CHECK(
            contraction == hidden_in,
            "grouped GEMM contraction dimensions must match");
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
    std::vector<GroupedElement*> ptr_a;
    std::vector<GroupedElement*> ptr_b;
    std::vector<GroupedElement*> ptr_c;
    std::vector<GroupedElement*> ptr_d;
    problems.reserve(num_experts);
    lda.reserve(num_experts);
    ldb.reserve(num_experts);
    ldc.reserve(num_experts);
    ldd.reserve(num_experts);
    ptr_a.reserve(num_experts);
    ptr_b.reserve(num_experts);
    ptr_c.reserve(num_experts);
    ptr_d.reserve(num_experts);

    auto* a_data =
        reinterpret_cast<GroupedElement*>(a.data_ptr());
    auto* b_data =
        reinterpret_cast<GroupedElement*>(b.data_ptr());
    auto* output_data =
        reinterpret_cast<GroupedElement*>(output.data_ptr());
    int64_t offset_a = 0;
    int64_t offset_b = 0;
    int64_t offset_output = 0;

    for (int64_t expert = 0; expert < num_experts; ++expert)
    {
        const int64_t group_rows = sizes[expert];
        cutlass::gemm::GemmCoord problem;
        if constexpr (TransposeA)
        {
            problem = cutlass::gemm::GemmCoord(
                hidden_in,
                hidden_out,
                group_rows);
        }
        else
        {
            problem = cutlass::gemm::GemmCoord(
                group_rows,
                hidden_out,
                hidden_in);
        }

        cutlass::gemm::GemmCoord layout_problem = problem;
        if constexpr (TransposeA)
        {
            if (group_rows == 0)
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
            offset_a += group_rows * hidden_in;
            offset_b += group_rows * hidden_out;
            offset_output += hidden_in * hidden_out;
        }
        else
        {
            offset_a += group_rows * hidden_in;
            offset_b += b.size(1) * b.size(2);
            offset_output += group_rows * hidden_out;
        }
    }

    const int threadblock_count = Operator::sufficient(
        problems.data(),
        static_cast<int>(num_experts));
    TORCH_CHECK(
        threadblock_count > 0,
        "CUTLASS grouped GEMM has insufficient resources");

    torch::Tensor device_problems =
        copy_grouped_metadata_to_device(
            problems,
            a.device());
    torch::Tensor device_lda =
        copy_grouped_metadata_to_device(lda, a.device());
    torch::Tensor device_ldb =
        copy_grouped_metadata_to_device(ldb, a.device());
    torch::Tensor device_ldc =
        copy_grouped_metadata_to_device(ldc, a.device());
    torch::Tensor device_ldd =
        copy_grouped_metadata_to_device(ldd, a.device());
    torch::Tensor device_ptr_a =
        copy_grouped_metadata_to_device(ptr_a, a.device());
    torch::Tensor device_ptr_b =
        copy_grouped_metadata_to_device(ptr_b, a.device());
    torch::Tensor device_ptr_c =
        copy_grouped_metadata_to_device(ptr_c, a.device());
    torch::Tensor device_ptr_d =
        copy_grouped_metadata_to_device(ptr_d, a.device());

    typename Operator::EpilogueOutputOp::Params epilogue(
        1.0f,
        0.0f);
    typename Operator::Arguments arguments(
        reinterpret_cast<cutlass::gemm::GemmCoord*>(
            device_problems.data_ptr()),
        static_cast<int>(num_experts),
        threadblock_count,
        epilogue,
        reinterpret_cast<GroupedElement**>(
            device_ptr_a.data_ptr()),
        reinterpret_cast<GroupedElement**>(
            device_ptr_b.data_ptr()),
        reinterpret_cast<GroupedElement**>(
            device_ptr_c.data_ptr()),
        reinterpret_cast<GroupedElement**>(
            device_ptr_d.data_ptr()),
        reinterpret_cast<int64_t*>(device_lda.data_ptr()),
        reinterpret_cast<int64_t*>(device_ldb.data_ptr()),
        reinterpret_cast<int64_t*>(device_ldc.data_ptr()),
        reinterpret_cast<int64_t*>(device_ldd.data_ptr()),
        problems.data());

    Operator grouped_gemm;
    const int64_t workspace_bytes =
        static_cast<int64_t>(
            grouped_gemm.get_workspace_size(arguments));
    torch::Tensor workspace = torch::empty(
        {workspace_bytes},
        torch::TensorOptions()
            .dtype(torch::kInt8)
            .device(a.device()));
    cutlass::Status status = grouped_gemm.initialize(
        arguments,
        workspace.data_ptr());
    TORCH_CHECK(
        status == cutlass::Status::kSuccess,
        "failed to initialize CUTLASS grouped GEMM: ",
        cutlassGetStatusString(status));
    status = grouped_gemm.run(
        c10::cuda::getCurrentCUDAStream());
    TORCH_CHECK(
        status == cutlass::Status::kSuccess,
        "failed to run CUTLASS grouped GEMM: ",
        cutlassGetStatusString(status));
    return output;
}

inline torch::Tensor grouped_gemm(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor batch_sizes,
    bool transpose_a,
    bool transpose_b)
{
    TORCH_CHECK(
        !(transpose_a && transpose_b),
        "transpose_a and transpose_b cannot both be true");
    if (transpose_a)
    {
        return run_grouped_gemm<true, false>(
            a,
            b,
            batch_sizes);
    }
    if (transpose_b)
    {
        return run_grouped_gemm<false, true>(
            a,
            b,
            batch_sizes);
    }
    return run_grouped_gemm<false, false>(
        a,
        b,
        batch_sizes);
}

}  // namespace lora_moe
