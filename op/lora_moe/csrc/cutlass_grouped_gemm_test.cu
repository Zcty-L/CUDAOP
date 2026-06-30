#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"

namespace
{

using Element = cutlass::bfloat16_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

using Epilogue = cutlass::epilogue::thread::LinearCombination<
    Element,
    128 / cutlass::sizeof_bits<Element>::value,
    float,
    float>;

using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    Element,
    LayoutA,
    cutlass::ComplexTransform::kNone,
    8,
    Element,
    LayoutB,
    cutlass::ComplexTransform::kNone,
    8,
    Element,
    LayoutC,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    Epilogue,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    4>::GemmKernel;

using GroupedGemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

bool check_cuda(cudaError_t status, const char* operation)
{
    if (status == cudaSuccess)
    {
        return true;
    }

    std::cerr
        << operation
        << " failed: "
        << cudaGetErrorString(status)
        << std::endl;
    return false;
}

bool check_cutlass(cutlass::Status status, const char* operation)
{
    if (status == cutlass::Status::kSuccess)
    {
        return true;
    }

    std::cerr
        << operation
        << " failed: "
        << cutlassGetStatusString(status)
        << std::endl;
    return false;
}

template <typename T>
bool allocate_and_copy(
    T** device,
    const std::vector<T>& host,
    const char* operation)
{
    const size_t bytes = host.size() * sizeof(T);
    if (!check_cuda(cudaMalloc(device, bytes), operation))
    {
        return false;
    }
    return check_cuda(
        cudaMemcpy(
            *device,
            host.data(),
            bytes,
            cudaMemcpyHostToDevice),
        operation);
}

template <typename T>
void release(T*& pointer)
{
    if (pointer != nullptr)
    {
        cudaFree(pointer);
        pointer = nullptr;
    }
}

}  // namespace

int main()
{
    constexpr int num_experts = 8;
    constexpr int hidden_in = 256;
    constexpr int hidden_out = 32;
    const std::vector<int> tokens_per_expert = {
        4,
        0,
        3,
        1,
        5,
        2,
        0,
        4,
    };

    std::vector<cutlass::gemm::GemmCoord> problem_sizes;
    std::vector<int64_t> lda;
    std::vector<int64_t> ldb;
    std::vector<int64_t> ldc;
    std::vector<int64_t> ldd;
    std::vector<int64_t> offset_a;
    std::vector<int64_t> offset_b;
    std::vector<int64_t> offset_d;

    int64_t elements_a = 0;
    int64_t elements_b = 0;
    int64_t elements_d = 0;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        const int tokens = tokens_per_expert[expert];
        const cutlass::gemm::GemmCoord problem(
            tokens,
            hidden_out,
            hidden_in);
        problem_sizes.push_back(problem);
        lda.push_back(LayoutA::packed(
            {problem.m(), problem.k()}).stride(0));
        ldb.push_back(LayoutB::packed(
            {problem.k(), problem.n()}).stride(0));
        ldc.push_back(LayoutC::packed(
            {problem.m(), problem.n()}).stride(0));
        ldd.push_back(ldc.back());
        offset_a.push_back(elements_a);
        offset_b.push_back(elements_b);
        offset_d.push_back(elements_d);
        elements_a += static_cast<int64_t>(tokens) * hidden_in;
        elements_b += static_cast<int64_t>(hidden_in) * hidden_out;
        elements_d += static_cast<int64_t>(tokens) * hidden_out;
    }

    std::vector<Element> host_a(elements_a, Element(1.0f));
    std::vector<Element> host_b(elements_b, Element(1.0f));
    std::vector<Element> host_c(elements_d, Element(0.0f));
    std::vector<Element> host_d(elements_d, Element(0.0f));

    Element* device_a = nullptr;
    Element* device_b = nullptr;
    Element* device_c = nullptr;
    Element* device_d = nullptr;
    cutlass::gemm::GemmCoord* device_problem_sizes = nullptr;
    int64_t* device_lda = nullptr;
    int64_t* device_ldb = nullptr;
    int64_t* device_ldc = nullptr;
    int64_t* device_ldd = nullptr;
    Element** device_ptr_a = nullptr;
    Element** device_ptr_b = nullptr;
    Element** device_ptr_c = nullptr;
    Element** device_ptr_d = nullptr;
    uint8_t* workspace = nullptr;

    bool success =
        allocate_and_copy(&device_a, host_a, "copy A") &&
        allocate_and_copy(&device_b, host_b, "copy B") &&
        allocate_and_copy(&device_c, host_c, "copy C") &&
        allocate_and_copy(&device_d, host_d, "copy D") &&
        allocate_and_copy(
            &device_problem_sizes,
            problem_sizes,
            "copy problem sizes") &&
        allocate_and_copy(&device_lda, lda, "copy lda") &&
        allocate_and_copy(&device_ldb, ldb, "copy ldb") &&
        allocate_and_copy(&device_ldc, ldc, "copy ldc") &&
        allocate_and_copy(&device_ldd, ldd, "copy ldd");

    std::vector<Element*> host_ptr_a(num_experts);
    std::vector<Element*> host_ptr_b(num_experts);
    std::vector<Element*> host_ptr_c(num_experts);
    std::vector<Element*> host_ptr_d(num_experts);
    if (success)
    {
        for (int expert = 0; expert < num_experts; ++expert)
        {
            host_ptr_a[expert] = device_a + offset_a[expert];
            host_ptr_b[expert] = device_b + offset_b[expert];
            host_ptr_c[expert] = device_c + offset_d[expert];
            host_ptr_d[expert] = device_d + offset_d[expert];
        }

        success =
            allocate_and_copy(
                &device_ptr_a,
                host_ptr_a,
                "copy A pointers") &&
            allocate_and_copy(
                &device_ptr_b,
                host_ptr_b,
                "copy B pointers") &&
            allocate_and_copy(
                &device_ptr_c,
                host_ptr_c,
                "copy C pointers") &&
            allocate_and_copy(
                &device_ptr_d,
                host_ptr_d,
                "copy D pointers");
    }

    if (success)
    {
        const int threadblock_count = GroupedGemm::sufficient(
            problem_sizes.data(),
            num_experts);
        if (threadblock_count <= 0)
        {
            std::cerr
                << "CUTLASS grouped GEMM has insufficient resources"
                << std::endl;
            success = false;
        }
        else
        {
            typename GroupedGemm::EpilogueOutputOp::Params epilogue(
                1.0f,
                0.0f);
            typename GroupedGemm::Arguments arguments(
                device_problem_sizes,
                num_experts,
                threadblock_count,
                epilogue,
                device_ptr_a,
                device_ptr_b,
                device_ptr_c,
                device_ptr_d,
                device_lda,
                device_ldb,
                device_ldc,
                device_ldd,
                problem_sizes.data());

            GroupedGemm grouped_gemm;
            const size_t workspace_bytes =
                grouped_gemm.get_workspace_size(arguments);
            if (workspace_bytes > 0)
            {
                success = check_cuda(
                    cudaMalloc(&workspace, workspace_bytes),
                    "allocate workspace");
            }
            if (success)
            {
                success = check_cutlass(
                    grouped_gemm.initialize(arguments, workspace),
                    "initialize grouped GEMM");
            }
            if (success)
            {
                success = check_cutlass(
                    grouped_gemm.run(),
                    "run grouped GEMM");
            }
            if (success)
            {
                success = check_cuda(
                    cudaDeviceSynchronize(),
                    "synchronize grouped GEMM");
            }
        }
    }

    if (success)
    {
        success = check_cuda(
            cudaMemcpy(
                host_d.data(),
                device_d,
                host_d.size() * sizeof(Element),
                cudaMemcpyDeviceToHost),
            "copy D to host");
    }

    if (success)
    {
        const float expected = static_cast<float>(hidden_in);
        for (const Element value : host_d)
        {
            if (std::abs(static_cast<float>(value) - expected) > 1.0f)
            {
                std::cerr
                    << "incorrect output: expected "
                    << expected
                    << ", got "
                    << static_cast<float>(value)
                    << std::endl;
                success = false;
                break;
            }
        }
    }

    release(workspace);
    release(device_ptr_d);
    release(device_ptr_c);
    release(device_ptr_b);
    release(device_ptr_a);
    release(device_ldd);
    release(device_ldc);
    release(device_ldb);
    release(device_lda);
    release(device_problem_sizes);
    release(device_d);
    release(device_c);
    release(device_b);
    release(device_a);

    if (!success)
    {
        return 1;
    }

    std::cout
        << "CUTLASS BF16 grouped GEMM test passed"
        << std::endl;
    return 0;
}
