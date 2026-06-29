#pragma once

#include <cstdint>

#include <c10/cuda/CUDAStream.h>
#include <cub/cub.cuh>
#include <torch/extension.h>

#include "common.h"

namespace lora_moe
{

template <typename T>
void run_cumsum(
    const torch::Tensor& input,
    torch::Tensor& output,
    bool inclusive)
{
    size_t scratchpad_bytes = 0;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    if (inclusive)
    {
        LORA_MOE_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
            nullptr,
            scratchpad_bytes,
            input.data_ptr<T>(),
            output.data_ptr<T>(),
            input.size(1),
            stream));
    }
    else
    {
        LORA_MOE_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            nullptr,
            scratchpad_bytes,
            input.data_ptr<T>(),
            output.data_ptr<T>(),
            input.size(1),
            stream));
    }

    const auto options = torch::TensorOptions()
                             .dtype(torch::kInt8)
                             .device(input.device());
    torch::Tensor scratchpad = torch::empty(
        {static_cast<int64_t>(scratchpad_bytes * input.size(0))},
        options);

    for (int64_t row = 0; row < input.size(0); ++row)
    {
        void* row_scratchpad =
            scratchpad.data_ptr<int8_t>() + scratchpad_bytes * row;
        T* input_row = input.data_ptr<T>() + input.size(1) * row;
        T* output_row = output.data_ptr<T>() + output.size(1) * row;

        if (inclusive)
        {
            LORA_MOE_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
                row_scratchpad,
                scratchpad_bytes,
                input_row,
                output_row,
                input.size(1),
                stream));
        }
        else
        {
            LORA_MOE_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
                row_scratchpad,
                scratchpad_bytes,
                input_row,
                output_row,
                input.size(1),
                stream));
        }
    }
}

inline void check_cumsum_arguments(
    const torch::Tensor& input,
    int64_t dim,
    const torch::Tensor& output)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be two-dimensional");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(
        input.scalar_type() == torch::kInt16 ||
            input.scalar_type() == torch::kInt32 ||
            input.scalar_type() == torch::kInt64,
        "input dtype must be int16, int32, or int64");
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(output.dim() == 2, "output must be two-dimensional");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(
        output.sizes() == input.sizes(),
        "input and output shapes must match");
    TORCH_CHECK(
        output.scalar_type() == input.scalar_type(),
        "input and output dtypes must match");
    TORCH_CHECK(dim == 1, "only dim=1 is supported");
}

inline void dispatch_cumsum(
    const torch::Tensor& input,
    torch::Tensor& output,
    bool inclusive)
{
    switch (input.scalar_type())
    {
        case torch::kInt16:
            run_cumsum<int16_t>(input, output, inclusive);
            break;
        case torch::kInt32:
            run_cumsum<int32_t>(input, output, inclusive);
            break;
        case torch::kInt64:
            run_cumsum<int64_t>(input, output, inclusive);
            break;
        default:
            TORCH_CHECK(false, "unsupported cumsum dtype");
    }
}

inline void exclusive_cumsum(
    torch::Tensor input,
    int64_t dim,
    torch::Tensor output)
{
    check_cumsum_arguments(input, dim, output);
    dispatch_cumsum(input, output, false);
}

inline void inclusive_cumsum(
    torch::Tensor input,
    int64_t dim,
    torch::Tensor output)
{
    check_cumsum_arguments(input, dim, output);
    dispatch_cumsum(input, output, true);
}

}  // namespace lora_moe
