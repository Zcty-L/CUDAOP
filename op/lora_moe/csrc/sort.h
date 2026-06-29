#pragma once

#include <cstdint>

#include <c10/cuda/CUDAStream.h>
#include <cub/cub.cuh>
#include <torch/extension.h>

#include "common.h"

namespace lora_moe
{

template <typename T>
void run_sort(
    const torch::Tensor& input,
    int64_t end_bit,
    torch::Tensor& sorted,
    torch::Tensor& indices)
{
    torch::Tensor input_indices = torch::arange(
        input.numel(),
        input.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    size_t scratchpad_bytes = 0;

    LORA_MOE_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr,
        scratchpad_bytes,
        input.data_ptr<T>(),
        sorted.data_ptr<T>(),
        input_indices.data_ptr<T>(),
        indices.data_ptr<T>(),
        input.numel(),
        0,
        end_bit,
        stream));

    torch::Tensor scratchpad = torch::empty(
        {static_cast<int64_t>(scratchpad_bytes)},
        torch::TensorOptions()
            .dtype(torch::kInt8)
            .device(input.device()));
    LORA_MOE_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        scratchpad.data_ptr(),
        scratchpad_bytes,
        input.data_ptr<T>(),
        sorted.data_ptr<T>(),
        input_indices.data_ptr<T>(),
        indices.data_ptr<T>(),
        input.numel(),
        0,
        end_bit,
        stream));
}

inline void sort(
    torch::Tensor input,
    int64_t end_bit,
    torch::Tensor sorted,
    torch::Tensor indices)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 1, "input must be one-dimensional");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(
        input.scalar_type() == torch::kInt16 ||
            input.scalar_type() == torch::kInt32 ||
            input.scalar_type() == torch::kInt64,
        "input dtype must be int16, int32, or int64");
    TORCH_CHECK(sorted.is_cuda(), "sorted must be a CUDA tensor");
    TORCH_CHECK(
        sorted.dim() == 1 && sorted.is_contiguous(),
        "sorted must be a contiguous one-dimensional tensor");
    TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
    TORCH_CHECK(
        indices.dim() == 1 && indices.is_contiguous(),
        "indices must be a contiguous one-dimensional tensor");
    TORCH_CHECK(
        sorted.sizes() == input.sizes() &&
            indices.sizes() == input.sizes(),
        "input, sorted, and indices shapes must match");
    TORCH_CHECK(
        sorted.scalar_type() == input.scalar_type() &&
            indices.scalar_type() == input.scalar_type(),
        "input, sorted, and indices dtypes must match");
    TORCH_CHECK(end_bit > 0, "end_bit must be positive");
    TORCH_CHECK(
        end_bit <= input.element_size() * 8,
        "end_bit exceeds the input dtype width");

    if (input.numel() == 0)
    {
        return;
    }

    switch (input.scalar_type())
    {
        case torch::kInt16:
            run_sort<int16_t>(input, end_bit, sorted, indices);
            break;
        case torch::kInt32:
            run_sort<int32_t>(input, end_bit, sorted, indices);
            break;
        case torch::kInt64:
            run_sort<int64_t>(input, end_bit, sorted, indices);
            break;
        default:
            TORCH_CHECK(false, "unsupported sort dtype");
    }
}

}  // namespace lora_moe
