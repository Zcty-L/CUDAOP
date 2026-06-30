#pragma once

#include <cmath>
#include <cstdint>

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "common.h"

namespace lora_moe
{
namespace detail
{

constexpr int kIndexThreads = 32;

__global__ void construct_indices_kernel(
    int16_t* __restrict__ indices,
    int32_t num_columns,
    int32_t block_size,
    const int32_t* __restrict__ padded_bins)
{
    int32_t start = 0;
    if (blockIdx.x > 0)
    {
        start = __ldg(padded_bins + blockIdx.x - 1);
    }
    const int32_t end = __ldg(padded_bins + blockIdx.x);

    start /= block_size;
    const int32_t end_block = end / block_size;
    indices +=
        (start + blockIdx.y) * num_columns + threadIdx.x;

    int32_t bin_offset = blockIdx.y;
    int32_t num_rows = end_block - start;
    for (; bin_offset < num_rows; num_rows -= gridDim.y)
    {
        int16_t* output = indices;
        for (
            int32_t column = threadIdx.x;
            column < num_columns;
            column += kIndexThreads)
        {
            *output = static_cast<int16_t>(
                column + blockIdx.x * num_columns);
            output += kIndexThreads;
        }
        indices += gridDim.y * num_columns;
    }
}

}  // namespace detail

inline void indices(
    torch::Tensor padded_bins,
    int64_t block_size,
    int64_t output_block_rows,
    int64_t output_block_columns,
    torch::Tensor output)
{
    TORCH_CHECK(
        padded_bins.is_cuda(),
        "padded_bins must be a CUDA tensor");
    TORCH_CHECK(
        padded_bins.dim() == 1 &&
            padded_bins.scalar_type() == torch::kInt32,
        "padded_bins must be a one-dimensional int32 tensor");
    TORCH_CHECK(
        padded_bins.is_contiguous(),
        "padded_bins must be contiguous");
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(
        output.dim() == 1 &&
            output.scalar_type() == torch::kInt16,
        "output must be a one-dimensional int16 tensor");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(
        output.numel() ==
            output_block_rows * output_block_columns,
        "output has an invalid number of elements");

    if (output.numel() == 0)
    {
        return;
    }

    const int64_t num_bins = padded_bins.numel();
    TORCH_CHECK(num_bins > 0, "padded_bins must not be empty");
    const int64_t grid_y =
        (output_block_rows + num_bins - 1) / num_bins;
    TORCH_CHECK(grid_y > 0, "output_block_rows must be positive");

    const dim3 grid(
        static_cast<uint32_t>(num_bins),
        static_cast<uint32_t>(grid_y));
    detail::construct_indices_kernel<<<
        grid,
        detail::kIndexThreads,
        0,
        c10::cuda::getCurrentCUDAStream()>>>(
        output.data_ptr<int16_t>(),
        static_cast<int32_t>(output_block_columns),
        static_cast<int32_t>(block_size),
        padded_bins.data_ptr<int32_t>());
    LORA_MOE_CUDA_CHECK(cudaGetLastError());
}

}  // namespace lora_moe
