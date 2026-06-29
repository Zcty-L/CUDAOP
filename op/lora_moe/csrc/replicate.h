#pragma once

#include <cstdint>

#include <c10/cuda/CUDAStream.h>
#include <c10/util/Half.h>
#include <cub/cub.cuh>
#include <torch/extension.h>

#include "common.h"

namespace lora_moe
{
namespace detail
{

template <typename T, int kThreads>
__global__ void replicate_forward_kernel(
    const T* __restrict__ input,
    const int32_t* __restrict__ bins,
    T* __restrict__ output,
    int32_t columns)
{
    const int32_t batch = blockIdx.y;
    const int32_t num_bins = gridDim.x;
    input += batch * num_bins;
    output += batch * columns;

    const int32_t bin = blockIdx.x;
    int32_t start = 0;
    if (bin > 0)
    {
        start = __ldg(bins + bin - 1);
    }
    const int32_t end = __ldg(bins + bin);
    const T value = __ldg(input + bin);

    int32_t offset = blockIdx.z * kThreads + threadIdx.x;
    output += start + offset;
    const int32_t stride = gridDim.z * kThreads;
    const int32_t count = end - start;
    for (; offset < count; offset += stride)
    {
        *output = value;
        output += stride;
    }
}

template <typename T>
void launch_replicate_forward(
    const torch::Tensor& input,
    const torch::Tensor& bins,
    torch::Tensor& output)
{
    constexpr int kThreads = 64;
    const int64_t groups =
        (output.size(1) + bins.numel() * kThreads - 1) /
        (bins.numel() * kThreads);
    const dim3 grid(
        static_cast<uint32_t>(bins.numel()),
        static_cast<uint32_t>(input.size(0)),
        static_cast<uint32_t>(groups));
    replicate_forward_kernel<T, kThreads><<<
        grid,
        kThreads,
        0,
        c10::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<T>(),
        bins.data_ptr<int32_t>(),
        output.data_ptr<T>(),
        static_cast<int32_t>(output.size(1)));
    LORA_MOE_CUDA_CHECK(cudaGetLastError());
}

inline void segmented_sum(
    const torch::Tensor& gradient,
    const torch::Tensor& bins,
    torch::Tensor& output)
{
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    torch::Tensor offsets = torch::empty(
        {bins.numel() + 1},
        bins.options());
    LORA_MOE_CUDA_CHECK(cudaMemsetAsync(
        offsets.data_ptr<int32_t>(),
        0,
        sizeof(int32_t),
        stream));
    LORA_MOE_CUDA_CHECK(cudaMemcpyAsync(
        offsets.data_ptr<int32_t>() + 1,
        bins.data_ptr<int32_t>(),
        bins.numel() * sizeof(int32_t),
        cudaMemcpyDeviceToDevice,
        stream));

    size_t scratchpad_bytes = 0;
    LORA_MOE_CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(
        nullptr,
        scratchpad_bytes,
        gradient.data_ptr<c10::Half>(),
        output.data_ptr<c10::Half>(),
        bins.numel(),
        offsets.data_ptr<int32_t>(),
        offsets.data_ptr<int32_t>() + 1,
        stream));
    torch::Tensor scratchpad = torch::empty(
        {static_cast<int64_t>(scratchpad_bytes)},
        torch::TensorOptions()
            .dtype(torch::kInt8)
            .device(gradient.device()));

    for (int64_t batch = 0; batch < gradient.size(0); ++batch)
    {
        LORA_MOE_CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(
            scratchpad.data_ptr(),
            scratchpad_bytes,
            gradient.data_ptr<c10::Half>() +
                batch * gradient.size(1),
            output.data_ptr<c10::Half>() +
                batch * output.size(1),
            bins.numel(),
            offsets.data_ptr<int32_t>(),
            offsets.data_ptr<int32_t>() + 1,
            stream));
    }
}

}  // namespace detail

inline void replicate_forward(
    torch::Tensor input,
    torch::Tensor bins,
    torch::Tensor output)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be two-dimensional");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat16 ||
            input.scalar_type() == torch::kInt16 ||
            input.scalar_type() == torch::kInt32,
        "input dtype must be float16, int16, or int32");
    TORCH_CHECK(
        bins.is_cuda() && bins.dim() == 1 &&
            bins.scalar_type() == torch::kInt32,
        "bins must be a one-dimensional CUDA int32 tensor");
    TORCH_CHECK(bins.is_contiguous(), "bins must be contiguous");
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(output.dim() == 2, "output must be two-dimensional");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(
        output.scalar_type() == input.scalar_type(),
        "input and output dtypes must match");
    TORCH_CHECK(
        input.size(0) == output.size(0),
        "input and output batch sizes must match");
    TORCH_CHECK(
        input.size(1) == bins.numel(),
        "input must contain one value per bin");

    if (output.numel() == 0)
    {
        return;
    }
    TORCH_CHECK(bins.numel() > 0, "bins must not be empty");

    switch (input.scalar_type())
    {
        case torch::kFloat16:
            detail::launch_replicate_forward<c10::Half>(
                input,
                bins,
                output);
            break;
        case torch::kInt16:
            detail::launch_replicate_forward<int16_t>(
                input,
                bins,
                output);
            break;
        case torch::kInt32:
            detail::launch_replicate_forward<int32_t>(
                input,
                bins,
                output);
            break;
        default:
            TORCH_CHECK(false, "unsupported replicate dtype");
    }
}

inline void replicate_backward(
    torch::Tensor gradient,
    torch::Tensor bins,
    torch::Tensor output)
{
    TORCH_CHECK(
        gradient.is_cuda() && gradient.dim() == 2 &&
            gradient.scalar_type() == torch::kFloat16,
        "gradient must be a two-dimensional CUDA float16 tensor");
    TORCH_CHECK(
        gradient.is_contiguous(),
        "gradient must be contiguous");
    TORCH_CHECK(
        bins.is_cuda() && bins.dim() == 1 &&
            bins.scalar_type() == torch::kInt32,
        "bins must be a one-dimensional CUDA int32 tensor");
    TORCH_CHECK(bins.is_contiguous(), "bins must be contiguous");
    TORCH_CHECK(
        output.is_cuda() && output.dim() == 2 &&
            output.scalar_type() == torch::kFloat16,
        "output must be a two-dimensional CUDA float16 tensor");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(
        gradient.size(0) == output.size(0),
        "gradient and output batch sizes must match");
    TORCH_CHECK(
        output.size(1) == bins.numel(),
        "output must contain one value per bin");

    if (output.numel() == 0)
    {
        return;
    }
    detail::segmented_sum(gradient, bins, output);
}

}  // namespace lora_moe
