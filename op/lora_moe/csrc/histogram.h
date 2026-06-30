#pragma once

#include <cstdint>

#include <c10/cuda/CUDAStream.h>
#include <cub/cub.cuh>
#include <torch/extension.h>

#include "common.h"

namespace lora_moe
{

template <typename T>
torch::Tensor run_histogram(torch::Tensor input, int64_t num_bins)
{
    const auto options = torch::TensorOptions()
                             .dtype(torch::kInt32)
                             .device(input.device());
    torch::Tensor output = torch::empty(
        {input.size(0), num_bins},
        options);
    if (output.numel() == 0)
    {
        return output;
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    size_t scratchpad_bytes = 0;
    LORA_MOE_CUDA_CHECK(cub::DeviceHistogram::HistogramEven(
        nullptr,
        scratchpad_bytes,
        input.data_ptr<T>(),
        output.data_ptr<int32_t>(),
        static_cast<int>(num_bins + 1),
        static_cast<T>(0),
        static_cast<T>(num_bins),
        input.size(1),
        stream));

    torch::Tensor scratchpad = torch::empty(
        {static_cast<int64_t>(scratchpad_bytes)},
        torch::TensorOptions()
            .dtype(torch::kInt8)
            .device(input.device()));
    for (int64_t row = 0; row < input.size(0); ++row)
    {
        LORA_MOE_CUDA_CHECK(cub::DeviceHistogram::HistogramEven(
            scratchpad.data_ptr(),
            scratchpad_bytes,
            input.data_ptr<T>() + input.size(1) * row,
            output.data_ptr<int32_t>() + output.size(1) * row,
            static_cast<int>(num_bins + 1),
            static_cast<T>(0),
            static_cast<T>(num_bins),
            input.size(1),
            stream));
    }
    return output;
}

inline torch::Tensor histogram(torch::Tensor input, int64_t num_bins)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(
        input.dim() == 1 || input.dim() == 2,
        "input must be one- or two-dimensional");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(num_bins >= 0, "num_bins must be non-negative");
    TORCH_CHECK(
        input.scalar_type() == torch::kInt16 ||
            input.scalar_type() == torch::kInt32 ||
            input.scalar_type() == torch::kInt64,
        "input dtype must be int16, int32, or int64");

    const bool unbatched = input.dim() == 1;
    if (unbatched)
    {
        input = input.view({1, input.numel()});
    }

    torch::Tensor output;
    switch (input.scalar_type())
    {
        case torch::kInt16:
            output = run_histogram<int16_t>(input, num_bins);
            break;
        case torch::kInt32:
            output = run_histogram<int32_t>(input, num_bins);
            break;
        case torch::kInt64:
            output = run_histogram<int64_t>(input, num_bins);
            break;
        default:
            TORCH_CHECK(false, "unsupported histogram dtype");
    }
    return unbatched ? output.flatten() : output;
}

}  // namespace lora_moe
