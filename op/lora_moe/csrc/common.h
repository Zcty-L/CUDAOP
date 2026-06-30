#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

#define LORA_MOE_CUDA_CHECK(expression)                                  \
    do                                                                   \
    {                                                                    \
        const cudaError_t status = (expression);                         \
        TORCH_CHECK(                                                     \
            status == cudaSuccess,                                       \
            cudaGetErrorString(status));                                 \
    } while (false)
