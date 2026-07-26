#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace cudaop
{

// Select the K largest values from each row of a row-major float32 matrix.
// Results are sorted by value descending, then by index ascending.
void topk_cpu(
    const float* input,
    float* values,
    int32_t* indices,
    int rows,
    int columns,
    int k);

// Device pointers must not overlap. The operation is enqueued on stream.
cudaError_t topk_cuda(
    const float* input,
    float* values,
    int32_t* indices,
    int rows,
    int columns,
    int k,
    cudaStream_t stream = nullptr);

}  // namespace cudaop
