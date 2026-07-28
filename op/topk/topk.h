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

// 基于 PyTorch radix-select 流程的完整 Top-K。
//
// largest=true 选择较大的 K 个元素，否则选择较小的 K 个元素。
// sorted=true 按 radix key 和原始列索引对每行结果排序。
// values 和 indices 的形状均为 [rows, k]，设备指针之间不能重叠。
cudaError_t topk_radix_cuda(
    const float* input,
    float* values,
    int32_t* indices,
    int rows,
    int columns,
    int k,
    bool largest = true,
    bool sorted = true,
    cudaStream_t stream = nullptr);

}  // namespace cudaop
