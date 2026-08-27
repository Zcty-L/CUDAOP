#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace cudaop
{

inline constexpr int kSoftmaxWarpSize = 32;
inline constexpr int kSoftmaxBlockThreads = 256;
inline constexpr int kSoftmaxMaxColumnsPerThread = 32;
inline constexpr int kSoftmaxBlockMaxColumns =
    kSoftmaxBlockThreads * kSoftmaxMaxColumnsPerThread;
inline constexpr int kSoftmaxOnlineVectorSize = 4;
inline constexpr int kSoftmaxInt8MaxColumns = 1024;
inline constexpr float kSoftmaxInt8OutputScale = 1.0f / 256.0f;
inline constexpr int32_t kSoftmaxInt8OutputZeroPoint = -128;

cudaError_t launch_softmax(
    const float *source,
    float *destination,
    int64_t rows,
    int64_t cols,
    cudaStream_t stream = nullptr);

cudaError_t launch_softmax_int8_to_float(
    const int8_t *source,
    float *destination,
    int64_t rows,
    int64_t cols,
    float input_scale,
    int32_t input_zero_point,
    cudaStream_t stream = nullptr);

cudaError_t launch_softmax_int8_to_int8(
    const int8_t *source,
    int8_t *destination,
    int64_t rows,
    int64_t cols,
    float input_scale,
    int32_t input_zero_point,
    cudaStream_t stream = nullptr);

} // namespace cudaop
