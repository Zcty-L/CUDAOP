#include "op/softmax/softmax.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <cuda_runtime.h>
#include <math_constants.h>

namespace
{

constexpr int WARP_SIZE = cudaop::kSoftmaxWarpSize;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int ROWS_PER_WARP = 2;
constexpr int BLOCK_REDUCE_THREADS = cudaop::kSoftmaxBlockThreads;
constexpr int BLOCK_REDUCE_WARPS = BLOCK_REDUCE_THREADS / WARP_SIZE;
constexpr int BLOCK_MAX_COLS = cudaop::kSoftmaxBlockMaxColumns;
constexpr int ONLINE_VECTOR_SIZE = cudaop::kSoftmaxOnlineVectorSize;
constexpr int MAX_INT8_COLS = cudaop::kSoftmaxInt8MaxColumns;
constexpr int MAX_GRID_SIZE = 65535;

constexpr int32_t INT8_MIN_VALUE = -128;
constexpr int32_t INT8_MAX_VALUE = 127;
constexpr float INT8_OUTPUT_SCALE = cudaop::kSoftmaxInt8OutputScale;
constexpr int32_t INT8_OUTPUT_ZERO_POINT =
    cudaop::kSoftmaxInt8OutputZeroPoint;

__device__ __forceinline__ void update_online_softmax(
    float value,
    float &running_max,
    float &running_sum)
{
    if (value > running_max)
    {
        running_sum = running_sum * expf(running_max - value) + 1.0f;
        running_max = value;
    }
    else
    {
        running_sum += expf(value - running_max);
    }
}

template <int COLS_PER_THREAD>
__global__ void softmax_warp_kernel(
    const float *__restrict__ source,
    float *__restrict__ destination,
    int64_t rows,
    int64_t cols)
{
    const int lane_id = threadIdx.x;
    const int64_t global_warp_id =
        static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    const int64_t warp_stride =
        static_cast<int64_t>(gridDim.x) * blockDim.y;

    for (int64_t first_row = global_warp_id * ROWS_PER_WARP;
         first_row < rows;
         first_row += warp_stride * ROWS_PER_WARP)
    {
#pragma unroll
        for (int row_index = 0; row_index < ROWS_PER_WARP; row_index++)
        {
            const int64_t row = first_row + row_index;
            if (row >= rows)
            {
                continue;
            }

            float values[COLS_PER_THREAD];
            float thread_max = -CUDART_INF_F;

#pragma unroll
            for (int index = 0; index < COLS_PER_THREAD; index++)
            {
                const int col = lane_id + index * WARP_SIZE;
                const float value =
                    col < cols ? source[row * cols + col] : -CUDART_INF_F;
                values[index] = value;
                thread_max = fmaxf(thread_max, value);
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2)
            {
                thread_max = fmaxf(
                    thread_max,
                    __shfl_xor_sync(0xffffffffU, thread_max, mask));
            }

            float thread_sum = 0.0f;
#pragma unroll
            for (int index = 0; index < COLS_PER_THREAD; index++)
            {
                const int col = lane_id + index * WARP_SIZE;
                const float value =
                    col < cols ? expf(values[index] - thread_max) : 0.0f;
                values[index] = value;
                thread_sum += value;
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2)
            {
                thread_sum +=
                    __shfl_xor_sync(0xffffffffU, thread_sum, mask);
            }

            const float reciprocal_sum = 1.0f / thread_sum;
#pragma unroll
            for (int index = 0; index < COLS_PER_THREAD; index++)
            {
                const int col = lane_id + index * WARP_SIZE;
                if (col < cols)
                {
                    destination[row * cols + col] =
                        values[index] * reciprocal_sum;
                }
            }
        }
    }
}

// 每个 block 处理一行，先归约最大值，再计算指数并归约指数和。
template <int COLS_PER_THREAD>
__global__ void softmax_block_kernel(
    const float *__restrict__ source,
    float *__restrict__ destination,
    int64_t cols)
{
    static_assert(COLS_PER_THREAD <= cudaop::kSoftmaxMaxColumnsPerThread);

    __shared__ float warp_results[BLOCK_REDUCE_WARPS];

    const int64_t row = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int lane_id = thread_id % WARP_SIZE;
    const int warp_id = thread_id / WARP_SIZE;
    const int64_t row_offset = row * cols;

    float values[COLS_PER_THREAD];
    float thread_max = -CUDART_INF_F;

#pragma unroll
    for (int index = 0; index < COLS_PER_THREAD; index++)
    {
        const int64_t col = thread_id + index * BLOCK_REDUCE_THREADS;
        const float value = col < cols ? source[row_offset + col] : -CUDART_INF_F;
        values[index] = value;
        thread_max = fmaxf(thread_max, value);
    }

#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2)
    {
        thread_max = fmaxf(
            thread_max,
            __shfl_down_sync(0xffffffffU, thread_max, mask));
    }

    if (lane_id == 0)
    {
        warp_results[warp_id] = thread_max;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        float block_max = lane_id < BLOCK_REDUCE_WARPS ? warp_results[lane_id] : -CUDART_INF_F;
#pragma unroll
        for (int mask = BLOCK_REDUCE_WARPS / 2; mask > 0; mask /= 2)
        {
            block_max = fmaxf(
                block_max,
                __shfl_down_sync(0xffffffffU, block_max, mask));
        }
        if (lane_id == 0)
        {
            warp_results[0] = block_max;
        }
    }
    __syncthreads();

    const float block_max = warp_results[0];
    float thread_sum = 0.0f;
#pragma unroll
    for (int index = 0; index < COLS_PER_THREAD; index++)
    {
        const int64_t col = thread_id + index * BLOCK_REDUCE_THREADS;
        const float value = col < cols ? expf(values[index] - block_max) : 0.0f;
        values[index] = value;
        thread_sum += value;
    }

#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2)
    {
        thread_sum += __shfl_down_sync(0xffffffffU, thread_sum, mask);
    }

    if (lane_id == 0)
    {
        warp_results[warp_id] = thread_sum;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        float block_sum = lane_id < BLOCK_REDUCE_WARPS ? warp_results[lane_id] : 0.0f;
#pragma unroll
        for (int mask = BLOCK_REDUCE_WARPS / 2; mask > 0; mask /= 2)
        {
            block_sum += __shfl_down_sync(0xffffffffU, block_sum, mask);
        }
        if (lane_id == 0)
        {
            warp_results[0] = block_sum;
        }
    }
    __syncthreads();

    const float reciprocal_sum = 1.0f / warp_results[0];
#pragma unroll
    for (int index = 0; index < COLS_PER_THREAD; index++)
    {
        const int64_t col = thread_id + index * BLOCK_REDUCE_THREADS;
        if (col < cols)
        {
            destination[row_offset + col] = values[index] * reciprocal_sum;
        }
    }
}

// 每个 block 处理一行。第一遍流式计算线程局部 online max/sum，
// 循环结束后执行 block reduce；第二遍重新加载输入并写出归一化结果。
__global__ void softmax_online_kernel(
    const float *__restrict__ source,
    float *__restrict__ destination,
    int64_t cols)
{
    static_assert(ONLINE_VECTOR_SIZE == 4);
    static_assert(sizeof(float4) == ONLINE_VECTOR_SIZE * sizeof(float));

    __shared__ float warp_results[BLOCK_REDUCE_WARPS];

    const int64_t row = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int lane_id = thread_id % WARP_SIZE;
    const int warp_id = thread_id / WARP_SIZE;
    const int64_t row_offset = row * cols;
    const float *row_source = source + row_offset;
    float *row_destination = destination + row_offset;

    const uintptr_t source_address = reinterpret_cast<uintptr_t>(row_source);
    int scalar_prefix = static_cast<int>(((sizeof(float4) - (source_address & (sizeof(float4) - 1))) & (sizeof(float4) - 1)) / sizeof(float));
    if (scalar_prefix > cols)
    {
        scalar_prefix = static_cast<int>(cols);
    }

    const int64_t vector_count = (cols - scalar_prefix) / ONLINE_VECTOR_SIZE;
    const int64_t tail_offset = scalar_prefix + vector_count * ONLINE_VECTOR_SIZE;
    const int scalar_tail = static_cast<int>(cols - tail_offset);
    const float4 *vector_source = reinterpret_cast<const float4 *>(row_source + scalar_prefix);

    float thread_max = -CUDART_INF_F;
    float thread_sum = 0.0f;

    if (thread_id < scalar_prefix)
    {
        update_online_softmax(row_source[thread_id], thread_max, thread_sum);
    }

    for (int64_t vector_index = thread_id; vector_index < vector_count; vector_index += BLOCK_REDUCE_THREADS)
    {
        const float4 values = vector_source[vector_index];
        update_online_softmax(values.x, thread_max, thread_sum);
        update_online_softmax(values.y, thread_max, thread_sum);
        update_online_softmax(values.z, thread_max, thread_sum);
        update_online_softmax(values.w, thread_max, thread_sum);
    }

    if (thread_id < scalar_tail)
    {
        update_online_softmax(row_source[tail_offset + thread_id], thread_max, thread_sum);
    }

    const float thread_local_max = thread_max;
    float warp_max = thread_max;
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2)
    {
        warp_max = fmaxf(
            warp_max,
            __shfl_down_sync(0xffffffffU, warp_max, mask));
    }

    if (lane_id == 0)
    {
        warp_results[warp_id] = warp_max;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        float block_max =
            lane_id < BLOCK_REDUCE_WARPS ?
            warp_results[lane_id] :
            -CUDART_INF_F;
#pragma unroll
        for (int mask = BLOCK_REDUCE_WARPS / 2; mask > 0; mask /= 2)
        {
            block_max = fmaxf(
                block_max,
                __shfl_down_sync(0xffffffffU, block_max, mask));
        }
        if (lane_id == 0)
        {
            warp_results[0] = block_max;
        }
    }
    __syncthreads();

    const float block_max = warp_results[0];
    thread_sum *= expf(thread_local_max - block_max);

#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2)
    {
        thread_sum += __shfl_down_sync(0xffffffffU, thread_sum, mask);
    }

    if (lane_id == 0)
    {
        warp_results[warp_id] = thread_sum;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        float block_sum =
            lane_id < BLOCK_REDUCE_WARPS ? warp_results[lane_id] : 0.0f;
#pragma unroll
        for (int mask = BLOCK_REDUCE_WARPS / 2; mask > 0; mask /= 2)
        {
            block_sum += __shfl_down_sync(0xffffffffU, block_sum, mask);
        }
        if (lane_id == 0)
        {
            warp_results[0] = block_sum;
        }
    }
    __syncthreads();

    const float reciprocal_sum = 1.0f / warp_results[0];
    const uintptr_t destination_vector_address =
        reinterpret_cast<uintptr_t>(row_destination + scalar_prefix);
    const bool vector_store_aligned =
        (destination_vector_address & (sizeof(float4) - 1)) == 0;

    if (vector_store_aligned)
    {
        if (thread_id < scalar_prefix)
        {
            const float value = row_source[thread_id];
            row_destination[thread_id] =
                expf(value - block_max) * reciprocal_sum;
        }

        float4 *vector_destination = reinterpret_cast<float4 *>(
            row_destination + scalar_prefix);
        for (int64_t vector_index = thread_id;
             vector_index < vector_count;
             vector_index += BLOCK_REDUCE_THREADS)
        {
            const float4 values = vector_source[vector_index];
            float4 probabilities;
            probabilities.x = expf(values.x - block_max) * reciprocal_sum;
            probabilities.y = expf(values.y - block_max) * reciprocal_sum;
            probabilities.z = expf(values.z - block_max) * reciprocal_sum;
            probabilities.w = expf(values.w - block_max) * reciprocal_sum;
            vector_destination[vector_index] = probabilities;
        }

        if (thread_id < scalar_tail)
        {
            const int64_t col = tail_offset + thread_id;
            row_destination[col] =
                expf(row_source[col] - block_max) * reciprocal_sum;
        }
    }
    else
    {
        for (int64_t col = thread_id;
             col < cols;
             col += BLOCK_REDUCE_THREADS)
        {
            row_destination[col] =
                expf(row_source[col] - block_max) * reciprocal_sum;
        }
    }
}

template <int COLS_PER_THREAD, typename OutputType>
__global__ void softmax_int8_warp_kernel(
    const int8_t *__restrict__ source,
    OutputType *__restrict__ destination,
    int64_t rows,
    int64_t cols,
    float input_scale)
{
    const int lane_id = threadIdx.x;
    const int64_t global_warp_id =
        static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    const int64_t warp_stride =
        static_cast<int64_t>(gridDim.x) * blockDim.y;

    for (int64_t first_row = global_warp_id * ROWS_PER_WARP;
         first_row < rows;
         first_row += warp_stride * ROWS_PER_WARP)
    {
#pragma unroll
        for (int row_index = 0; row_index < ROWS_PER_WARP; row_index++)
        {
            const int64_t row = first_row + row_index;
            if (row >= rows)
            {
                continue;
            }

            int32_t quantized_values[COLS_PER_THREAD];
            int32_t thread_max = INT8_MIN_VALUE;

#pragma unroll
            for (int index = 0; index < COLS_PER_THREAD; index++)
            {
                const int col = lane_id + index * WARP_SIZE;
                const int32_t value =
                    col < cols ?
                    static_cast<int32_t>(source[row * cols + col]) :
                    INT8_MIN_VALUE;
                quantized_values[index] = value;
                thread_max = max(thread_max, value);
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2)
            {
                thread_max = max(
                    thread_max,
                    __shfl_xor_sync(0xffffffffU, thread_max, mask));
            }

            float exponentials[COLS_PER_THREAD];
            float thread_sum = 0.0f;
#pragma unroll
            for (int index = 0; index < COLS_PER_THREAD; index++)
            {
                const int col = lane_id + index * WARP_SIZE;
                const float value =
                    col < cols ?
                    expf(
                        static_cast<float>(
                            quantized_values[index] - thread_max) *
                        input_scale) :
                    0.0f;
                exponentials[index] = value;
                thread_sum += value;
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2)
            {
                thread_sum +=
                    __shfl_xor_sync(0xffffffffU, thread_sum, mask);
            }

            const float reciprocal_sum = 1.0f / thread_sum;
#pragma unroll
            for (int index = 0; index < COLS_PER_THREAD; index++)
            {
                const int col = lane_id + index * WARP_SIZE;
                if (col >= cols)
                {
                    continue;
                }

                const float probability =
                    exponentials[index] * reciprocal_sum;
                if constexpr (std::is_same_v<OutputType, float>)
                {
                    destination[row * cols + col] = probability;
                }
                else
                {
                    int32_t quantized = __float2int_rn(
                        probability / INT8_OUTPUT_SCALE);
                    quantized += INT8_OUTPUT_ZERO_POINT;
                    quantized = max(
                        INT8_MIN_VALUE,
                        min(
                            INT8_MAX_VALUE,
                            quantized));
                    destination[row * cols + col] =
                        static_cast<int8_t>(quantized);
                }
            }
        }
    }
}

template <int COLS_PER_THREAD>
cudaError_t launch_softmax_kernel(
    const float *source,
    float *destination,
    int64_t rows,
    int64_t cols,
    cudaStream_t stream)
{
    constexpr int ROWS_PER_BLOCK = WARPS_PER_BLOCK * ROWS_PER_WARP;
    const int64_t required_blocks = (rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    const int grid_size = static_cast<int>(std::min<int64_t>(required_blocks, MAX_GRID_SIZE));

    const dim3 block(WARP_SIZE, WARPS_PER_BLOCK);
    softmax_warp_kernel<COLS_PER_THREAD>
        <<<grid_size, block, 0, stream>>>(
            source,
            destination,
            rows,
            cols);
    return cudaGetLastError();
}

template <int COLS_PER_THREAD>
cudaError_t launch_softmax_block_kernel(
    const float *source,
    float *destination,
    int64_t rows,
    int64_t cols,
    cudaStream_t stream)
{
    softmax_block_kernel<COLS_PER_THREAD>
        <<<static_cast<unsigned int>(rows),
           BLOCK_REDUCE_THREADS,
           0,
           stream>>>(source, destination, cols);
    return cudaGetLastError();
}

cudaError_t launch_softmax_online_kernel(
    const float *source,
    float *destination,
    int64_t rows,
    int64_t cols,
    cudaStream_t stream)
{
    softmax_online_kernel
        <<<static_cast<unsigned int>(rows),
           BLOCK_REDUCE_THREADS,
           0,
           stream>>>(source, destination, cols);
    return cudaGetLastError();
}

cudaError_t launch_softmax_impl(
    const float *source,
    float *destination,
    int64_t rows,
    int64_t cols,
    cudaStream_t stream = nullptr)
{
    if (source == nullptr || destination == nullptr ||
        rows <= 0 || cols <= 0)
    {
        return cudaErrorInvalidValue;
    }

    if (cols <= 32)
    {
        return launch_softmax_kernel<1>(
            source,
            destination,
            rows,
            cols,
            stream);
    }
    if (cols <= 64)
    {
        return launch_softmax_kernel<2>(
            source,
            destination,
            rows,
            cols,
            stream);
    }
    if (cols <= 128)
    {
        return launch_softmax_kernel<4>(
            source,
            destination,
            rows,
            cols,
            stream);
    }
    if (cols <= 256)
    {
        return launch_softmax_kernel<8>(
            source,
            destination,
            rows,
            cols,
            stream);
    }
    if (cols <= 512)
    {
        return launch_softmax_kernel<16>(
            source,
            destination,
            rows,
            cols,
            stream);
    }
    if (cols <= 1024)
    {
        return launch_softmax_kernel<32>(
            source,
            destination,
            rows,
            cols,
            stream);
    }

    if (rows > std::numeric_limits<int>::max())
    {
        return cudaErrorInvalidValue;
    }
    if (cols <= 2048)
    {
        return launch_softmax_block_kernel<8>(
            source,
            destination,
            rows,
            cols,
            stream);
    }
    if (cols <= 4096)
    {
        return launch_softmax_block_kernel<16>(
            source,
            destination,
            rows,
            cols,
            stream);
    }

    if (cols <= BLOCK_MAX_COLS)
    {
        return launch_softmax_block_kernel<32>(
            source,
            destination,
            rows,
            cols,
            stream);
    }

    return launch_softmax_online_kernel(
        source,
        destination,
        rows,
        cols,
        stream);
}

template <int COLS_PER_THREAD, typename OutputType>
cudaError_t launch_softmax_int8_kernel(
    const int8_t *source,
    OutputType *destination,
    int64_t rows,
    int64_t cols,
    float input_scale,
    cudaStream_t stream)
{
    constexpr int ROWS_PER_BLOCK = WARPS_PER_BLOCK * ROWS_PER_WARP;
    const int64_t required_blocks = (rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    const int grid_size = static_cast<int>(std::min<int64_t>(required_blocks, MAX_GRID_SIZE));

    const dim3 block(WARP_SIZE, WARPS_PER_BLOCK);
    softmax_int8_warp_kernel<COLS_PER_THREAD, OutputType>
        <<<grid_size, block, 0, stream>>>(
            source,
            destination,
            rows,
            cols,
            input_scale);
    return cudaGetLastError();
}

template <typename OutputType>
cudaError_t launch_softmax_int8_impl(
    const int8_t *source,
    OutputType *destination,
    int64_t rows,
    int64_t cols,
    float input_scale,
    int32_t input_zero_point,
    cudaStream_t stream)
{
    if (source == nullptr || destination == nullptr ||
        rows <= 0 || cols <= 0 || cols > MAX_INT8_COLS ||
        !std::isfinite(input_scale) || input_scale <= 0.0f ||
        input_zero_point < std::numeric_limits<int8_t>::min() ||
        input_zero_point > std::numeric_limits<int8_t>::max())
    {
        return cudaErrorInvalidValue;
    }

    // 对同一张量使用统一 zero point 时，减去行最大值后 zero point 抵消。
    if (cols <= 32)
    {
        return launch_softmax_int8_kernel<1>(
            source,
            destination,
            rows,
            cols,
            input_scale,
            stream);
    }
    if (cols <= 64)
    {
        return launch_softmax_int8_kernel<2>(
            source,
            destination,
            rows,
            cols,
            input_scale,
            stream);
    }
    if (cols <= 128)
    {
        return launch_softmax_int8_kernel<4>(
            source,
            destination,
            rows,
            cols,
            input_scale,
            stream);
    }
    if (cols <= 256)
    {
        return launch_softmax_int8_kernel<8>(
            source,
            destination,
            rows,
            cols,
            input_scale,
            stream);
    }
    if (cols <= 512)
    {
        return launch_softmax_int8_kernel<16>(
            source,
            destination,
            rows,
            cols,
            input_scale,
            stream);
    }

    return launch_softmax_int8_kernel<32>(
        source,
        destination,
        rows,
        cols,
        input_scale,
        stream);
}

} // namespace

namespace cudaop
{

cudaError_t launch_softmax(
    const float *source,
    float *destination,
    int64_t rows,
    int64_t cols,
    cudaStream_t stream)
{
    return launch_softmax_impl(source, destination, rows, cols, stream);
}

cudaError_t launch_softmax_int8_to_float(
    const int8_t *source,
    float *destination,
    int64_t rows,
    int64_t cols,
    float input_scale,
    int32_t input_zero_point,
    cudaStream_t stream)
{
    return launch_softmax_int8_impl(
        source,
        destination,
        rows,
        cols,
        input_scale,
        input_zero_point,
        stream);
}

cudaError_t launch_softmax_int8_to_int8(
    const int8_t *source,
    int8_t *destination,
    int64_t rows,
    int64_t cols,
    float input_scale,
    int32_t input_zero_point,
    cudaStream_t stream)
{
    return launch_softmax_int8_impl(
        source,
        destination,
        rows,
        cols,
        input_scale,
        input_zero_point,
        stream);
}

} // namespace cudaop
