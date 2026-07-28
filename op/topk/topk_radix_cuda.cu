#include "op/topk/SortingRadixSelect.cuh"
#include "op/topk/topk.h"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace cudaop
{
namespace
{

// 单 block 路径与 PyTorch single-block Top-K 一样，每行由一个 block
// 处理。固定 256 个线程可以简化 warp 级 prefix scan 和归约。
constexpr int kBlockThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kBlockThreads / kWarpSize;

// 比较两个已经收集的 Top-K 二元组。数值比较使用与 radix-select 相同的
// key，key 相同时让原始列索引较小的元素排在前面，使排序结果确定。
__device__ __forceinline__ bool radix_pair_is_better(
    float left_value,
    int left_index,
    float right_value,
    int right_index,
    bool largest)
{
    if (left_index < 0)
    {
        return false;
    }
    if (right_index < 0)
    {
        return true;
    }

    const uint32_t left_radix =
        topk_radix::FloatRadixConfig::convert(left_value);
    const uint32_t right_radix =
        topk_radix::FloatRadixConfig::convert(right_value);
    if (left_radix != right_radix)
    {
        return largest
            ? left_radix > right_radix
            : left_radix < right_radix;
    }
    return left_index < right_index;
}

// 对 predicate 执行 block 级 exclusive binary prefix scan。
//
// 返回值是当前线程之前 predicate=true 的线程数量，total_count 返回整个
// block 的命中数量。每个 warp 先通过 ballot/popcount 得到局部前缀，
// 再由线程 0 对最多 8 个 warp 的计数做串行前缀和。
__device__ __forceinline__ int exclusive_binary_prefix_scan(
    bool predicate,
    int* warp_counts,
    int* warp_offsets,
    int* shared_total,
    int& total_count)
{
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    const unsigned int active_mask = __activemask();
    const unsigned int votes =
        __ballot_sync(active_mask, predicate);
    const unsigned int lower_lane_mask =
        lane == 0
        ? 0U
        : (1U << lane) - 1U;
    const int warp_prefix =
        __popc(votes & lower_lane_mask);

    if (lane == 0)
    {
        warp_counts[warp] = __popc(votes);
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        int running_total = 0;
        for (int warp_index = 0;
             warp_index < kWarpsPerBlock;
             ++warp_index)
        {
            warp_offsets[warp_index] = running_total;
            running_total += warp_counts[warp_index];
        }
        *shared_total = running_total;
    }
    __syncthreads();

    total_count = *shared_total;
    return warp_offsets[warp] + warp_prefix;
}

__device__ __forceinline__ void reduce_warp_best(
    float& value,
    int& index,
    int& position,
    bool largest)
{
    constexpr unsigned int kFullWarpMask = 0xffffffffU;

    for (int offset = kWarpSize / 2;
         offset > 0;
         offset /= 2)
    {
        const float other_value =
            __shfl_down_sync(kFullWarpMask, value, offset);
        const int other_index =
            __shfl_down_sync(kFullWarpMask, index, offset);
        const int other_position =
            __shfl_down_sync(kFullWarpMask, position, offset);
        if (radix_pair_is_better(
                other_value,
                other_index,
                value,
                index,
                largest))
        {
            value = other_value;
            index = other_index;
            position = other_position;
        }
    }
}

// 两级归约得到当前未排序区间中的最优二元组及其输出位置。
__device__ __forceinline__ void reduce_block_best(
    float& value,
    int& index,
    int& position,
    bool largest,
    float* warp_values,
    int* warp_indices,
    int* warp_positions)
{
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;

    reduce_warp_best(value, index, position, largest);
    if (lane == 0)
    {
        warp_values[warp] = value;
        warp_indices[warp] = index;
        warp_positions[warp] = position;
    }
    __syncthreads();

    if (warp == 0)
    {
        if (lane < kWarpsPerBlock)
        {
            value = warp_values[lane];
            index = warp_indices[lane];
            position = warp_positions[lane];
        }
        else
        {
            value = 0.0F;
            index = -1;
            position = -1;
        }
        reduce_warp_best(value, index, position, largest);
        if (lane == 0)
        {
            warp_values[0] = value;
            warp_indices[0] = index;
            warp_positions[0] = position;
        }
    }
    __syncthreads();

    value = warp_values[0];
    index = warp_indices[0];
    position = warp_positions[0];
}

// 对已经收集到输出缓冲区的 K 个二元组执行原地选择排序。
//
// PyTorch 在 sorted=true 时会调用独立的排序实现。本项目为了保持接口
// 独立且不申请 workspace，使用 block 归约逐轮选择最优元素。该路径适合
// 小 K，复杂度约为 O(K^2 / blockDim.x + K log blockDim.x)。
__device__ void sort_selected_pairs(
    float* values,
    int32_t* indices,
    int k,
    bool largest,
    float* warp_values,
    int* warp_indices,
    int* warp_positions)
{
    for (int rank = 0; rank < k; ++rank)
    {
        float local_value = 0.0F;
        int local_index = -1;
        int local_position = -1;

        for (int position = rank + threadIdx.x;
             position < k;
             position += blockDim.x)
        {
            const float candidate_value = values[position];
            const int candidate_index = indices[position];
            if (radix_pair_is_better(
                    candidate_value,
                    candidate_index,
                    local_value,
                    local_index,
                    largest))
            {
                local_value = candidate_value;
                local_index = candidate_index;
                local_position = position;
            }
        }

        reduce_block_best(
            local_value,
            local_index,
            local_position,
            largest,
            warp_values,
            warp_indices,
            warp_positions);

        if (threadIdx.x == 0 &&
            local_position != rank)
        {
            const float rank_value = values[rank];
            const int32_t rank_index = indices[rank];
            values[rank] = local_value;
            indices[rank] = local_index;
            values[local_position] = rank_value;
            indices[local_position] = rank_index;
        }
        __syncthreads();
    }
}

// 完整的单 block radix Top-K：
// 1. radix-select 找到第 K 个阈值；
// 2. prefix scan 收集严格优于阈值的元素；
// 3. 从等于阈值的元素中按输入顺序补足到 K 个；
// 4. sorted=true 时排序收集到的 K 个值和索引。
__global__ void topk_radix_float_kernel(
    const float* __restrict__ input,
    float* __restrict__ values,
    int32_t* __restrict__ indices,
    int columns,
    int k,
    bool largest,
    bool sorted)
{
    __shared__ int radix_storage[topk_radix::kRadixSize];
    __shared__ int scan_warp_counts[kWarpsPerBlock];
    __shared__ int scan_warp_offsets[kWarpsPerBlock];
    __shared__ int scan_total;
    __shared__ float sort_warp_values[kWarpsPerBlock];
    __shared__ int sort_warp_indices[kWarpsPerBlock];
    __shared__ int sort_warp_positions[kWarpsPerBlock];

    const int row = blockIdx.x;
    const float* row_input = input + static_cast<size_t>(row) * columns;
    float* row_values = values + static_cast<size_t>(row) * k;
    int32_t* row_indices = indices + static_cast<size_t>(row) * k;

    float threshold = 0.0F;
    topk_radix::radix_select(
        row_input,
        k,
        largest,
        columns,
        1,
        radix_storage,
        &threshold);
    const uint32_t threshold_radix = topk_radix::FloatRadixConfig::convert(threshold);

    // 先收集所有严格优于阈值的元素。它们的总数必然小于 K。
    int write_index_start = 0;
    for (int base = 0; base < columns; base += kBlockThreads)
    {
        const int column = base + threadIdx.x;
        const bool in_range = column < columns;
        float value = 0.0F;
        uint32_t radix = 0;
        if (in_range)
        {
            value = row_input[column];
            radix = topk_radix::FloatRadixConfig::convert(value);
        }

        const bool has_topk = in_range && (largest ? radix > threshold_radix: radix < threshold_radix);
        int batch_count = 0;
        const int prefix = exclusive_binary_prefix_scan(
            has_topk,
            scan_warp_counts,
            scan_warp_offsets,
            &scan_total,
            batch_count);

        if (has_topk)
        {
            const int output_index = write_index_start + prefix;
            if (output_index < k)
            {
                row_values[output_index] = value;
                row_indices[output_index] = column;
            }
        }
        __syncthreads();
        write_index_start += batch_count;
    }

    // 阈值可能重复。只收集足够数量的等值元素，使输出数量严格等于 K。
    int topk_remaining = k - write_index_start;
    for (int base = 0; base < columns && topk_remaining > 0; base += kBlockThreads)
    {
        const int column = base + threadIdx.x;
        const bool in_range = column < columns;
        float value = 0.0F;
        uint32_t radix = 0;
        if (in_range)
        {
            value = row_input[column];
            radix = topk_radix::FloatRadixConfig::convert(value);
        }

        const bool equals_threshold = in_range && radix == threshold_radix;
        int batch_count = 0;
        const int prefix = exclusive_binary_prefix_scan(
            equals_threshold,
            scan_warp_counts,
            scan_warp_offsets,
            &scan_total,
            batch_count);

        if (equals_threshold && prefix < topk_remaining)
        {
            const int output_index = write_index_start + prefix;
            row_values[output_index] = value;
            row_indices[output_index] = column;
        }
        __syncthreads();

        const int accepted = batch_count < topk_remaining ? batch_count : topk_remaining;
        write_index_start += accepted;
        topk_remaining -= accepted;
    }
    __syncthreads();

    if (sorted)
    {
        sort_selected_pairs(
            row_values,
            row_indices,
            k,
            largest,
            sort_warp_values,
            sort_warp_indices,
            sort_warp_positions);
    }
}

}  // namespace

cudaError_t topk_radix_cuda(
    const float* input,
    float* values,
    int32_t* indices,
    int rows,
    int columns,
    int k,
    bool largest,
    bool sorted,
    cudaStream_t stream)
{
    if (input == nullptr ||
        values == nullptr ||
        indices == nullptr)
    {
        return cudaErrorInvalidValue;
    }
    if (rows <= 0 ||
        columns <= 0 ||
        k <= 0 ||
        k > columns)
    {
        return cudaErrorInvalidValue;
    }

    topk_radix_float_kernel<<<
        rows,
        kBlockThreads,
        0,
        stream>>>(
        input,
        values,
        indices,
        columns,
        k,
        largest,
        sorted);
    return cudaGetLastError();
}

}  // namespace cudaop
