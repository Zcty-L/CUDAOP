#include "op/topk/topk.h"

#include <cmath>
#include <cstdint>

#include <math_constants.h>

namespace cudaop
{
namespace
{

// 每个 CUDA block 负责输入矩阵的一行。
// 256 个线程组成 8 个 warp，用于两级最大值归约。
constexpr int kBlockThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kBlockThreads / kWarpSize;

// 判断 left 是否应在 Top-K 顺序中排在 right 前面。
// index == -1 表示当前线程没有有效候选元素。
// 有效元素按 NaN 优先、数值降序、列索引升序排列。
__device__ __forceinline__ bool pair_is_better(
    float left_value,
    int left_index,
    float right_value,
    int right_index)
{
    // 无效的 left 候选永远不能胜出。
    if (left_index < 0)
    {
        return false;
    }

    // left 有效而 right 无效时，直接选择 left。
    if (right_index < 0)
    {
        return true;
    }

    const bool left_nan = isnan(left_value);
    const bool right_nan = isnan(right_value);

    // 只有一侧为 NaN 时，将 NaN 视为更大的元素。
    if (left_nan != right_nan)
    {
        return left_nan;
    }

    // 两侧均为普通数值时，较大的数值排在前面。
    if (!left_nan && left_value != right_value)
    {
        return left_value > right_value;
    }

    // 相同数值或两侧均为 NaN 时，较小的列索引优先。
    return left_index < right_index;
}

// 使用 warp shuffle 对一个 warp 内的候选二元组执行归约。
// 归约结束后，lane 0 持有该 warp 中排序最靠前的候选。
__device__ __forceinline__ void reduce_warp_best(float& value, int& index)
{
    constexpr unsigned int kFullWarpMask = 0xffffffffU;

    // 每轮把距离 offset 的候选向低 lane 合并，最终归约到 lane 0。
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2)
    {
        const float other_value = __shfl_down_sync(kFullWarpMask, value, offset);
        const int other_index = __shfl_down_sync(kFullWarpMask, index, offset);
        if (pair_is_better(
                other_value,
                other_index,
                value,
                index))
        {
            value = other_value;
            index = other_index;
        }
    }
}

__device__ __forceinline__ void reduce_block_best(
    float& value,
    int& index,
    float* warp_values,
    int* warp_indices)
{
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;

    // 第一级：分别归约每个 warp，并将 warp 结果写入共享内存。
    reduce_warp_best(value, index);
    if (lane == 0)
    {
        warp_values[warp] = value;
        warp_indices[warp] = index;
    }
    __syncthreads();

    // 第二级：由第一个 warp 归约所有 warp 的结果。
    if (warp == 0)
    {
        if (lane < kWarpsPerBlock)
        {
            value = warp_values[lane];
            index = warp_indices[lane];
        }
        else
        {
            value = -CUDART_INF_F;
            index = -1;
        }
        reduce_warp_best(value, index);
        if (lane == 0)
        {
            warp_values[0] = value;
            warp_indices[0] = index;
        }
    }
    __syncthreads();

    // 将 block 最优候选广播给所有线程。
    value = warp_values[0];
    index = warp_indices[0];
}

// 对行主序 float32 矩阵逐行计算 Top-K。
//
// 每个线程维护其步进列序列中的当前最优候选。每轮先在 block
// 内选出所有线程候选中的最优元素，再仅推进胜出线程的候选。
// 该过程等价于合并 256 个按 Top-K 规则排列的候选序列。
// 算法不需要额外 workspace，适合 k 较小的批量行 Top-K。
__global__ void topk_float_kernel(
    const float* __restrict__ input,
    float* __restrict__ values,
    int32_t* __restrict__ indices,
    int columns,
    int k)
{
    // 每个 warp 只向共享内存提交一个候选值及其列索引。
    __shared__ float warp_values[kWarpsPerBlock];
    __shared__ int warp_indices[kWarpsPerBlock];

    const int row_index = blockIdx.x;
    const int thread_index = threadIdx.x;
    const float* row = input + static_cast<size_t>(row_index) * columns;

    // 初次扫描：每个线程从自己负责的跨步列中找出首个候选。
    float local_value = -CUDART_INF_F;
    int local_index = -1;
    for (int column = thread_index; column < columns; column += kBlockThreads)
    {
        const float candidate = row[column];
        if (pair_is_better(candidate, column, local_value, local_index))
        {
            local_value = candidate;
            local_index = column;
        }
    }

    // 每轮输出一个元素，因此循环结束后结果已经按比较规则排序。
    for (int rank = 0; rank < k; ++rank)
    {
        float block_value = local_value;
        int block_index = local_index;

        // 从所有线程当前的候选中选出本轮全局最优元素。
        reduce_block_best(
            block_value,
            block_index,
            warp_values,
            warp_indices);

        // 只由线程 0 写出 block 的统一归约结果，避免重复写入。
        if (thread_index == 0)
        {
            const size_t output_offset = static_cast<size_t>(row_index) * k + rank;
            values[output_offset] = block_value;
            indices[output_offset] = block_index;
        }

        // 只有拥有胜出列的线程需要推进自己的候选。
        // 其他线程的当前候选仍未被选中，可以保留到下一轮。
        if (local_index == block_index)
        {
            float next_value = -CUDART_INF_F;
            int next_index = -1;
            for (int column = thread_index;
                 column < columns;
                 column += kBlockThreads)
            {
                const float candidate = row[column];
                const bool follows_current = pair_is_better(
                    local_value,
                    local_index,
                    candidate,
                    column);

                // 只考虑严格排在当前候选之后的元素，从而排除当前元素
                // 以及该线程在之前轮次中已经输出的所有候选。
                if (follows_current &&
                    pair_is_better(
                        candidate,
                        column,
                        next_value,
                        next_index))
                {
                    next_value = candidate;
                    next_index = column;
                }
            }
            local_value = next_value;
            local_index = next_index;
        }

        // 确保胜出线程完成候选更新后，再开始下一轮 block 归约。
        __syncthreads();
    }
}

}  // namespace

// 将 Top-K kernel 异步提交到指定 CUDA stream。
//
// input   : [rows, columns] 设备端输入矩阵。
// values  : [rows, k] 设备端 Top-K 数值。
// indices : [rows, k] 设备端原始列索引。
cudaError_t topk_cuda(
    const float* input,
    float* values,
    int32_t* indices,
    int rows,
    int columns,
    int k,
    cudaStream_t stream)
{
    // 在 kernel 启动前完成基础参数检查。
    if (input == nullptr || values == nullptr || indices == nullptr)
    {
        return cudaErrorInvalidValue;
    }
    if (rows <= 0 || columns <= 0 || k <= 0 || k > columns)
    {
        return cudaErrorInvalidValue;
    }

    // 一个 block 处理一行；调用立即返回，不在此处同步 stream。
    topk_float_kernel<<<rows, kBlockThreads, 0, stream>>>(
        input,
        values,
        indices,
        columns,
        k);

    // 这里只报告 kernel 启动错误。异步执行错误需要调用方同步后检查。
    return cudaGetLastError();
}

}  // namespace cudaop
