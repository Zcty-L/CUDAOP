#pragma once

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

// 本文件改编自 PyTorch 2.9.1：
// aten/src/ATen/native/cuda/SortingRadixSelect.cuh
// 完整来源链接及算法关系见 docs/topk.md。
//
// 原实现依赖 ATen/c10 内部工具。这里仅保留 CUDAOP Top-K 所需的 float32
// radix-select 核心逻辑，并用 CUDA 原生 intrinsic 替换相关依赖。PyTorch
// 的许可证声明保存在同目录的 PYTORCH_LICENSE 中。

namespace cudaop
{
namespace topk_radix
{

// 每轮检查 2 个 bit，因此共有 2^2 = 4 个桶。
// float32 一共需要检查 32 / 2 = 16 轮。
constexpr int kRadixBits = 2;
constexpr int kRadixSize = 1 << kRadixBits;
constexpr uint32_t kRadixMask = kRadixSize - 1;

// 将 float 的 IEEE 754 bit pattern 映射为可按无符号整数比较的 key。
//
// 普通 float 的原始 bit 不能直接按 uint32_t 排序：
// - 正数的 bit 随数值递增；
// - 负数的符号位为 1，并且绝对值越大，bit 反而越大。
//
// 映射规则：
// - 负数按位取反，使负数区间恢复数值升序；
// - 非负数翻转符号位，使它们排在负数之后；
// - 所有 NaN 映射到 UINT32_MAX，使 largest=true 时 NaN 优先。
//
// 映射后满足：若普通数值 a < b，则 convert(a) < convert(b)。
struct FloatRadixConfig
{
    using RadixType = uint32_t;

    __device__ __forceinline__ static RadixType convert(float value)
    {
        const RadixType bits =
            static_cast<RadixType>(__float_as_int(value));
        const RadixType mask =
            (bits & 0x80000000U) != 0U
            ? 0xffffffffU
            : 0x80000000U;

        return isnan(value) ? 0xffffffffU : bits ^ mask;
    }

    // convert 的逆变换。radix-select 最终只得到完整 radix key 时，
    // 使用该函数恢复第 K 个元素的 float 值。
    __device__ __forceinline__ static float deconvert(RadixType radix)
    {
        const RadixType mask =
            (radix & 0x80000000U) != 0U
            ? 0x80000000U
            : 0xffffffffU;
        return __int_as_float(static_cast<int>(radix ^ mask));
    }
};

// 修改 value 中从 bit_position 开始的 kRadixBits 个 bit。
// 每轮确定一个 radix digit 后，desired 和 desired_mask 都通过它推进。
__device__ __forceinline__ uint32_t set_radix_digit(
    uint32_t value,
    uint32_t digit,
    int bit_position)
{
    const uint32_t shifted_mask = kRadixMask << bit_position;
    return (value & ~shifted_mask) |
           ((digit & kRadixMask) << bit_position);
}

// 统计当前 2-bit digit 的四个桶各有多少元素。
//
// 只有满足以下前缀条件的元素才参与本轮统计：
//
//     (radix_key & desired_mask) == desired
//
// desired_mask 表示高位中已经确定的 bit，desired 保存这些 bit 的目标值。
// 算法从最高位向最低位推进，所以候选集合会逐轮缩小。
//
// 每个线程先使用 warp ballot 得到本 warp 的桶计数；每个 warp 的 lane 0
// 再将计数原子累加到共享内存。相比每个元素都执行 atomicAdd，这样每个
// warp、每个桶、每轮最多只需要一次共享内存原子操作。
template <typename IndexType>
__device__ __forceinline__ void count_radix_using_mask(
    int (&thread_counts)[kRadixSize],
    int* shared_counts,
    uint32_t desired,
    uint32_t desired_mask,
    int bit_position,
    IndexType slice_size,
    IndexType within_slice_stride,
    const float* data)
{
#pragma unroll
    for (int bucket = 0; bucket < kRadixSize; ++bucket)
    {
        thread_counts[bucket] = 0;
    }

    if (threadIdx.x < kRadixSize)
    {
        shared_counts[threadIdx.x] = 0;
    }
    __syncthreads();

    const IndexType iteration_count =
        (slice_size + static_cast<IndexType>(blockDim.x) - 1) /
        static_cast<IndexType>(blockDim.x);
    const int lane = threadIdx.x & (warpSize - 1);

    // 所有线程执行相同的 iteration_count，保证 warp 中的线程都能参与
    // ballot。越过 slice_size 的线程只投 false，不读取越界数据。
    for (IndexType iteration = 0;
         iteration < iteration_count;
         ++iteration)
    {
        const IndexType index =
            static_cast<IndexType>(threadIdx.x) +
            iteration * static_cast<IndexType>(blockDim.x);
        const bool in_range = index < slice_size;

        uint32_t digit = 0;
        bool matches_prefix = false;
        if (in_range)
        {
            const float value = data[index * within_slice_stride];
            const uint32_t radix = FloatRadixConfig::convert(value);
            matches_prefix = (radix & desired_mask) == desired;
            digit = (radix >> bit_position) & kRadixMask;
        }

        const unsigned int active_mask = __activemask();
#pragma unroll
        for (uint32_t bucket = 0;
             bucket < static_cast<uint32_t>(kRadixSize);
             ++bucket)
        {
            const bool vote = matches_prefix && digit == bucket;
            const unsigned int votes = __ballot_sync(active_mask, vote);
            if (lane == 0)
            {
                thread_counts[bucket] += __popc(votes);
            }
        }
    }

    // 现在 thread_counts 只在每个 warp 的 lane 0 中有效。
    // 将所有 warp 的结果合并为 block 级计数。
    if (lane == 0)
    {
#pragma unroll
        for (int bucket = 0; bucket < kRadixSize; ++bucket)
        {
            atomicAdd(&shared_counts[bucket], thread_counts[bucket]);
        }
    }
    __syncthreads();

    // PyTorch 的后续分支由整个 block 一致执行，因此把共享内存计数广播
    // 到每个线程的局部数组。
#pragma unroll
    for (int bucket = 0; bucket < kRadixSize; ++bucket)
    {
        thread_counts[bucket] = shared_counts[bucket];
    }
    __syncthreads();
}

// 当某轮发现目标桶中只剩一个元素时，根据已确定的 radix 前缀从原输入
// 找回该元素。此提前结束路径可以跳过剩余的低位 radix 轮次。
//
// shared_storage 至少需要 kRadixSize 个 int。这里复用前两个位置保存
// found 标志和 float 结果，不增加额外共享内存。
template <typename IndexType>
__device__ __forceinline__ float find_pattern(
    int* shared_storage,
    const float* data,
    IndexType slice_size,
    IndexType within_slice_stride,
    uint32_t desired,
    uint32_t desired_mask)
{
    int* found_flag = &shared_storage[0];
    float* found_value =
        reinterpret_cast<float*>(&shared_storage[1]);

    if (threadIdx.x == 0)
    {
        *found_flag = 0;
        *found_value = 0.0F;
    }
    __syncthreads();

    const IndexType iteration_count =
        (slice_size + static_cast<IndexType>(blockDim.x) - 1) /
        static_cast<IndexType>(blockDim.x);
    for (IndexType iteration = 0;
         iteration < iteration_count;
         ++iteration)
    {
        const IndexType index =
            static_cast<IndexType>(threadIdx.x) +
            iteration * static_cast<IndexType>(blockDim.x);
        if (index < slice_size)
        {
            const float value = data[index * within_slice_stride];
            const uint32_t radix = FloatRadixConfig::convert(value);
            if ((radix & desired_mask) == desired)
            {
                // 调用方已经确认该前缀只匹配一个元素，因此不会有多个
                // 线程同时写 found_value。
                *found_value = value;
                *found_flag = 1;
            }
        }
        __syncthreads();

        if (*found_flag != 0)
        {
            // found_flag 对整个 block 可见，所以所有线程一致返回。
            return *found_value;
        }
        __syncthreads();
    }

    // 正常情况下不可能到达这里。返回值只用于防御意外的无匹配输入。
    return 0.0F;
}

// 找到切片中的第 K 大或第 K 小值。
//
// 参数约束：
// - k 使用从 1 开始的排名，必须满足 1 <= k <= slice_size；
// - 整个 block 必须以完全一致的参数调用；
// - shared_storage 至少包含 kRadixSize 个 int；
// - 函数只返回第 K 个阈值，不负责收集 Top-K 的值和索引。
//
// 算法从 float radix key 的最高 2 bit 开始。每轮统计四个桶后：
// 1. 按 largest 指定的方向依次检查桶；
// 2. 跳过不足以覆盖第 K 个元素的桶，并递减 k_to_find；
// 3. 选中包含目标排名的桶，把该 digit 加入 desired 前缀；
// 4. 继续检查下一组低位 bit。
template <typename IndexType>
__device__ __forceinline__ void radix_select(
    const float* data,
    IndexType k,
    bool largest,
    IndexType slice_size,
    IndexType within_slice_stride,
    int* shared_storage,
    float* top_k)
{
    int counts[kRadixSize];
    uint32_t desired = 0;
    uint32_t desired_mask = 0;
    IndexType k_to_find = k;

    for (int bit_position = 32 - kRadixBits;
         bit_position >= 0;
         bit_position -= kRadixBits)
    {
        count_radix_using_mask(
            counts,
            shared_storage,
            desired,
            desired_mask,
            bit_position,
            slice_size,
            within_slice_stride,
            data);

        const int first_bucket = largest ? kRadixSize - 1 : 0;
        const int bucket_step = largest ? -1 : 1;

        for (int offset = 0; offset < kRadixSize; ++offset)
        {
            const int bucket = first_bucket + offset * bucket_step;
            const int count = counts[bucket];

            // 目标桶中只剩一个元素，直接按前缀从输入中找回它。
            if (count == 1 && k_to_find == static_cast<IndexType>(1))
            {
                desired = set_radix_digit(
                    desired,
                    static_cast<uint32_t>(bucket),
                    bit_position);
                desired_mask = set_radix_digit(
                    desired_mask,
                    kRadixMask,
                    bit_position);
                *top_k = find_pattern(
                    shared_storage,
                    data,
                    slice_size,
                    within_slice_stride,
                    desired,
                    desired_mask);
                return;
            }

            // 第 K 个元素位于当前桶中，固定本轮 digit，继续检查低位。
            if (count >= k_to_find)
            {
                desired = set_radix_digit(
                    desired,
                    static_cast<uint32_t>(bucket),
                    bit_position);
                desired_mask = set_radix_digit(
                    desired_mask,
                    kRadixMask,
                    bit_position);
                break;
            }

            // 整个桶都排在目标元素之前，将其数量从目标排名中扣除。
            k_to_find -= static_cast<IndexType>(count);
        }
    }

    // 所有 32 个 bit 都已确定。若有多个相同元素，无法走唯一值提前返回
    // 分支，但 desired 已经是完整 radix key，可以直接反变换。
    *top_k = FloatRadixConfig::deconvert(desired);
}

}  // namespace topk_radix
}  // namespace cudaop
