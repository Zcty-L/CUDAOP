#pragma once

#include <cuda_fp16.h>

using SpikeScalar = __half;

__device__ __forceinline__ SpikeScalar spike_zero()
{
    return __float2half(0.0F);
}

__device__ __forceinline__ SpikeScalar spike_add(
    SpikeScalar lhs,
    SpikeScalar rhs)
{
    return __hadd(lhs, rhs);
}

__device__ __forceinline__ SpikeScalar spike_shuffle(
    SpikeScalar value,
    int source_lane)
{
    const unsigned int bits = __half_as_ushort(value);
    const unsigned int shuffled =
        __shfl_sync(0xFFFFFFFFU, bits, source_lane);
    return __ushort_as_half(static_cast<unsigned short>(shuffled));
}

inline SpikeScalar spike_from_float(float value)
{
    return __float2half(value);
}

inline SpikeScalar spike_host_add(SpikeScalar lhs, SpikeScalar rhs)
{
    return __hadd(lhs, rhs);
}

inline float spike_to_float(SpikeScalar value)
{
    return __half2float(value);
}

constexpr float spike_tolerance()
{
    return 1.0e-2F;
}

constexpr const char *spike_dtype_name()
{
    return "fp16";
}

#include "dwconv_spike_common.cuh"
