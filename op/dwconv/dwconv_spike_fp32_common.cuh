#pragma once

#include <cuda_runtime.h>

using SpikeScalar = float;

__device__ __forceinline__ SpikeScalar spike_zero()
{
    return 0.0F;
}

__device__ __forceinline__ SpikeScalar spike_add(
    SpikeScalar lhs,
    SpikeScalar rhs)
{
    return lhs + rhs;
}

__device__ __forceinline__ SpikeScalar spike_shuffle(
    SpikeScalar value,
    int source_lane)
{
    return __shfl_sync(0xFFFFFFFFU, value, source_lane);
}

inline SpikeScalar spike_from_float(float value)
{
    return value;
}

inline SpikeScalar spike_host_add(SpikeScalar lhs, SpikeScalar rhs)
{
    return lhs + rhs;
}

inline float spike_to_float(SpikeScalar value)
{
    return value;
}

constexpr float spike_tolerance()
{
    return 1.0e-4F;
}

constexpr const char *spike_dtype_name()
{
    return "fp32";
}

#include "dwconv_spike_common.cuh"
