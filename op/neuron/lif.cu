#include "lif.cuh"

namespace LIFNode
{
    // charge
    template<typename T>
    struct noDecayInput
    {
        __device__ __forceinline__ T operator()(const T &last_v, const T &v, const T &v_reset, const T &decay) const
        { return last_v + v - (last_v - v_reset) * decay; }
    };

    template<typename T>
    struct DecayInput
    {
        __device__ __forceinline__ T operator()(const T &last_v, const T &v, const T &v_reset, const T &decay) const
        { return last_v + (v - (last_v - v_reset)) * decay; }
    };

    template<template<typename> class ChargeFunc, typename T>
    __inline__ __device__ T NeuronCharge(T last_v, T v, T v_reset, T decay)
    {
        return ChargeFunc<T>()(last_v, v, v_reset, decay);
    }

    // reset
    template<typename T>
    struct HardReset
    {
        __device__ __forceinline__ T operator()(const T &v, const T &spike, const T &v_reset, const T &v_th) const
        { return v - spike * v + spike * v_reset; }
    };

    template<typename T>
    struct SoftReset
    {
        __device__ __forceinline__ T operator()(const T &v, const T &spike, const T &v_reset, const T &v_th) const
        { return v - spike * v_th; }
    };

    template<template<typename> class ResetFunc, typename T>
    __inline__ __device__ T NeuronReset(T v, T spike, T v_reset, T v_th)
    {
        v = ResetFunc<T>()(v, spike, v_reset, v_th);
        return v;
    }

    // grad_v_to_h
    template<typename T>
    struct GradVToHHardReset;

    template<>
    struct GradVToHHardReset<float>
    {
        __device__ __forceinline__ float operator()(float over_th) const
        { return over_th < 0; }
    };

    template<>
    struct GradVToHHardReset<half2>
    {
        __device__ __forceinline__ half2 operator()(half2 over_th) const
        { return __hle2(over_th, __float2half2_rn(0)); }
    };

    template<typename T>
    struct GradVToHSoftReset;

    template<>
    struct GradVToHSoftReset<float>
    {
        __device__ __forceinline__ float operator()(float over_th) const
        { return 1.0f; }
    };

    template<>
    struct GradVToHSoftReset<half2>
    {
        __device__ __forceinline__ half2 operator()(half2 over_th) const
        { return __float2half2_rn(1.0f); }
    };

    template<template<typename> class GradVToHFunc, typename T>
    __device__ T GradVToH(T over_th)
    { return GradVToHFunc<T>()(over_th); }

}


// --- --- --- --- --- --- --- --- LIFNode Forward FLOAT --- --- --- --- --- --- --- --- --- ---
template<
        template<typename> class ChargeFunc,
        template<typename> class ResetFunc,
        bool padding
>
__global__ void LIFNodeFPTTFLOATKernel(
        float *__restrict__ inputs,
        float *__restrict__ spikes_seq,
        float *__restrict__ v_seq,
        float *__restrict__ h_seq,
        const float v_th, const float v_reset,
        const float decay, const int64_t numel, const int64_t time_step)
{
    int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx = idx << 2;
    if (idx >= numel) return;

    bool isLegalIndex = idx + 3 < numel;
    int64_t edgeIndex = numel - idx;

    float v[4], spikes[4], last_v[4];

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        last_v[i] = 0;
    }

    int64_t index;
    for (int64_t t = 0; t < time_step; t++)
    {
        index = idx + numel * t;

        if (!padding || isLegalIndex)
        {
            FETCH_FLOAT4(v[0]) = FETCH_FLOAT4(inputs[index]);
        }
        else
        {
            for (int i = 0; i < edgeIndex; i++)
            {
                v[i] = inputs[index + i];
            }
        }

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            v[i] = LIFNode::NeuronCharge<ChargeFunc, float>(last_v[i], v[i], v_reset, decay);
        }

        if (!padding || isLegalIndex)
        {
            FETCH_FLOAT4(h_seq[index]) = FETCH_FLOAT4(v[0]);
        }
        else
        {
            for (int i = 0; i < edgeIndex; i++)
            {
                h_seq[index + i] = v[i];
            }
        }

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            spikes[i] = v[i] >= v_th;
            last_v[i] = LIFNode::NeuronReset<ResetFunc, float>(v[i], spikes[i], v_reset, v_th);
        }

        if (!padding || isLegalIndex)
        {
            FETCH_FLOAT4(spikes_seq[index]) = FETCH_FLOAT4(spikes[0]);
            FETCH_FLOAT4(v_seq[index]) = FETCH_FLOAT4(last_v[0]);
        }
        else
        {
            for (int i = 0; i < edgeIndex; i++)
            {
                spikes_seq[index + i] = spikes[i];
                v_seq[index + i] = last_v[i];
            }
        }
    }
}

// --- --- --- --- --- --- --- --- LIFNode Forward HALF --- --- --- --- --- --- --- --- --- ---
template<
        template<typename> class ChargeFunc,
        template<typename> class ResetFunc,
        bool padding
>
__global__ void LIFNodeFPTTHALFKernel(
        half *__restrict__ inputs,
        half *__restrict__ spikes_seq,
        half *__restrict__ v_seq,
        half *__restrict__ h_seq,
        const half2 v_th, const half2 v_reset,
        const half2 decay, const int64_t numel, const int64_t time_step)
{
    int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx = idx << 3;
    if (idx >= numel) return;

    bool isLegalIndex = idx + 7 < numel;
    int64_t edgeIndex = numel - idx;

    half2 v[4], spikes[4], last_v[4];

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        last_v[i] = __float2half2_rn(0);
    }

    int64_t index;
    for (int64_t t = 0; t < time_step; t++)
    {
        index = idx + numel * t;

        if (!padding || isLegalIndex)
        {
            FETCH_FLOAT4(v[0]) = FETCH_FLOAT4(inputs[index]);
        }
        else
        {
            auto *ptr = (half *) &v[0];
            for (int i = 0; i < edgeIndex; i++)
            {
                ptr[i] = inputs[index + i];
            }
        }

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            v[i] = LIFNode::NeuronCharge<ChargeFunc, half2>(last_v[i], v[i], v_reset, decay);
        }

        if (!padding || isLegalIndex)
        {
            FETCH_FLOAT4(h_seq[index]) = FETCH_FLOAT4(v[0]);
        }
        else
        {
            auto *ptr = (half *) &v[0];
            for (int i = 0; i < edgeIndex; i++)
            {
                h_seq[index + i] = ptr[i];
            }
        }

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            spikes[i] = __hge2(v[i], v_th);
            last_v[i] = LIFNode::NeuronReset<ResetFunc, half2>(v[i], spikes[i], v_reset, v_th);
        }

        if (!padding || isLegalIndex)
        {
            FETCH_FLOAT4(spikes_seq[index]) = FETCH_FLOAT4(spikes[0]);
            FETCH_FLOAT4(v_seq[index]) = FETCH_FLOAT4(last_v[0]);
        }
        else
        {
            auto *spike_ptr = (half *) &spikes[0];
            auto *v_ptr = (half *) &last_v[0];
            for (int i = 0; i < edgeIndex; i++)
            {
                spikes_seq[index + i] = spike_ptr[i];
                v_seq[index + i] = v_ptr[i];
            }
        }
    }
}

// --- --- --- --- --- --- --- --- LIFNode Backward FLOAT --- --- --- --- --- --- --- --- --- ---
template<
        template<typename> class GradVToHFunc, SurrogateFunc surrogateFunc,
        bool decay_input, bool detach_reset, bool padding
>
__global__ void LIFNodeBPTTFLOATKernel(
        float *__restrict__ grad_spike_seq,
        float *__restrict__ grad_v_seq,
        float *__restrict__ h_seq,
        float *__restrict__ grad_x_seq,
        const float v_th, const float v_reset, const float decay,
        const float alpha, const float args, const int64_t numel,
        const int64_t time_step)
{
    int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx = idx << 2;
    if (idx >= numel) return;

    bool isLegalIndex = idx + 3 < numel;
    int64_t edgeIndex = numel - idx;

    float load[4];
    float var[4], grad_v_to_h[4], grad_h[4];

    const float grad_h_to_x = decay_input ? decay : 1.0f;
    const float grad_h_next_to_v = 1.0f - decay;

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        grad_h[i] = 0;
    }

    int64_t index;
    for (int64_t t = time_step - 1; t >= 0; t--)
    {
        index = numel * t + idx;

        if (!padding || isLegalIndex)
        {
            FETCH_FLOAT4(load[0]) = FETCH_FLOAT4(h_seq[index]);
        }
        else
        {
            for (int i = 0; i < edgeIndex; i++)
            {
                load[i] = h_seq[index + i];
            }
        }

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            var[i] = load[i] - v_th;     // var = over_th | load = h
            grad_v_to_h[i] = LIFNode::GradVToH<GradVToHFunc, float>(var[i]);
        }

        if (surrogateFunc == SurrogateFunc::ATan)
        {
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                // 2alpha / (4 + (math.pi * alpha * x).pow_(2)) * grad_output
                // alpha_ / (4 +                   pai * x * x) * grad_output
                var[i] = alpha / (4.0f + args * var[i] * var[i]);  // var = grad_s_to_h
            }
        }
        else // SurrogateFunc::Sigmoid
        {
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                var[i] = 1.0f / (1.0f + expf(-alpha * var[i])); // 1.0f / (1.0f + expf(-alpha * over_th));
                var[i] = (1.0f - var[i]) * var[i] * alpha;      // var = grad_s_to_h
            }
        }

        if (!detach_reset)
        {
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                grad_v_to_h[i] += (v_reset - load[i]) * var[i]; // var = grad_s_to_h
            }
        }

        if (!padding || isLegalIndex)
        {
            FETCH_FLOAT4(load[0]) = FETCH_FLOAT4(grad_spike_seq[index]);
        }
        else
        {
            for (int i = 0; i < edgeIndex; i++)
            {
                load[i] = grad_spike_seq[index + i];
            }
        }

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            var[i] = var[i] * load[i]; // var = grad_s_to_h(var) * grad_spike(load)
        }

        if (!padding || isLegalIndex)
        {
            FETCH_FLOAT4(load[0]) = FETCH_FLOAT4(grad_v_seq[index]);
        }
        else
        {
            for (int i = 0; i < edgeIndex; i++)
            {
                load[i] = grad_v_seq[index + i];
            }
        }

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            grad_h[i] = grad_h[i] * grad_h_next_to_v + load[i];
            grad_h[i] = grad_h[i] * grad_v_to_h[i] + var[i];

            var[i] = grad_h[i] * grad_h_to_x;
        }

        if (!padding || isLegalIndex)
        {
            FETCH_FLOAT4(grad_x_seq[index]) = FETCH_FLOAT4(var[0]);
        }
        else
        {
            for (int i = 0; i < edgeIndex; i++)
            {
                grad_x_seq[index + i] = var[i];
            }
        }
    }
}

// --- --- --- --- --- --- --- --- LIFNode Backward HALF --- --- --- --- --- --- --- --- --- ---
template<
        template<typename> class GradVToHFunc, SurrogateFunc surrogateFunc,
        bool decay_input, bool detach_reset, bool padding
>
__global__ void LIFNodeBPTTHALFKernel(
        half *__restrict__ grad_spike_seq,
        half *__restrict__ grad_v_seq,
        half *__restrict__ h_seq,
        half *__restrict__ grad_x_seq,
        const half2 v_th, const half2 v_reset, const half2 decay,
        const half2 alpha, const half2 args, const int64_t numel,
        const int64_t time_step)
{
    int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx = idx << 3;
    if (idx >= numel) return;

    bool isLegalIndex = idx + 7 < numel;
    int64_t edgeIndex = numel - idx;

    half2 load[4];
    half2 var[4], grad_v_to_h[4], grad_h[4];

    const half2 grad_h_to_x = decay_input ? decay : __float2half2_rn(1);
    const half2 grad_h_next_to_v = __float2half2_rn(1) - decay;

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        grad_h[i] = __float2half2_rn(0);
    }

    int64_t index;
    for (int64_t t = time_step - 1; t >= 0; t--)
    {
        index = numel * t + idx;

        if (!padding || isLegalIndex)
        {
            FETCH_FLOAT4(load[0]) = FETCH_FLOAT4(h_seq[index]);
        }
        else
        {
            auto *ptr = (half *) &load[0];
            for (int i = 0; i < edgeIndex; i++)
            {
                ptr[i] = h_seq[index + i];
            }
        }

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            var[i] = load[i] - v_th;      // var = over_th
            grad_v_to_h[i] = LIFNode::GradVToH<GradVToHFunc, half2>(var[i]);
        }

        if (surrogateFunc == SurrogateFunc::ATan)
        {
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                // 2alpha / (4 + (math.pi * alpha * x).pow_(2)) * grad_output
                // alpha_ / (4 +                   pai * x * x) * grad_output
                var[i] = alpha / (__float2half2_rn(4.0f) + args * var[i] * var[i]);  // grad_s_to_h
            }
        }
        else
        {
#pragma unroll
            for (int i = 0; i < 4; i++)
            {    // 1 / (1 + exp(-alpha * over_th));
                var[i] = __float2half2_rn(1) / (__float2half2_rn(1) + h2exp(-alpha * var[i]));
                var[i] = (__float2half2_rn(1) - var[i]) * var[i] * alpha;  // grad_s_to_h
            }
        }

        if (!detach_reset)
        {
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                grad_v_to_h[i] += (v_reset - load[i]) * var[i]; // var = grad_s_to_h
            }
        }

        if (!padding || isLegalIndex)
        {
            FETCH_FLOAT4(load[0]) = FETCH_FLOAT4(grad_spike_seq[index]);
        }
        else
        {
            auto *ptr = (half *) &load[0];
            for (int i = 0; i < edgeIndex; i++)
            {
                ptr[i] = grad_spike_seq[index + i];
            }
        }

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            var[i] = var[i] * load[i]; // var = grad_s_to_h(var) * grad_spike(load)
        }

        if (!padding || isLegalIndex)
        {
            FETCH_FLOAT4(load[0]) = FETCH_FLOAT4(grad_v_seq[index]);
        }
        else
        {
            auto *ptr = (half *) &load[0];
            for (int i = 0; i < edgeIndex; i++)
            {
                ptr[i] = grad_v_seq[index + i];
            }
        }

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            grad_h[i] = grad_h[i] * grad_h_next_to_v + load[i];
            grad_h[i] = grad_h[i] * grad_v_to_h[i] + var[i];

            var[i] = grad_h[i] * grad_h_to_x;
        }

        if (!padding || isLegalIndex)
        {
            FETCH_FLOAT4(grad_x_seq[index]) = FETCH_FLOAT4(var[0]);
        }
        else
        {
            auto *ptr = (half *) &var[0];
            for (int i = 0; i < edgeIndex; i++)
            {
                grad_x_seq[index + i] = ptr[i];
            }
        }
    }
}


// --- --- --- --- --- --- --- --- LIFNode Forward Launch --- --- --- --- --- --- --- --- ---
std::vector<torch::Tensor> LIFNodeFPTTFLOATLaunch(
        const torch::Tensor &inputs, const float v_th, const float v_reset, const float decay,
        ResetType resetType, bool decay_input, const int threads)
{
    const int64_t T = inputs.size(0);         // [T, ...]
    const int64_t numel = inputs.numel() / T;

    torch::Tensor spike_seq = torch::empty(inputs.sizes(), inputs.options());
    torch::Tensor v_seq = torch::empty(inputs.sizes(), inputs.options());
    torch::Tensor h_seq = torch::empty(inputs.sizes(), inputs.options());

    auto *inputs_ptr = inputs.data_ptr<float>();
    auto *spike_seq_ptr = spike_seq.data_ptr<float>();
    auto *v_seq_ptr = v_seq.data_ptr<float>();
    auto *h_seq_ptr = h_seq.data_ptr<float>();

    bool padding = numel % 4 != 0 ? true : false;
    int64_t blocks = (numel / 4 + threads - 1) / threads;

    if (decay_input && resetType == ResetType::HardReset)
    {
        if (padding)
        {
            LIFNodeFPTTFLOATKernel<LIFNode::DecayInput, LIFNode::HardReset, true><<<blocks, threads>>>(
                    inputs_ptr, spike_seq_ptr, v_seq_ptr, h_seq_ptr, v_th, v_reset, decay, numel, T);
        }
        else
        {
            LIFNodeFPTTFLOATKernel<LIFNode::DecayInput, LIFNode::HardReset, false><<<blocks, threads>>>(
                    inputs_ptr, spike_seq_ptr, v_seq_ptr, h_seq_ptr, v_th, v_reset, decay, numel, T);
        }
    }
    else if (decay_input && resetType == ResetType::SoftReset)
    {
        if (padding)
        {
            LIFNodeFPTTFLOATKernel<LIFNode::DecayInput, LIFNode::SoftReset, true><<<blocks, threads>>>(
                    inputs_ptr, spike_seq_ptr, v_seq_ptr, h_seq_ptr, v_th, v_reset, decay, numel, T);
        }
        else
        {
            LIFNodeFPTTFLOATKernel<LIFNode::DecayInput, LIFNode::SoftReset, false><<<blocks, threads>>>(
                    inputs_ptr, spike_seq_ptr, v_seq_ptr, h_seq_ptr, v_th, v_reset, decay, numel, T);
        }
    }
    else if ((!decay_input) && resetType == ResetType::HardReset)
    {
        if (padding)
        {
            LIFNodeFPTTFLOATKernel<LIFNode::noDecayInput, LIFNode::HardReset, true><<<blocks, threads>>>(
                    inputs_ptr, spike_seq_ptr, v_seq_ptr, h_seq_ptr, v_th, v_reset, decay, numel, T);
        }
        else
        {
            LIFNodeFPTTFLOATKernel<LIFNode::noDecayInput, LIFNode::HardReset, false><<<blocks, threads>>>(
                    inputs_ptr, spike_seq_ptr, v_seq_ptr, h_seq_ptr, v_th, v_reset, decay, numel, T);
        }
    }
    else
    {
        if (padding)
        {
            LIFNodeFPTTFLOATKernel<LIFNode::noDecayInput, LIFNode::SoftReset, true><<<blocks, threads>>>(
                    inputs_ptr, spike_seq_ptr, v_seq_ptr, h_seq_ptr, v_th, v_reset, decay, numel, T);
        }
        else
        {
            LIFNodeFPTTFLOATKernel<LIFNode::noDecayInput, LIFNode::SoftReset, false><<<blocks, threads>>>(
                    inputs_ptr, spike_seq_ptr, v_seq_ptr, h_seq_ptr, v_th, v_reset, decay, numel, T);
        }
    }

    return {spike_seq, v_seq, h_seq};
}

std::vector<torch::Tensor> LIFNodeFPTTHALFLaunch(
        const torch::Tensor &inputs, const float v_th, const float v_reset, const float decay,
        ResetType resetType, bool decay_input, const int threads)
{
    const int64_t T = inputs.size(0);         // [T, ...]
    const int64_t numel = inputs.numel() / T;

    torch::Tensor spike_seq = torch::empty(inputs.sizes(), inputs.options());
    torch::Tensor v_seq = torch::empty(inputs.sizes(), inputs.options());
    torch::Tensor h_seq = torch::empty(inputs.sizes(), inputs.options());

    half2 v_th_half2 = __float2half2_rn(v_th);
    half2 v_reset_half2 = __float2half2_rn(v_reset);
    half2 decay_half2 = __float2half2_rn(decay);

    auto *inputs_ptr = inputs.data_ptr<at::Half>();
    auto *spike_seq_ptr = spike_seq.data_ptr<at::Half>();
    auto *v_seq_ptr = v_seq.data_ptr<at::Half>();
    auto *h_seq_ptr = h_seq.data_ptr<at::Half>();

    bool padding = numel % 8 != 0 ? true : false;
    int64_t blocks = (numel / 8 + threads - 1) / threads;

    if (decay_input && resetType == ResetType::HardReset)
    {
        if (padding)
        {
            LIFNodeFPTTHALFKernel<LIFNode::DecayInput, LIFNode::HardReset, true><<<blocks, threads>>>(
                    reinterpret_cast<half *>(inputs_ptr), reinterpret_cast<half *> (spike_seq_ptr),
                    reinterpret_cast<half *>(v_seq_ptr), reinterpret_cast<half *> (h_seq_ptr),
                    v_th_half2, v_reset_half2, decay_half2, numel, T);
        }
        else
        {
            LIFNodeFPTTHALFKernel<LIFNode::DecayInput, LIFNode::HardReset, false><<<blocks, threads>>>(
                    reinterpret_cast<half *>(inputs_ptr), reinterpret_cast<half *> (spike_seq_ptr),
                    reinterpret_cast<half *>(v_seq_ptr), reinterpret_cast<half *> (h_seq_ptr),
                    v_th_half2, v_reset_half2, decay_half2, numel, T);
        }
    }
    else if (decay_input && resetType == ResetType::SoftReset)
    {
        if (padding)
        {
            LIFNodeFPTTHALFKernel<LIFNode::DecayInput, LIFNode::SoftReset, true><<<blocks, threads>>>(
                    reinterpret_cast<half *>(inputs_ptr), reinterpret_cast<half *> (spike_seq_ptr),
                    reinterpret_cast<half *>(v_seq_ptr), reinterpret_cast<half *> (h_seq_ptr),
                    v_th_half2, v_reset_half2, decay_half2, numel, T);
        }
        else
        {
            LIFNodeFPTTHALFKernel<LIFNode::DecayInput, LIFNode::SoftReset, false><<<blocks, threads>>>(
                    reinterpret_cast<half *>(inputs_ptr), reinterpret_cast<half *> (spike_seq_ptr),
                    reinterpret_cast<half *>(v_seq_ptr), reinterpret_cast<half *> (h_seq_ptr),
                    v_th_half2, v_reset_half2, decay_half2, numel, T);
        }
    }
    else if ((!decay_input) && resetType == ResetType::HardReset)
    {
        if (padding)
        {
            LIFNodeFPTTHALFKernel<LIFNode::noDecayInput, LIFNode::HardReset, true><<<blocks, threads>>>(
                    reinterpret_cast<half *>(inputs_ptr), reinterpret_cast<half *> (spike_seq_ptr),
                    reinterpret_cast<half *>(v_seq_ptr), reinterpret_cast<half *> (h_seq_ptr),
                    v_th_half2, v_reset_half2, decay_half2, numel, T);
        }
        else
        {
            LIFNodeFPTTHALFKernel<LIFNode::noDecayInput, LIFNode::HardReset, false><<<blocks, threads>>>(
                    reinterpret_cast<half *>(inputs_ptr), reinterpret_cast<half *> (spike_seq_ptr),
                    reinterpret_cast<half *>(v_seq_ptr), reinterpret_cast<half *> (h_seq_ptr),
                    v_th_half2, v_reset_half2, decay_half2, numel, T);
        }
    }
    else
    {
        if (padding)
        {
            LIFNodeFPTTHALFKernel<LIFNode::noDecayInput, LIFNode::SoftReset, true><<<blocks, threads>>>(
                    reinterpret_cast<half *>(inputs_ptr), reinterpret_cast<half *> (spike_seq_ptr),
                    reinterpret_cast<half *>(v_seq_ptr), reinterpret_cast<half *> (h_seq_ptr),
                    v_th_half2, v_reset_half2, decay_half2, numel, T);
        }
        else
        {
            LIFNodeFPTTHALFKernel<LIFNode::noDecayInput, LIFNode::SoftReset, false><<<blocks, threads>>>(
                    reinterpret_cast<half *>(inputs_ptr), reinterpret_cast<half *> (spike_seq_ptr),
                    reinterpret_cast<half *>(v_seq_ptr), reinterpret_cast<half *> (h_seq_ptr),
                    v_th_half2, v_reset_half2, decay_half2, numel, T);
        }
    }

    return {spike_seq, v_seq, h_seq};
}

// --- --- --- --- --- --- --- --- LIFNode Backward Launch Sigmoid --- --- --- --- --- --- --- --- ---
std::vector<torch::Tensor> LIFNodeBPTTSigmoidFLOATLaunch(
        const torch::Tensor &grad_spike_seq, const torch::Tensor &h_seq, const torch::Tensor &grad_v_seq,
        const float v_th, const float v_reset, const float decay, const float alpha,
        const bool decay_input, const bool detach_reset, ResetType resetType, const int threads)
{
    const int64_t T = grad_spike_seq.size(0);         // [T, ...]
    const int64_t numel = grad_spike_seq.numel() / T;

    torch::Tensor grad_x_seq = torch::empty(grad_spike_seq.sizes(), grad_spike_seq.options());

    auto *grad_spike_ptr = grad_spike_seq.data_ptr<float>();
    auto *grad_v_ptr = grad_v_seq.data_ptr<float>();
    auto *h_seq_ptr = h_seq.data_ptr<float>();
    auto *grad_x_ptr = grad_x_seq.data_ptr<float>();

    bool padding = numel % 4 != 0 ? true : false;
    int64_t blocks = (numel / 4 + threads - 1) / threads;

    if (resetType == ResetType::HardReset)
    {
        if (decay_input && detach_reset)
        {
            if (padding)
            {
                LIFNodeBPTTFLOATKernel<LIFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, true, true, true> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, decay, alpha, 0, numel, T);
            }
            else
            {
                LIFNodeBPTTFLOATKernel<LIFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, true, true, false> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, decay, alpha, 0, numel, T);
            }
        }
        else if (decay_input && (!detach_reset))
        {
            if (padding)
            {
                LIFNodeBPTTFLOATKernel<LIFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, true, false, true> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, decay, alpha, 0, numel, T);
            }
            else
            {
                LIFNodeBPTTFLOATKernel<LIFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, true, false, false> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, decay, alpha, 0, numel, T);
            }
        }
        else if ((!decay_input) && detach_reset)
        {
            if (padding)
            {
                LIFNodeBPTTFLOATKernel<LIFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, false, true, true> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, decay, alpha, 0, numel, T);
            }
            else
            {
                LIFNodeBPTTFLOATKernel<LIFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, false, true, false> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, decay, alpha, 0, numel, T);
            }
        }
        else
        {
            if (padding)
            {
                LIFNodeBPTTFLOATKernel<LIFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, false, false, true> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, decay, alpha, 0, numel, T);
            }
            else
            {
                LIFNodeBPTTFLOATKernel<LIFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, false, false, false> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, decay, alpha, 0, numel, T);
            }
        }
    }
    else
    {
        if (decay_input && detach_reset)
        {
            if (padding)
            {
                LIFNodeBPTTFLOATKernel<LIFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, true, true, true> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, decay, alpha, 0, numel, T);
            }
            else
            {
                LIFNodeBPTTFLOATKernel<LIFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, true, true, false> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, decay, alpha, 0, numel, T);
            }
        }
        else if (decay_input && (!detach_reset))
        {
            if (padding)
            {
                LIFNodeBPTTFLOATKernel<LIFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, true, false, true> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, decay, alpha, 0, numel, T);
            }
            else
            {
                LIFNodeBPTTFLOATKernel<LIFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, true, false, false> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, decay, alpha, 0, numel, T);
            }
        }
        else if ((!decay_input) && detach_reset)
        {
            if (padding)
            {
                LIFNodeBPTTFLOATKernel<LIFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, false, true, true> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, decay, alpha, 0, numel, T);
            }
            else
            {
                LIFNodeBPTTFLOATKernel<LIFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, false, true, false> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, decay, alpha, 0, numel, T);
            }
        }
        else
        {
            if (padding)
            {
                LIFNodeBPTTFLOATKernel<LIFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, false, false, true> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, decay, alpha, 0, numel, T);
            }
            else
            {
                LIFNodeBPTTFLOATKernel<LIFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, false, false, false> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, decay, alpha, 0, numel, T);
            }
        }
    }

    return {grad_x_seq};
}

std::vector<torch::Tensor> LIFNodeBPTTSigmoidHALFLaunch(
        const torch::Tensor &grad_spike_seq, const torch::Tensor &h_seq, const torch::Tensor &grad_v_seq,
        const float v_th, const float v_reset, const float decay, const float alpha,
        const bool decay_input, const bool detach_reset, ResetType resetType, const int threads)
{
    const int64_t T = grad_spike_seq.size(0);         // [T, ...]
    const int64_t numel = grad_spike_seq.numel() / T;

    torch::Tensor grad_x_seq = torch::empty(grad_spike_seq.sizes(), grad_spike_seq.options());

    half2 v_th_half2 = __float2half2_rn(v_th);
    half2 v_reset_half2 = __float2half2_rn(v_reset);
    half2 decay_half2 = __float2half2_rn(decay);
    half2 alpha_half2 = __float2half2_rn(alpha);
    half2 args = __float2half2_rn(0);

    auto *grad_spike_ptr = grad_spike_seq.data_ptr<at::Half>();
    auto *grad_v_ptr = grad_v_seq.data_ptr<at::Half>();
    auto *h_seq_ptr = h_seq.data_ptr<at::Half>();
    auto *grad_x_ptr = grad_x_seq.data_ptr<at::Half>();

    bool padding = numel % 8 != 0 ? true : false;
    int64_t blocks = (numel / 8 + threads - 1) / threads;

    if (resetType == ResetType::HardReset)
    {
        if (decay_input && detach_reset)
        {
            if (padding)
            {
                LIFNodeBPTTHALFKernel<LIFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, true, true, true> <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, args, numel, T);
            }
            else
            {
                LIFNodeBPTTHALFKernel<LIFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, true, true, false> <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, args, numel, T);
            }
        }
        else if (decay_input && (!detach_reset))
        {
            if (padding)
            {
                LIFNodeBPTTHALFKernel<LIFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, true, false, true> <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, args, numel, T);
            }
            else
            {
                LIFNodeBPTTHALFKernel<LIFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, true, false, false> <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, args, numel, T);
            }
        }
        else if ((!decay_input) && detach_reset)
        {
            if (padding)
            {
                LIFNodeBPTTHALFKernel<LIFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, false, true, true> <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, args, numel, T);
            }
            else
            {
                LIFNodeBPTTHALFKernel<LIFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, false, true, false> <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, args, numel, T);
            }
        }
        else
        {
            if (padding)
            {
                LIFNodeBPTTHALFKernel<LIFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, false, false, true> <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, args, numel, T);
            }
            else
            {
                LIFNodeBPTTHALFKernel<LIFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, false, false, false> <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, args, numel, T);
            }

        }
    }
    else
    {
        if (decay_input && detach_reset)
        {
            if (padding)
            {
                LIFNodeBPTTHALFKernel<LIFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, true, true, true> <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, args, numel, T);
            }
            else
            {
                LIFNodeBPTTHALFKernel<LIFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, true, true, false> <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, args, numel, T);
            }
        }
        else if (decay_input && (!detach_reset))
        {
            if (padding)
            {
                LIFNodeBPTTHALFKernel<LIFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, true, false, true> <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, args, numel, T);
            }
            else
            {
                LIFNodeBPTTHALFKernel<LIFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, true, false, false> <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, args, numel, T);
            }
        }
        else if ((!decay_input) && detach_reset)
        {
            if (padding)
            {
                LIFNodeBPTTHALFKernel<LIFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, false, true, true> <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, args, numel, T);
            }
            else
            {
                LIFNodeBPTTHALFKernel<LIFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, false, true, false> <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, args, numel, T);
            }
        }
        else
        {
            if (padding)
            {
                LIFNodeBPTTHALFKernel<LIFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, false, false, true> <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, args, numel, T);
            }
            else
            {
                LIFNodeBPTTHALFKernel<LIFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, false, false, false> <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, args, numel, T);
            }
        }
    }

    return {grad_x_seq};
}

// --- --- --- --- --- --- --- --- LIFNode Backward Launch ATan --- --- --- --- --- --- --- --- ---
std::vector<torch::Tensor> LIFNodeBPTTATanFLOATLaunch(
        const torch::Tensor &grad_spike_seq, const torch::Tensor &h_seq, const torch::Tensor &grad_v_seq,
        const float v_th, const float v_reset, const float decay, float alpha,
        const bool decay_input, const bool detach_reset, ResetType resetType, const int threads)
{
    const int64_t T = grad_spike_seq.size(0);         // [T, B, C, H, W]
    const int64_t numel = grad_spike_seq.numel() / T; // B*C*H*W

    torch::Tensor grad_x_seq = torch::empty(grad_spike_seq.sizes(), grad_spike_seq.options());

    auto *grad_spike_ptr = grad_spike_seq.data_ptr<float>();
    auto *grad_v_ptr = grad_v_seq.data_ptr<float>();
    auto *h_seq_ptr = h_seq.data_ptr<float>();
    auto *grad_x_ptr = grad_x_seq.data_ptr<float>();

    const float pai = 3.14159265358979323846f * 3.14159265358979323846f * alpha * alpha;
    alpha = 2.0f * alpha;

    bool padding = numel % 4 != 0 ? true : false;
    int64_t blocks = (numel / 4 + threads - 1) / threads;

    if (resetType == ResetType::HardReset)
    {
        if (decay_input && detach_reset)
        {
            if (padding)
            {
                LIFNodeBPTTFLOATKernel<LIFNode::GradVToHHardReset, SurrogateFunc::ATan, true, true, true> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, decay, alpha, pai, numel, T);
            }
            else
            {
                LIFNodeBPTTFLOATKernel<LIFNode::GradVToHHardReset, SurrogateFunc::ATan, true, true, false> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, decay, alpha, pai, numel, T);
            }
        }
        else if (decay_input && (!detach_reset))
        {
            if (padding)
            {
                LIFNodeBPTTFLOATKernel<LIFNode::GradVToHHardReset, SurrogateFunc::ATan, true, false, true> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, decay, alpha, pai, numel, T);
            }
            else
            {
                LIFNodeBPTTFLOATKernel<LIFNode::GradVToHHardReset, SurrogateFunc::ATan, true, false, false> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, decay, alpha, pai, numel, T);
            }
        }
        else if ((!decay_input) && detach_reset)
        {
            if (padding)
            {
                LIFNodeBPTTFLOATKernel<LIFNode::GradVToHHardReset, SurrogateFunc::ATan, false, true, true> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, decay, alpha, pai, numel, T);
            }
            else
            {
                LIFNodeBPTTFLOATKernel<LIFNode::GradVToHHardReset, SurrogateFunc::ATan, false, true, false> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, decay, alpha, pai, numel, T);
            }
        }
        else
        {
            if (padding)
            {
                LIFNodeBPTTFLOATKernel<LIFNode::GradVToHHardReset, SurrogateFunc::ATan, false, false, true> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, decay, alpha, pai, numel, T);
            }
            else
            {
                LIFNodeBPTTFLOATKernel<LIFNode::GradVToHHardReset, SurrogateFunc::ATan, false, false, false> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, decay, alpha, pai, numel, T);
            }
        }
    }
    else
    {
        if (decay_input && detach_reset)
        {
            if (padding)
            {
                LIFNodeBPTTFLOATKernel<LIFNode::GradVToHSoftReset, SurrogateFunc::ATan, true, true, true> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, decay, alpha, pai, numel, T);
            }
            else
            {
                LIFNodeBPTTFLOATKernel<LIFNode::GradVToHSoftReset, SurrogateFunc::ATan, true, true, false> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, decay, alpha, pai, numel, T);
            }
        }
        else if (decay_input && (!detach_reset))
        {
            if (padding)
            {
                LIFNodeBPTTFLOATKernel<LIFNode::GradVToHSoftReset, SurrogateFunc::ATan, true, false, true> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, decay, alpha, pai, numel, T);
            }
            else
            {
                LIFNodeBPTTFLOATKernel<LIFNode::GradVToHSoftReset, SurrogateFunc::ATan, true, false, false> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, decay, alpha, pai, numel, T);
            }
        }
        else if ((!decay_input) && detach_reset)
        {
            if (padding)
            {
                LIFNodeBPTTFLOATKernel<LIFNode::GradVToHSoftReset, SurrogateFunc::ATan, false, true, true> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, decay, alpha, pai, numel, T);
            }
            else
            {
                LIFNodeBPTTFLOATKernel<LIFNode::GradVToHSoftReset, SurrogateFunc::ATan, false, true, false> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, decay, alpha, pai, numel, T);
            }
        }
        else
        {
            if (padding)
            {
                LIFNodeBPTTFLOATKernel<LIFNode::GradVToHSoftReset, SurrogateFunc::ATan, false, false, true> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, decay, alpha, pai, numel, T);
            }
            else
            {
                LIFNodeBPTTFLOATKernel<LIFNode::GradVToHSoftReset, SurrogateFunc::ATan, false, false, false> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, decay, alpha, pai, numel, T);
            }
        }
    }

    return {grad_x_seq};
}

std::vector<torch::Tensor> LIFNodeBPTTATanHALFLaunch(
        const torch::Tensor &grad_spike_seq, const torch::Tensor &h_seq, const torch::Tensor &grad_v_seq,
        const float v_th, const float v_reset, const float decay, const float alpha,
        const bool decay_input, const bool detach_reset, ResetType resetType, const int threads)
{
    const int64_t T = grad_spike_seq.size(0);         // [T, ...]
    const int64_t numel = grad_spike_seq.numel() / T;

    torch::Tensor grad_x_seq = torch::empty(grad_spike_seq.sizes(), grad_spike_seq.options());

    half2 v_th_half2 = __float2half2_rn(v_th);
    half2 v_reset_half2 = __float2half2_rn(v_reset);
    half2 decay_half2 = __float2half2_rn(decay);
    half2 alpha_half2 = __float2half2_rn(2.0f * alpha);
    half2 pai_half2 = __float2half2_rn(3.14159265358979323846f * 3.14159265358979323846f * alpha * alpha);

    auto *grad_spike_ptr = grad_spike_seq.data_ptr<at::Half>();
    auto *grad_v_ptr = grad_v_seq.data_ptr<at::Half>();
    auto *h_seq_ptr = h_seq.data_ptr<at::Half>();
    auto *grad_x_ptr = grad_x_seq.data_ptr<at::Half>();

    bool padding = numel % 8 != 0 ? true : false;
    int64_t blocks = (numel / 8 + threads - 1) / threads;

    if (resetType == ResetType::HardReset)
    {
        if (decay_input && detach_reset)
        {
            if (padding)
            {
                LIFNodeBPTTHALFKernel<LIFNode::GradVToHHardReset, SurrogateFunc::ATan, true, true, true> <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, pai_half2, numel, T);
            }
            else
            {
                LIFNodeBPTTHALFKernel<LIFNode::GradVToHHardReset, SurrogateFunc::ATan, true, true, false> <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, pai_half2, numel, T);
            }
        }
        else if (decay_input && (!detach_reset))
        {
            if (padding)
            {
                LIFNodeBPTTHALFKernel<LIFNode::GradVToHHardReset, SurrogateFunc::ATan, true, false, true> <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, pai_half2, numel, T);
            }
            else
            {
                LIFNodeBPTTHALFKernel<LIFNode::GradVToHHardReset, SurrogateFunc::ATan, true, false, false> <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, pai_half2, numel, T);
            }
        }
        else if ((!decay_input) && detach_reset)
        {
            if (padding)
            {
                LIFNodeBPTTHALFKernel<LIFNode::GradVToHHardReset, SurrogateFunc::ATan, false, true, true> <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, pai_half2, numel, T);
            }
            else
            {
                LIFNodeBPTTHALFKernel<LIFNode::GradVToHHardReset, SurrogateFunc::ATan, false, true, false> <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, pai_half2, numel, T);
            }
        }
        else
        {
            if (padding)
            {
                LIFNodeBPTTHALFKernel<LIFNode::GradVToHHardReset, SurrogateFunc::ATan, false, false, true> <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, pai_half2, numel, T);
            }
            else
            {
                LIFNodeBPTTHALFKernel<LIFNode::GradVToHHardReset, SurrogateFunc::ATan, false, false, false> <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, pai_half2, numel, T);
            }
        }
    }
    else
    {
        if (decay_input && detach_reset)
        {
            if (padding)
            {
                LIFNodeBPTTHALFKernel<LIFNode::GradVToHSoftReset, SurrogateFunc::ATan, true, true, true> <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, pai_half2, numel, T);
            }
            else
            {
                LIFNodeBPTTHALFKernel<LIFNode::GradVToHSoftReset, SurrogateFunc::ATan, true, true, false> <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, pai_half2, numel, T);
            }
        }
        else if (decay_input && (!detach_reset))
        {
            if (padding)
            {
                LIFNodeBPTTHALFKernel<LIFNode::GradVToHSoftReset, SurrogateFunc::ATan, true, false, true> <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, pai_half2, numel, T);
            }
            else
            {
                LIFNodeBPTTHALFKernel<LIFNode::GradVToHSoftReset, SurrogateFunc::ATan, true, false, false> <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, pai_half2, numel, T);
            }
        }
        else if ((!decay_input) && detach_reset)
        {
            if (padding)
            {
                LIFNodeBPTTHALFKernel<LIFNode::GradVToHSoftReset, SurrogateFunc::ATan, false, true, true> <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, pai_half2, numel, T);
            }
            else
            {
                LIFNodeBPTTHALFKernel<LIFNode::GradVToHSoftReset, SurrogateFunc::ATan, false, true, false> <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, pai_half2, numel, T);
            }
        }
        else
        {
            if (padding)
            {
                LIFNodeBPTTHALFKernel<LIFNode::GradVToHSoftReset, SurrogateFunc::ATan, false, false, true> <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, pai_half2, numel, T);
            }
            else
            {
                LIFNodeBPTTHALFKernel<LIFNode::GradVToHSoftReset, SurrogateFunc::ATan, false, false, false> <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, pai_half2, numel, T);
            }
        }
    }

    return {grad_x_seq};
}


void lif_main()
{
    std::cout << " CUDA is available: " << torch::cuda::is_available() << std::endl;
    std::cout << "cuDNN is available: " << torch::cuda::cudnn_is_available() << std::endl;
    std::cout << "Device count : " << torch::cuda::device_count() << std::endl;

    int T, B, C, H, W;
    T = 4, B = 16, C = 64, H = 40, W = 40;
    constexpr float v_reset = 0;
    constexpr float v_th = 1.0f;
    constexpr float decay = 0.5f;
    constexpr bool decay_input = false;
    ResetType resetType = ResetType::HardReset;

    constexpr float alpha = 2.0f;
    constexpr bool detach_reset = true;

    auto inputTensor = torch::rand(
            {T, B, C, H, W},
            torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA).requires_grad(true));
    // auto inputTensor = torch::rand(
    // {T, 17},
    // torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(true));
    // inputTensor.mul_(10);
    std::cout << "Options: " << inputTensor.options() << std::endl;

    // test
    torch::Tensor spike_seq = torch::empty(inputTensor.sizes(), inputTensor.options());
    torch::Tensor v_seq = torch::empty(inputTensor.sizes(), inputTensor.options());
    torch::Tensor h_seq = torch::empty(inputTensor.sizes(), inputTensor.options());

    torch::Tensor last_v = torch::zeros(inputTensor[0].sizes(), inputTensor.options());
    torch::Tensor v = torch::empty(inputTensor[0].sizes(), inputTensor.options());
    for (int t = 0; t < T; t++)
    {
        // neuron charge
        if (decay_input)
        {
            v = last_v + (inputTensor[t] - (last_v - v_reset)) * decay;
        }
        else
        {
            v = last_v + inputTensor[t] - (last_v - v_reset) * decay;
        }

        h_seq[t] = v;
        spike_seq[t] = v >= v_th;

        // reset
        if (resetType == ResetType::HardReset)
        {
            v = (1 - spike_seq[t]) * v + spike_seq[t] * v_reset;
        }
        else
        {
            v = v - spike_seq[t] * v_th;
        }

        v_seq[t] = v;
        last_v = v.detach();
    }
    // end test

    std::cout << " Launch CUDA Kernel... " << std::endl;
    std::vector<torch::Tensor> outputTensor; // spike_seq, v_seq, h_seq

    std::cout << " Forward CUDA Kernel... " << std::endl;
    if (inputTensor.options().dtype() == torch::kFloat16)
    {
        outputTensor = LIFNodeFPTTHALFLaunch(inputTensor, v_th, v_reset, decay, resetType, decay_input, 256);
    }
    else
    {
        outputTensor = LIFNodeFPTTFLOATLaunch(inputTensor, v_th, v_reset, decay, resetType, decay_input, 256);
    }

    // std::cout << "Backward CUDA Kernel... " << std::endl;
    // if (inputTensor.options().dtype() == torch::kFloat16)
    // {
    //     auto tensor = LIFNodeBPTTSigmoidHALFLaunch(
    //         outputTensor[0], outputTensor[1], outputTensor[2], v_th, v_reset, decay, alpha, decay_input, detach_reset, resetType, 256);
    // }
    // else
    // {
    //     auto tensor = LIFNodeBPTTSigmoidFLOATLaunch(
    //         outputTensor[0], outputTensor[1], outputTensor[2], v_th, v_reset, decay, alpha, decay_input, detach_reset, resetType, 256);
    // }
    // std::cout << outputTensor[0].sizes() << std::endl; // [4, 2, 4, 20, 20]
    // std::cout << outputTensor[1].sizes() << std::endl;
    // std::cout << outputTensor[2].sizes() << std::endl;

    torch::Tensor sub0 = outputTensor[0] - spike_seq;
    std::cout << sub0.max() << std::endl;

    torch::Tensor sub1 = outputTensor[1] - v_seq;
    std::cout << sub1.max() << std::endl;

    torch::Tensor sub2 = outputTensor[2] - h_seq;
    std::cout << sub2.max() << std::endl;

    // std::cout << "Test Time... " << std::endl;
    // float time_elapsed = 0;
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start);

    // for (int i = 0; i < 10; i++)
    // {
    //     // auto tensor = LIFNodeFPTTFLOATLaunch(inputTensor, v_th, v_reset, decay, resetType, decay_input, 256);

    //     auto tensor = LIFNodeBPTTSigmoidFLOATLaunch<decay_input, detach_reset>(
    //         outputTensor[0], outputTensor[1], outputTensor[2], v_th, v_reset, decay, alpha, resetType, 256);
    // }

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&time_elapsed, start, stop);
    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);

    // float 0.232246
    // half2 0.024195

    //
    // printf("Time: %.6lf\n", time_elapsed / 10);
}

