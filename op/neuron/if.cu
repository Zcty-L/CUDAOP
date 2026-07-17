#include "if.cuh"


namespace IFNode
{
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


// --- --- --- --- --- --- --- --- IFNode Forward FLOAT --- --- --- --- --- --- --- --- --- ---
template<
        template<typename> class ResetFunc, bool padding
>
__global__ void IFNodeFPTTFLOATKernel(
        float *__restrict__ inputs,
        float *__restrict__ spikes_seq,
        float *__restrict__ v_seq,
        float *__restrict__ h_seq,
        const float v_th, const float v_reset,
        const int64_t numel, const int64_t time_step)
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
            v[i] = last_v[i] + v[i];
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
            last_v[i] = IFNode::NeuronReset<ResetFunc, float>(v[i], spikes[i], v_reset, v_th);
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

// --- --- --- --- --- --- --- --- IFNode Forward HALF --- --- --- --- --- --- --- --- --- ---
template<
        template<typename> class ResetFunc, bool padding
>
__global__ void IFNodeFPTTHALFKernel(
        half *__restrict__ inputs,
        half *__restrict__ spikes_seq,
        half *__restrict__ v_seq,
        half *__restrict__ h_seq,
        const half2 v_th, const half2 v_reset,
        const int64_t numel, const int64_t time_step)
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
            v[i] = last_v[i] + v[i];
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
            last_v[i] = IFNode::NeuronReset<ResetFunc, half2>(v[i], spikes[i], v_reset, v_th);
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

// --- --- --- --- --- --- --- --- IFNode Backward FLOAT --- --- --- --- --- --- --- --- --- ---
template<
        template<typename> class GradVToHFunc, SurrogateFunc surrogateFunc,
        bool detach_reset, bool padding
>
__global__ void IFNodeBPTTFLOATKernel(
        float *__restrict__ grad_spike_seq,
        float *__restrict__ grad_v_seq,
        float *__restrict__ h_seq,
        float *__restrict__ grad_x_seq,
        const float v_th, const float v_reset, const float alpha,
        const float args, const int64_t numel, const int64_t time_step)
{
    int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx = idx << 2;
    if (idx >= numel) return;

    bool isLegalIndex = idx + 3 < numel;
    int64_t edgeIndex = numel - idx;

    float load[4];
    float var[4], grad_v_to_h[4], grad_h[4];

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
            grad_v_to_h[i] = IFNode::GradVToH<GradVToHFunc, float>(var[i]);
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
            for (int i = 0; i < 4; i++) // no detach reset
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
            grad_h[i] = grad_h[i] + load[i]; // grad_h[i] * grad_h_next_to_v(1.0f) + grad_v[i];
            grad_h[i] = grad_h[i] * grad_v_to_h[i] + var[i];
            // grad_x_seq = grad_h[i] * grad_h_to_x(1.0f)
        }

        if (!padding || isLegalIndex)
        {
            FETCH_FLOAT4(grad_x_seq[index]) = FETCH_FLOAT4(grad_h[0]);
        }
        else
        {
            for (int i = 0; i < edgeIndex; i++)
            {
                grad_x_seq[index + i] = grad_h[i];
            }
        }
    }
}

// --- --- --- --- --- --- --- --- IFNode Backward HALF --- --- --- --- --- --- --- --- --- ---
template<
        template<typename> class GradVToHFunc, SurrogateFunc surrogateFunc,
        bool detach_reset, bool padding
>
__global__ void IFNodeBPTTHALFKernel(
        half *__restrict__ grad_spike_seq,
        half *__restrict__ grad_v_seq,
        half *__restrict__ h_seq,
        half *__restrict__ grad_x_seq,
        const half2 v_th, const half2 v_reset, const half2 alpha,
        const half2 args, const int64_t numel, const int64_t time_step)
{
    int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx = idx << 3;
    if (idx >= numel) return;

    bool isLegalIndex = idx + 7 < numel;
    int64_t edgeIndex = numel - idx;

    half2 load[4];
    half2 var[4], grad_v_to_h[4], grad_h[4];

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
            grad_v_to_h[i] = IFNode::GradVToH<GradVToHFunc, half2>(var[i]);
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
            for (int i = 0; i < 4; i++) // no detach reset
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
            grad_h[i] = grad_h[i] + load[i];
            grad_h[i] = grad_h[i] * grad_v_to_h[i] + var[i];
        }

        if (!padding || isLegalIndex)
        {
            FETCH_FLOAT4(grad_x_seq[index]) = FETCH_FLOAT4(grad_h[0]);
        }
        else
        {
            auto *ptr = (half *) &grad_h[0];
            for (int i = 0; i < edgeIndex; i++)
            {
                grad_x_seq[index + i] = ptr[i];
            }
        }
    }
}


// --- --- --- --- --- --- --- --- IFNode Forward Launch --- --- --- --- --- --- --- --- ---
std::vector<torch::Tensor> IFNodeFPTTFLOATLaunch(
        const torch::Tensor &inputs, const float v_th, const float v_reset,
        ResetType resetType, const int threads)
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

    if (resetType == ResetType::HardReset)
    {
        if (padding)
        {
            IFNodeFPTTFLOATKernel<IFNode::HardReset, true><<<blocks, threads>>>(
                    inputs_ptr, spike_seq_ptr, v_seq_ptr, h_seq_ptr, v_th, v_reset, numel, T);
        }
        else
        {
            IFNodeFPTTFLOATKernel<IFNode::HardReset, false><<<blocks, threads>>>(
                    inputs_ptr, spike_seq_ptr, v_seq_ptr, h_seq_ptr, v_th, v_reset, numel, T);
        }
    }
    else
    {
        if (padding)
        {
            IFNodeFPTTFLOATKernel<IFNode::SoftReset, true><<<blocks, threads>>>(
                    inputs_ptr, spike_seq_ptr, v_seq_ptr, h_seq_ptr, v_th, v_reset, numel, T);
        }
        else
        {
            IFNodeFPTTFLOATKernel<IFNode::SoftReset, false><<<blocks, threads>>>(
                    inputs_ptr, spike_seq_ptr, v_seq_ptr, h_seq_ptr, v_th, v_reset, numel, T);
        }
    }

    return {spike_seq, v_seq, h_seq};
}

std::vector<torch::Tensor> IFNodeFPTTHALFLaunch(
        const torch::Tensor &inputs, const float v_th, const float v_reset,
        ResetType resetType, const int threads)
{
    const int64_t T = inputs.size(0);         // [T, ...]
    const int64_t numel = inputs.numel() / T;

    torch::Tensor spike_seq = torch::empty(inputs.sizes(), inputs.options());
    torch::Tensor v_seq = torch::empty(inputs.sizes(), inputs.options());
    torch::Tensor h_seq = torch::empty(inputs.sizes(), inputs.options());

    half2 v_th_half2 = __float2half2_rn(v_th);
    half2 v_reset_half2 = __float2half2_rn(v_reset);

    auto *inputs_ptr = inputs.data_ptr<at::Half>();
    auto *spike_seq_ptr = spike_seq.data_ptr<at::Half>();
    auto *v_seq_ptr = v_seq.data_ptr<at::Half>();
    auto *h_seq_ptr = h_seq.data_ptr<at::Half>();

    bool padding = numel % 8 != 0 ? true : false;
    int64_t blocks = (numel / 8 + threads - 1) / threads;

    if (resetType == ResetType::HardReset)
    {
        if (padding)
        {
            IFNodeFPTTHALFKernel<IFNode::HardReset, true><<<blocks, threads>>>(
                    reinterpret_cast<half *>(inputs_ptr), reinterpret_cast<half *> (spike_seq_ptr),
                    reinterpret_cast<half *>(v_seq_ptr), reinterpret_cast<half *> (h_seq_ptr),
                    v_th_half2, v_reset_half2, numel, T);
        }
        else
        {
            IFNodeFPTTHALFKernel<IFNode::HardReset, false><<<blocks, threads>>>(
                    reinterpret_cast<half *>(inputs_ptr), reinterpret_cast<half *> (spike_seq_ptr),
                    reinterpret_cast<half *>(v_seq_ptr), reinterpret_cast<half *> (h_seq_ptr),
                    v_th_half2, v_reset_half2, numel, T);
        }
    }
    else
    {
        if (padding)
        {
            IFNodeFPTTHALFKernel<IFNode::SoftReset, true><<<blocks, threads>>>(
                    reinterpret_cast<half *>(inputs_ptr), reinterpret_cast<half *> (spike_seq_ptr),
                    reinterpret_cast<half *>(v_seq_ptr), reinterpret_cast<half *> (h_seq_ptr),
                    v_th_half2, v_reset_half2, numel, T);
        }
        else
        {
            IFNodeFPTTHALFKernel<IFNode::SoftReset, false><<<blocks, threads>>>(
                    reinterpret_cast<half *>(inputs_ptr), reinterpret_cast<half *> (spike_seq_ptr),
                    reinterpret_cast<half *>(v_seq_ptr), reinterpret_cast<half *> (h_seq_ptr),
                    v_th_half2, v_reset_half2, numel, T);
        }
    }

    return {spike_seq, v_seq, h_seq};
}

// --- --- --- --- --- --- --- --- IFNode Backward Launch Sigmoid --- --- --- --- --- --- --- --- ---
std::vector<torch::Tensor> IFNodeBPTTSigmoidFLOATLaunch(
        const torch::Tensor &grad_spike_seq, const torch::Tensor &h_seq, const torch::Tensor &grad_v_seq,
        const float v_th, const float v_reset, const float alpha,
        ResetType resetType, const bool detach_reset, const int threads)
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

    if (resetType == ResetType::HardReset && detach_reset)
    {
        if (padding)
        {
            IFNodeBPTTFLOATKernel<IFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, true, true> <<<blocks, threads>>>(
                    grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, alpha, 0, numel, T);
        }
        else
        {
            IFNodeBPTTFLOATKernel<IFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, true, false> <<<blocks, threads>>>(
                    grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, alpha, 0, numel, T);
        }
    }
    else if (resetType == ResetType::HardReset && (!detach_reset))
    {
        if (padding)
        {
            IFNodeBPTTFLOATKernel<IFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, false, true> <<<blocks, threads>>>(
                    grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, alpha, 0, numel, T);
        }
        else
        {
            IFNodeBPTTFLOATKernel<IFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, false, false> <<<blocks, threads>>>(
                    grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, alpha, 0, numel, T);
        }
    }
    else if (resetType == ResetType::SoftReset && detach_reset)
    {
        if (padding)
        {
            IFNodeBPTTFLOATKernel<IFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, true, true> <<<blocks, threads>>>(
                    grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, alpha, 0, numel, T);
        }
        else
        {
            IFNodeBPTTFLOATKernel<IFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, true, false> <<<blocks, threads>>>(
                    grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, alpha, 0, numel, T);
        }
    }
    else
    {
        if (padding)
        {
            IFNodeBPTTFLOATKernel<IFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, false, true> <<<blocks, threads>>>(
                    grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, alpha, 0, numel, T);
        }
        else
        {
            IFNodeBPTTFLOATKernel<IFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, false, false> <<<blocks, threads>>>(
                    grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, alpha, 0, numel, T);
        }
    }

    return {grad_x_seq};
}

std::vector<torch::Tensor> IFNodeBPTTSigmoidHALFLaunch(
        const torch::Tensor &grad_spike_seq, const torch::Tensor &h_seq, const torch::Tensor &grad_v_seq,
        const float v_th, const float v_reset, const float alpha,
        ResetType resetType, const bool detach_reset, const int threads)
{
    const int64_t T = grad_spike_seq.size(0);         // [T, ...]
    const int64_t numel = grad_spike_seq.numel() / T;

    torch::Tensor grad_x_seq = torch::empty(grad_spike_seq.sizes(), grad_spike_seq.options());

    half2 v_th_half2 = __float2half2_rn(v_th);
    half2 v_reset_half2 = __float2half2_rn(v_reset);
    half2 alpha_half2 = __float2half2_rn(alpha);
    half2 args = __float2half2_rn(0);

    auto *grad_spike_ptr = grad_spike_seq.data_ptr<at::Half>();
    auto *grad_v_ptr = grad_v_seq.data_ptr<at::Half>();
    auto *h_seq_ptr = h_seq.data_ptr<at::Half>();
    auto *grad_x_ptr = grad_x_seq.data_ptr<at::Half>();

    bool padding = numel % 8 != 0 ? true : false;
    int64_t blocks = (numel / 8 + threads - 1) / threads;

    if (resetType == ResetType::HardReset && detach_reset)
    {
        if (padding)
        {
            IFNodeBPTTHALFKernel<IFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, true, true> <<<blocks, threads>>>(
                    reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                    reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                    v_th_half2, v_reset_half2, alpha_half2, args, numel, T);
        }
        else
        {
            IFNodeBPTTHALFKernel<IFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, true, false> <<<blocks, threads>>>(
                    reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                    reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                    v_th_half2, v_reset_half2, alpha_half2, args, numel, T);
        }
    }
    else if (resetType == ResetType::HardReset && (!detach_reset))
    {
        if (padding)
        {
            IFNodeBPTTHALFKernel<IFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, false, true> <<<blocks, threads>>>(
                    reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                    reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                    v_th_half2, v_reset_half2, alpha_half2, args, numel, T);
        }
        else
        {
            IFNodeBPTTHALFKernel<IFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, false, false> <<<blocks, threads>>>(
                    reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                    reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                    v_th_half2, v_reset_half2, alpha_half2, args, numel, T);
        }
    }
    else if (resetType == ResetType::SoftReset && detach_reset)
    {
        if (padding)
        {
            IFNodeBPTTHALFKernel<IFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, true, true> <<<blocks, threads>>>(
                    reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                    reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                    v_th_half2, v_reset_half2, alpha_half2, args, numel, T);
        }
        else
        {
            IFNodeBPTTHALFKernel<IFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, true, false> <<<blocks, threads>>>(
                    reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                    reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                    v_th_half2, v_reset_half2, alpha_half2, args, numel, T);
        }
    }
    else
    {
        if (padding)
        {
            IFNodeBPTTHALFKernel<IFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, false, true> <<<blocks, threads>>>(
                    reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                    reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                    v_th_half2, v_reset_half2, alpha_half2, args, numel, T);
        }
        else
        {
            IFNodeBPTTHALFKernel<IFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, false, false> <<<blocks, threads>>>(
                    reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                    reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                    v_th_half2, v_reset_half2, alpha_half2, args, numel, T);
        }
    }

    return {grad_x_seq};
}

// --- --- --- --- --- --- --- --- IFNode Backward Launch ATan --- --- --- --- --- --- --- --- ---
std::vector<torch::Tensor> IFNodeBPTTATanFLOATLaunch(
        const torch::Tensor &grad_spike_seq, const torch::Tensor &h_seq, const torch::Tensor &grad_v_seq,
        const float v_th, const float v_reset, float alpha,
        ResetType resetType, const bool detach_reset, const int threads)
{
    const int64_t T = grad_spike_seq.size(0);         // [T, ...]
    const int64_t numel = grad_spike_seq.numel() / T;

    torch::Tensor grad_x_seq = torch::empty(grad_spike_seq.sizes(), grad_spike_seq.options());

    auto *grad_spike_ptr = grad_spike_seq.data_ptr<float>();
    auto *grad_v_ptr = grad_v_seq.data_ptr<float>();
    auto *h_seq_ptr = h_seq.data_ptr<float>();
    auto *grad_x_ptr = grad_x_seq.data_ptr<float>();

    const float pai = 3.14159265358979323846f * 3.14159265358979323846f * alpha * alpha;
    alpha = 2.0f * alpha;

    bool padding = numel % 4 != 0 ? true : false;
    int64_t blocks = (numel / 4 + threads - 1) / threads;

    if (resetType == ResetType::HardReset && detach_reset)
    {
        if (padding)
        {
            IFNodeBPTTFLOATKernel<IFNode::GradVToHHardReset, SurrogateFunc::ATan, true, true> <<<blocks, threads>>>(
                    grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, alpha, pai, numel, T);
        }
        else
        {
            IFNodeBPTTFLOATKernel<IFNode::GradVToHHardReset, SurrogateFunc::ATan, true, false> <<<blocks, threads>>>(
                    grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, alpha, pai, numel, T);
        }
    }
    else if (resetType == ResetType::HardReset && (!detach_reset))
    {
        if (padding)
        {
            IFNodeBPTTFLOATKernel<IFNode::GradVToHHardReset, SurrogateFunc::ATan, false, true> <<<blocks, threads>>>(
                    grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, alpha, pai, numel, T);
        }
        else
        {
            IFNodeBPTTFLOATKernel<IFNode::GradVToHHardReset, SurrogateFunc::ATan, false, false> <<<blocks, threads>>>(
                    grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, alpha, pai, numel, T);
        }
    }
    else if (resetType == ResetType::SoftReset && detach_reset)
    {
        if (padding)
        {
            IFNodeBPTTFLOATKernel<IFNode::GradVToHSoftReset, SurrogateFunc::ATan, true, true> <<<blocks, threads>>>(
                    grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, alpha, pai, numel, T);
        }
        else
        {
            IFNodeBPTTFLOATKernel<IFNode::GradVToHSoftReset, SurrogateFunc::ATan, true, false> <<<blocks, threads>>>(
                    grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, alpha, pai, numel, T);
        }
    }
    else
    {
        if (padding)
        {
            IFNodeBPTTFLOATKernel<IFNode::GradVToHSoftReset, SurrogateFunc::ATan, false, true> <<<blocks, threads>>>(
                    grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, alpha, pai, numel, T);
        }
        else
        {
            IFNodeBPTTFLOATKernel<IFNode::GradVToHSoftReset, SurrogateFunc::ATan, false, false> <<<blocks, threads>>>(
                    grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr, v_th, v_reset, alpha, pai, numel, T);
        }
    }

    return {grad_x_seq};
}

std::vector<torch::Tensor> IFNodeBPTTATanHALFLaunch(
        const torch::Tensor &grad_spike_seq, const torch::Tensor &h_seq, const torch::Tensor &grad_v_seq,
        const float v_th, const float v_reset, const float alpha,
        ResetType resetType, const bool detach_reset, const int threads)
{
    const int64_t T = grad_spike_seq.size(0);         // [T, ...]
    const int64_t numel = grad_spike_seq.numel() / T;

    torch::Tensor grad_x_seq = torch::empty(grad_spike_seq.sizes(), grad_spike_seq.options());

    half2 v_th_half2 = __float2half2_rn(v_th);
    half2 v_reset_half2 = __float2half2_rn(v_reset);
    half2 alpha_half2 = __float2half2_rn(2.0f * alpha);
    half2 pai_half2 = __float2half2_rn(3.14159265358979323846f * 3.14159265358979323846f * alpha * alpha);

    auto *grad_spike_ptr = grad_spike_seq.data_ptr<at::Half>();
    auto *grad_v_ptr = grad_v_seq.data_ptr<at::Half>();
    auto *h_seq_ptr = h_seq.data_ptr<at::Half>();
    auto *grad_x_ptr = grad_x_seq.data_ptr<at::Half>();

    bool padding = numel % 8 != 0 ? true : false;
    int64_t blocks = (numel / 8 + threads - 1) / threads;

    if (resetType == ResetType::HardReset && detach_reset)
    {
        if (padding)
        {
            IFNodeBPTTHALFKernel<IFNode::GradVToHHardReset, SurrogateFunc::ATan, true, true> <<<blocks, threads>>>(
                    reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                    reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                    v_th_half2, v_reset_half2, alpha_half2, pai_half2, numel, T);
        }
        else
        {
            IFNodeBPTTHALFKernel<IFNode::GradVToHHardReset, SurrogateFunc::ATan, true, false> <<<blocks, threads>>>(
                    reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                    reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                    v_th_half2, v_reset_half2, alpha_half2, pai_half2, numel, T);
        }
    }
    else if (resetType == ResetType::HardReset && (!detach_reset))
    {
        if (padding)
        {
            IFNodeBPTTHALFKernel<IFNode::GradVToHHardReset, SurrogateFunc::ATan, false, true> <<<blocks, threads>>>(
                    reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                    reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                    v_th_half2, v_reset_half2, alpha_half2, pai_half2, numel, T);
        }
        else
        {
            IFNodeBPTTHALFKernel<IFNode::GradVToHHardReset, SurrogateFunc::ATan, false, false> <<<blocks, threads>>>(
                    reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                    reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                    v_th_half2, v_reset_half2, alpha_half2, pai_half2, numel, T);
        }
    }
    else if (resetType == ResetType::SoftReset && detach_reset)
    {
        if (padding)
        {
            IFNodeBPTTHALFKernel<IFNode::GradVToHSoftReset, SurrogateFunc::ATan, true, true> <<<blocks, threads>>>(
                    reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                    reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                    v_th_half2, v_reset_half2, alpha_half2, pai_half2, numel, T);
        }
        else
        {
            IFNodeBPTTHALFKernel<IFNode::GradVToHSoftReset, SurrogateFunc::ATan, true, false> <<<blocks, threads>>>(
                    reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                    reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                    v_th_half2, v_reset_half2, alpha_half2, pai_half2, numel, T);
        }
    }
    else
    {
        if (padding)
        {
            IFNodeBPTTHALFKernel<IFNode::GradVToHSoftReset, SurrogateFunc::ATan, false, true> <<<blocks, threads>>>(
                    reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                    reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                    v_th_half2, v_reset_half2, alpha_half2, pai_half2, numel, T);
        }
        else
        {
            IFNodeBPTTHALFKernel<IFNode::GradVToHSoftReset, SurrogateFunc::ATan, false, false> <<<blocks, threads>>>(
                    reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                    reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (grad_x_ptr),
                    v_th_half2, v_reset_half2, alpha_half2, pai_half2, numel, T);
        }
    }

    return {grad_x_seq};
}


void if_main()
{


}

