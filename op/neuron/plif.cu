#include "plif.cuh"


namespace PLIFNode
{
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


// --- --- --- --- --- --- --- --- LIFNode Backward FLOAT --- --- --- --- --- --- --- --- --- ---
template<
        template<typename> class GradVToHFunc, SurrogateFunc surrogateFunc,
        bool decay_input, bool detach_reset, bool padding
>
__global__ void PLIFNodeBPTTFLOATKernel(
        float *__restrict__ grad_spike_seq,
        float *__restrict__ grad_v_seq,
        float *__restrict__ h_seq,
        float *__restrict__ v_seq,
        float *__restrict__ grad_x_seq,
        float *__restrict__ grad_tau_seq,
        const float v_th, const float v_reset, const float decay,
        const float alpha, const float args, const int64_t numel,
        const int64_t time_step)
{
    int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx = idx << 2;

    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    __shared__ float smem[8];

    bool isLegalIndex = idx + 3 < numel;
    int64_t edgeIndex = numel - idx;

    float h[4], load[4];
    float var[4], grad_v_to_h[4], grad_h[4];
    float grad_tau = 0;

    const float grad_h_to_x = decay_input ? decay : 1.0f;
    const float grad_h_next_to_v = 1.0f - decay;

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        h[i] = 0;
        var[i] = 0;
        grad_h[i] = 0;
    }

    if (idx < numel)
    {
        int64_t index;
        for (int64_t t = time_step - 1; t >= 0; t--)
        {
            index = numel * t + idx;

            if (!padding || isLegalIndex)
            {
                FETCH_FLOAT4(h[0]) = FETCH_FLOAT4(h_seq[index]);
                FETCH_FLOAT4(load[0]) = FETCH_FLOAT4(grad_spike_seq[index]);
            }
            else
            {
                for (int i = 0; i < edgeIndex; i++)
                {
                    h[i] = h_seq[index + i];
                    load[i] = grad_spike_seq[index + i];
                }
            }

#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                var[i] = h[i] - v_th;     // var = over_th
                grad_v_to_h[i] = PLIFNode::GradVToH<GradVToHFunc, float>(var[i]);
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
                    grad_v_to_h[i] += (v_reset - h[i]) * var[i]; // var = grad_s_to_h
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

#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                load[i] = 0;
            }
            if (t > 0)
            {
                if (!padding || isLegalIndex)
                {
                    FETCH_FLOAT4(load[0]) = FETCH_FLOAT4(v_seq[index - numel]);
                }
                else
                {
                    for (int i = 0; i < edgeIndex; i++)
                    {
                        load[i] = v_seq[index - numel + i];
                    }
                }
            }

            if (decay_input)
            {
#pragma unroll
                for (int i = 0; i < 4; i++)
                {
                    var[i] = (h[i] - load[i]) * grad_h[i]; // load = v
                    var[i] = var[i] / decay;

                    grad_tau += var[i];
                }
            }
            else
            {
#pragma unroll
                for (int i = 0; i < 4; i++)
                {
                    var[i] = (v_reset - load[i]) * grad_h[i]; // load = v

                    grad_tau += var[i];
                }
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        grad_tau += __shfl_xor_sync(0xFFFFFFFF, grad_tau, offset);

    if (lane_id == 0)
    {
        smem[warp_id] = grad_tau;
    }
    __syncthreads();

    if (threadIdx.x < 8)
    {
        grad_tau = smem[threadIdx.x];
    }
    __syncthreads();

    if (warp_id == 0)
    {
        grad_tau += __shfl_xor_sync(0xFF, grad_tau, 4, 8);
        grad_tau += __shfl_xor_sync(0xFF, grad_tau, 2, 8);
        grad_tau += __shfl_xor_sync(0xFF, grad_tau, 1, 8);
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(grad_tau_seq, grad_tau);
    }
}

// --- --- --- --- --- --- --- --- PLIFNode Backward HALF --- --- --- --- --- --- --- --- --- ---
template<
        template<typename> class GradVToHFunc, SurrogateFunc surrogateFunc,
        bool decay_input, bool detach_reset, bool padding
>
__global__ void PLIFNodeBPTTHALFKernel(
        half *__restrict__ grad_spike_seq,
        half *__restrict__ grad_v_seq,
        half *__restrict__ h_seq,
        half *__restrict__ v_seq,
        half *__restrict__ grad_x_seq,
        float *__restrict__ grad_tau_seq,
        const half2 v_th, const half2 v_reset, const half2 decay,
        const half2 alpha, const half2 args, const int64_t numel,
        const int64_t time_step)
{
    int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx = idx << 3;

    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    __shared__ float smem[8];

    bool isLegalIndex = idx + 7 < numel;
    int64_t edgeIndex = numel - idx;

    half2 h[4], load[4];
    half2 var[4], grad_v_to_h[4], grad_h[4];
    float grad_tau = 0;

    const half2 grad_h_to_x = decay_input ? decay : __float2half2_rn(1);
    const half2 grad_h_next_to_v = __float2half2_rn(1) - decay;

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        h[i] = __float2half2_rn(0);
        var[i] = __float2half2_rn(0);
        grad_h[i] = __float2half2_rn(0);
    }

    if (idx < numel)
    {
        int64_t index;
        for (int64_t t = time_step - 1; t >= 0; t--)
        {
            index = numel * t + idx;

            if (!padding || isLegalIndex)
            {
                FETCH_FLOAT4(h[0]) = FETCH_FLOAT4(h_seq[index]);
                FETCH_FLOAT4(load[0]) = FETCH_FLOAT4(grad_spike_seq[index]);
            }
            else
            {
                auto *h_ptr = (half *) &h[0];
                auto *grad_spike_ptr = (half *) &load[0];
                for (int i = 0; i < edgeIndex; i++)
                {
                    h_ptr[i] = h_seq[index + i];
                    grad_spike_ptr[i] = grad_spike_seq[index + i];
                }
            }

#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                var[i] = h[i] - v_th;      // var = over_th
                grad_v_to_h[i] = PLIFNode::GradVToH<GradVToHFunc, half2>(var[i]);
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
                    grad_v_to_h[i] += (v_reset - h[i]) * var[i]; // var = grad_s_to_h
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

#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                load[i] = __float2half2_rn(0);
            }
            if (t > 0)
            {
                if (!padding || isLegalIndex)
                {
                    FETCH_FLOAT4(load[0]) = FETCH_FLOAT4(v_seq[index - numel]);
                }
                else
                {
                    auto *ptr = (half *) &load[0];
                    for (int i = 0; i < edgeIndex; i++)
                    {
                        ptr[i] = v_seq[index - numel + i];
                    }
                }
            }

            if (decay_input)
            {
#pragma unroll
                for (int i = 0; i < 4; i++)
                {
                    var[i] = (h[i] - load[i]) * grad_h[i]; // load = v
                    var[i] = var[i] / decay;

                    grad_tau += __half2float(var[i].x), grad_tau += __half2float(var[i].y);
                }
            }
            else
            {
#pragma unroll
                for (int i = 0; i < 4; i++)
                {
                    var[i] = (v_reset - load[i]) * grad_h[i]; // load = v

                    grad_tau += __half2float(var[i].x), grad_tau += __half2float(var[i].y);
                }
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        grad_tau += __shfl_xor_sync(0xFFFFFFFF, grad_tau, offset);

    if (lane_id == 0)
    {
        smem[warp_id] = grad_tau;
    }
    __syncthreads();

    if (threadIdx.x < 8)
    {
        grad_tau = smem[threadIdx.x];
    }
    __syncthreads();

    if (warp_id == 0)
    {
        grad_tau += __shfl_xor_sync(0xFF, grad_tau, 4, 8);
        grad_tau += __shfl_xor_sync(0xFF, grad_tau, 2, 8);
        grad_tau += __shfl_xor_sync(0xFF, grad_tau, 1, 8);
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(grad_tau_seq, grad_tau);
    }
}


// --- --- --- --- --- --- --- --- PLIFNode Backward Launch Sigmoid --- --- --- --- --- --- --- --- ---
std::vector<torch::Tensor> PLIFNodeBPTTSigmoidFLOATLaunch(
        const torch::Tensor &grad_spike_seq, const torch::Tensor &h_seq, const torch::Tensor &grad_v_seq,
        const torch::Tensor &v_seq, const float v_th, const float v_reset, const float tau, const float alpha,
        const bool decay_input, ResetType resetType, const bool detach_reset, const int threads)
{
    const int64_t T = grad_spike_seq.size(0);         // [T, B, C, H, W]
    const int64_t numel = grad_spike_seq.numel() / T; // B*C*H*W

    torch::Tensor grad_x_seq = torch::empty(grad_spike_seq.sizes(), grad_spike_seq.options());
    torch::Tensor grad_tau_seq = torch::zeros({1}, grad_spike_seq.options());

    auto *grad_spike_ptr = grad_spike_seq.data_ptr<float>();
    auto *grad_v_ptr = grad_v_seq.data_ptr<float>();
    auto *h_seq_ptr = h_seq.data_ptr<float>();
    auto *v_seq_ptr = v_seq.data_ptr<float>();

    auto *grad_x_ptr = grad_x_seq.data_ptr<float>();
    auto *grad_tau_ptr = grad_tau_seq.data_ptr<float>();

    bool padding = numel % 4 != 0 ? true : false;
    int64_t blocks = (numel / 4 + threads - 1) / threads;

    if (resetType == ResetType::HardReset)
    {
        if (decay_input && detach_reset)
        {
            if (padding)
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, true, true, true> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, v_seq_ptr, grad_x_ptr, grad_tau_ptr,
                        v_th, v_reset, tau, alpha, 0, numel, T);
            }
            else
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, true, true, false> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, v_seq_ptr, grad_x_ptr, grad_tau_ptr,
                        v_th, v_reset, tau, alpha, 0, numel, T);
            }
        }
        else if (decay_input && (!detach_reset))
        {
            if (padding)
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, true, false, true> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, v_seq_ptr, grad_x_ptr, grad_tau_ptr,
                        v_th, v_reset, tau, alpha, 0, numel, T);
            }
            else
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, true, false, false> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, v_seq_ptr, grad_x_ptr, grad_tau_ptr,
                        v_th, v_reset, tau, alpha, 0, numel, T);
            }
        }
        else if ((!decay_input) && detach_reset)
        {
            if (padding)
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, false, true, true> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, v_seq_ptr, grad_x_ptr, grad_tau_ptr,
                        v_th, v_reset, tau, alpha, 0, numel, T);
            }
            else
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, false, true, false> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, v_seq_ptr, grad_x_ptr, grad_tau_ptr,
                        v_th, v_reset, tau, alpha, 0, numel, T);
            }
        }
        else
        {
            if (padding)
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, false, false, true> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, v_seq_ptr, grad_x_ptr, grad_tau_ptr,
                        v_th, v_reset, tau, alpha, 0, numel, T);
            }
            else
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, false, false, false> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, v_seq_ptr, grad_x_ptr, grad_tau_ptr,
                        v_th, v_reset, tau, alpha, 0, numel, T);
            }
        }
    }
    else
    {
        if (decay_input && detach_reset)
        {
            if (padding)
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, true, true, true> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, v_seq_ptr, grad_x_ptr, grad_tau_ptr,
                        v_th, v_reset, tau, alpha, 0, numel, T);
            }
            else
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, true, true, false> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, v_seq_ptr, grad_x_ptr, grad_tau_ptr,
                        v_th, v_reset, tau, alpha, 0, numel, T);
            }
        }
        else if (decay_input && (!detach_reset))
        {
            if (padding)
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, true, false, true> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, v_seq_ptr, grad_x_ptr, grad_tau_ptr,
                        v_th, v_reset, tau, alpha, 0, numel, T);
            }
            else
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, true, false, false> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, v_seq_ptr, grad_x_ptr, grad_tau_ptr,
                        v_th, v_reset, tau, alpha, 0, numel, T);
            }
        }
        else if ((!decay_input) && detach_reset)
        {
            if (padding)
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, false, true, true> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, v_seq_ptr, grad_x_ptr, grad_tau_ptr,
                        v_th, v_reset, tau, alpha, 0, numel, T);
            }
            else
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, false, true, false> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, v_seq_ptr, grad_x_ptr, grad_tau_ptr,
                        v_th, v_reset, tau, alpha, 0, numel, T);
            }
        }
        else
        {
            if (padding)
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, false, false, true> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, v_seq_ptr, grad_x_ptr, grad_tau_ptr,
                        v_th, v_reset, tau, alpha, 0, numel, T);
            }
            else
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, false, false, false> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, v_seq_ptr, grad_x_ptr, grad_tau_ptr,
                        v_th, v_reset, tau, alpha, 0, numel, T);
            }
        }
    }

    return {grad_x_seq, grad_tau_seq};
}

std::vector<torch::Tensor> PLIFNodeBPTTSigmoidHALFLaunch(
        const torch::Tensor &grad_spike_seq, const torch::Tensor &h_seq, const torch::Tensor &grad_v_seq,
        const torch::Tensor &v_seq, const float v_th, const float v_reset, const float tau, const float alpha,
        const bool decay_input, ResetType resetType, const bool detach_reset, const int threads)
{
    const int64_t T = grad_spike_seq.size(0);         // [T, ...]
    const int64_t numel = grad_spike_seq.numel() / T;

    torch::Tensor grad_x_seq = torch::empty(grad_spike_seq.sizes(), grad_spike_seq.options());
    torch::Tensor grad_tau_seq = torch::zeros({1}, grad_spike_seq.options().dtype(torch::kFloat32));

    half2 v_th_half2 = __float2half2_rn(v_th);
    half2 v_reset_half2 = __float2half2_rn(v_reset);
    half2 tau_half2 = __float2half2_rn(tau);
    half2 alpha_half2 = __float2half2_rn(alpha);
    half2 args_half2 = __float2half2_rn(0);

    auto *grad_spike_ptr = grad_spike_seq.data_ptr<at::Half>();
    auto *grad_v_ptr = grad_v_seq.data_ptr<at::Half>();
    auto *h_seq_ptr = h_seq.data_ptr<at::Half>();
    auto *v_seq_ptr = v_seq.data_ptr<at::Half>();

    auto *grad_x_ptr = grad_x_seq.data_ptr<at::Half>();
    auto *grad_tau = grad_tau_seq.data_ptr<float>();

    bool padding = numel % 8 != 0 ? true : false;
    int64_t blocks = (numel / 8 + threads - 1) / threads;

    if (resetType == ResetType::HardReset)
    {
        if (decay_input && detach_reset)
        {
            if (padding)
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, true, true, true>  <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (v_seq_ptr),
                        reinterpret_cast<half *> (grad_x_ptr), grad_tau,
                        v_th_half2, v_reset_half2, tau_half2, alpha_half2, args_half2, numel, T);
            }
            else
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, true, true, false>  <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (v_seq_ptr),
                        reinterpret_cast<half *> (grad_x_ptr), grad_tau,
                        v_th_half2, v_reset_half2, tau_half2, alpha_half2, args_half2, numel, T);
            }
        }
        else if (decay_input && (!detach_reset))
        {
            if (padding)
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, true, false, true>  <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (v_seq_ptr),
                        reinterpret_cast<half *> (grad_x_ptr), grad_tau,
                        v_th_half2, v_reset_half2, tau_half2, alpha_half2, args_half2, numel, T);
            }
            else
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, true, false, false>  <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (v_seq_ptr),
                        reinterpret_cast<half *> (grad_x_ptr), grad_tau,
                        v_th_half2, v_reset_half2, tau_half2, alpha_half2, args_half2, numel, T);
            }
        }
        else if ((!decay_input) && detach_reset)
        {
            if (padding)
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, false, true, true>  <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (v_seq_ptr),
                        reinterpret_cast<half *> (grad_x_ptr), grad_tau,
                        v_th_half2, v_reset_half2, tau_half2, alpha_half2, args_half2, numel, T);
            }
            else
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, false, true, false>  <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (v_seq_ptr),
                        reinterpret_cast<half *> (grad_x_ptr), grad_tau,
                        v_th_half2, v_reset_half2, tau_half2, alpha_half2, args_half2, numel, T);
            }
        }
        else
        {
            if (padding)
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, false, false, true>  <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (v_seq_ptr),
                        reinterpret_cast<half *> (grad_x_ptr), grad_tau,
                        v_th_half2, v_reset_half2, tau_half2, alpha_half2, args_half2, numel, T);
            }
            else
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHHardReset, SurrogateFunc::Sigmoid, false, false, false>  <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (v_seq_ptr),
                        reinterpret_cast<half *> (grad_x_ptr), grad_tau,
                        v_th_half2, v_reset_half2, tau_half2, alpha_half2, args_half2, numel, T);
            }
        }
    }
    else
    {
        if (decay_input && detach_reset)
        {
            if (padding)
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, true, true, true>  <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (v_seq_ptr),
                        reinterpret_cast<half *> (grad_x_ptr), grad_tau,
                        v_th_half2, v_reset_half2, tau_half2, alpha_half2, args_half2, numel, T);
            }
            else
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, true, true, false>  <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (v_seq_ptr),
                        reinterpret_cast<half *> (grad_x_ptr), grad_tau,
                        v_th_half2, v_reset_half2, tau_half2, alpha_half2, args_half2, numel, T);
            }
        }
        else if (decay_input && (!detach_reset))
        {
            if (padding)
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, true, false, true>  <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (v_seq_ptr),
                        reinterpret_cast<half *> (grad_x_ptr), grad_tau,
                        v_th_half2, v_reset_half2, tau_half2, alpha_half2, args_half2, numel, T);
            }
            else
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, true, false, false>  <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (v_seq_ptr),
                        reinterpret_cast<half *> (grad_x_ptr), grad_tau,
                        v_th_half2, v_reset_half2, tau_half2, alpha_half2, args_half2, numel, T);
            }
        }
        else if ((!decay_input) && detach_reset)
        {
            if (padding)
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, false, true, true>  <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (v_seq_ptr),
                        reinterpret_cast<half *> (grad_x_ptr), grad_tau,
                        v_th_half2, v_reset_half2, tau_half2, alpha_half2, args_half2, numel, T);
            }
            else
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, false, true, false>  <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (v_seq_ptr),
                        reinterpret_cast<half *> (grad_x_ptr), grad_tau,
                        v_th_half2, v_reset_half2, tau_half2, alpha_half2, args_half2, numel, T);
            }
        }
        else
        {
            if (padding)
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, false, false, true>  <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (v_seq_ptr),
                        reinterpret_cast<half *> (grad_x_ptr), grad_tau,
                        v_th_half2, v_reset_half2, tau_half2, alpha_half2, args_half2, numel, T);
            }
            else
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHSoftReset, SurrogateFunc::Sigmoid, false, false, false>  <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (v_seq_ptr),
                        reinterpret_cast<half *> (grad_x_ptr), grad_tau,
                        v_th_half2, v_reset_half2, tau_half2, alpha_half2, args_half2, numel, T);
            }
        }
    }

    return {grad_x_seq, grad_tau_seq};
}


// --- --- --- --- --- --- --- --- PLIFNode Backward Launch ATan --- --- --- --- --- --- --- --- ---
std::vector<torch::Tensor> PLIFNodeBPTTATanFLOATLaunch(
        const torch::Tensor &grad_spike_seq, const torch::Tensor &h_seq, const torch::Tensor &grad_v_seq,
        const torch::Tensor &v_seq, const float v_th, const float v_reset, const float tau, float alpha,
        const bool decay_input, ResetType resetType, const bool detach_reset, const int threads)
{
    const int64_t T = grad_spike_seq.size(0);
    const int64_t numel = grad_spike_seq.numel() / T;

    torch::Tensor grad_x_seq = torch::empty(grad_spike_seq.sizes(), grad_spike_seq.options());
    torch::Tensor grad_tau_seq = torch::zeros({1}, grad_spike_seq.options());

    auto *grad_spike_ptr = grad_spike_seq.data_ptr<float>();
    auto *grad_v_ptr = grad_v_seq.data_ptr<float>();
    auto *h_seq_ptr = h_seq.data_ptr<float>();
    auto *v_seq_ptr = v_seq.data_ptr<float>();

    auto *grad_x_ptr = grad_x_seq.data_ptr<float>();
    auto *grad_tau_ptr = grad_tau_seq.data_ptr<float>();

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
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHHardReset, SurrogateFunc::ATan, true, true, true> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, v_seq_ptr, grad_x_ptr, grad_tau_ptr,
                        v_th, v_reset, tau, alpha, pai, numel, T);
            }
            else
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHHardReset, SurrogateFunc::ATan, true, true, false> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, v_seq_ptr, grad_x_ptr, grad_tau_ptr,
                        v_th, v_reset, tau, alpha, pai, numel, T);
            }
        }
        else if (decay_input && (!detach_reset))
        {
            if (padding)
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHHardReset, SurrogateFunc::ATan, true, false, true> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, v_seq_ptr, grad_x_ptr, grad_tau_ptr,
                        v_th, v_reset, tau, alpha, pai, numel, T);
            }
            else
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHHardReset, SurrogateFunc::ATan, true, false, false> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, v_seq_ptr, grad_x_ptr, grad_tau_ptr,
                        v_th, v_reset, tau, alpha, pai, numel, T);
            }
        }
        else if ((!decay_input) && detach_reset)
        {
            if (padding)
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHHardReset, SurrogateFunc::ATan, false, true, true> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, v_seq_ptr, grad_x_ptr, grad_tau_ptr,
                        v_th, v_reset, tau, alpha, pai, numel, T);
            }
            else
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHHardReset, SurrogateFunc::ATan, false, true, false> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, v_seq_ptr, grad_x_ptr, grad_tau_ptr,
                        v_th, v_reset, tau, alpha, pai, numel, T);
            }
        }
        else
        {
            if (padding)
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHHardReset, SurrogateFunc::ATan, false, false, true> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, v_seq_ptr, grad_x_ptr, grad_tau_ptr,
                        v_th, v_reset, tau, alpha, pai, numel, T);
            }
            else
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHHardReset, SurrogateFunc::ATan, false, false, false> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, v_seq_ptr, grad_x_ptr, grad_tau_ptr,
                        v_th, v_reset, tau, alpha, pai, numel, T);
            }
        }
    }
    else
    {
        if (decay_input && detach_reset)
        {
            if (padding)
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHSoftReset, SurrogateFunc::ATan, true, true, true> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, v_seq_ptr, grad_x_ptr, grad_tau_ptr,
                        v_th, v_reset, tau, alpha, pai, numel, T);
            }
            else
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHSoftReset, SurrogateFunc::ATan, true, true, false> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, v_seq_ptr, grad_x_ptr, grad_tau_ptr,
                        v_th, v_reset, tau, alpha, pai, numel, T);
            }
        }
        else if (decay_input && (!detach_reset))
        {
            if (padding)
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHSoftReset, SurrogateFunc::ATan, true, false, true> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, v_seq_ptr, grad_x_ptr, grad_tau_ptr,
                        v_th, v_reset, tau, alpha, pai, numel, T);
            }
            else
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHSoftReset, SurrogateFunc::ATan, true, false, false> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, v_seq_ptr, grad_x_ptr, grad_tau_ptr,
                        v_th, v_reset, tau, alpha, pai, numel, T);
            }
        }
        else if ((!decay_input) && detach_reset)
        {
            if (padding)
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHSoftReset, SurrogateFunc::ATan, false, true, true> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, v_seq_ptr, grad_x_ptr, grad_tau_ptr,
                        v_th, v_reset, tau, alpha, pai, numel, T);
            }
            else
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHSoftReset, SurrogateFunc::ATan, false, true, false> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, v_seq_ptr, grad_x_ptr, grad_tau_ptr,
                        v_th, v_reset, tau, alpha, pai, numel, T);
            }
        }
        else
        {
            if (padding)
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHSoftReset, SurrogateFunc::ATan, false, false, true> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, v_seq_ptr, grad_x_ptr, grad_tau_ptr,
                        v_th, v_reset, tau, alpha, pai, numel, T);
            }
            else
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHSoftReset, SurrogateFunc::ATan, false, false, false> <<<blocks, threads>>>(
                        grad_spike_ptr, grad_v_ptr, h_seq_ptr, v_seq_ptr, grad_x_ptr, grad_tau_ptr,
                        v_th, v_reset, tau, alpha, pai, numel, T);
            }
        }
    }

    return {grad_x_seq, grad_tau_seq};
}

std::vector<torch::Tensor> PLIFNodeBPTTATanHALFLaunch(
        const torch::Tensor &grad_spike_seq, const torch::Tensor &h_seq, const torch::Tensor &grad_v_seq,
        const torch::Tensor &v_seq, const float v_th, const float v_reset, const float tau, const float alpha,
        const bool decay_input, ResetType resetType, const bool detach_reset, const int threads)
{
    const int64_t T = grad_spike_seq.size(0);
    const int64_t numel = grad_spike_seq.numel() / T;

    torch::Tensor grad_x_seq = torch::empty(grad_spike_seq.sizes(), grad_spike_seq.options());
    torch::Tensor grad_tau_seq = torch::zeros({1}, grad_spike_seq.options().dtype(torch::kFloat32));

    half2 v_th_half2 = __float2half2_rn(v_th);
    half2 v_reset_half2 = __float2half2_rn(v_reset);
    half2 tau_half2 = __float2half2_rn(tau);
    half2 alpha_half2 = __float2half2_rn(2.0f * alpha);
    half2 pai_half2 = __float2half2_rn(3.14159265358979323846f * 3.14159265358979323846f * alpha * alpha);

    auto *grad_spike_ptr = grad_spike_seq.data_ptr<at::Half>();
    auto *grad_v_ptr = grad_v_seq.data_ptr<at::Half>();
    auto *h_seq_ptr = h_seq.data_ptr<at::Half>();
    auto *v_seq_ptr = v_seq.data_ptr<at::Half>();

    auto *grad_x_ptr = grad_x_seq.data_ptr<at::Half>();
    auto *grad_tau = grad_tau_seq.data_ptr<float>();

    bool padding = numel % 8 != 0 ? true : false;
    int64_t blocks = (numel / 8 + threads - 1) / threads;

    if (resetType == ResetType::HardReset)
    {
        if (decay_input && detach_reset)
        {
            if (padding)
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHHardReset, SurrogateFunc::ATan, true, true, true>  <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (v_seq_ptr),
                        reinterpret_cast<half *> (grad_x_ptr), grad_tau,
                        v_th_half2, v_reset_half2, tau_half2, alpha_half2, pai_half2, numel, T);
            }
            else
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHHardReset, SurrogateFunc::ATan, true, true, false>  <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (v_seq_ptr),
                        reinterpret_cast<half *> (grad_x_ptr), grad_tau,
                        v_th_half2, v_reset_half2, tau_half2, alpha_half2, pai_half2, numel, T);
            }
        }
        else if (decay_input && (!detach_reset))
        {
            if (padding)
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHHardReset, SurrogateFunc::ATan, true, false, true>  <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (v_seq_ptr),
                        reinterpret_cast<half *> (grad_x_ptr), grad_tau,
                        v_th_half2, v_reset_half2, tau_half2, alpha_half2, pai_half2, numel, T);
            }
            else
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHHardReset, SurrogateFunc::ATan, true, false, false>  <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (v_seq_ptr),
                        reinterpret_cast<half *> (grad_x_ptr), grad_tau,
                        v_th_half2, v_reset_half2, tau_half2, alpha_half2, pai_half2, numel, T);
            }
        }
        else if ((!decay_input) && detach_reset)
        {
            if (padding)
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHHardReset, SurrogateFunc::ATan, false, true, true>  <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (v_seq_ptr),
                        reinterpret_cast<half *> (grad_x_ptr), grad_tau,
                        v_th_half2, v_reset_half2, tau_half2, alpha_half2, pai_half2, numel, T);
            }
            else
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHHardReset, SurrogateFunc::ATan, false, true, false>  <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (v_seq_ptr),
                        reinterpret_cast<half *> (grad_x_ptr), grad_tau,
                        v_th_half2, v_reset_half2, tau_half2, alpha_half2, pai_half2, numel, T);
            }
        }
        else
        {
            if (padding)
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHHardReset, SurrogateFunc::ATan, false, false, true>  <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (v_seq_ptr),
                        reinterpret_cast<half *> (grad_x_ptr), grad_tau,
                        v_th_half2, v_reset_half2, tau_half2, alpha_half2, pai_half2, numel, T);
            }
            else
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHHardReset, SurrogateFunc::ATan, false, false, false>  <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (v_seq_ptr),
                        reinterpret_cast<half *> (grad_x_ptr), grad_tau,
                        v_th_half2, v_reset_half2, tau_half2, alpha_half2, pai_half2, numel, T);
            }
        }
    }
    else
    {
        if (decay_input && detach_reset)
        {
            if (padding)
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHSoftReset, SurrogateFunc::ATan, true, true, true>  <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (v_seq_ptr),
                        reinterpret_cast<half *> (grad_x_ptr), grad_tau,
                        v_th_half2, v_reset_half2, tau_half2, alpha_half2, pai_half2, numel, T);
            }
            else
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHSoftReset, SurrogateFunc::ATan, true, true, false>  <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (v_seq_ptr),
                        reinterpret_cast<half *> (grad_x_ptr), grad_tau,
                        v_th_half2, v_reset_half2, tau_half2, alpha_half2, pai_half2, numel, T);
            }
        }
        else if (decay_input && (!detach_reset))
        {
            if (padding)
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHSoftReset, SurrogateFunc::ATan, true, false, true>  <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (v_seq_ptr),
                        reinterpret_cast<half *> (grad_x_ptr), grad_tau,
                        v_th_half2, v_reset_half2, tau_half2, alpha_half2, pai_half2, numel, T);
            }
            else
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHSoftReset, SurrogateFunc::ATan, true, false, false>  <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (v_seq_ptr),
                        reinterpret_cast<half *> (grad_x_ptr), grad_tau,
                        v_th_half2, v_reset_half2, tau_half2, alpha_half2, pai_half2, numel, T);
            }
        }
        else if ((!decay_input) && detach_reset)
        {
            if (padding)
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHSoftReset, SurrogateFunc::ATan, false, true, true>  <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (v_seq_ptr),
                        reinterpret_cast<half *> (grad_x_ptr), grad_tau,
                        v_th_half2, v_reset_half2, tau_half2, alpha_half2, pai_half2, numel, T);
            }
            else
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHSoftReset, SurrogateFunc::ATan, false, true, false>  <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (v_seq_ptr),
                        reinterpret_cast<half *> (grad_x_ptr), grad_tau,
                        v_th_half2, v_reset_half2, tau_half2, alpha_half2, pai_half2, numel, T);
            }
        }
        else
        {
            if (padding)
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHSoftReset, SurrogateFunc::ATan, false, false, true>  <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (v_seq_ptr),
                        reinterpret_cast<half *> (grad_x_ptr), grad_tau,
                        v_th_half2, v_reset_half2, tau_half2, alpha_half2, pai_half2, numel, T);
            }
            else
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHSoftReset, SurrogateFunc::ATan, false, false, false>  <<<blocks, threads>>>(
                        reinterpret_cast<half *>(grad_spike_ptr), reinterpret_cast<half *> (grad_v_ptr),
                        reinterpret_cast<half *>(h_seq_ptr), reinterpret_cast<half *> (v_seq_ptr),
                        reinterpret_cast<half *> (grad_x_ptr), grad_tau,
                        v_th_half2, v_reset_half2, tau_half2, alpha_half2, pai_half2, numel, T);
            }
        }
    }

    return {grad_x_seq, grad_tau_seq};
}


void plif_main()
{


}


