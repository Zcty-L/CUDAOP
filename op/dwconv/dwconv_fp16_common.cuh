#pragma once

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include "config.h"

#ifndef DWCONV_KERNEL_CAPACITY
#error "DWCONV_KERNEL_CAPACITY must be defined"
#endif

#ifndef DWCONV_KERNEL_NAME
#error "DWCONV_KERNEL_NAME must be defined"
#endif

#ifndef DWCONV_DB_KERNEL_NAME
#error "DWCONV_DB_KERNEL_NAME must be defined"
#endif

#ifndef DWCONV_TARGET_NAME
#error "DWCONV_TARGET_NAME must be defined"
#endif

#define CUDA_CHECK(call)                                                   \
{                                                                          \
    cudaError_t status = (call);                                           \
    if (status != cudaSuccess)                                             \
    {                                                                      \
        std::cout << "[ERROR] CUDA: " << cudaGetErrorString(status)        \
                  << std::endl;                                            \
        std::exit(1);                                                      \
    }                                                                      \
}

#define CUDNN_CHECK(call)                                                  \
{                                                                          \
    cudnnStatus_t status = (call);                                         \
    if (status != CUDNN_STATUS_SUCCESS)                                    \
    {                                                                      \
        std::cout << "[ERROR] cuDNN: " << cudnnGetErrorString(status)      \
                  << std::endl;                                            \
        std::exit(1);                                                      \
    }                                                                      \
}

template <bool UseBiasShuffle>
__global__ void
DWCONV_KERNEL_NAME(
    const __half *inputs,
    const __half2 *packed_weights,
    const __half *bias,
    __half *outputs,
    Conv2DParam param)
{
    __shared__ __half2 smem_weights[4 * DWCONV_KERNEL_CAPACITY];
    __shared__ __half2 smem_bias[4];

    int tid = threadIdx.x;
    int out_pos = blockIdx.x * blockDim.x + tid;
    int posh_ori = (out_pos / static_cast<int>(param.out_w))
        * static_cast<int>(param.Sh) - static_cast<int>(param.Ph);
    int posw_ori = (out_pos % static_cast<int>(param.out_w))
        * static_cast<int>(param.Sw) - static_cast<int>(param.Pw);

    for (int index = tid;
         index < 4 * DWCONV_KERNEL_CAPACITY;
         index += blockDim.x)
    {
        int channel_pair = index / DWCONV_KERNEL_CAPACITY;
        int kernel_pos = index % DWCONV_KERNEL_CAPACITY;
        __half2 value = __float2half2_rn(0.0f);
        if (kernel_pos < static_cast<int>(param.KhKw))
        {
            int weight_offset = blockIdx.y * 4 * param.KhKw
                + channel_pair * param.KhKw + kernel_pos;
            value = packed_weights[weight_offset];
        }
        smem_weights[index] = value;
    }
    __syncthreads();

    const __half *input_ptr = inputs
        + blockIdx.z * param.inBatchNumel
        + blockIdx.y * 8 * param.inHW;
    __half2 output_frag[4];

#pragma unroll
    for (int channel_pair = 0; channel_pair < 4; ++channel_pair)
    {
        output_frag[channel_pair] = __float2half2_rn(0.0f);
    }

    if constexpr (UseBiasShuffle)
    {
        int lane = tid & 31;
        __half2 bias_value = __float2half2_rn(0.0f);
        if (lane < 4)
        {
            int channel = blockIdx.y * 8 + lane * 2;
            bias_value = __halves2half2(bias[channel], bias[channel + 1]);
        }
#pragma unroll
        for (int channel_pair = 0; channel_pair < 4; ++channel_pair)
        {
            output_frag[channel_pair] = __hadd2(
                output_frag[channel_pair],
                __shfl_sync(0xFFFFFFFF, bias_value, channel_pair));
        }
    }
    else
    {
        if (tid < 4)
        {
            int channel = blockIdx.y * 8 + tid * 2;
            smem_bias[tid] = __halves2half2(
                bias[channel],
                bias[channel + 1]);
        }
        __syncthreads();
#pragma unroll
        for (int channel_pair = 0; channel_pair < 4; ++channel_pair)
        {
            output_frag[channel_pair] = __hadd2(
                output_frag[channel_pair],
                smem_bias[channel_pair]);
        }
    }

    for (int k = 0; k < static_cast<int>(param.KhKw); k += 4)
    {
        __half2 input_frag[4][4];

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int kernel_pos = k + i;
            int cur_h = posh_ori
                + kernel_pos / static_cast<int>(param.Kw);
            int cur_w = posw_ori
                + kernel_pos % static_cast<int>(param.Kw);
            bool valid = kernel_pos < static_cast<int>(param.KhKw)
                && cur_h >= 0
                && cur_h < static_cast<int>(param.in_h)
                && cur_w >= 0
                && cur_w < static_cast<int>(param.in_w);

#pragma unroll
            for (int channel_pair = 0; channel_pair < 4; ++channel_pair)
            {
                if (valid)
                {
                    int input_offset = cur_h * param.in_w + cur_w
                        + channel_pair * 2 * param.inHW;
                    input_frag[channel_pair][i] = __halves2half2(
                        input_ptr[input_offset],
                        input_ptr[input_offset + param.inHW]);
                }
                else
                {
                    input_frag[channel_pair][i] = __float2half2_rn(0.0f);
                }
            }
        }

#pragma unroll
        for (int channel_pair = 0; channel_pair < 4; ++channel_pair)
        {
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                __half2 weight =
                    smem_weights[channel_pair * DWCONV_KERNEL_CAPACITY + k + i];
                output_frag[channel_pair] = __hfma2(
                    weight,
                    input_frag[channel_pair][i],
                    output_frag[channel_pair]);
            }
        }
    }

    if (out_pos < static_cast<int>(param.outHW))
    {
        int out_offset = blockIdx.z * param.outBatchNumel
            + blockIdx.y * 8 * param.outHW + out_pos;
#pragma unroll
        for (int channel_pair = 0; channel_pair < 4; ++channel_pair)
        {
            outputs[out_offset + channel_pair * 2 * param.outHW] =
                output_frag[channel_pair].x;
            outputs[out_offset + (channel_pair * 2 + 1) * param.outHW] =
                output_frag[channel_pair].y;
        }
    }
}

__global__ void
DWCONV_DB_KERNEL_NAME(
    const __half *inputs,
    const __half2 *packed_weights,
    const __half *bias,
    __half *outputs,
    Conv2DParam param)
{
    __shared__ __half2 weight_buffers[2][4][4];

    int tid = threadIdx.x;
    int out_pos = blockIdx.x * blockDim.x + tid;
    int posh_ori = (out_pos / static_cast<int>(param.out_w))
        * static_cast<int>(param.Sh) - static_cast<int>(param.Ph);
    int posw_ori = (out_pos % static_cast<int>(param.out_w))
        * static_cast<int>(param.Sw) - static_cast<int>(param.Pw);
    const __half *input_ptr = inputs
        + blockIdx.z * param.inBatchNumel
        + blockIdx.y * 8 * param.inHW;

    bool is_weight_loader = tid < 16;
    int loader_channel_pair = tid / 4;
    int loader_position = tid % 4;
    if (is_weight_loader)
    {
        int weight_offset = blockIdx.y * 4 * param.KhKw
            + loader_channel_pair * param.KhKw + loader_position;
        weight_buffers[0][loader_channel_pair][loader_position] =
            loader_position < static_cast<int>(param.KhKw)
            ? packed_weights[weight_offset]
            : __float2half2_rn(0.0f);
    }
    __syncthreads();

    __half2 output_frag[4];
#pragma unroll
    for (int channel_pair = 0; channel_pair < 4; ++channel_pair)
    {
        output_frag[channel_pair] = __float2half2_rn(0.0f);
    }

    int current_buffer = 0;
    int next_buffer = 1;
    for (int k = 0; k < static_cast<int>(param.KhKw); k += 4)
    {
        int next_k = k + 4;
        if (next_k < static_cast<int>(param.KhKw) && is_weight_loader)
        {
            int load_position = next_k + loader_position;
            __half2 value = __float2half2_rn(0.0f);
            if (load_position < static_cast<int>(param.KhKw))
            {
                int weight_offset = blockIdx.y * 4 * param.KhKw
                    + loader_channel_pair * param.KhKw + load_position;
                value = packed_weights[weight_offset];
            }
            weight_buffers[next_buffer]
                [loader_channel_pair][loader_position] = value;
        }

        __half2 input_frag[4][4];
        __half2 weight_frag[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int kernel_pos = k + i;
            int cur_h = posh_ori
                + kernel_pos / static_cast<int>(param.Kw);
            int cur_w = posw_ori
                + kernel_pos % static_cast<int>(param.Kw);
            bool valid = kernel_pos < static_cast<int>(param.KhKw)
                && cur_h >= 0
                && cur_h < static_cast<int>(param.in_h)
                && cur_w >= 0
                && cur_w < static_cast<int>(param.in_w);
#pragma unroll
            for (int channel_pair = 0; channel_pair < 4; ++channel_pair)
            {
                weight_frag[channel_pair][i] =
                    weight_buffers[current_buffer][channel_pair][i];
                if (valid)
                {
                    int input_offset = cur_h * param.in_w + cur_w
                        + channel_pair * 2 * param.inHW;
                    input_frag[channel_pair][i] = __halves2half2(
                        input_ptr[input_offset],
                        input_ptr[input_offset + param.inHW]);
                }
                else
                {
                    input_frag[channel_pair][i] = __float2half2_rn(0.0f);
                }
            }
        }
        __syncthreads();

#pragma unroll
        for (int channel_pair = 0; channel_pair < 4; ++channel_pair)
        {
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                output_frag[channel_pair] = __hfma2(
                    weight_frag[channel_pair][i],
                    input_frag[channel_pair][i],
                    output_frag[channel_pair]);
            }
        }
        current_buffer ^= 1;
        next_buffer ^= 1;
    }

    int lane = tid & 31;
    __half2 bias_value = __float2half2_rn(0.0f);
    if (lane < 4)
    {
        int channel = blockIdx.y * 8 + lane * 2;
        bias_value = __halves2half2(bias[channel], bias[channel + 1]);
    }
#pragma unroll
    for (int channel_pair = 0; channel_pair < 4; ++channel_pair)
    {
        output_frag[channel_pair] = __hadd2(
            output_frag[channel_pair],
            __shfl_sync(0xFFFFFFFF, bias_value, channel_pair));
    }

    if (out_pos < static_cast<int>(param.outHW))
    {
        int out_offset = blockIdx.z * param.outBatchNumel
            + blockIdx.y * 8 * param.outHW + out_pos;
#pragma unroll
        for (int channel_pair = 0; channel_pair < 4; ++channel_pair)
        {
            outputs[out_offset + channel_pair * 2 * param.outHW] =
                output_frag[channel_pair].x;
            outputs[out_offset + (channel_pair * 2 + 1) * param.outHW] =
                output_frag[channel_pair].y;
        }
    }
}

__global__ void compare_fp16_outputs(
    const __half *actual,
    const __half *expected,
    int numel,
    float tolerance,
    unsigned int *mismatch_count)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numel)
    {
        float actual_value = __half2float(actual[index]);
        float expected_value = __half2float(expected[index]);
        if (fabsf(actual_value - expected_value) > tolerance)
        {
            atomicAdd(mismatch_count, 1U);
        }
    }
}

using KernelLaunch = void (*)(
    const __half *,
    const __half2 *,
    const __half *,
    __half *,
    const Conv2DParam &,
    int);

static void launch_custom(
    const __half *inputs,
    const __half2 *packed_weights,
    const __half *bias,
    __half *outputs,
    const Conv2DParam &param,
    int n)
{
    dim3 block(256);
    dim3 grid(
        (param.outHW + block.x - 1) / block.x,
        param.out_ch / 8,
        n);
    DWCONV_KERNEL_NAME<false><<<grid, block>>>(
        inputs,
        packed_weights,
        bias,
        outputs,
        param);
}

static void launch_biasopt(
    const __half *inputs,
    const __half2 *packed_weights,
    const __half *bias,
    __half *outputs,
    const Conv2DParam &param,
    int n)
{
    dim3 block(256);
    dim3 grid(
        (param.outHW + block.x - 1) / block.x,
        param.out_ch / 8,
        n);
    DWCONV_KERNEL_NAME<true><<<grid, block>>>(
        inputs,
        packed_weights,
        bias,
        outputs,
        param);
}

static void launch_db(
    const __half *inputs,
    const __half2 *packed_weights,
    const __half *bias,
    __half *outputs,
    const Conv2DParam &param,
    int n)
{
    dim3 block(256);
    dim3 grid(
        (param.outHW + block.x - 1) / block.x,
        param.out_ch / 8,
        n);
    DWCONV_DB_KERNEL_NAME<<<grid, block>>>(
        inputs,
        packed_weights,
        bias,
        outputs,
        param);
}

static float benchmark_kernel(
    KernelLaunch launch,
    const __half *inputs,
    const __half2 *packed_weights,
    const __half *bias,
    __half *outputs,
    const Conv2DParam &param,
    int n,
    int warmup,
    int iters)
{
    for (int i = 0; i < warmup; ++i)
    {
        launch(inputs, packed_weights, bias, outputs, param, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i)
    {
        launch(inputs, packed_weights, bias, outputs, param, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaGetLastError());
    return elapsed_ms / iters;
}

static float benchmark_cudnn(
    cudnnHandle_t handle,
    cudnnTensorDescriptor_t input_desc,
    cudnnFilterDescriptor_t filter_desc,
    cudnnConvolutionDescriptor_t conv_desc,
    cudnnTensorDescriptor_t output_desc,
    cudnnConvolutionFwdAlgo_t algo,
    const __half *inputs,
    const __half *weights,
    __half *outputs,
    void *workspace,
    size_t workspace_size,
    int warmup,
    int iters)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    for (int i = 0; i < warmup; ++i)
    {
        CUDNN_CHECK(cudnnConvolutionForward(
            handle,
            &alpha,
            input_desc,
            inputs,
            filter_desc,
            weights,
            conv_desc,
            algo,
            workspace,
            workspace_size,
            &beta,
            output_desc,
            outputs));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i)
    {
        CUDNN_CHECK(cudnnConvolutionForward(
            handle,
            &alpha,
            input_desc,
            inputs,
            filter_desc,
            weights,
            conv_desc,
            algo,
            workspace,
            workspace_size,
            &beta,
            output_desc,
            outputs));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed_ms / iters;
}

static Conv2DParam make_param(
    int c,
    int h,
    int w,
    int r,
    int stride,
    int padding)
{
    int out_h = (h - r + 2 * padding) / stride + 1;
    int out_w = (w - r + 2 * padding) / stride + 1;

    Conv2DParam param {};
    param.in_h = h;
    param.in_w = w;
    param.in_ch = c;
    param.inHW = h * w;
    param.inChKhKw = c * r * r;
    param.inBatchNumel = c * h * w;
    param.out_ch = c;
    param.out_h = out_h;
    param.out_w = out_w;
    param.outHW = out_h * out_w;
    param.outBatchNumel = c * out_h * out_w;
    param.Kh = r;
    param.Kw = r;
    param.KhKw = r * r;
    param.Sh = stride;
    param.Sw = stride;
    param.Ph = padding;
    param.Pw = padding;
    return param;
}

static void fill_input_nhwc(
    const __half *input_nchw,
    __half *input_nhwc,
    int n,
    int c,
    int h,
    int w)
{
    for (int ni = 0; ni < n; ++ni)
    {
        for (int hi = 0; hi < h; ++hi)
        {
            for (int wi = 0; wi < w; ++wi)
            {
                for (int ci = 0; ci < c; ++ci)
                {
                    int nchw_index = ni * c * h * w
                        + ci * h * w + hi * w + wi;
                    int nhwc_index = ni * h * w * c
                        + hi * w * c + wi * c + ci;
                    input_nhwc[nhwc_index] = input_nchw[nchw_index];
                }
            }
        }
    }
}

static void pack_weights(
    const __half *weights,
    __half2 *packed_weights,
    int c,
    int kernel_elements)
{
    for (int channel = 0; channel < c; channel += 2)
    {
        int channel_pair = channel / 2;
        for (int k = 0; k < kernel_elements; ++k)
        {
            packed_weights[channel_pair * kernel_elements + k] =
                __halves2half2(
                    weights[channel * kernel_elements + k],
                    weights[(channel + 1) * kernel_elements + k]);
        }
    }
}

static void write_result(
    std::ofstream &csv,
    int r,
    int n,
    int c,
    int h,
    const std::string &kernel,
    float time_ms,
    double flops,
    double arithmetic_intensity)
{
    double gflops = flops / (time_ms / 1000.0);
    csv << r << "," << n << "," << c << "," << h << ","
        << kernel << "," << time_ms << "," << gflops << ","
        << arithmetic_intensity << std::endl;

    std::cout << std::left << std::setw(6) << r << " |"
              << std::setw(4) << n << " |"
              << std::setw(4) << c << " |"
              << std::setw(4) << h << " |"
              << std::setw(12) << kernel << " |"
              << std::fixed << std::setprecision(6)
              << std::setw(12) << time_ms << " |"
              << std::setw(12) << gflops << " |"
              << std::setw(12) << arithmetic_intensity << std::endl;
}

static void verify_output(
    const std::string &implementation,
    const __half *custom_output,
    const __half *cudnn_output,
    int out_numel)
{
    unsigned int *mismatch_count_d = nullptr;
    CUDA_CHECK(cudaMalloc(&mismatch_count_d, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(mismatch_count_d, 0, sizeof(unsigned int)));
    compare_fp16_outputs<<<(out_numel + 255) / 256, 256>>>(
        custom_output,
        cudnn_output,
        out_numel,
        0.1f,
        mismatch_count_d);

    unsigned int mismatch_count = 0;
    CUDA_CHECK(cudaMemcpy(
        &mismatch_count,
        mismatch_count_d,
        sizeof(unsigned int),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(mismatch_count_d));

    std::cout << "[VERIFY] " << implementation
              << " vs cuDNN NCHW: mismatch="
              << mismatch_count << "/" << out_numel
              << " tolerance=0.1" << std::endl;
    if (mismatch_count != 0)
    {
        std::cout << "[ERROR] FP16 correctness check failed" << std::endl;
        std::exit(1);
    }
}

static void run_case(
    cudnnHandle_t cudnn,
    cudnnConvolutionFwdAlgo_t algo,
    std::ofstream &csv,
    int r,
    int n,
    int c,
    int h,
    int warmup,
    int iters)
{
    int w = h;
    int stride = 2;
    int padding = r / 2;
    Conv2DParam param = make_param(c, h, w, r, stride, padding);
    int in_numel = n * param.inBatchNumel;
    int out_numel = n * param.outBatchNumel;
    int weight_numel = c * r * r;

    double flops = static_cast<double>(out_numel) * r * r * 2.0 / 1e9;
    double total_bytes = static_cast<double>(
        in_numel + out_numel + weight_numel + c) * sizeof(__half);
    double arithmetic_intensity = flops * 1e9 / total_bytes;

    __half *input_h = nullptr;
    __half *input_nhwc_h = nullptr;
    __half *weight_h = nullptr;
    __half2 *packed_weight_h = nullptr;
    __half *bias_h = nullptr;
    CUDA_CHECK(cudaMallocHost(&input_h, in_numel * sizeof(__half)));
    CUDA_CHECK(cudaMallocHost(&input_nhwc_h, in_numel * sizeof(__half)));
    CUDA_CHECK(cudaMallocHost(&weight_h, weight_numel * sizeof(__half)));
    CUDA_CHECK(cudaMallocHost(
        &packed_weight_h,
        weight_numel / 2 * sizeof(__half2)));
    CUDA_CHECK(cudaMallocHost(&bias_h, c * sizeof(__half)));

    for (int i = 0; i < in_numel; ++i)
    {
        float value = static_cast<float>((std::rand() & 255) - 128) / 512.0f;
        input_h[i] = __float2half_rn(value);
    }
    for (int i = 0; i < weight_numel; ++i)
    {
        float value = static_cast<float>((std::rand() & 255) - 128) / 512.0f;
        weight_h[i] = __float2half_rn(value);
    }
    for (int i = 0; i < c; ++i)
    {
        float value = static_cast<float>((std::rand() & 127) - 64) / 512.0f;
        bias_h[i] = __float2half_rn(value);
    }
    fill_input_nhwc(input_h, input_nhwc_h, n, c, h, w);
    pack_weights(weight_h, packed_weight_h, c, r * r);

    __half *input_d = nullptr;
    __half *input_nhwc_d = nullptr;
    __half *weight_d = nullptr;
    __half2 *packed_weight_d = nullptr;
    __half *bias_d = nullptr;
    __half *custom_output_d = nullptr;
    __half *biasopt_output_d = nullptr;
    __half *db_output_d = nullptr;
    __half *cudnn_output_d = nullptr;
    CUDA_CHECK(cudaMalloc(&input_d, in_numel * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&input_nhwc_d, in_numel * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&weight_d, weight_numel * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(
        &packed_weight_d,
        weight_numel / 2 * sizeof(__half2)));
    CUDA_CHECK(cudaMalloc(&bias_d, c * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&custom_output_d, out_numel * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&biasopt_output_d, out_numel * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&db_output_d, out_numel * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&cudnn_output_d, out_numel * sizeof(__half)));

    CUDA_CHECK(cudaMemcpy(
        input_d,
        input_h,
        in_numel * sizeof(__half),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        input_nhwc_d,
        input_nhwc_h,
        in_numel * sizeof(__half),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        weight_d,
        weight_h,
        weight_numel * sizeof(__half),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        packed_weight_d,
        packed_weight_h,
        weight_numel / 2 * sizeof(__half2),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        bias_d,
        bias_h,
        c * sizeof(__half),
        cudaMemcpyHostToDevice));

    float custom_time = benchmark_kernel(
        launch_custom,
        input_d,
        packed_weight_d,
        bias_d,
        custom_output_d,
        param,
        n,
        warmup,
        iters);
    write_result(
        csv,
        r,
        n,
        c,
        h,
        "custom",
        custom_time,
        flops,
        arithmetic_intensity);

    float biasopt_time = benchmark_kernel(
        launch_biasopt,
        input_d,
        packed_weight_d,
        bias_d,
        biasopt_output_d,
        param,
        n,
        warmup,
        iters);
    write_result(
        csv,
        r,
        n,
        c,
        h,
        "biasopt",
        biasopt_time,
        flops,
        arithmetic_intensity);

    float db_time = benchmark_kernel(
        launch_db,
        input_d,
        packed_weight_d,
        bias_d,
        db_output_d,
        param,
        n,
        warmup,
        iters);
    write_result(
        csv,
        r,
        n,
        c,
        h,
        "db",
        db_time,
        flops,
        arithmetic_intensity);

    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnTensorDescriptor_t input_nhwc_desc;
    cudnnTensorDescriptor_t output_nhwc_desc;
    cudnnTensorDescriptor_t bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnFilterDescriptor_t filter_nhwc_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_nhwc_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_nhwc_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_nhwc_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        input_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_HALF,
        n,
        c,
        h,
        w));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        output_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_HALF,
        n,
        c,
        param.out_h,
        param.out_w));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        bias_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_HALF,
        1,
        c,
        1,
        1));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        filter_desc,
        CUDNN_DATA_HALF,
        CUDNN_TENSOR_NCHW,
        c,
        1,
        r,
        r));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        conv_desc,
        padding,
        padding,
        stride,
        stride,
        1,
        1,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_HALF));
    CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc, c));

    size_t workspace_size = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        algo,
        &workspace_size));
    void *workspace = nullptr;
    if (workspace_size > 0)
    {
        CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    }
    float nchw_time = benchmark_cudnn(
        cudnn,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        algo,
        input_d,
        weight_d,
        cudnn_output_d,
        workspace,
        workspace_size,
        warmup,
        iters);
    write_result(
        csv,
        r,
        n,
        c,
        h,
        "cudnn_nchw",
        nchw_time,
        flops,
        arithmetic_intensity);
    float alpha = 1.0f;
    CUDNN_CHECK(cudnnAddTensor(
        cudnn,
        &alpha,
        bias_desc,
        bias_d,
        &alpha,
        output_desc,
        cudnn_output_d));
    verify_output(
        "custom",
        custom_output_d,
        cudnn_output_d,
        out_numel);
    verify_output(
        "biasopt",
        biasopt_output_d,
        cudnn_output_d,
        out_numel);
    verify_output(
        "db",
        db_output_d,
        cudnn_output_d,
        out_numel);

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        input_nhwc_desc,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_HALF,
        n,
        c,
        h,
        w));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        output_nhwc_desc,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_HALF,
        n,
        c,
        param.out_h,
        param.out_w));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        filter_nhwc_desc,
        CUDNN_DATA_HALF,
        CUDNN_TENSOR_NHWC,
        c,
        1,
        r,
        r));

    size_t workspace_nhwc_size = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,
        input_nhwc_desc,
        filter_nhwc_desc,
        conv_desc,
        output_nhwc_desc,
        algo,
        &workspace_nhwc_size));
    void *workspace_nhwc = nullptr;
    if (workspace_nhwc_size > 0)
    {
        CUDA_CHECK(cudaMalloc(&workspace_nhwc, workspace_nhwc_size));
    }
    float nhwc_time = benchmark_cudnn(
        cudnn,
        input_nhwc_desc,
        filter_nhwc_desc,
        conv_desc,
        output_nhwc_desc,
        algo,
        input_nhwc_d,
        weight_d,
        cudnn_output_d,
        workspace_nhwc,
        workspace_nhwc_size,
        warmup,
        iters);
    write_result(
        csv,
        r,
        n,
        c,
        h,
        "cudnn_nhwc",
        nhwc_time,
        flops,
        arithmetic_intensity);

    if (workspace != nullptr)
    {
        CUDA_CHECK(cudaFree(workspace));
    }
    if (workspace_nhwc != nullptr)
    {
        CUDA_CHECK(cudaFree(workspace_nhwc));
    }
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_nhwc_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_nhwc_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_nhwc_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));

    CUDA_CHECK(cudaFree(input_d));
    CUDA_CHECK(cudaFree(input_nhwc_d));
    CUDA_CHECK(cudaFree(weight_d));
    CUDA_CHECK(cudaFree(packed_weight_d));
    CUDA_CHECK(cudaFree(bias_d));
    CUDA_CHECK(cudaFree(custom_output_d));
    CUDA_CHECK(cudaFree(biasopt_output_d));
    CUDA_CHECK(cudaFree(db_output_d));
    CUDA_CHECK(cudaFree(cudnn_output_d));
    CUDA_CHECK(cudaFreeHost(input_h));
    CUDA_CHECK(cudaFreeHost(input_nhwc_h));
    CUDA_CHECK(cudaFreeHost(weight_h));
    CUDA_CHECK(cudaFreeHost(packed_weight_h));
    CUDA_CHECK(cudaFreeHost(bias_h));
}

int main(int argc, char *argv[])
{
    std::string csv_path = std::string("benchmark_")
        + DWCONV_TARGET_NAME + ".csv";
    int iters = 100;
    int warmup = 10;
    bool quick = false;

    for (int argi = 1; argi < argc; ++argi)
    {
        if (std::strcmp(argv[argi], "--csv") == 0 && argi + 1 < argc)
        {
            csv_path = argv[++argi];
        }
        else if (std::strcmp(argv[argi], "--iters") == 0
                 && argi + 1 < argc)
        {
            iters = std::atoi(argv[++argi]);
        }
        else if (std::strcmp(argv[argi], "--warmup") == 0
                 && argi + 1 < argc)
        {
            warmup = std::atoi(argv[++argi]);
        }
        else if (std::strcmp(argv[argi], "--quick") == 0)
        {
            quick = true;
        }
    }
    if (iters <= 0 || warmup < 0)
    {
        std::cout << "[ERROR] invalid benchmark iteration config" << std::endl;
        return 1;
    }

    std::vector<int> ns = quick
        ? std::vector<int> {1, 16}
        : std::vector<int> {1, 2, 4, 8, 16, 32};
    std::vector<int> cs = quick
        ? std::vector<int> {32, 128}
        : std::vector<int> {32, 64, 128, 256};
    std::vector<int> hs = quick
        ? std::vector<int> {40, 80}
        : std::vector<int> {40, 80, 160};
    std::vector<int> kernel_sizes;
    if (DWCONV_KERNEL_CAPACITY == 32)
    {
        kernel_sizes = {3, 5};
    }
    else if (DWCONV_KERNEL_CAPACITY == 64)
    {
        kernel_sizes = quick
            ? std::vector<int> {3, 7}
            : std::vector<int> {3, 5, 7};
    }
    else
    {
        kernel_sizes = quick
            ? std::vector<int> {7, 11}
            : std::vector<int> {3, 5, 7, 9, 11};
    }

    std::ofstream csv(csv_path);
    if (!csv.is_open())
    {
        std::cout << "[ERROR] failed to open csv: " << csv_path << std::endl;
        return 1;
    }
    csv << "k_size,n,c,h,kernel,time_ms,gflops,arith_intensity"
        << std::endl;

    std::cout << "[CONFIG] target=" << DWCONV_TARGET_NAME
              << " capacity=" << DWCONV_KERNEL_CAPACITY
              << " csv=" << csv_path
              << " iters=" << iters
              << " warmup=" << warmup
              << " quick=" << (quick ? "true" : "false")
              << std::endl << std::endl;
    std::cout << std::left << std::setw(6) << "k_size" << " |"
              << std::setw(4) << "n" << " |"
              << std::setw(4) << "c" << " |"
              << std::setw(4) << "h" << " |"
              << std::setw(12) << "kernel" << " |"
              << std::setw(12) << "time_ms" << " |"
              << std::setw(12) << "gflops" << " |"
              << std::setw(12) << "ai" << std::endl;

    std::srand(42);
    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));
    cudnnConvolutionFwdAlgo_t algo =
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    for (int r : kernel_sizes)
    {
        for (int n : ns)
        {
            for (int c : cs)
            {
                for (int h : hs)
                {
                    run_case(cudnn, algo, csv, r, n, c, h, warmup, iters);
                }
            }
        }
    }
    CUDNN_CHECK(cudnnDestroy(cudnn));
    csv.close();

    std::cout << std::endl
              << "[SUCCESS] benchmark finished" << std::endl;
    return 0;
}
