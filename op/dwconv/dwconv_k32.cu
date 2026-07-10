#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include <cuda_runtime.h>
#include <cudnn.h>

#include "config.h"
#include "ptx_utils.cuh"

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

__global__ void
conv2d_4x32x256_groups_kernel(
    float *inputs,
    float *weights,
    float *bias,
    float *outputs,
    Conv2DParam param)
{
    __shared__ __align__(128)
    char smem[4 * 32 * 4 + 4 * 4];
    auto *smemweight = reinterpret_cast<float *>(smem);

    int out_pos = blockIdx.x * 256 + threadIdx.x;
    int posh_ori = (out_pos / param.out_w) * param.Sh - param.Ph;
    int posw_ori = (out_pos % param.out_w) * param.Sw - param.Pw;

    uint32_t weights_sts_addr =
        ptx::smem_u32addr(smemweight + threadIdx.x);
    uint32_t weights_lds_addr = ptx::smem_u32addr(smemweight);

    const char *weight_ldg_ptr = reinterpret_cast<const char *>(
        weights + blockIdx.y * 4 * param.KhKw
        + threadIdx.x / 32 * param.KhKw
        + threadIdx.x % 32);
    auto *input_ptr = inputs
        + blockIdx.z * param.inBatchNumel
        + blockIdx.y * 4 * param.inHW;

    float weight_ldg_reg;
    float weight_frag[16];
    float input_frag[4][4];
    float output_frag[4];

#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        output_frag[i] = 0.0f;
    }

    if (threadIdx.x < 128)
    {
        ptx::ldg_nc_0(
            weight_ldg_reg,
            weight_ldg_ptr,
            threadIdx.x % 32 < param.KhKw);
        ptx::sts32(weight_ldg_reg, weights_sts_addr);
    }
    __syncthreads();

    for (int k = 0; k < param.KhKw; k += 4)
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int tap = k + i;
            int curH = posh_ori + tap / param.Kw;
            int curW = posw_ori + tap % param.Kw;
            int in_offset = curH * param.in_w + curW;
            bool guard = curH >= 0 && curW >= 0
                && curW < param.in_w && curH < param.in_h;

            if (guard)
            {
                input_frag[0][i] = input_ptr[in_offset];
                input_frag[1][i] = input_ptr[in_offset + param.inHW];
                input_frag[2][i] = input_ptr[in_offset + param.inHW * 2];
                input_frag[3][i] = input_ptr[in_offset + param.inHW * 3];
            }
            else
            {
                input_frag[0][i] = 0.0f;
                input_frag[1][i] = 0.0f;
                input_frag[2][i] = 0.0f;
                input_frag[3][i] = 0.0f;
            }
        }

        ptx::lds128(
            weight_frag[0],
            weight_frag[1],
            weight_frag[2],
            weight_frag[3],
            weights_lds_addr);
        ptx::lds128(
            weight_frag[4],
            weight_frag[5],
            weight_frag[6],
            weight_frag[7],
            weights_lds_addr + 32 * sizeof(float));
        ptx::lds128(
            weight_frag[8],
            weight_frag[9],
            weight_frag[10],
            weight_frag[11],
            weights_lds_addr + 64 * sizeof(float));
        ptx::lds128(
            weight_frag[12],
            weight_frag[13],
            weight_frag[14],
            weight_frag[15],
            weights_lds_addr + 96 * sizeof(float));
        __syncthreads();

        weights_lds_addr += 4 * sizeof(float);

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            output_frag[0] += weight_frag[i] * input_frag[0][i];
            output_frag[1] += weight_frag[i + 4] * input_frag[1][i];
            output_frag[2] += weight_frag[i + 8] * input_frag[2][i];
            output_frag[3] += weight_frag[i + 12] * input_frag[3][i];
        }
    }

    auto *smembias = reinterpret_cast<float *>(smem);
    if (threadIdx.x < 4)
    {
        smembias[threadIdx.x] = bias[blockIdx.y * 4 + threadIdx.x];
    }
    __syncthreads();

    output_frag[0] += smembias[0];
    output_frag[1] += smembias[1];
    output_frag[2] += smembias[2];
    output_frag[3] += smembias[3];

    int outOffset = blockIdx.z * param.outBatchNumel
        + blockIdx.y * 4 * param.outHW
        + out_pos;
    if (out_pos < param.outHW)
    {
        outputs[outOffset + param.outHW * 0] = output_frag[0];
        outputs[outOffset + param.outHW * 1] = output_frag[1];
        outputs[outOffset + param.outHW * 2] = output_frag[2];
        outputs[outOffset + param.outHW * 3] = output_frag[3];
    }
}

__global__ void
conv2d_4x32x256_groups_kernel_biasopt(
    float *inputs,
    float *weights,
    float *bias,
    float *outputs,
    Conv2DParam param)
{
    __shared__ __align__(128)
    char smem[4 * 32 * 4 + 4 * 4];
    auto *smemweight = reinterpret_cast<float *>(smem);

    int out_pos = blockIdx.x * 256 + threadIdx.x;
    int posh_ori = (out_pos / param.out_w) * param.Sh - param.Ph;
    int posw_ori = (out_pos % param.out_w) * param.Sw - param.Pw;

    uint32_t weights_sts_addr =
        ptx::smem_u32addr(smemweight + threadIdx.x);
    uint32_t weights_lds_addr = ptx::smem_u32addr(smemweight);

    const char *weight_ldg_ptr = reinterpret_cast<const char *>(
        weights + blockIdx.y * 4 * param.KhKw
        + threadIdx.x / 32 * param.KhKw
        + threadIdx.x % 32);
    auto *input_ptr = inputs
        + blockIdx.z * param.inBatchNumel
        + blockIdx.y * 4 * param.inHW;

    float weight_ldg_reg;
    float weight_frag[16];
    float input_frag[4][4];
    float output_frag[4];

#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        output_frag[i] = 0.0f;
    }

    if (threadIdx.x < 128)
    {
        ptx::ldg_nc_0(
            weight_ldg_reg,
            weight_ldg_ptr,
            threadIdx.x % 32 < param.KhKw);
        ptx::sts32(weight_ldg_reg, weights_sts_addr);
    }
    __syncthreads();

    for (int k = 0; k < param.KhKw; k += 4)
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int tap = k + i;
            int curH = posh_ori + tap / param.Kw;
            int curW = posw_ori + tap % param.Kw;
            int in_offset = curH * param.in_w + curW;
            bool guard = curH >= 0 && curW >= 0
                && curW < param.in_w && curH < param.in_h;

            if (guard)
            {
                input_frag[0][i] = input_ptr[in_offset];
                input_frag[1][i] = input_ptr[in_offset + param.inHW];
                input_frag[2][i] = input_ptr[in_offset + param.inHW * 2];
                input_frag[3][i] = input_ptr[in_offset + param.inHW * 3];
            }
            else
            {
                input_frag[0][i] = 0.0f;
                input_frag[1][i] = 0.0f;
                input_frag[2][i] = 0.0f;
                input_frag[3][i] = 0.0f;
            }
        }

        ptx::lds128(
            weight_frag[0],
            weight_frag[1],
            weight_frag[2],
            weight_frag[3],
            weights_lds_addr);
        ptx::lds128(
            weight_frag[4],
            weight_frag[5],
            weight_frag[6],
            weight_frag[7],
            weights_lds_addr + 32 * sizeof(float));
        ptx::lds128(
            weight_frag[8],
            weight_frag[9],
            weight_frag[10],
            weight_frag[11],
            weights_lds_addr + 64 * sizeof(float));
        ptx::lds128(
            weight_frag[12],
            weight_frag[13],
            weight_frag[14],
            weight_frag[15],
            weights_lds_addr + 96 * sizeof(float));
        __syncthreads();

        weights_lds_addr += 4 * sizeof(float);

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            output_frag[0] += weight_frag[i] * input_frag[0][i];
            output_frag[1] += weight_frag[i + 4] * input_frag[1][i];
            output_frag[2] += weight_frag[i + 8] * input_frag[2][i];
            output_frag[3] += weight_frag[i + 12] * input_frag[3][i];
        }
    }

    int lane = threadIdx.x & 31;
    float bias_val = 0.0f;
    if (lane < 4)
    {
        bias_val = bias[blockIdx.y * 4 + lane];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        output_frag[i] += __shfl_sync(0xFFFFFFFF, bias_val, i);
    }

    int outOffset = blockIdx.z * param.outBatchNumel
        + blockIdx.y * 4 * param.outHW
        + out_pos;
    if (out_pos < param.outHW)
    {
        outputs[outOffset + param.outHW * 0] = output_frag[0];
        outputs[outOffset + param.outHW * 1] = output_frag[1];
        outputs[outOffset + param.outHW * 2] = output_frag[2];
        outputs[outOffset + param.outHW * 3] = output_frag[3];
    }
}

__global__ void
conv2d_4x32x256_groups_kernel_db(
    float *inputs,
    float *weights,
    float *bias,
    float *outputs,
    Conv2DParam param)
{
    constexpr int kWeightBufferElements = 4 * 32;
    __shared__ __align__(1024)
    float weight_buffers[2][kWeightBufferElements];

    uint32_t buffer_addr[2] = {
        ptx::smem_u32addr(weight_buffers[0]),
        ptx::smem_u32addr(weight_buffers[1])};

    int out_pos = blockIdx.x * 256 + threadIdx.x;
    int posh_ori = (out_pos / param.out_w) * param.Sh - param.Ph;
    int posw_ori = (out_pos % param.out_w) * param.Sw - param.Pw;

    const char *weight_group_base = reinterpret_cast<const char *>(
        weights + blockIdx.y * 4 * param.KhKw);
    auto *input_ptr = inputs
        + blockIdx.z * param.inBatchNumel
        + blockIdx.y * 4 * param.inHW;

    float weight_ldg_reg = 0.0f;
    float weight_frag[16];
    float input_frag[4][4];
    float output_frag[4];

#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        output_frag[i] = 0.0f;
    }

    bool is_weight_loader = threadIdx.x < 128;
    int weight_channel = threadIdx.x / 32;
    int weight_position = threadIdx.x % 32;
    if (is_weight_loader && weight_position < 4)
    {
        ptx::ldg_nc_0(
            weight_ldg_reg,
            weight_group_base
                + (weight_channel * param.KhKw + weight_position)
                    * sizeof(float),
            weight_position < param.KhKw);
        ptx::sts32(
            weight_ldg_reg,
            buffer_addr[0]
                + (weight_channel * 32 + weight_position) * sizeof(float));
    }
    __syncthreads();

    int current_buffer = 0;
    int next_buffer = 1;
    for (int k = 0; k < param.KhKw; k += 4)
    {
        int next_k = k + 4;
        if (next_k < param.KhKw
            && is_weight_loader && weight_position < 4)
        {
            int load_position = next_k + weight_position;
            ptx::ldg_nc_0(
                weight_ldg_reg,
                weight_group_base
                    + (weight_channel * param.KhKw + load_position)
                        * sizeof(float),
                load_position < param.KhKw);
            ptx::sts32(
                weight_ldg_reg,
                buffer_addr[next_buffer]
                    + (weight_channel * 32 + weight_position)
                        * sizeof(float));
        }

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int tap = k + i;
            int curH = posh_ori + tap / param.Kw;
            int curW = posw_ori + tap % param.Kw;
            int in_offset = curH * param.in_w + curW;
            bool guard = curH >= 0 && curW >= 0
                && curW < param.in_w && curH < param.in_h;

            if (guard)
            {
                input_frag[0][i] = input_ptr[in_offset];
                input_frag[1][i] = input_ptr[in_offset + param.inHW];
                input_frag[2][i] = input_ptr[in_offset + param.inHW * 2];
                input_frag[3][i] = input_ptr[in_offset + param.inHW * 3];
            }
            else
            {
                input_frag[0][i] = 0.0f;
                input_frag[1][i] = 0.0f;
                input_frag[2][i] = 0.0f;
                input_frag[3][i] = 0.0f;
            }
        }

        ptx::lds128(
            weight_frag[0],
            weight_frag[1],
            weight_frag[2],
            weight_frag[3],
            buffer_addr[current_buffer]);
        ptx::lds128(
            weight_frag[4],
            weight_frag[5],
            weight_frag[6],
            weight_frag[7],
            buffer_addr[current_buffer] + 32 * sizeof(float));
        ptx::lds128(
            weight_frag[8],
            weight_frag[9],
            weight_frag[10],
            weight_frag[11],
            buffer_addr[current_buffer] + 64 * sizeof(float));
        ptx::lds128(
            weight_frag[12],
            weight_frag[13],
            weight_frag[14],
            weight_frag[15],
            buffer_addr[current_buffer] + 96 * sizeof(float));
        __syncthreads();

        current_buffer ^= 1;
        next_buffer ^= 1;

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            output_frag[0] += weight_frag[i] * input_frag[0][i];
            output_frag[1] += weight_frag[i + 4] * input_frag[1][i];
            output_frag[2] += weight_frag[i + 8] * input_frag[2][i];
            output_frag[3] += weight_frag[i + 12] * input_frag[3][i];
        }
    }

    int lane = threadIdx.x & 31;
    float bias_val = 0.0f;
    if (lane < 4)
    {
        bias_val = bias[blockIdx.y * 4 + lane];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        output_frag[i] += __shfl_sync(0xFFFFFFFF, bias_val, i);
    }

    int outOffset = blockIdx.z * param.outBatchNumel
        + blockIdx.y * 4 * param.outHW
        + out_pos;
    if (out_pos < param.outHW)
    {
        outputs[outOffset + param.outHW * 0] = output_frag[0];
        outputs[outOffset + param.outHW * 1] = output_frag[1];
        outputs[outOffset + param.outHW * 2] = output_frag[2];
        outputs[outOffset + param.outHW * 3] = output_frag[3];
    }
}

static void launch_custom(
    void *inputs,
    void *weights,
    void *bias,
    void *outputs,
    Conv2DParam param,
    uint32_t n)
{
    dim3 block(256);
    dim3 grid((param.outHW + 255) / 256, param.out_ch / 4, n);
    conv2d_4x32x256_groups_kernel<<<grid, block>>>(
        static_cast<float *>(inputs),
        static_cast<float *>(weights),
        static_cast<float *>(bias),
        static_cast<float *>(outputs),
        param);
}

static void launch_biasopt(
    void *inputs,
    void *weights,
    void *bias,
    void *outputs,
    Conv2DParam param,
    uint32_t n)
{
    dim3 block(256);
    dim3 grid((param.outHW + 255) / 256, param.out_ch / 4, n);
    conv2d_4x32x256_groups_kernel_biasopt<<<grid, block>>>(
        static_cast<float *>(inputs),
        static_cast<float *>(weights),
        static_cast<float *>(bias),
        static_cast<float *>(outputs),
        param);
}

static void launch_db(
    void *inputs,
    void *weights,
    void *bias,
    void *outputs,
    Conv2DParam param,
    uint32_t n)
{
    dim3 block(256);
    dim3 grid((param.outHW + 255) / 256, param.out_ch / 4, n);
    conv2d_4x32x256_groups_kernel_db<<<grid, block>>>(
        static_cast<float *>(inputs),
        static_cast<float *>(weights),
        static_cast<float *>(bias),
        static_cast<float *>(outputs),
        param);
}

__global__ void compare_outputs_kernel(
    const float *reference,
    const float *actual,
    int numel,
    float tolerance,
    unsigned int *max_abs_error_bits,
    unsigned long long *mismatch_count)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < numel; i += stride)
    {
        float abs_error = fabsf(reference[i] - actual[i]);
        atomicMax(max_abs_error_bits, __float_as_uint(abs_error));
        if (!(abs_error <= tolerance))
        {
            atomicAdd(mismatch_count, 1ULL);
        }
    }
}

static void validate_db(
    void *inputs,
    void *weights,
    void *bias,
    void *outputs,
    Conv2DParam param,
    uint32_t n,
    int out_numel)
{
    float *reference_d = nullptr;
    unsigned int *max_abs_error_bits_d = nullptr;
    unsigned long long *mismatch_count_d = nullptr;
    CUDA_CHECK(cudaMalloc(&reference_d, out_numel * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&max_abs_error_bits_d, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&mismatch_count_d, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(max_abs_error_bits_d, 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(mismatch_count_d, 0, sizeof(unsigned long long)));

    launch_biasopt(inputs, weights, bias, reference_d, param, n);
    launch_db(inputs, weights, bias, outputs, param, n);
    int compare_blocks = (out_numel + 255) / 256;
    compare_blocks = compare_blocks < 1024 ? compare_blocks : 1024;
    compare_outputs_kernel<<<compare_blocks, 256>>>(
        reference_d,
        static_cast<float *>(outputs),
        out_numel,
        1.0e-5f,
        max_abs_error_bits_d,
        mismatch_count_d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned int max_abs_error_bits = 0;
    unsigned long long mismatch_count = 0;
    CUDA_CHECK(cudaMemcpy(
        &max_abs_error_bits,
        max_abs_error_bits_d,
        sizeof(unsigned int),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        &mismatch_count,
        mismatch_count_d,
        sizeof(unsigned long long),
        cudaMemcpyDeviceToHost));
    float max_abs_error = 0.0f;
    std::memcpy(
        &max_abs_error,
        &max_abs_error_bits,
        sizeof(max_abs_error));

    CUDA_CHECK(cudaFree(reference_d));
    CUDA_CHECK(cudaFree(max_abs_error_bits_d));
    CUDA_CHECK(cudaFree(mismatch_count_d));

    std::cout << "[VERIFY] db vs biasopt: max_abs_error="
              << std::scientific << max_abs_error
              << " mismatch=" << mismatch_count << "/" << out_numel
              << std::fixed << std::endl;
    if (mismatch_count != 0)
    {
        std::cout << "[ERROR] db correctness check failed" << std::endl;
        std::exit(1);
    }
}

using KernelLaunch = void (*)(
    void *,
    void *,
    void *,
    void *,
    Conv2DParam,
    uint32_t);

static float benchmark_kernel(
    KernelLaunch launch,
    void *inputs,
    void *weights,
    void *bias,
    void *outputs,
    Conv2DParam param,
    uint32_t n,
    int warmup,
    int iters)
{
    for (int i = 0; i < warmup; ++i)
    {
        launch(inputs, weights, bias, outputs, param, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t ev0;
    cudaEvent_t ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));
    for (int i = 0; i < iters; ++i)
    {
        launch(inputs, weights, bias, outputs, param, n);
    }
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    CUDA_CHECK(cudaGetLastError());

    return ms / iters;
}

static float benchmark_cudnn(
    cudnnHandle_t handle,
    cudnnTensorDescriptor_t input_desc,
    cudnnFilterDescriptor_t filter_desc,
    cudnnConvolutionDescriptor_t conv_desc,
    cudnnTensorDescriptor_t output_desc,
    cudnnConvolutionFwdAlgo_t algo,
    const float *inputs,
    const float *weights,
    float *outputs,
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

    cudaEvent_t ev0;
    cudaEvent_t ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));
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
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));

    return ms / iters;
}

static Conv2DParam make_param(
    int n,
    int c,
    int h,
    int w,
    int r,
    int s,
    int u,
    int v,
    int p,
    int q)
{
    int out_h = (h - r + 2 * p) / u + 1;
    int out_w = (w - s + 2 * q) / v + 1;

    Conv2DParam param {};
    param.in_h = h;
    param.in_w = w;
    param.in_ch = c;
    param.inHW = h * w;
    param.inChKhKw = c * r * s;
    param.inBatchNumel = c * h * w;
    param.out_ch = c;
    param.out_h = out_h;
    param.out_w = out_w;
    param.outHW = out_h * out_w;
    param.outBatchNumel = c * out_h * out_w;
    param.Kh = r;
    param.Kw = s;
    param.KhKw = r * s;
    param.Sh = u;
    param.Sw = v;
    param.Ph = p;
    param.Pw = q;
    (void)n;

    return param;
}

static void fill_input_nhwc(
    const float *input_nchw,
    float *input_nhwc,
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
                    input_nhwc[ni * h * w * c + hi * w * c + wi * c + ci] =
                        input_nchw[ni * c * h * w + ci * h * w + hi * w + wi];
                }
            }
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
    double arith_intensity)
{
    double gflops = flops / (time_ms / 1000.0);
    csv << r << "," << n << "," << c << "," << h << ","
        << kernel << "," << time_ms << "," << gflops << ","
        << arith_intensity << std::endl;

    std::cout << std::left << std::setw(6) << r << " |"
              << std::setw(4) << n << " |"
              << std::setw(4) << c << " |"
              << std::setw(4) << h << " |"
              << std::setw(12) << kernel << " |"
              << std::fixed << std::setprecision(6)
              << std::setw(12) << time_ms << " |"
              << std::setw(12) << gflops << " |"
              << std::setw(12) << arith_intensity << std::endl;
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
    int s = r;
    int u = 2;
    int v = 2;
    int p = r / 2;
    int q = s / 2;
    int out_h = (h - r + 2 * p) / u + 1;
    int out_w = (w - s + 2 * q) / v + 1;
    int in_numel = n * c * h * w;
    int out_numel = n * c * out_h * out_w;
    int wt_numel = c * r * s;

    double flops =
        static_cast<double>(n) * out_h * out_w * c * r * s * 2.0 / 1e9;
    double total_bytes = static_cast<double>(wt_numel) * 4.0
        + static_cast<double>(in_numel) * 4.0
        + static_cast<double>(out_numel) * 4.0;
    double arith_intensity = flops * 1e9 / total_bytes;

    float *input_h = nullptr;
    float *input_nhwc_h = nullptr;
    float *weight_h = nullptr;
    float *bias_h = nullptr;
    float *input_d = nullptr;
    float *input_nhwc_d = nullptr;
    float *weight_d = nullptr;
    float *bias_d = nullptr;
    float *output_d = nullptr;

    CUDA_CHECK(cudaMallocHost(&input_h, in_numel * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&input_nhwc_h, in_numel * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&weight_h, wt_numel * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&bias_h, c * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&input_d, in_numel * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&input_nhwc_d, in_numel * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&weight_d, wt_numel * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bias_d, c * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&output_d, out_numel * sizeof(float)));

    for (int i = 0; i < in_numel; ++i)
    {
        input_h[i] = (std::rand() & 1023) / 1024.0f;
    }
    for (int i = 0; i < wt_numel; ++i)
    {
        weight_h[i] = (std::rand() & 1023) / 1024.0f;
    }
    for (int i = 0; i < c; ++i)
    {
        bias_h[i] = (std::rand() & 1023) / 1024.0f;
    }

    fill_input_nhwc(input_h, input_nhwc_h, n, c, h, w);

    CUDA_CHECK(cudaMemcpy(
        input_d,
        input_h,
        in_numel * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        input_nhwc_d,
        input_nhwc_h,
        in_numel * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        weight_d,
        weight_h,
        wt_numel * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        bias_d,
        bias_h,
        c * sizeof(float),
        cudaMemcpyHostToDevice));

    Conv2DParam param = make_param(n, c, h, w, r, s, u, v, p, q);

    validate_db(
        input_d,
        weight_d,
        bias_d,
        output_d,
        param,
        n,
        out_numel);

    float t_custom = benchmark_kernel(
        launch_custom,
        input_d,
        weight_d,
        bias_d,
        output_d,
        param,
        n,
        warmup,
        iters);
    write_result(csv, r, n, c, h, "custom", t_custom, flops, arith_intensity);

    float t_biasopt = benchmark_kernel(
        launch_biasopt,
        input_d,
        weight_d,
        bias_d,
        output_d,
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
        t_biasopt,
        flops,
        arith_intensity);

    float t_db = benchmark_kernel(
        launch_db,
        input_d,
        weight_d,
        bias_d,
        output_d,
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
        t_db,
        flops,
        arith_intensity);

    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnTensorDescriptor_t input_nhwc_desc;
    cudnnTensorDescriptor_t output_nhwc_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnFilterDescriptor_t filter_nhwc_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_nhwc_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_nhwc_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_nhwc_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        input_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        n,
        c,
        h,
        w));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        output_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        n,
        c,
        out_h,
        out_w));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        filter_desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        c,
        1,
        r,
        s));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        conv_desc,
        p,
        q,
        u,
        v,
        1,
        1,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));
    CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc, c));

    size_t workspace_size = 0;
    void *workspace = nullptr;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        algo,
        &workspace_size));
    if (workspace_size > 0)
    {
        CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    }

    float t_nchw = benchmark_cudnn(
        cudnn,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        algo,
        input_d,
        weight_d,
        output_d,
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
        t_nchw,
        flops,
        arith_intensity);

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        input_nhwc_desc,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_FLOAT,
        n,
        c,
        h,
        w));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        output_nhwc_desc,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_FLOAT,
        n,
        c,
        out_h,
        out_w));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        filter_nhwc_desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NHWC,
        c,
        1,
        r,
        s));

    size_t workspace_nhwc_size = 0;
    void *workspace_nhwc = nullptr;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,
        input_nhwc_desc,
        filter_nhwc_desc,
        conv_desc,
        output_nhwc_desc,
        algo,
        &workspace_nhwc_size));
    if (workspace_nhwc_size > 0)
    {
        CUDA_CHECK(cudaMalloc(&workspace_nhwc, workspace_nhwc_size));
    }

    float t_nhwc = benchmark_cudnn(
        cudnn,
        input_nhwc_desc,
        filter_nhwc_desc,
        conv_desc,
        output_nhwc_desc,
        algo,
        input_nhwc_d,
        weight_d,
        output_d,
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
        t_nhwc,
        flops,
        arith_intensity);

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
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_nhwc_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));

    CUDA_CHECK(cudaFree(input_d));
    CUDA_CHECK(cudaFree(input_nhwc_d));
    CUDA_CHECK(cudaFree(weight_d));
    CUDA_CHECK(cudaFree(bias_d));
    CUDA_CHECK(cudaFree(output_d));
    CUDA_CHECK(cudaFreeHost(input_h));
    CUDA_CHECK(cudaFreeHost(input_nhwc_h));
    CUDA_CHECK(cudaFreeHost(weight_h));
    CUDA_CHECK(cudaFreeHost(bias_h));
}

int main(int argc, char *argv[])
{
    std::string csv_path = "benchmark_dwconv_k32_fp32.csv";
    int iters = 100;
    int warmup = 10;
    bool quick = false;

    for (int argi = 1; argi < argc; ++argi)
    {
        if (std::strcmp(argv[argi], "--csv") == 0 && argi + 1 < argc)
        {
            csv_path = argv[++argi];
        }
        else if (std::strcmp(argv[argi], "--iters") == 0 && argi + 1 < argc)
        {
            iters = std::atoi(argv[++argi]);
        }
        else if (std::strcmp(argv[argi], "--warmup") == 0 && argi + 1 < argc)
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

    int ns[] = {1, 2, 4, 8, 16, 32};
    int cs[] = {32, 64, 128, 256};
    int hs[] = {40, 80, 160};
    int kernel_sizes[] = {3, 5};

    int ns_quick[] = {1, 16};
    int cs_quick[] = {32, 128};
    int hs_quick[] = {40, 80};
    int kernel_sizes_quick[] = {3, 5};

    int *active_ns = quick ? ns_quick : ns;
    int *active_cs = quick ? cs_quick : cs;
    int *active_hs = quick ? hs_quick : hs;
    int *active_kernel_sizes = quick ? kernel_sizes_quick : kernel_sizes;
    int ns_count = quick ? 2 : 6;
    int cs_count = quick ? 2 : 4;
    int hs_count = quick ? 2 : 3;
    int kernel_sizes_count = 2;

    std::ofstream csv(csv_path);
    if (!csv.is_open())
    {
        std::cout << "[ERROR] failed to open csv: " << csv_path << std::endl;
        return 1;
    }
    csv << "k_size,n,c,h,kernel,time_ms,gflops,arith_intensity"
        << std::endl;

    std::cout << "[CONFIG] csv=" << csv_path
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

    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));
    cudnnConvolutionFwdAlgo_t algo =
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

    for (int r_idx = 0; r_idx < kernel_sizes_count; ++r_idx)
    {
        int r = active_kernel_sizes[r_idx];
        for (int n_idx = 0; n_idx < ns_count; ++n_idx)
        {
            int n = active_ns[n_idx];
            for (int c_idx = 0; c_idx < cs_count; ++c_idx)
            {
                int c = active_cs[c_idx];
                for (int h_idx = 0; h_idx < hs_count; ++h_idx)
                {
                    int h = active_hs[h_idx];
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
