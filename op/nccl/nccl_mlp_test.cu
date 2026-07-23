#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include "op/config.h"

namespace
{

constexpr int kWorldSize = 2;
constexpr float kAlpha = 1.0F;
constexpr float kBeta = 0.0F;

void check_cuda(
    cudaError_t status,
    const char *expression,
    const char *file,
    int line)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            std::string("CUDA error at ") + file + ":" +
            std::to_string(line) + " for " + expression + ": " +
            cudaGetErrorString(status));
    }
}

void check_cublas(
    cublasStatus_t status,
    const char *expression,
    const char *file,
    int line)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error(
            std::string("cuBLAS error at ") + file + ":" +
            std::to_string(line) + " for " + expression +
            ", status=" + std::to_string(static_cast<int>(status)));
    }
}

void check_nccl(
    ncclResult_t status,
    const char *expression,
    const char *file,
    int line)
{
    if (status != ncclSuccess)
    {
        throw std::runtime_error(
            std::string("NCCL error at ") + file + ":" +
            std::to_string(line) + " for " + expression + ": " +
            ncclGetErrorString(status));
    }
}

#define CUDA_CHECK(expression) \
    check_cuda((expression), #expression, __FILE__, __LINE__)
#define CUBLAS_CHECK(expression) \
    check_cublas((expression), #expression, __FILE__, __LINE__)
#define NCCL_CHECK(expression) \
    check_nccl((expression), #expression, __FILE__, __LINE__)

struct DeviceContext
{
    int device = 0;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cublasHandle_t cublas = nullptr;
    float *x = nullptr;
    float *a = nullptr;
    float *hidden = nullptr;
    float *b = nullptr;
    float *output = nullptr;
};

__global__ void gelu_kernel(float *values, std::size_t count)
{
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count)
    {
        constexpr float kInvSqrtTwo = 0.70710678118654752440F;
        const float value = values[index];
        values[index] =
            0.5F * value * (1.0F + erff(value * kInvSqrtTwo));
    }
}

void row_major_gemm(
    cublasHandle_t handle,
    const float *left,
    const float *right,
    float *output,
    int rows,
    int inner,
    int columns)
{
    // cuBLAS uses column-major storage. Reversing the operands computes the
    // transpose of the requested row-major result without extra transposes.
    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        columns,
        rows,
        inner,
        &kAlpha,
        right,
        columns,
        left,
        inner,
        &kBeta,
        output,
        columns));
}

void launch_gelu(
    float *values,
    std::size_t count,
    cudaStream_t stream)
{
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
    gelu_kernel<<<blocks, kThreads, 0, stream>>>(values, count);
    CUDA_CHECK(cudaGetLastError());
}

void run_distributed(
    std::array<DeviceContext, kWorldSize> &contexts,
    const std::array<ncclComm_t, kWorldSize> &communicators,
    const NcclMlpParam &config)
{
    const int rows = static_cast<int>(config.batch_size);
    const int input = static_cast<int>(config.input_size);
    const int local_hidden =
        static_cast<int>(config.hidden_size / kWorldSize);
    const int output = static_cast<int>(config.output_size);
    const std::size_t hidden_count =
        static_cast<std::size_t>(rows) * local_hidden;
    const std::size_t output_count =
        static_cast<std::size_t>(rows) * output;

    for (DeviceContext &context : contexts)
    {
        CUDA_CHECK(cudaSetDevice(context.device));
        row_major_gemm(
            context.cublas,
            context.x,
            context.a,
            context.hidden,
            rows,
            input,
            local_hidden);
        launch_gelu(context.hidden, hidden_count, context.stream);
        row_major_gemm(
            context.cublas,
            context.hidden,
            context.b,
            context.output,
            rows,
            local_hidden,
            output);
    }

    NCCL_CHECK(ncclGroupStart());
    for (int rank = 0; rank < kWorldSize; ++rank)
    {
        DeviceContext &context = contexts[rank];
        CUDA_CHECK(cudaSetDevice(context.device));
        NCCL_CHECK(ncclAllReduce(
            context.output,
            context.output,
            output_count,
            ncclFloat,
            ncclSum,
            communicators[rank],
            context.stream));
    }
    NCCL_CHECK(ncclGroupEnd());
}

void synchronize_all(
    const std::array<DeviceContext, kWorldSize> &contexts)
{
    for (const DeviceContext &context : contexts)
    {
        CUDA_CHECK(cudaSetDevice(context.device));
        CUDA_CHECK(cudaStreamSynchronize(context.stream));
    }
}

std::vector<float> build_reference(
    const std::vector<float> &host_x,
    const std::vector<float> &host_a,
    const std::vector<float> &host_b,
    const NcclMlpParam &config)
{
    CUDA_CHECK(cudaSetDevice(0));

    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;
    float *device_x = nullptr;
    float *device_a = nullptr;
    float *device_hidden = nullptr;
    float *device_b = nullptr;
    float *device_output = nullptr;

    const std::size_t x_bytes = host_x.size() * sizeof(float);
    const std::size_t a_bytes = host_a.size() * sizeof(float);
    const std::size_t hidden_bytes =
        static_cast<std::size_t>(config.batch_size) * config.hidden_size *
        sizeof(float);
    const std::size_t b_bytes = host_b.size() * sizeof(float);
    const std::size_t output_count =
        static_cast<std::size_t>(config.batch_size) * config.output_size;
    const std::size_t output_bytes = output_count * sizeof(float);

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUDA_CHECK(cudaMalloc(&device_x, x_bytes));
    CUDA_CHECK(cudaMalloc(&device_a, a_bytes));
    CUDA_CHECK(cudaMalloc(&device_hidden, hidden_bytes));
    CUDA_CHECK(cudaMalloc(&device_b, b_bytes));
    CUDA_CHECK(cudaMalloc(&device_output, output_bytes));

    CUDA_CHECK(cudaMemcpyAsync(
        device_x,
        host_x.data(),
        x_bytes,
        cudaMemcpyHostToDevice,
        stream));
    CUDA_CHECK(cudaMemcpyAsync(
        device_a,
        host_a.data(),
        a_bytes,
        cudaMemcpyHostToDevice,
        stream));
    CUDA_CHECK(cudaMemcpyAsync(
        device_b,
        host_b.data(),
        b_bytes,
        cudaMemcpyHostToDevice,
        stream));

    row_major_gemm(
        handle,
        device_x,
        device_a,
        device_hidden,
        static_cast<int>(config.batch_size),
        static_cast<int>(config.input_size),
        static_cast<int>(config.hidden_size));
    launch_gelu(
        device_hidden,
        static_cast<std::size_t>(config.batch_size) * config.hidden_size,
        stream);
    row_major_gemm(
        handle,
        device_hidden,
        device_b,
        device_output,
        static_cast<int>(config.batch_size),
        static_cast<int>(config.hidden_size),
        static_cast<int>(config.output_size));

    std::vector<float> reference(output_count);
    CUDA_CHECK(cudaMemcpyAsync(
        reference.data(),
        device_output,
        output_bytes,
        cudaMemcpyDeviceToHost,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(device_output));
    CUDA_CHECK(cudaFree(device_b));
    CUDA_CHECK(cudaFree(device_hidden));
    CUDA_CHECK(cudaFree(device_a));
    CUDA_CHECK(cudaFree(device_x));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return reference;
}

void initialize_inputs(
    std::vector<float> &x,
    std::vector<float> &a,
    std::vector<float> &b)
{
    std::mt19937 generator(20260720);
    std::uniform_real_distribution<float> distribution(-0.02F, 0.02F);
    auto fill = [&](std::vector<float> &values)
    {
        std::generate(values.begin(), values.end(), [&]()
        {
            return distribution(generator);
        });
    };
    fill(x);
    fill(a);
    fill(b);
}

std::vector<float> shard_a(
    const std::vector<float> &a,
    const NcclMlpParam &config,
    int rank)
{
    const std::size_t local_hidden = config.hidden_size / kWorldSize;
    std::vector<float> shard(
        static_cast<std::size_t>(config.input_size) * local_hidden);
    for (std::size_t row = 0; row < config.input_size; ++row)
    {
        const std::size_t source_offset =
            row * config.hidden_size + rank * local_hidden;
        const std::size_t destination_offset = row * local_hidden;
        std::copy_n(
            a.begin() + source_offset,
            local_hidden,
            shard.begin() + destination_offset);
    }
    return shard;
}

std::vector<float> shard_b(
    const std::vector<float> &b,
    const NcclMlpParam &config,
    int rank)
{
    const std::size_t local_hidden = config.hidden_size / kWorldSize;
    const std::size_t offset =
        static_cast<std::size_t>(rank) * local_hidden * config.output_size;
    const std::size_t count = local_hidden * config.output_size;
    return std::vector<float>(b.begin() + offset, b.begin() + offset + count);
}

void initialize_devices(
    std::array<DeviceContext, kWorldSize> &contexts,
    const std::vector<float> &host_x,
    const std::vector<float> &host_a,
    const std::vector<float> &host_b,
    const NcclMlpParam &config)
{
    const std::size_t local_hidden = config.hidden_size / kWorldSize;
    const std::size_t x_bytes = host_x.size() * sizeof(float);
    const std::size_t a_bytes =
        static_cast<std::size_t>(config.input_size) * local_hidden *
        sizeof(float);
    const std::size_t hidden_bytes =
        static_cast<std::size_t>(config.batch_size) * local_hidden *
        sizeof(float);
    const std::size_t b_bytes =
        local_hidden * config.output_size * sizeof(float);
    const std::size_t output_bytes =
        static_cast<std::size_t>(config.batch_size) * config.output_size *
        sizeof(float);

    for (int rank = 0; rank < kWorldSize; ++rank)
    {
        DeviceContext &context = contexts[rank];
        context.device = rank;
        const std::vector<float> local_a = shard_a(host_a, config, rank);
        const std::vector<float> local_b = shard_b(host_b, config, rank);

        CUDA_CHECK(cudaSetDevice(context.device));
        CUDA_CHECK(cudaStreamCreateWithFlags(
            &context.stream,
            cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&context.start));
        CUDA_CHECK(cudaEventCreate(&context.stop));
        CUBLAS_CHECK(cublasCreate(&context.cublas));
        CUBLAS_CHECK(cublasSetStream(context.cublas, context.stream));
        CUDA_CHECK(cudaMalloc(&context.x, x_bytes));
        CUDA_CHECK(cudaMalloc(&context.a, a_bytes));
        CUDA_CHECK(cudaMalloc(&context.hidden, hidden_bytes));
        CUDA_CHECK(cudaMalloc(&context.b, b_bytes));
        CUDA_CHECK(cudaMalloc(&context.output, output_bytes));
        CUDA_CHECK(cudaMemcpyAsync(
            context.x,
            host_x.data(),
            x_bytes,
            cudaMemcpyHostToDevice,
            context.stream));
        CUDA_CHECK(cudaMemcpyAsync(
            context.a,
            local_a.data(),
            a_bytes,
            cudaMemcpyHostToDevice,
            context.stream));
        CUDA_CHECK(cudaMemcpyAsync(
            context.b,
            local_b.data(),
            b_bytes,
            cudaMemcpyHostToDevice,
            context.stream));
        CUDA_CHECK(cudaStreamSynchronize(context.stream));
    }
}

void destroy_devices(
    std::array<DeviceContext, kWorldSize> &contexts)
{
    for (DeviceContext &context : contexts)
    {
        CUDA_CHECK(cudaSetDevice(context.device));
        if (context.output != nullptr)
        {
            CUDA_CHECK(cudaFree(context.output));
            CUDA_CHECK(cudaFree(context.b));
            CUDA_CHECK(cudaFree(context.hidden));
            CUDA_CHECK(cudaFree(context.a));
            CUDA_CHECK(cudaFree(context.x));
            CUBLAS_CHECK(cublasDestroy(context.cublas));
            CUDA_CHECK(cudaEventDestroy(context.stop));
            CUDA_CHECK(cudaEventDestroy(context.start));
            CUDA_CHECK(cudaStreamDestroy(context.stream));
        }
    }
}

struct ErrorMetrics
{
    float max_absolute = 0.0F;
    float max_relative = 0.0F;
};

ErrorMetrics compare_outputs(
    const std::vector<float> &actual,
    const std::vector<float> &expected)
{
    ErrorMetrics metrics;
    for (std::size_t index = 0; index < actual.size(); ++index)
    {
        const float absolute = std::abs(actual[index] - expected[index]);
        const float denominator = std::max(std::abs(expected[index]), 1.0e-6F);
        metrics.max_absolute = std::max(metrics.max_absolute, absolute);
        metrics.max_relative =
            std::max(metrics.max_relative, absolute / denominator);
    }
    return metrics;
}

void write_field_label(const std::string &label)
{
    constexpr std::size_t kTargetWidth = 30;
    std::size_t display_width = 0;
    for (const unsigned char character : label)
    {
        if (character < 0x80)
        {
            ++display_width;
        }
        else if ((character & 0xC0) != 0x80)
        {
            display_width += 2;
        }
    }

    std::cout << label;
    if (display_width < kTargetWidth)
    {
        std::cout << std::string(kTargetWidth - display_width, ' ');
    }
}

float benchmark(
    std::array<DeviceContext, kWorldSize> &contexts,
    const std::array<ncclComm_t, kWorldSize> &communicators,
    const NcclMlpParam &config)
{
    for (std::uint32_t iteration = 0;
         iteration < config.warmup_iterations;
         ++iteration)
    {
        run_distributed(contexts, communicators, config);
    }
    synchronize_all(contexts);

    for (DeviceContext &context : contexts)
    {
        CUDA_CHECK(cudaSetDevice(context.device));
        CUDA_CHECK(cudaEventRecord(context.start, context.stream));
    }
    for (std::uint32_t iteration = 0;
         iteration < config.benchmark_iterations;
         ++iteration)
    {
        run_distributed(contexts, communicators, config);
    }
    for (DeviceContext &context : contexts)
    {
        CUDA_CHECK(cudaSetDevice(context.device));
        CUDA_CHECK(cudaEventRecord(context.stop, context.stream));
    }

    float maximum_ms = 0.0F;
    for (DeviceContext &context : contexts)
    {
        CUDA_CHECK(cudaSetDevice(context.device));
        CUDA_CHECK(cudaEventSynchronize(context.stop));
        float elapsed_ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(
            &elapsed_ms,
            context.start,
            context.stop));
        maximum_ms = std::max(maximum_ms, elapsed_ms);
    }
    return maximum_ms / config.benchmark_iterations;
}

void write_configuration(const NcclMlpParam &config)
{
    std::cout << "[配置]" << std::endl;
    write_field_label("GPU 数量");
    std::cout << kWorldSize << std::endl;
    write_field_label("X");
    std::cout << config.batch_size << " x " << config.input_size
              << std::endl;
    write_field_label("A");
    std::cout << config.input_size << " x " << config.hidden_size
              << "（按列切分）" << std::endl;
    write_field_label("B");
    std::cout << config.hidden_size << " x " << config.output_size
              << "（按行切分）" << std::endl;
    write_field_label("Y");
    std::cout << config.batch_size << " x " << config.output_size
              << std::endl;
    write_field_label("数据类型");
    std::cout << "FP32" << std::endl;
    write_field_label("GEMM");
    std::cout << "cuBLAS SGEMM" << std::endl;
    write_field_label("通信");
    std::cout << "NCCL AllReduce(SUM)" << std::endl;
}

int run_test()
{
    const NcclMlpParam config =
    {
        512,
        1024,
        2048,
        1024,
        10,
        50,
    };
    if (config.hidden_size % kWorldSize != 0)
    {
        throw std::runtime_error("hidden_size 必须能被 GPU 数量整除");
    }

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count < kWorldSize)
    {
        throw std::runtime_error(
            "测试需要至少 2 张可见 CUDA GPU，当前仅有 " +
            std::to_string(device_count) + " 张");
    }

    std::cout << std::endl;
    write_configuration(config);

    const std::size_t x_count =
        static_cast<std::size_t>(config.batch_size) * config.input_size;
    const std::size_t a_count =
        static_cast<std::size_t>(config.input_size) * config.hidden_size;
    const std::size_t b_count =
        static_cast<std::size_t>(config.hidden_size) * config.output_size;
    const std::size_t output_count =
        static_cast<std::size_t>(config.batch_size) * config.output_size;
    std::vector<float> host_x(x_count);
    std::vector<float> host_a(a_count);
    std::vector<float> host_b(b_count);
    initialize_inputs(host_x, host_a, host_b);

    std::cout << std::endl;
    std::cout << "[阶段 1/4] 初始化两卡资源与 NCCL communicator"
              << std::endl;
    std::array<DeviceContext, kWorldSize> contexts;
    initialize_devices(contexts, host_x, host_a, host_b, config);
    std::array<ncclComm_t, kWorldSize> communicators =
    {
        nullptr,
        nullptr,
    };
    const int devices[kWorldSize] =
    {
        0,
        1,
    };
    NCCL_CHECK(ncclCommInitAll(
        communicators.data(),
        kWorldSize,
        devices));

    int nccl_version = 0;
    NCCL_CHECK(ncclGetVersion(&nccl_version));
    write_field_label("NCCL 版本");
    std::cout << nccl_version << std::endl;

    std::cout << std::endl;
    std::cout << "[阶段 2/4] 运行 Column Parallel + Row Parallel"
              << std::endl;
    run_distributed(contexts, communicators, config);
    synchronize_all(contexts);

    std::array<std::vector<float>, kWorldSize> rank_outputs;
    for (int rank = 0; rank < kWorldSize; ++rank)
    {
        rank_outputs[rank].resize(output_count);
        CUDA_CHECK(cudaSetDevice(rank));
        CUDA_CHECK(cudaMemcpy(
            rank_outputs[rank].data(),
            contexts[rank].output,
            output_count * sizeof(float),
            cudaMemcpyDeviceToHost));
    }

    std::cout << std::endl;
    std::cout << "[阶段 3/4] 与单卡完整 cuBLAS 结果进行精度验证"
              << std::endl;
    const std::vector<float> reference =
        build_reference(host_x, host_a, host_b, config);
    const ErrorMetrics rank_zero_error =
        compare_outputs(rank_outputs[0], reference);
    const ErrorMetrics rank_one_error =
        compare_outputs(rank_outputs[1], reference);
    const ErrorMetrics replica_error =
        compare_outputs(rank_outputs[0], rank_outputs[1]);
    std::cout << std::scientific << std::setprecision(6);
    write_field_label("Rank 0 最大绝对误差");
    std::cout << rank_zero_error.max_absolute << std::endl;
    write_field_label("Rank 0 最大相对误差");
    std::cout << rank_zero_error.max_relative << std::endl;
    write_field_label("Rank 1 最大绝对误差");
    std::cout << rank_one_error.max_absolute << std::endl;
    write_field_label("Rank 1 最大相对误差");
    std::cout << rank_one_error.max_relative << std::endl;
    write_field_label("两卡副本最大差异");
    std::cout << replica_error.max_absolute << std::endl;

    constexpr float kAbsoluteTolerance = 2.0e-4F;
    const bool correct =
        rank_zero_error.max_absolute <= kAbsoluteTolerance &&
        rank_one_error.max_absolute <= kAbsoluteTolerance &&
        replica_error.max_absolute == 0.0F;

    std::cout << std::endl;
    std::cout << "[阶段 4/4] 端到端性能测试" << std::endl;
    const float average_ms = benchmark(contexts, communicators, config);
    std::cout << std::fixed << std::setprecision(3);
    write_field_label("预热次数");
    std::cout << config.warmup_iterations << std::endl;
    write_field_label("测试次数");
    std::cout << config.benchmark_iterations << std::endl;
    write_field_label("平均端到端耗时");
    std::cout << average_ms << " ms" << std::endl;

    for (ncclComm_t communicator : communicators)
    {
        NCCL_CHECK(ncclCommDestroy(communicator));
    }
    destroy_devices(contexts);

    std::cout << std::endl;
    if (!correct)
    {
        throw std::runtime_error("精度验证失败");
    }
    std::cout << "[SUCCESS] 两卡 NCCL 张量并行测试通过" << std::endl;
    return 0;
}

}  // namespace

int main()
{
    try
    {
        return run_test();
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAILED] " << error.what() << std::endl;
        return 1;
    }
}
