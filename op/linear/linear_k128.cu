#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "config.h"
#include "ptx_utils.cuh"

namespace
{

constexpr int OUTPUT_TILE = 128;
constexpr int K_TILE = 8;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int FRAGMENT_SIZE = 8;
constexpr int WARP_SIZE = 32;
constexpr int WARMUP_ITERATIONS = 10;
constexpr int BENCHMARK_ITERATIONS = 50;
constexpr double FP32_TOLERANCE = 1.0e-3;
constexpr double FP16_TOLERANCE = 6.25e-2;

void check_cuda(cudaError_t result, const char *expression, const char *file, int line)
{
    if (result == cudaSuccess)
    {
        return;
    }

    std::cout << "CUDA error: " << cudaGetErrorString(result) << '\n'
              << "  expression : " << expression << '\n'
              << "  location   : " << file << ':' << line << std::endl;
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression)                                               \
    check_cuda((expression), #expression, __FILE__, __LINE__)

template <typename T>
class DeviceBuffer
{
public:
    explicit DeviceBuffer(std::size_t count)
        : count_(count)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&pointer_), count_ * sizeof(T)));
    }

    ~DeviceBuffer()
    {
        if (pointer_ != nullptr)
        {
            cudaFree(pointer_);
        }
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    T *get()
    {
        return pointer_;
    }

    const T *get() const
    {
        return pointer_;
    }

    std::size_t bytes() const
    {
        return count_ * sizeof(T);
    }

private:
    T *pointer_ = nullptr;
    std::size_t count_ = 0;
};

struct TestShape
{
    int time_steps;
    int m;
    int k;
    int n;
};

struct ValidationResult
{
    std::size_t mismatches = 0;
    double max_absolute_error = 0.0;

    bool passed() const
    {
        return mismatches == 0;
    }
};

struct BenchmarkResult
{
    std::string name;
    double latency_ms;
    double dense_equivalent_tflops;
};

__host__ __device__ constexpr int divide_up(int value, int divisor)
{
    return (value + divisor - 1) / divisor;
}

__host__ __device__ constexpr int round_up(int value, int multiple)
{
    return divide_up(value, multiple) * multiple;
}

LinearParam make_linear_param(const TestShape &shape, bool packed_input)
{
    LinearParam parameter{};
    parameter.in_ch = static_cast<std::uint32_t>(shape.m);
    parameter.in_dim = static_cast<std::uint32_t>(shape.k);
    parameter.out_dim = static_cast<std::uint32_t>(shape.n);
    parameter.out_dim_padded =
        static_cast<std::uint32_t>(round_up(shape.n, OUTPUT_TILE));
    parameter.inBatchNumel =
        packed_input ? 0U : static_cast<std::uint32_t>(shape.m * shape.k);
    parameter.outBatchNumel = static_cast<std::uint32_t>(shape.m * shape.n);

    return parameter;
}

dim3 make_grid(const TestShape &shape)
{
    return dim3(divide_up(shape.n, OUTPUT_TILE),
                divide_up(shape.m, OUTPUT_TILE),
                shape.time_steps);
}

template <typename T>
void copy_to_device(DeviceBuffer<T> &destination, const std::vector<T> &source)
{
    CUDA_CHECK(cudaMemcpy(
        destination.get(), source.data(), destination.bytes(), cudaMemcpyHostToDevice));
}

template <typename T>
void copy_to_host(std::vector<T> &destination, const DeviceBuffer<T> &source)
{
    CUDA_CHECK(cudaMemcpy(
        destination.data(), source.get(), source.bytes(), cudaMemcpyDeviceToHost));
}

template <typename T>
void pad_weights(const std::vector<T> &source,
                 std::vector<T> &destination,
                 int k,
                 int n,
                 int k_padded)
{
    for (int output = 0; output < n; ++output)
    {
        for (int input = 0; input < k; ++input)
        {
            destination[static_cast<std::size_t>(output) * k_padded + input] =
                source[static_cast<std::size_t>(output) * k + input];
        }
    }
}

std::vector<float> make_random_fp32(std::size_t count, std::uint32_t seed)
{
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-0.5F, 0.5F);
    std::vector<float> values(count);
    std::generate(values.begin(), values.end(), [&generator, &distribution]()
    {
        return distribution(generator);
    });

    return values;
}

std::vector<half> make_random_fp16(std::size_t count, std::uint32_t seed)
{
    const std::vector<float> fp32_values = make_random_fp32(count, seed);
    std::vector<half> values(count);
    std::transform(fp32_values.begin(), fp32_values.end(), values.begin(), [](float value)
    {
        return __float2half(value);
    });

    return values;
}

std::vector<std::uint8_t> make_random_spikes(
    std::size_t count, int time_steps, std::uint32_t seed)
{
    std::mt19937 generator(seed);
    std::bernoulli_distribution distribution(0.5);
    std::vector<std::uint8_t> values(count, 0);

    for (std::uint8_t &value : values)
    {
        for (int time = 0; time < time_steps; ++time)
        {
            if (distribution(generator))
            {
                value |= static_cast<std::uint8_t>(1U << time);
            }
        }
    }

    return values;
}

void update_validation(
    ValidationResult &result, double actual, double expected, double tolerance)
{
    const double absolute_error = std::abs(actual - expected);
    result.max_absolute_error = std::max(result.max_absolute_error, absolute_error);

    if (absolute_error > tolerance)
    {
        ++result.mismatches;
    }
}

void print_validation_result(const std::string &name,
                             const TestShape &shape,
                             const ValidationResult &result)
{
    std::cout << "  " << std::left << std::setw(10) << name
              << " T=" << std::right << std::setw(2) << shape.time_steps
              << " M=" << std::setw(4) << shape.m
              << " K=" << std::setw(4) << shape.k
              << " N=" << std::setw(4) << shape.n
              << "  mismatches=" << std::setw(7) << result.mismatches
              << "  max_abs_error=" << std::scientific
              << std::setprecision(3) << result.max_absolute_error
              << "  " << (result.passed() ? "PASS" : "FAIL")
              << std::endl;
}

} // namespace

// Computes output[t, m, n] = bias[n] + input[t, m, :] * weight[n, :].
__global__ void linear_128x128x8_kernel(
    const float *__restrict__ inputs,
    const float *__restrict__ weights,
    const float *__restrict__ bias,
    float *__restrict__ outputs,
    LinearParam parameter)
{
    __shared__ __align__(128) char shared_storage[16 * 1024];
    float *input_shared = reinterpret_cast<float *>(shared_storage);
    float *weight_shared = reinterpret_cast<float *>(shared_storage + 8 * 1024);
    float *bias_shared = reinterpret_cast<float *>(shared_storage);

    const int thread_id = threadIdx.x;
    const int warp_id = thread_id / WARP_SIZE;
    const int lane_id = thread_id % WARP_SIZE;

    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;
    const int lane_n = (lane_id / 2) % 8;
    const int lane_m = (lane_id / 16) * 2 + lane_id % 2;
    const int thread_m_base = warp_m * 32 + lane_m * FRAGMENT_SIZE;
    const int thread_n_base = warp_n * 64 + lane_n * FRAGMENT_SIZE;

    if (thread_id < OUTPUT_TILE)
    {
        const int global_n = blockIdx.x * OUTPUT_TILE + thread_id;
        bias_shared[thread_id] = global_n < parameter.out_dim ? bias[global_n] : 0.0F;
    }
    __syncthreads();

    float output_fragment[FRAGMENT_SIZE][FRAGMENT_SIZE];
#pragma unroll
    for (int i = 0; i < FRAGMENT_SIZE; ++i)
    {
        const int global_m = blockIdx.y * OUTPUT_TILE + thread_m_base + i;
        const float row_mask = global_m < parameter.in_ch ? 1.0F : 0.0F;
#pragma unroll
        for (int j = 0; j < FRAGMENT_SIZE; ++j)
        {
            output_fragment[i][j] = bias_shared[thread_n_base + j] * row_mask;
        }
    }
    __syncthreads();

    const int load_row = thread_id / 2;
    const int load_k = thread_id % 2 * 4;

    const int global_m = blockIdx.y * OUTPUT_TILE + load_row;
    const int safe_global_m = global_m < parameter.in_ch ? global_m : 0;
    const float *input_row =
        inputs + static_cast<std::size_t>(blockIdx.z) * parameter.inBatchNumel +
        static_cast<std::size_t>(safe_global_m) * parameter.in_dim;

    const int global_n = blockIdx.x * OUTPUT_TILE + load_row;
    const int k_padded = round_up(static_cast<int>(parameter.in_dim), K_TILE);
    const float *weight_row = weights + static_cast<std::size_t>(global_n) * k_padded;

    const int k_iterations = divide_up(static_cast<int>(parameter.in_dim), K_TILE);

    auto load_chunk = [&](int k_tile, int buffer)
    {
        const int current_k = k_tile * K_TILE + load_k;  // 8
        const int remaining = static_cast<int>(parameter.in_dim) - current_k;
        const int valid_elements = remaining > 4 ? 4 : (remaining > 0 ? remaining : 0);
        const int input_bytes = global_m < parameter.in_ch ? valid_elements * sizeof(float) : 0;
        const int safe_k = current_k < parameter.in_dim ? current_k : 0;

        const std::uint32_t input_destination = ptx::smem_u32addr(
            &input_shared[buffer * OUTPUT_TILE * K_TILE + load_row * K_TILE + load_k]);
        ptx::cp_async_cg(input_destination, input_row + safe_k, input_bytes);

        const int weight_bytes = global_n < parameter.out_dim_padded ? 16 : 0;
        const std::uint32_t weight_destination = ptx::smem_u32addr(
            &weight_shared[buffer * OUTPUT_TILE * K_TILE + load_row * K_TILE + load_k]);
        ptx::cp_async_ca(weight_destination, weight_row + current_k, weight_bytes);
    };

    if (k_iterations > 0)
    {
        load_chunk(0, 0);
        ptx::cp_async_commit_group();
    }

    for (int k_tile = 0; k_tile < k_iterations; ++k_tile)
    {
        if (k_tile + 1 < k_iterations)
        {
            load_chunk(k_tile + 1, (k_tile + 1) % 2);
            ptx::cp_async_commit_group();
            ptx::cp_async_wait_group<1>();
        }
        else
        {
            ptx::cp_async_wait_group<0>();
        }
        __syncthreads();

        float *current_input = &input_shared[k_tile % 2 * OUTPUT_TILE * K_TILE];
        float *current_weight = &weight_shared[k_tile % 2 * OUTPUT_TILE * K_TILE];

#pragma unroll
        for (int k = 0; k < K_TILE; ++k)
        {
            float input_fragment[FRAGMENT_SIZE];
            float weight_fragment[FRAGMENT_SIZE];
#pragma unroll
            for (int i = 0; i < FRAGMENT_SIZE; ++i)
            {
                input_fragment[i] = current_input[(thread_m_base + i) * K_TILE + k];
            }
#pragma unroll
            for (int j = 0; j < FRAGMENT_SIZE; ++j)
            {
                weight_fragment[j] = current_weight[(thread_n_base + j) * K_TILE + k];
            }
#pragma unroll
            for (int i = 0; i < FRAGMENT_SIZE; ++i)
            {
#pragma unroll
                for (int j = 0; j < FRAGMENT_SIZE; ++j)
                {
                    output_fragment[i][j] += input_fragment[i] * weight_fragment[j];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < FRAGMENT_SIZE; ++i)
    {
        const int output_m = blockIdx.y * OUTPUT_TILE + thread_m_base + i;
        if (output_m < parameter.in_ch)
        {
#pragma unroll
            for (int j = 0; j < FRAGMENT_SIZE; ++j)
            {
                const int output_n = blockIdx.x * OUTPUT_TILE + thread_n_base + j;
                if (output_n < parameter.out_dim)
                {
                    outputs[static_cast<std::size_t>(blockIdx.z) * parameter.outBatchNumel +
                            static_cast<std::size_t>(output_m) * parameter.out_dim + output_n] =
                        output_fragment[i][j];
                }
            }
        }
    }
}

// Computes all time steps from a uint8 bit-packed spike input.
__global__ void linear_128x128x8_S_kernel(
    const std::uint8_t *__restrict__ inputs,
    const float *__restrict__ weights,
    const float *__restrict__ bias,
    float *__restrict__ outputs,
    LinearParam parameter)
{
    __shared__ __align__(128) char shared_storage[16 * 1024];
    std::uint8_t *input_shared = reinterpret_cast<std::uint8_t *>(shared_storage);
    float *weight_shared = reinterpret_cast<float *>(shared_storage + 4 * 1024);
    float *bias_shared = reinterpret_cast<float *>(shared_storage);

    const int thread_id = threadIdx.x;
    const int warp_id = thread_id / WARP_SIZE;
    const int lane_id = thread_id % WARP_SIZE;

    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;
    const int lane_n = (lane_id / 2) % 8;
    const int lane_m = (lane_id / 16) * 2 + lane_id % 2;
    const int thread_m_base = warp_m * 32 + lane_m * FRAGMENT_SIZE;
    const int thread_n_base = warp_n * 64 + lane_n * FRAGMENT_SIZE;

    if (thread_id < OUTPUT_TILE)
    {
        const int global_n = blockIdx.x * OUTPUT_TILE + thread_id;
        bias_shared[thread_id] = global_n < parameter.out_dim ? bias[global_n] : 0.0F;
    }
    __syncthreads();

    float output_fragment[FRAGMENT_SIZE][FRAGMENT_SIZE];
#pragma unroll
    for (int i = 0; i < FRAGMENT_SIZE; ++i)
    {
        const int global_m = blockIdx.y * OUTPUT_TILE + thread_m_base + i;
        const float row_mask = global_m < parameter.in_ch ? 1.0F : 0.0F;
#pragma unroll
        for (int j = 0; j < FRAGMENT_SIZE; ++j)
        {
            output_fragment[i][j] = bias_shared[thread_n_base + j] * row_mask;
        }
    }
    __syncthreads();

    const int load_row = thread_id % OUTPUT_TILE;

    const int global_m = blockIdx.y * OUTPUT_TILE + load_row;
    const int safe_global_m = global_m < parameter.in_ch ? global_m : 0;
    const std::uint8_t *input_row =
        inputs + static_cast<std::size_t>(safe_global_m) * parameter.in_dim;

    const int weight_row_index = thread_id / 2;
    const int weight_k_offset = thread_id % 2 * 4;
    const int global_n = blockIdx.x * OUTPUT_TILE + weight_row_index;
    const int k_padded = round_up(static_cast<int>(parameter.in_dim), K_TILE);
    const float *weight_row = weights + static_cast<std::size_t>(global_n) * k_padded;

    const int k_iterations = divide_up(static_cast<int>(parameter.in_dim), K_TILE);

    auto load_chunk = [&](int k_tile, int buffer)
    {
        const int current_k = k_tile * K_TILE;

        if (thread_id < OUTPUT_TILE)
        {
            const std::uint32_t input_destination = ptx::smem_u32addr(
                &input_shared[buffer * OUTPUT_TILE * 16 + load_row * 16]);
            std::uint32_t input_registers[2] = {0U, 0U};

            if (global_m < parameter.in_ch)
            {
                const int remaining = static_cast<int>(parameter.in_dim) - current_k;
                if (remaining >= K_TILE)
                {
                    ptx::ldg32_nc_0(input_registers[0], input_row + current_k, true);
                    ptx::ldg32_nc_0(input_registers[1], input_row + current_k + 4, true);
                }
                else
                {
                    for (int i = 0; i < remaining; ++i)
                    {
                        const std::uint32_t value = input_row[current_k + i];
                        if (i < 4)
                        {
                            input_registers[0] |= value << (i * 8);
                        }
                        else
                        {
                            input_registers[1] |= value << ((i - 4) * 8);
                        }
                    }
                }
            }

            ptx::sts64(input_registers[0], input_registers[1], input_destination);
        }

        const int current_weight_k = current_k + weight_k_offset;
        const int weight_bytes = global_n < parameter.out_dim_padded ? 16 : 0;
        const std::uint32_t weight_destination = ptx::smem_u32addr(
            &weight_shared[buffer * OUTPUT_TILE * K_TILE +
                           weight_row_index * K_TILE + weight_k_offset]);
        ptx::cp_async_ca(weight_destination, weight_row + current_weight_k, weight_bytes);
    };

    if (k_iterations > 0)
    {
        load_chunk(0, 0);
        ptx::cp_async_commit_group();
    }

    for (int k_tile = 0; k_tile < k_iterations; ++k_tile)
    {
        if (k_tile + 1 < k_iterations)
        {
            load_chunk(k_tile + 1, (k_tile + 1) % 2);
            ptx::cp_async_commit_group();
            ptx::cp_async_wait_group<1>();
        }
        else
        {
            ptx::cp_async_wait_group<0>();
        }
        __syncthreads();

        std::uint8_t *current_input = &input_shared[k_tile % 2 * OUTPUT_TILE * 16];
        float *current_weight = &weight_shared[k_tile % 2 * OUTPUT_TILE * K_TILE];

#pragma unroll
        for (int k = 0; k < K_TILE; ++k)
        {
            float weight_fragment[FRAGMENT_SIZE];
#pragma unroll
            for (int j = 0; j < FRAGMENT_SIZE; ++j)
            {
                weight_fragment[j] = current_weight[(thread_n_base + j) * K_TILE + k];
            }
#pragma unroll
            for (int i = 0; i < FRAGMENT_SIZE; ++i)
            {
                const std::uint8_t packed_input =
                    current_input[(thread_m_base + i) * 16 + k];
                if ((packed_input >> blockIdx.z) & 1U)
                {
#pragma unroll
                    for (int j = 0; j < FRAGMENT_SIZE; ++j)
                    {
                        output_fragment[i][j] += weight_fragment[j];
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < FRAGMENT_SIZE; ++i)
    {
        const int output_m = blockIdx.y * OUTPUT_TILE + thread_m_base + i;
        if (output_m < parameter.in_ch)
        {
#pragma unroll
            for (int j = 0; j < FRAGMENT_SIZE; ++j)
            {
                const int output_n = blockIdx.x * OUTPUT_TILE + thread_n_base + j;
                if (output_n < parameter.out_dim)
                {
                    outputs[static_cast<std::size_t>(blockIdx.z) * parameter.outBatchNumel +
                            static_cast<std::size_t>(output_m) * parameter.out_dim + output_n] =
                        output_fragment[i][j];
                }
            }
        }
    }
}

__global__ void linear_128x128x8_FP16_kernel(
    const half *__restrict__ inputs,
    const half *__restrict__ weights,
    const half *__restrict__ bias,
    half *__restrict__ outputs,
    LinearParam parameter)
{
    __shared__ __align__(128) char shared_storage[8 * 1024];
    half *input_shared = reinterpret_cast<half *>(shared_storage);
    half *weight_shared = reinterpret_cast<half *>(shared_storage + 4 * 1024);
    half *bias_shared = reinterpret_cast<half *>(shared_storage);

    const int thread_id = threadIdx.x;
    const int warp_id = thread_id / WARP_SIZE;
    const int lane_id = thread_id % WARP_SIZE;

    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;
    const int lane_n = (lane_id / 2) % 8;
    const int lane_m = (lane_id / 16) * 2 + lane_id % 2;
    const int thread_m_base = warp_m * 32 + lane_m * FRAGMENT_SIZE;
    const int thread_n_base = warp_n * 64 + lane_n * FRAGMENT_SIZE;

    if (thread_id < OUTPUT_TILE)
    {
        const int global_n = blockIdx.x * OUTPUT_TILE + thread_id;
        bias_shared[thread_id] =
            global_n < parameter.out_dim ? bias[global_n] : __float2half(0.0F);
    }
    __syncthreads();

    float output_fragment[FRAGMENT_SIZE][FRAGMENT_SIZE];
#pragma unroll
    for (int i = 0; i < FRAGMENT_SIZE; ++i)
    {
        const int global_m = blockIdx.y * OUTPUT_TILE + thread_m_base + i;
        const float row_mask = global_m < parameter.in_ch ? 1.0F : 0.0F;
#pragma unroll
        for (int j = 0; j < FRAGMENT_SIZE; ++j)
        {
            output_fragment[i][j] = __half2float(bias_shared[thread_n_base + j]) * row_mask;
        }
    }
    __syncthreads();

    const int load_row = thread_id % OUTPUT_TILE;
    const bool load_weight = thread_id >= OUTPUT_TILE;

    const int global_m = blockIdx.y * OUTPUT_TILE + load_row;
    const int safe_global_m = global_m < parameter.in_ch ? global_m : 0;
    const half *input_row =
        inputs + static_cast<std::size_t>(blockIdx.z) * parameter.inBatchNumel +
        static_cast<std::size_t>(safe_global_m) * parameter.in_dim;

    const int global_n = blockIdx.x * OUTPUT_TILE + load_row;
    const int k_padded = round_up(static_cast<int>(parameter.in_dim), K_TILE);
    const half *weight_row = weights + static_cast<std::size_t>(global_n) * k_padded;

    const int k_iterations = divide_up(static_cast<int>(parameter.in_dim), K_TILE);

    auto load_chunk = [&](int k_tile, int buffer)
    {
        const int current_k = k_tile * K_TILE;

        if (!load_weight)
        {
            const int remaining = static_cast<int>(parameter.in_dim) - current_k;
            const int valid_elements =
                remaining > K_TILE ? K_TILE : (remaining > 0 ? remaining : 0);
            const int input_bytes =
                global_m < parameter.in_ch ? valid_elements * sizeof(half) : 0;
            const std::uint32_t destination = ptx::smem_u32addr(
                &input_shared[buffer * OUTPUT_TILE * K_TILE + load_row * K_TILE]);
            ptx::cp_async_cg(destination, input_row + current_k, input_bytes);
        }
        else
        {
            const int weight_bytes = global_n < parameter.out_dim_padded ? 16 : 0;
            const std::uint32_t destination = ptx::smem_u32addr(
                &weight_shared[buffer * OUTPUT_TILE * K_TILE + load_row * K_TILE]);
            ptx::cp_async_ca(destination, weight_row + current_k, weight_bytes);
        }
    };

    if (k_iterations > 0)
    {
        load_chunk(0, 0);
        ptx::cp_async_commit_group();
    }

    for (int k_tile = 0; k_tile < k_iterations; ++k_tile)
    {
        if (k_tile + 1 < k_iterations)
        {
            load_chunk(k_tile + 1, (k_tile + 1) % 2);
            ptx::cp_async_commit_group();
            ptx::cp_async_wait_group<1>();
        }
        else
        {
            ptx::cp_async_wait_group<0>();
        }
        __syncthreads();

        half *current_input = &input_shared[k_tile % 2 * OUTPUT_TILE * K_TILE];
        half *current_weight = &weight_shared[k_tile % 2 * OUTPUT_TILE * K_TILE];

#pragma unroll
        for (int k = 0; k < K_TILE; ++k)
        {
            float input_fragment[FRAGMENT_SIZE];
            float weight_fragment[FRAGMENT_SIZE];
#pragma unroll
            for (int i = 0; i < FRAGMENT_SIZE; ++i)
            {
                input_fragment[i] =
                    __half2float(current_input[(thread_m_base + i) * K_TILE + k]);
            }
#pragma unroll
            for (int j = 0; j < FRAGMENT_SIZE; ++j)
            {
                weight_fragment[j] =
                    __half2float(current_weight[(thread_n_base + j) * K_TILE + k]);
            }
#pragma unroll
            for (int i = 0; i < FRAGMENT_SIZE; ++i)
            {
#pragma unroll
                for (int j = 0; j < FRAGMENT_SIZE; ++j)
                {
                    output_fragment[i][j] += input_fragment[i] * weight_fragment[j];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < FRAGMENT_SIZE; ++i)
    {
        const int output_m = blockIdx.y * OUTPUT_TILE + thread_m_base + i;
        if (output_m < parameter.in_ch)
        {
#pragma unroll
            for (int j = 0; j < FRAGMENT_SIZE; ++j)
            {
                const int output_n = blockIdx.x * OUTPUT_TILE + thread_n_base + j;
                if (output_n < parameter.out_dim)
                {
                    outputs[static_cast<std::size_t>(blockIdx.z) * parameter.outBatchNumel +
                            static_cast<std::size_t>(output_m) * parameter.out_dim + output_n] =
                        __float2half(output_fragment[i][j]);
                }
            }
        }
    }
}

__global__ void linear_128x128x8_S_FP16_kernel(
    const std::uint8_t *__restrict__ inputs,
    const half *__restrict__ weights,
    const half *__restrict__ bias,
    half *__restrict__ outputs,
    LinearParam parameter)
{
    __shared__ __align__(128) char shared_storage[16 * 1024];
    std::uint8_t *input_shared = reinterpret_cast<std::uint8_t *>(shared_storage);
    half *weight_shared = reinterpret_cast<half *>(shared_storage + 4 * 1024);
    half *bias_shared = reinterpret_cast<half *>(shared_storage);

    const int thread_id = threadIdx.x;
    const int warp_id = thread_id / WARP_SIZE;
    const int lane_id = thread_id % WARP_SIZE;

    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;
    const int lane_n = (lane_id / 2) % 8;
    const int lane_m = (lane_id / 16) * 2 + lane_id % 2;
    const int thread_m_base = warp_m * 32 + lane_m * FRAGMENT_SIZE;
    const int thread_n_base = warp_n * 64 + lane_n * FRAGMENT_SIZE;

    if (thread_id < OUTPUT_TILE)
    {
        const int global_n = blockIdx.x * OUTPUT_TILE + thread_id;
        bias_shared[thread_id] =
            global_n < parameter.out_dim ? bias[global_n] : __float2half(0.0F);
    }
    __syncthreads();

    float output_fragment[FRAGMENT_SIZE][FRAGMENT_SIZE];
#pragma unroll
    for (int i = 0; i < FRAGMENT_SIZE; ++i)
    {
        const int global_m = blockIdx.y * OUTPUT_TILE + thread_m_base + i;
        const float row_mask = global_m < parameter.in_ch ? 1.0F : 0.0F;
#pragma unroll
        for (int j = 0; j < FRAGMENT_SIZE; ++j)
        {
            output_fragment[i][j] = __half2float(bias_shared[thread_n_base + j]) * row_mask;
        }
    }
    __syncthreads();

    const int load_row = thread_id % OUTPUT_TILE;
    const bool load_weight = thread_id >= OUTPUT_TILE;

    const int global_m = blockIdx.y * OUTPUT_TILE + load_row;
    const int safe_global_m = global_m < parameter.in_ch ? global_m : 0;
    const std::uint8_t *input_row =
        inputs + static_cast<std::size_t>(safe_global_m) * parameter.in_dim;

    const int global_n = blockIdx.x * OUTPUT_TILE + load_row;
    const int k_padded = round_up(static_cast<int>(parameter.in_dim), K_TILE);
    const half *weight_row = weights + static_cast<std::size_t>(global_n) * k_padded;

    const int k_iterations = divide_up(static_cast<int>(parameter.in_dim), K_TILE);

    auto load_chunk = [&](int k_tile, int buffer)
    {
        const int current_k = k_tile * K_TILE;

        if (!load_weight)
        {
            const std::uint32_t destination = ptx::smem_u32addr(
                &input_shared[buffer * OUTPUT_TILE * 16 + load_row * 16]);
            std::uint32_t input_registers[2] = {0U, 0U};

            if (global_m < parameter.in_ch)
            {
                const int remaining = static_cast<int>(parameter.in_dim) - current_k;
                if (remaining >= K_TILE)
                {
                    ptx::ldg32_nc_0(input_registers[0], input_row + current_k, true);
                    ptx::ldg32_nc_0(input_registers[1], input_row + current_k + 4, true);
                }
                else
                {
                    for (int i = 0; i < remaining; ++i)
                    {
                        const std::uint32_t value = input_row[current_k + i];
                        if (i < 4)
                        {
                            input_registers[0] |= value << (i * 8);
                        }
                        else
                        {
                            input_registers[1] |= value << ((i - 4) * 8);
                        }
                    }
                }
            }

            ptx::sts64(input_registers[0], input_registers[1], destination);
        }
        else
        {
            const int weight_bytes = global_n < parameter.out_dim_padded ? 16 : 0;
            const std::uint32_t destination = ptx::smem_u32addr(
                &weight_shared[buffer * OUTPUT_TILE * K_TILE + load_row * K_TILE]);
            ptx::cp_async_ca(destination, weight_row + current_k, weight_bytes);
        }
    };

    if (k_iterations > 0)
    {
        load_chunk(0, 0);
        ptx::cp_async_commit_group();
    }

    for (int k_tile = 0; k_tile < k_iterations; ++k_tile)
    {
        if (k_tile + 1 < k_iterations)
        {
            load_chunk(k_tile + 1, (k_tile + 1) % 2);
            ptx::cp_async_commit_group();
            ptx::cp_async_wait_group<1>();
        }
        else
        {
            ptx::cp_async_wait_group<0>();
        }
        __syncthreads();

        std::uint8_t *current_input = &input_shared[k_tile % 2 * OUTPUT_TILE * 16];
        half *current_weight = &weight_shared[k_tile % 2 * OUTPUT_TILE * K_TILE];

#pragma unroll
        for (int k = 0; k < K_TILE; ++k)
        {
            float weight_fragment[FRAGMENT_SIZE];
#pragma unroll
            for (int j = 0; j < FRAGMENT_SIZE; ++j)
            {
                weight_fragment[j] =
                    __half2float(current_weight[(thread_n_base + j) * K_TILE + k]);
            }
#pragma unroll
            for (int i = 0; i < FRAGMENT_SIZE; ++i)
            {
                const std::uint8_t packed_input =
                    current_input[(thread_m_base + i) * 16 + k];
                if ((packed_input >> blockIdx.z) & 1U)
                {
#pragma unroll
                    for (int j = 0; j < FRAGMENT_SIZE; ++j)
                    {
                        output_fragment[i][j] += weight_fragment[j];
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < FRAGMENT_SIZE; ++i)
    {
        const int output_m = blockIdx.y * OUTPUT_TILE + thread_m_base + i;
        if (output_m < parameter.in_ch)
        {
#pragma unroll
            for (int j = 0; j < FRAGMENT_SIZE; ++j)
            {
                const int output_n = blockIdx.x * OUTPUT_TILE + thread_n_base + j;
                if (output_n < parameter.out_dim)
                {
                    outputs[static_cast<std::size_t>(blockIdx.z) * parameter.outBatchNumel +
                            static_cast<std::size_t>(output_m) * parameter.out_dim + output_n] =
                        __float2half(output_fragment[i][j]);
                }
            }
        }
    }
}

namespace
{

ValidationResult validate_ann_fp32(const TestShape &shape)
{
    const int k_padded = round_up(shape.k, K_TILE);
    const int n_padded = round_up(shape.n, OUTPUT_TILE);

    const std::size_t input_count =
        static_cast<std::size_t>(shape.time_steps) * shape.m * shape.k;
    const std::size_t weight_count = static_cast<std::size_t>(shape.n) * shape.k;
    const std::size_t padded_weight_count = static_cast<std::size_t>(n_padded) * k_padded;
    const std::size_t output_count =
        static_cast<std::size_t>(shape.time_steps) * shape.m * shape.n;

    const std::vector<float> input = make_random_fp32(input_count, 101U);
    const std::vector<float> weight = make_random_fp32(weight_count, 102U);
    const std::vector<float> bias = make_random_fp32(shape.n, 103U);
    std::vector<float> padded_weight(padded_weight_count, 0.0F);
    std::vector<float> output(output_count, 0.0F);

    pad_weights(weight, padded_weight, shape.k, shape.n, k_padded);

    DeviceBuffer<float> device_input(input_count);
    DeviceBuffer<float> device_weight(padded_weight_count);
    DeviceBuffer<float> device_bias(shape.n);
    DeviceBuffer<float> device_output(output_count);

    copy_to_device(device_input, input);
    copy_to_device(device_weight, padded_weight);
    copy_to_device(device_bias, bias);

    linear_128x128x8_kernel<<<make_grid(shape), THREADS_PER_BLOCK>>>(
        device_input.get(),
        device_weight.get(),
        device_bias.get(),
        device_output.get(),
        make_linear_param(shape, false));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    copy_to_host(output, device_output);

    ValidationResult result;
    for (int time = 0; time < shape.time_steps; ++time)
    {
        for (int m = 0; m < shape.m; ++m)
        {
            for (int n = 0; n < shape.n; ++n)
            {
                float expected = bias[n];
                for (int k = 0; k < shape.k; ++k)
                {
                    const std::size_t input_index =
                        (static_cast<std::size_t>(time) * shape.m + m) * shape.k + k;
                    const std::size_t weight_index =
                        static_cast<std::size_t>(n) * shape.k + k;

                    expected += input[input_index] * weight[weight_index];
                }

                const std::size_t output_index =
                    (static_cast<std::size_t>(time) * shape.m + m) * shape.n + n;
                update_validation(result, output[output_index], expected, FP32_TOLERANCE);
            }
        }
    }

    return result;
}

ValidationResult validate_snn_fp32(const TestShape &shape)
{
    const int k_padded = round_up(shape.k, K_TILE);
    const int n_padded = round_up(shape.n, OUTPUT_TILE);

    const std::size_t input_count = static_cast<std::size_t>(shape.m) * shape.k;
    const std::size_t weight_count = static_cast<std::size_t>(shape.n) * shape.k;
    const std::size_t padded_weight_count = static_cast<std::size_t>(n_padded) * k_padded;
    const std::size_t output_count =
        static_cast<std::size_t>(shape.time_steps) * shape.m * shape.n;

    const std::vector<std::uint8_t> input =
        make_random_spikes(input_count, shape.time_steps, 201U);
    const std::vector<float> weight = make_random_fp32(weight_count, 202U);
    const std::vector<float> bias = make_random_fp32(shape.n, 203U);
    std::vector<float> padded_weight(padded_weight_count, 0.0F);
    std::vector<float> output(output_count, 0.0F);

    pad_weights(weight, padded_weight, shape.k, shape.n, k_padded);

    DeviceBuffer<std::uint8_t> device_input(input_count);
    DeviceBuffer<float> device_weight(padded_weight_count);
    DeviceBuffer<float> device_bias(shape.n);
    DeviceBuffer<float> device_output(output_count);

    copy_to_device(device_input, input);
    copy_to_device(device_weight, padded_weight);
    copy_to_device(device_bias, bias);

    linear_128x128x8_S_kernel<<<make_grid(shape), THREADS_PER_BLOCK>>>(
        device_input.get(),
        device_weight.get(),
        device_bias.get(),
        device_output.get(),
        make_linear_param(shape, true));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    copy_to_host(output, device_output);

    ValidationResult result;
    for (int time = 0; time < shape.time_steps; ++time)
    {
        for (int m = 0; m < shape.m; ++m)
        {
            for (int n = 0; n < shape.n; ++n)
            {
                float expected = bias[n];
                for (int k = 0; k < shape.k; ++k)
                {
                    const bool spike =
                        (input[static_cast<std::size_t>(m) * shape.k + k] >> time) & 1U;
                    if (spike)
                    {
                        expected += weight[static_cast<std::size_t>(n) * shape.k + k];
                    }
                }

                const std::size_t output_index =
                    (static_cast<std::size_t>(time) * shape.m + m) * shape.n + n;
                update_validation(result, output[output_index], expected, FP32_TOLERANCE);
            }
        }
    }

    return result;
}

ValidationResult validate_ann_fp16(const TestShape &shape)
{
    const int k_padded = round_up(shape.k, K_TILE);
    const int n_padded = round_up(shape.n, OUTPUT_TILE);

    const std::size_t input_count =
        static_cast<std::size_t>(shape.time_steps) * shape.m * shape.k;
    const std::size_t weight_count = static_cast<std::size_t>(shape.n) * shape.k;
    const std::size_t padded_weight_count = static_cast<std::size_t>(n_padded) * k_padded;
    const std::size_t output_count =
        static_cast<std::size_t>(shape.time_steps) * shape.m * shape.n;

    const std::vector<half> input = make_random_fp16(input_count, 301U);
    const std::vector<half> weight = make_random_fp16(weight_count, 302U);
    const std::vector<half> bias = make_random_fp16(shape.n, 303U);
    std::vector<half> padded_weight(padded_weight_count, __float2half(0.0F));
    std::vector<half> output(output_count, __float2half(0.0F));

    pad_weights(weight, padded_weight, shape.k, shape.n, k_padded);

    DeviceBuffer<half> device_input(input_count);
    DeviceBuffer<half> device_weight(padded_weight_count);
    DeviceBuffer<half> device_bias(shape.n);
    DeviceBuffer<half> device_output(output_count);

    copy_to_device(device_input, input);
    copy_to_device(device_weight, padded_weight);
    copy_to_device(device_bias, bias);

    linear_128x128x8_FP16_kernel<<<make_grid(shape), THREADS_PER_BLOCK>>>(
        device_input.get(),
        device_weight.get(),
        device_bias.get(),
        device_output.get(),
        make_linear_param(shape, false));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    copy_to_host(output, device_output);

    ValidationResult result;
    for (int time = 0; time < shape.time_steps; ++time)
    {
        for (int m = 0; m < shape.m; ++m)
        {
            for (int n = 0; n < shape.n; ++n)
            {
                float expected = __half2float(bias[n]);
                for (int k = 0; k < shape.k; ++k)
                {
                    const std::size_t input_index =
                        (static_cast<std::size_t>(time) * shape.m + m) * shape.k + k;
                    const std::size_t weight_index =
                        static_cast<std::size_t>(n) * shape.k + k;

                    expected += __half2float(input[input_index]) *
                                __half2float(weight[weight_index]);
                }

                const std::size_t output_index =
                    (static_cast<std::size_t>(time) * shape.m + m) * shape.n + n;
                update_validation(result,
                                  __half2float(output[output_index]),
                                  __half2float(__float2half(expected)),
                                  FP16_TOLERANCE);
            }
        }
    }

    return result;
}

ValidationResult validate_snn_fp16(const TestShape &shape)
{
    const int k_padded = round_up(shape.k, K_TILE);
    const int n_padded = round_up(shape.n, OUTPUT_TILE);

    const std::size_t input_count = static_cast<std::size_t>(shape.m) * shape.k;
    const std::size_t weight_count = static_cast<std::size_t>(shape.n) * shape.k;
    const std::size_t padded_weight_count = static_cast<std::size_t>(n_padded) * k_padded;
    const std::size_t output_count =
        static_cast<std::size_t>(shape.time_steps) * shape.m * shape.n;

    const std::vector<std::uint8_t> input =
        make_random_spikes(input_count, shape.time_steps, 401U);
    const std::vector<half> weight = make_random_fp16(weight_count, 402U);
    const std::vector<half> bias = make_random_fp16(shape.n, 403U);
    std::vector<half> padded_weight(padded_weight_count, __float2half(0.0F));
    std::vector<half> output(output_count, __float2half(0.0F));

    pad_weights(weight, padded_weight, shape.k, shape.n, k_padded);

    DeviceBuffer<std::uint8_t> device_input(input_count);
    DeviceBuffer<half> device_weight(padded_weight_count);
    DeviceBuffer<half> device_bias(shape.n);
    DeviceBuffer<half> device_output(output_count);

    copy_to_device(device_input, input);
    copy_to_device(device_weight, padded_weight);
    copy_to_device(device_bias, bias);

    linear_128x128x8_S_FP16_kernel<<<make_grid(shape), THREADS_PER_BLOCK>>>(
        device_input.get(),
        device_weight.get(),
        device_bias.get(),
        device_output.get(),
        make_linear_param(shape, true));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    copy_to_host(output, device_output);

    ValidationResult result;
    for (int time = 0; time < shape.time_steps; ++time)
    {
        for (int m = 0; m < shape.m; ++m)
        {
            for (int n = 0; n < shape.n; ++n)
            {
                float expected = __half2float(bias[n]);
                for (int k = 0; k < shape.k; ++k)
                {
                    const bool spike =
                        (input[static_cast<std::size_t>(m) * shape.k + k] >> time) & 1U;
                    if (spike)
                    {
                        expected += __half2float(
                            weight[static_cast<std::size_t>(n) * shape.k + k]);
                    }
                }

                const std::size_t output_index =
                    (static_cast<std::size_t>(time) * shape.m + m) * shape.n + n;
                update_validation(result,
                                  __half2float(output[output_index]),
                                  __half2float(__float2half(expected)),
                                  FP16_TOLERANCE);
            }
        }
    }

    return result;
}

template <typename Launcher>
BenchmarkResult benchmark_kernel(const std::string &name, const TestShape &shape, Launcher launch)
{
    for (int iteration = 0; iteration < WARMUP_ITERATIONS; ++iteration)
    {
        launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int iteration = 0; iteration < BENCHMARK_ITERATIONS; ++iteration)
    {
        launch();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float elapsed_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    const double average_ms = elapsed_ms / BENCHMARK_ITERATIONS;
    const double operation_count = 2.0 * shape.time_steps * shape.m * shape.n * shape.k;

    BenchmarkResult result;
    result.name = name;
    result.latency_ms = average_ms;
    result.dense_equivalent_tflops = operation_count / (average_ms * 1.0e9);

    return result;
}

BenchmarkResult benchmark_ann_fp32(const TestShape &shape)
{
    const int k_padded = round_up(shape.k, K_TILE);
    const int n_padded = round_up(shape.n, OUTPUT_TILE);

    const std::size_t input_count =
        static_cast<std::size_t>(shape.time_steps) * shape.m * shape.k;
    const std::size_t weight_count = static_cast<std::size_t>(n_padded) * k_padded;
    const std::size_t output_count =
        static_cast<std::size_t>(shape.time_steps) * shape.m * shape.n;

    const std::vector<float> input = make_random_fp32(input_count, 501U);
    const std::vector<float> weight = make_random_fp32(weight_count, 502U);
    const std::vector<float> bias = make_random_fp32(shape.n, 503U);

    DeviceBuffer<float> device_input(input_count);
    DeviceBuffer<float> device_weight(weight_count);
    DeviceBuffer<float> device_bias(shape.n);
    DeviceBuffer<float> device_output(output_count);

    copy_to_device(device_input, input);
    copy_to_device(device_weight, weight);
    copy_to_device(device_bias, bias);

    const LinearParam parameter = make_linear_param(shape, false);
    const dim3 grid = make_grid(shape);

    return benchmark_kernel("ANN FP32", shape, [&]()
    {
        linear_128x128x8_kernel<<<grid, THREADS_PER_BLOCK>>>(
            device_input.get(),
            device_weight.get(),
            device_bias.get(),
            device_output.get(),
            parameter);
    });
}

BenchmarkResult benchmark_snn_fp32(const TestShape &shape)
{
    const int k_padded = round_up(shape.k, K_TILE);
    const int n_padded = round_up(shape.n, OUTPUT_TILE);

    const std::size_t input_count = static_cast<std::size_t>(shape.m) * shape.k;
    const std::size_t weight_count = static_cast<std::size_t>(n_padded) * k_padded;
    const std::size_t output_count =
        static_cast<std::size_t>(shape.time_steps) * shape.m * shape.n;

    const std::vector<std::uint8_t> input =
        make_random_spikes(input_count, shape.time_steps, 601U);
    const std::vector<float> weight = make_random_fp32(weight_count, 602U);
    const std::vector<float> bias = make_random_fp32(shape.n, 603U);

    DeviceBuffer<std::uint8_t> device_input(input_count);
    DeviceBuffer<float> device_weight(weight_count);
    DeviceBuffer<float> device_bias(shape.n);
    DeviceBuffer<float> device_output(output_count);

    copy_to_device(device_input, input);
    copy_to_device(device_weight, weight);
    copy_to_device(device_bias, bias);

    const LinearParam parameter = make_linear_param(shape, true);
    const dim3 grid = make_grid(shape);

    return benchmark_kernel("SNN FP32", shape, [&]()
    {
        linear_128x128x8_S_kernel<<<grid, THREADS_PER_BLOCK>>>(
            device_input.get(),
            device_weight.get(),
            device_bias.get(),
            device_output.get(),
            parameter);
    });
}

BenchmarkResult benchmark_ann_fp16(const TestShape &shape)
{
    const int k_padded = round_up(shape.k, K_TILE);
    const int n_padded = round_up(shape.n, OUTPUT_TILE);

    const std::size_t input_count =
        static_cast<std::size_t>(shape.time_steps) * shape.m * shape.k;
    const std::size_t weight_count = static_cast<std::size_t>(n_padded) * k_padded;
    const std::size_t output_count =
        static_cast<std::size_t>(shape.time_steps) * shape.m * shape.n;

    const std::vector<half> input = make_random_fp16(input_count, 701U);
    const std::vector<half> weight = make_random_fp16(weight_count, 702U);
    const std::vector<half> bias = make_random_fp16(shape.n, 703U);

    DeviceBuffer<half> device_input(input_count);
    DeviceBuffer<half> device_weight(weight_count);
    DeviceBuffer<half> device_bias(shape.n);
    DeviceBuffer<half> device_output(output_count);

    copy_to_device(device_input, input);
    copy_to_device(device_weight, weight);
    copy_to_device(device_bias, bias);

    const LinearParam parameter = make_linear_param(shape, false);
    const dim3 grid = make_grid(shape);

    return benchmark_kernel("ANN FP16", shape, [&]()
    {
        linear_128x128x8_FP16_kernel<<<grid, THREADS_PER_BLOCK>>>(
            device_input.get(),
            device_weight.get(),
            device_bias.get(),
            device_output.get(),
            parameter);
    });
}

BenchmarkResult benchmark_snn_fp16(const TestShape &shape)
{
    const int k_padded = round_up(shape.k, K_TILE);
    const int n_padded = round_up(shape.n, OUTPUT_TILE);

    const std::size_t input_count = static_cast<std::size_t>(shape.m) * shape.k;
    const std::size_t weight_count = static_cast<std::size_t>(n_padded) * k_padded;
    const std::size_t output_count =
        static_cast<std::size_t>(shape.time_steps) * shape.m * shape.n;

    const std::vector<std::uint8_t> input =
        make_random_spikes(input_count, shape.time_steps, 801U);
    const std::vector<half> weight = make_random_fp16(weight_count, 802U);
    const std::vector<half> bias = make_random_fp16(shape.n, 803U);

    DeviceBuffer<std::uint8_t> device_input(input_count);
    DeviceBuffer<half> device_weight(weight_count);
    DeviceBuffer<half> device_bias(shape.n);
    DeviceBuffer<half> device_output(output_count);

    copy_to_device(device_input, input);
    copy_to_device(device_weight, weight);
    copy_to_device(device_bias, bias);

    const LinearParam parameter = make_linear_param(shape, true);
    const dim3 grid = make_grid(shape);

    return benchmark_kernel("SNN FP16", shape, [&]()
    {
        linear_128x128x8_S_FP16_kernel<<<grid, THREADS_PER_BLOCK>>>(
            device_input.get(),
            device_weight.get(),
            device_bias.get(),
            device_output.get(),
            parameter);
    });
}

void print_benchmark_result(const BenchmarkResult &result)
{
    std::cout << "  " << std::left << std::setw(10) << result.name
              << std::right << std::fixed << std::setprecision(4)
              << " latency=" << std::setw(9) << result.latency_ms << " ms"
              << "  dense-equivalent=" << std::setw(8)
              << std::setprecision(3) << result.dense_equivalent_tflops
              << " TFLOP/s" << std::endl;
}

} // namespace

int main()
{
    const TestShape aligned_shape{1, 128, 128, 128};
    const TestShape boundary_ann_shape{4, 130, 72, 140};
    const TestShape boundary_snn_shape{8, 130, 72, 140};

    const TestShape ann_benchmark_shape{4, 512, 1024, 768};
    const TestShape snn_benchmark_shape{8, 512, 1024, 768};

    std::cout << "Linear 128x128x8 CUDA kernel test and benchmark\n\n"
              << "[Configuration]\n"
              << "  output tile          : " << OUTPUT_TILE << " x " << OUTPUT_TILE << '\n'
              << "  K tile               : " << K_TILE << '\n'
              << "  input K requirement  : multiple of " << K_TILE << '\n'
              << "  threads per block    : " << THREADS_PER_BLOCK << '\n'
              << "  FP32 tolerance       : " << std::scientific << FP32_TOLERANCE << '\n'
              << "  FP16 tolerance       : " << FP16_TOLERANCE << '\n'
              << "  warmup iterations    : " << std::fixed << WARMUP_ITERATIONS << '\n'
              << "  benchmark iterations : " << BENCHMARK_ITERATIONS << '\n'
              << "  ANN benchmark shape  : T=" << ann_benchmark_shape.time_steps
              << ", M=" << ann_benchmark_shape.m
              << ", K=" << ann_benchmark_shape.k
              << ", N=" << ann_benchmark_shape.n << '\n'
              << "  SNN benchmark shape  : T=" << snn_benchmark_shape.time_steps
              << ", M=" << snn_benchmark_shape.m
              << ", K=" << snn_benchmark_shape.k
              << ", N=" << snn_benchmark_shape.n << std::endl;

    std::cout << "\n[Stage 1] Correctness validation" << std::endl;
    bool all_passed = true;

    ValidationResult result = validate_ann_fp32(aligned_shape);
    print_validation_result("ANN FP32", aligned_shape, result);
    all_passed = all_passed && result.passed();
    result = validate_ann_fp32(boundary_ann_shape);
    print_validation_result("ANN FP32", boundary_ann_shape, result);
    all_passed = all_passed && result.passed();

    result = validate_snn_fp32(aligned_shape);
    print_validation_result("SNN FP32", aligned_shape, result);
    all_passed = all_passed && result.passed();
    result = validate_snn_fp32(boundary_snn_shape);
    print_validation_result("SNN FP32", boundary_snn_shape, result);
    all_passed = all_passed && result.passed();

    result = validate_ann_fp16(aligned_shape);
    print_validation_result("ANN FP16", aligned_shape, result);
    all_passed = all_passed && result.passed();
    result = validate_ann_fp16(boundary_ann_shape);
    print_validation_result("ANN FP16", boundary_ann_shape, result);
    all_passed = all_passed && result.passed();

    result = validate_snn_fp16(aligned_shape);
    print_validation_result("SNN FP16", aligned_shape, result);
    all_passed = all_passed && result.passed();
    result = validate_snn_fp16(boundary_snn_shape);
    print_validation_result("SNN FP16", boundary_snn_shape, result);
    all_passed = all_passed && result.passed();

    if (!all_passed)
    {
        std::cout << "\n[FAILED] Linear correctness validation failed" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "\n[Stage 2] Performance benchmark" << std::endl;
    print_benchmark_result(benchmark_ann_fp32(ann_benchmark_shape));
    print_benchmark_result(benchmark_snn_fp32(snn_benchmark_shape));
    print_benchmark_result(benchmark_ann_fp16(ann_benchmark_shape));
    print_benchmark_result(benchmark_snn_fp16(snn_benchmark_shape));

    std::cout << "\n[SUCCESS] Linear correctness and benchmark completed" << std::endl;
    return EXIT_SUCCESS;
}
