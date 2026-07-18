#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace
{

constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 2048;

constexpr int M_TILE = 128;
constexpr int N_TILE = 128;
constexpr int K_TILE = 16;
constexpr int INT8_VALUES_PER_PACK = 4;
constexpr int K_PACKS = K_TILE / INT8_VALUES_PER_PACK;
constexpr int THREADS = 256;
constexpr int THREAD_TILE_M = 8;
constexpr int THREAD_TILE_N = 8;

void check_cuda(
    cudaError_t result,
    const char *expression,
    const char *file,
    int line)
{
    if (result == cudaSuccess)
    {
        return;
    }

    std::cout << "CUDA error: " << cudaGetErrorString(result) << '\n'
              << "  expression: " << expression << '\n'
              << "  location:   " << file << ':' << line << std::endl;
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression)                                                \
    check_cuda((expression), #expression, __FILE__, __LINE__)

template <typename T>
void allocate_device(T **pointer, std::size_t count)
{
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(pointer),
        count * sizeof(T)));
}

__global__ void quantize_symmetric_per_tensor_kernel(
    const float *__restrict__ input,
    std::int8_t *__restrict__ output,
    std::size_t count,
    float inverse_scale)
{
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index < count)
    {
        int value = __float2int_rn(input[index] * inverse_scale);
        value = max(-127, min(127, value));
        output[index] = static_cast<std::int8_t>(value);
    }
}

// A: [M, K], B: [N, K], C: [M, N]. This kernel computes C = A * B^T.
// M, N, and K are assumed to be exact multiples of their tile sizes.
__global__ void gemm_int8_128x128x16_kernel(
    const std::int8_t *__restrict__ a,
    const std::int8_t *__restrict__ b,
    std::int32_t *__restrict__ c,
    int k_size)
{
    // Each int32 stores four consecutive signed int8 values along K.
    __shared__ std::int32_t a_shared[M_TILE][K_PACKS];
    __shared__ std::int32_t b_shared[N_TILE][K_PACKS];

    const int thread_id = threadIdx.x;
    const int thread_tile_row = thread_id / 16;
    const int thread_tile_col = thread_id % 16;
    const int local_m_base = thread_tile_row * THREAD_TILE_M;
    const int local_n_base = thread_tile_col * THREAD_TILE_N;
    const int global_m_base = blockIdx.y * M_TILE;
    const int global_n_base = blockIdx.x * N_TILE;
    const int k_size_packed = k_size / INT8_VALUES_PER_PACK;
    const std::int32_t *a_packed_global =
        reinterpret_cast<const std::int32_t *>(a);
    const std::int32_t *b_packed_global =
        reinterpret_cast<const std::int32_t *>(b);

    std::int32_t accumulators[THREAD_TILE_M][THREAD_TILE_N] = {};

    for (int k_base = 0; k_base < k_size; k_base += K_TILE)
    {
        // Each block loads 128 * 4 packed int32 values from each operand.
        // Every packed value contains four consecutive signed int8 values.
        for (int index = thread_id;
             index < M_TILE * K_PACKS;
             index += THREADS)
        {
            const int row = index / K_PACKS;
            const int k_pack = index % K_PACKS;
            const int global_k_pack =
                k_base / INT8_VALUES_PER_PACK + k_pack;
            a_shared[row][k_pack] = a_packed_global[
                (global_m_base + row) * k_size_packed + global_k_pack];
            b_shared[row][k_pack] = b_packed_global[
                (global_n_base + row) * k_size_packed + global_k_pack];
        }
        __syncthreads();

#pragma unroll 1
        for (int k_pack = 0; k_pack < K_PACKS; ++k_pack)
        {
            std::int32_t a_fragment[THREAD_TILE_M];
            std::int32_t b_fragment[THREAD_TILE_N];

#pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i)
            {
                a_fragment[i] =
                    a_shared[local_m_base + i][k_pack];
            }

#pragma unroll
            for (int j = 0; j < THREAD_TILE_N; ++j)
            {
                b_fragment[j] =
                    b_shared[local_n_base + j][k_pack];
            }

#pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i)
            {
#pragma unroll
                for (int j = 0; j < THREAD_TILE_N; ++j)
                {
                    accumulators[i][j] = __dp4a(
                        a_fragment[i],
                        b_fragment[j],
                        accumulators[i][j]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < THREAD_TILE_M; ++i)
    {
#pragma unroll
        for (int j = 0; j < THREAD_TILE_N; ++j)
        {
            const int global_m = global_m_base + local_m_base + i;
            const int global_n = global_n_base + local_n_base + j;
            c[global_m * N + global_n] = accumulators[i][j];
        }
    }
}

__global__ void dequantize_per_tensor_kernel(
    const std::int32_t *__restrict__ input,
    float *__restrict__ output,
    std::size_t count,
    float output_scale)
{
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index < count)
    {
        output[index] = static_cast<float>(input[index]) * output_scale;
    }
}

std::int32_t reference_int8_dot(
    const std::vector<std::int8_t> &a,
    const std::vector<std::int8_t> &b,
    int row,
    int column)
{
    std::int32_t result = 0;
    for (int k = 0; k < K; ++k)
    {
        result += static_cast<std::int32_t>(a[row * K + k]) *
                  static_cast<std::int32_t>(b[column * K + k]);
    }
    return result;
}

float reference_fp32_dot(
    const std::vector<float> &a,
    const std::vector<float> &b,
    int row,
    int column,
    double *absolute_product_sum)
{
    double result = 0.0;
    double product_sum = 0.0;
    for (int k = 0; k < K; ++k)
    {
        const double product =
            static_cast<double>(a[row * K + k]) *
            static_cast<double>(b[column * K + k]);
        result += product;
        product_sum += std::abs(product);
    }

    *absolute_product_sum = product_sum;
    return static_cast<float>(result);
}

} // namespace

int main()
{
    static_assert(M % M_TILE == 0);
    static_assert(N % N_TILE == 0);
    static_assert(K % K_TILE == 0);
    static_assert(K_TILE % INT8_VALUES_PER_PACK == 0);
    static_assert(
        THREADS * THREAD_TILE_M * THREAD_TILE_N == M_TILE * N_TILE);

    constexpr std::size_t A_COUNT = static_cast<std::size_t>(M) * K;
    constexpr std::size_t B_COUNT = static_cast<std::size_t>(N) * K;
    constexpr std::size_t C_COUNT = static_cast<std::size_t>(M) * N;

    std::mt19937 generator(20260717);
    std::uniform_real_distribution<float> input_distribution(-1.0F, 1.0F);
    std::uniform_real_distribution<float> scale_distribution(0.008F, 0.012F);

    const float a_scale = scale_distribution(generator);
    const float b_scale = scale_distribution(generator);
    const float output_scale = a_scale * b_scale;

    std::vector<float> host_a_fp32(A_COUNT);
    std::vector<float> host_b_fp32(B_COUNT);
    std::vector<std::int8_t> host_a_int8(A_COUNT);
    std::vector<std::int8_t> host_b_int8(B_COUNT);

    std::cout << "INT8 GEMM basic test\n\n"
              << "[Configuration]\n"
              << "  A layout          : [M, K] = [" << M << ", " << K
              << "]\n"
              << "  B layout          : [N, K] = [" << N << ", " << K
              << "]\n"
              << "  C layout          : [M, N] = [" << M << ", " << N
              << "]\n"
              << "  block tile        : " << M_TILE << " x " << N_TILE
              << " x " << K_TILE << '\n'
              << "  threads per block : " << THREADS << '\n'
              << "  outputs per thread: "
              << THREAD_TILE_M * THREAD_TILE_N << '\n'
              << std::fixed << std::setprecision(8)
              << "  A scale           : " << a_scale << '\n'
              << "  B scale           : " << b_scale << '\n'
              << "  output scale      : " << output_scale << std::endl;

    std::cout << "\n[Stage 1] Generate FP32 inputs" << std::endl;
    std::generate(
        host_a_fp32.begin(),
        host_a_fp32.end(),
        [&generator, &input_distribution]()
        {
            return input_distribution(generator);
        });
    std::generate(
        host_b_fp32.begin(),
        host_b_fp32.end(),
        [&generator, &input_distribution]()
        {
            return input_distribution(generator);
        });

    float *device_a_fp32 = nullptr;
    float *device_b_fp32 = nullptr;
    float *device_c_fp32 = nullptr;
    std::int8_t *device_a_int8 = nullptr;
    std::int8_t *device_b_int8 = nullptr;
    std::int32_t *device_c_int32 = nullptr;

    allocate_device(&device_a_fp32, A_COUNT);
    allocate_device(&device_b_fp32, B_COUNT);
    allocate_device(&device_c_fp32, C_COUNT);
    allocate_device(&device_a_int8, A_COUNT);
    allocate_device(&device_b_int8, B_COUNT);
    allocate_device(&device_c_int32, C_COUNT);

    CUDA_CHECK(cudaMemcpy(
        device_a_fp32,
        host_a_fp32.data(),
        A_COUNT * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        device_b_fp32,
        host_b_fp32.data(),
        B_COUNT * sizeof(float),
        cudaMemcpyHostToDevice));

    std::cout << "\n[Stage 2] Symmetric per-tensor FP32 -> INT8 quantization"
              << std::endl;
    constexpr int ELEMENT_THREADS = 256;
    const int a_blocks = static_cast<int>(
        (A_COUNT + ELEMENT_THREADS - 1) / ELEMENT_THREADS);
    const int b_blocks = static_cast<int>(
        (B_COUNT + ELEMENT_THREADS - 1) / ELEMENT_THREADS);
    const int c_blocks = static_cast<int>(
        (C_COUNT + ELEMENT_THREADS - 1) / ELEMENT_THREADS);

    quantize_symmetric_per_tensor_kernel<<<a_blocks, ELEMENT_THREADS>>>(
        device_a_fp32,
        device_a_int8,
        A_COUNT,
        1.0F / a_scale);
    quantize_symmetric_per_tensor_kernel<<<b_blocks, ELEMENT_THREADS>>>(
        device_b_fp32,
        device_b_int8,
        B_COUNT,
        1.0F / b_scale);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(
        host_a_int8.data(),
        device_a_int8,
        A_COUNT * sizeof(std::int8_t),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        host_b_int8.data(),
        device_b_int8,
        B_COUNT * sizeof(std::int8_t),
        cudaMemcpyDeviceToHost));

    std::cout << "\n[Stage 3] INT32 += INT8 * INT8 CUDA Core GEMM"
              << std::endl;
    const dim3 block(THREADS);
    const dim3 grid(N / N_TILE, M / M_TILE);

    gemm_int8_128x128x16_kernel<<<grid, block>>>(
        device_a_int8,
        device_b_int8,
        device_c_int32,
        K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    constexpr int BENCHMARK_ITERATIONS = 5;
    CUDA_CHECK(cudaEventRecord(start));
    for (int iteration = 0; iteration < BENCHMARK_ITERATIONS; ++iteration)
    {
        gemm_int8_128x128x16_kernel<<<grid, block>>>(
            device_a_int8,
            device_b_int8,
            device_c_int32,
            K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float elapsed_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    const float average_ms = elapsed_ms / BENCHMARK_ITERATIONS;
    const double operation_count =
        2.0 * static_cast<double>(M) * N * K;
    const double effective_tops =
        operation_count / (static_cast<double>(average_ms) * 1.0e9);

    std::cout << std::fixed << std::setprecision(3)
              << "  average latency   : " << average_ms << " ms\n"
              << "  effective compute : " << effective_tops << " TOPS"
              << std::endl;

    std::cout << "\n[Stage 4] INT32 -> FP32 dequantization" << std::endl;
    dequantize_per_tensor_kernel<<<c_blocks, ELEMENT_THREADS>>>(
        device_c_int32,
        device_c_fp32,
        C_COUNT,
        output_scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "\n[Stage 5] Check sampled output elements" << std::endl;
    constexpr int CHECK_COUNT = 32;
    std::uniform_int_distribution<int> row_distribution(0, M - 1);
    std::uniform_int_distribution<int> column_distribution(0, N - 1);

    bool int32_exact = true;
    double max_normalized_quantization_error = 0.0;
    for (int check = 0; check < CHECK_COUNT; ++check)
    {
        const int row = row_distribution(generator);
        const int column = column_distribution(generator);
        const std::size_t output_index =
            static_cast<std::size_t>(row) * N + column;

        std::int32_t gpu_int32 = 0;
        float gpu_fp32 = 0.0F;
        CUDA_CHECK(cudaMemcpy(
            &gpu_int32,
            device_c_int32 + output_index,
            sizeof(gpu_int32),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            &gpu_fp32,
            device_c_fp32 + output_index,
            sizeof(gpu_fp32),
            cudaMemcpyDeviceToHost));

        const std::int32_t reference_int32 = reference_int8_dot(
            host_a_int8,
            host_b_int8,
            row,
            column);
        int32_exact = int32_exact && gpu_int32 == reference_int32;

        double absolute_product_sum = 0.0;
        const float reference_fp32 = reference_fp32_dot(
            host_a_fp32,
            host_b_fp32,
            row,
            column,
            &absolute_product_sum);
        const double normalized_error =
            std::abs(static_cast<double>(gpu_fp32) - reference_fp32) /
            std::max(absolute_product_sum, 1.0e-12);
        max_normalized_quantization_error = std::max(
            max_normalized_quantization_error,
            normalized_error);
    }

    constexpr double QUANTIZATION_ERROR_LIMIT = 0.02;
    const bool quantization_error_ok =
        max_normalized_quantization_error < QUANTIZATION_ERROR_LIMIT;

    std::cout << std::boolalpha
              << "  sampled INT32 exact match       : " << int32_exact << '\n'
              << std::scientific << std::setprecision(6)
              << "  max normalized quantization err : "
              << max_normalized_quantization_error << '\n'
              << "  quantization error limit        : "
              << QUANTIZATION_ERROR_LIMIT << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(device_a_fp32));
    CUDA_CHECK(cudaFree(device_b_fp32));
    CUDA_CHECK(cudaFree(device_c_fp32));
    CUDA_CHECK(cudaFree(device_a_int8));
    CUDA_CHECK(cudaFree(device_b_int8));
    CUDA_CHECK(cudaFree(device_c_int32));

    if (!int32_exact || !quantization_error_ok)
    {
        std::cout << "\n[FAILED] INT8 GEMM validation failed" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "\n[SUCCESS] INT8 GEMM validation passed" << std::endl;
    return EXIT_SUCCESS;
}
