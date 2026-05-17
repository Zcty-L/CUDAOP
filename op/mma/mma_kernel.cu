#include <cublas_v2.h>
#include <mma.h>
#include <random>
#include <cstdio>

#include "cpu/cpu_ops.h"

using namespace nvcuda;

template<uint32_t B, uint32_t M, uint32_t S>
__device__ __forceinline__ uint32_t swizzle(const uint32_t addr)
{
    constexpr auto Bmask = ((1 << B) - 1) << M;
    return ((addr >> S) & Bmask) ^ addr;
}

__device__ __forceinline__ void mma_m16n8k16_rowcol_f32f16f16f32(
        uint32_t &D0, uint32_t &D1, uint32_t &D2, uint32_t &D3,
        uint32_t &A0, uint32_t &A1, uint32_t &A2, uint32_t &A3,
        uint32_t &B0, uint32_t &B1,
        uint32_t &C0, uint32_t &C1, uint32_t &C2, uint32_t &C3
)
{
    asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13}; "
            :"=r"(D0), "=r"(D1), "=r"(D2), "=r"(D3)
            :
            "r"(A0), "r"(A1), "r"(A2), "r"(A3),
            "r"(B0), "r"(B1),
            "r"(C0), "r"(C1), "r"(C2), "r"(C3)
            );
}

template<typename T>
static void random_matrix(T *data, int m, int n, float low = 0.0f, float high = 1.0f)
{
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(low, high);

    for (int i = 0; i < m * n; i++)
    {
        data[i] = static_cast<T>(dist(gen));
    }
}

// gemm  128x32 @ 32x128
__global__ void mma_kernel_1_0(__half *A, __half *B, float *C, const int M, const int N, const int K)
{
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    __shared__ __align__(16)
    char smem[128 * 32 * sizeof(__half) + 128 * 32 * sizeof(__half)];

    __half *A_smem_ptr = reinterpret_cast<__half *>(smem);
    __half *B_smem_ptr = A_smem_ptr + 128 * 32;

    __half *A_sts_ptr = A_smem_ptr + warp_id * 32 * 32 + swizzle<2, 3, 3>(lane_id * 8);
    __half *B_sts_ptr = B_smem_ptr + warp_id * 32 * 32 + swizzle<2, 3, 3>(lane_id * 8);

    __half *A_lds_ptr = A_smem_ptr + warp_id * 32 * 32 + swizzle<2, 3, 3>(lane_id % 16 * 16 + lane_id / 16 * 8);
    __half *B_lds_ptr = B_smem_ptr + swizzle<2, 3, 3>((lane_id % 8) * 16 + ((lane_id / 8) % 2) * 8 + (lane_id / 16) * 16 * 8);

    __half *A_gmem_ptr = A + warp_id * 32 * K + lane_id / 2 * K + lane_id % 2 * 8;
    __half *B_gmem_ptr = B + warp_id * 32 * K + lane_id / 2 * K + lane_id % 2 * 8;

    // A: ldg --> sts
#pragma unroll
    for (int i = 0; i < 2; i++) // 32/16 = 2
    {
#pragma unroll
        for (int j = 0; j < 2; j++) // 32/16 = 2
        {
            uint32_t A_sts_addr = __cvta_generic_to_shared(A_sts_ptr + i * 16 * 32 + j * 16 * 16);
            asm volatile(
                    "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                    : : "r"(A_sts_addr), "l"(A_gmem_ptr + i * 16 * K + j * 16)
                    );
        }
    }

    // B: ldg --> sts
#pragma unroll
    for (int i = 0; i < 2; i++) // 32/16 = 2
    {
#pragma unroll
        for (int j = 0; j < 2; j++) // 32/16 = 2
        {
            uint32_t B_sts_addr = __cvta_generic_to_shared(B_sts_ptr + i * 16 * 32 + j * 16 * 16);
            asm volatile(
                    "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                    : : "r"(B_sts_addr), "l"(B_gmem_ptr + i * 16 * K + j * 16)
                    );
        }
    }

    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();

    uint32_t RA[4 * 4];
    uint32_t RB[4 * 2];
    uint32_t RC[8];

    // lds A   32x32 = 16x16 x4
    uint32_t A_lds_addr;

    A_lds_addr = __cvta_generic_to_shared(A_lds_ptr + 0 * 16 * 32 + 0 * 16 * 16);
    asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(RA[0 * 4 + 0]), "=r"(RA[0 * 4 + 1]), "=r"(RA[0 * 4 + 2]), "=r"(RA[0 * 4 + 3])
            : "r"(A_lds_addr)
            );

    A_lds_addr = __cvta_generic_to_shared(A_lds_ptr + 0 * 16 * 32 + 1 * 16 * 16);
    asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(RA[1 * 4 + 0]), "=r"(RA[1 * 4 + 1]), "=r"(RA[1 * 4 + 2]), "=r"(RA[1 * 4 + 3])
            : "r"(A_lds_addr)
            );

    A_lds_addr = __cvta_generic_to_shared(A_lds_ptr + 1 * 16 * 32 + 0 * 16 * 16);
    asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(RA[2 * 4 + 0]), "=r"(RA[2 * 4 + 1]), "=r"(RA[2 * 4 + 2]), "=r"(RA[2 * 4 + 3])
            : "r"(A_lds_addr)
            );

    A_lds_addr = __cvta_generic_to_shared(A_lds_ptr + 1 * 16 * 32 + 1 * 16 * 16);
    asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(RA[3 * 4 + 0]), "=r"(RA[3 * 4 + 1]), "=r"(RA[3 * 4 + 2]), "=r"(RA[3 * 4 + 3])
            : "r"(A_lds_addr)
            );

    for (int n_ = 0; n_ < N; n_ += 16)
    {
        uint32_t B_lds_addr;

        B_lds_addr = __cvta_generic_to_shared(B_lds_ptr + n_ * 32 + 0 * 16 * 16);
        asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                : "=r"(RB[0]), "=r"(RB[1]), "=r"(RB[2]), "=r"(RB[3])
                : "r"(B_lds_addr)
                );

        B_lds_addr = __cvta_generic_to_shared(B_lds_ptr + n_ * 32 + 1 * 16 * 16);
        asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                : "=r"(RB[1 * 4 + 0]), "=r"(RB[1 * 4 + 1]), "=r"(RB[1 * 4 + 2]), "=r"(RB[1 * 4 + 3])
                : "r"(B_lds_addr)
                );

#pragma unroll
        for (int i = 0; i < 2; i++)
        {
#pragma unroll
            for (int ii = 0; ii < 8; ii++)
            {
                RC[ii] = 0;
            }

            mma_m16n8k16_rowcol_f32f16f16f32(
                    RC[0], RC[1], RC[2], RC[3],
                    RA[i * 8 + 0], RA[i * 8 + 1], RA[i * 8 + 2], RA[i * 8 + 3],
                    RB[0], RB[1],
                    RC[0], RC[1], RC[2], RC[3]
            );
            mma_m16n8k16_rowcol_f32f16f16f32(
                    RC[4], RC[5], RC[6], RC[7],
                    RA[i * 8 + 0], RA[i * 8 + 1], RA[i * 8 + 2], RA[i * 8 + 3],
                    RB[2], RB[3],
                    RC[4], RC[5], RC[6], RC[7]
            );

            mma_m16n8k16_rowcol_f32f16f16f32(
                    RC[0], RC[1], RC[2], RC[3],
                    RA[i * 8 + 4], RA[i * 8 + 5], RA[i * 8 + 6], RA[i * 8 + 7],
                    RB[4], RB[5],
                    RC[0], RC[1], RC[2], RC[3]
            );
            mma_m16n8k16_rowcol_f32f16f16f32(
                    RC[4], RC[5], RC[6], RC[7],
                    RA[i * 8 + 4], RA[i * 8 + 5], RA[i * 8 + 6], RA[i * 8 + 7],
                    RB[6], RB[7],
                    RC[4], RC[5], RC[6], RC[7]
            );

            const int ldc = N;
            uint32_t *c_stg_ptr = reinterpret_cast<uint32_t *>(C + (warp_id * 32 + i * 16) * ldc + n_);

            uint32_t c_addr_0 = lane_id / 4 * ldc + lane_id % 4 * 2;
            uint32_t c_addr_1 = (lane_id / 4 + 8) * ldc + lane_id % 4 * 2;
            uint32_t c_addr_2 = lane_id / 4 * ldc + lane_id % 4 * 2 + 8;
            uint32_t c_addr_3 = (lane_id / 4 + 8) * ldc + lane_id % 4 * 2 + 8;

            c_stg_ptr[c_addr_0] = RC[0], c_stg_ptr[c_addr_0 + 1] = RC[1];
            c_stg_ptr[c_addr_1] = RC[2], c_stg_ptr[c_addr_1 + 1] = RC[3];
            c_stg_ptr[c_addr_2] = RC[4], c_stg_ptr[c_addr_2 + 1] = RC[5];
            c_stg_ptr[c_addr_3] = RC[6], c_stg_ptr[c_addr_3 + 1] = RC[7];
        }
    }
}


static void run_test(const char *name, void (*kernel)(__half*, __half*, float*, int, int, int),
                     float *matrixA, float *matrixB, float *matrixC, float *matrixC_cpu,
                     __half *matrixA_h, __half *matrixB_h,
                     __half *matrixA_device_h, __half *matrixB_device_h, float *matrixC_device,
                     int M, int N, int K)
{
    cudaMemcpy(matrixA_device_h, matrixA_h, M * K * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(matrixB_device_h, matrixB_h, K * N * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemset(matrixC_device, 0, M * N * sizeof(float));

    kernel<<<1, 32 * 4>>>(matrixA_device_h, matrixB_device_h, matrixC_device, M, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(matrixC_cpu, matrixC_device, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("=================== %s ===================\n", name);
    gemm_mma_cpu(matrixA, matrixB, matrixC, M, N, K);

    printf("================ start verify ================\n");
    bool passed = true;
    for (int i = 0; i < M * N; i++)
    {
        if (fabsf(matrixC_cpu[i] - matrixC[i]) > 0.1f)
        {
            printf("Error: position: %d, gpu:%.4f, cpu:%.4f\n", i, matrixC_cpu[i], matrixC[i]);
            passed = false;
            break;
        }
    }
    if (passed)
    {
        printf("All tests PASSED!\n");
    }

    printf("First 4 results:\n");
    printf("  GPU: %.4f %.4f %.4f %.4f\n", matrixC_cpu[0], matrixC_cpu[1], matrixC_cpu[2], matrixC_cpu[3]);
    printf("  CPU: %.4f %.4f %.4f %.4f\n", matrixC[0], matrixC[1], matrixC[2], matrixC[3]);
}

int main()
{
    const int M = 128;
    const int N = 128;
    const int K = 32;

    float *matrixA, *matrixB, *matrixC, *matrixC_cpu;
    __half *matrixA_h, *matrixB_h;
    cudaMallocHost((void **) &matrixA, M * K * sizeof(float));
    cudaMallocHost((void **) &matrixB, K * N * sizeof(float));
    cudaMallocHost((void **) &matrixC, M * N * sizeof(float));
    cudaMallocHost((void **) &matrixC_cpu, M * N * sizeof(float));
    cudaMallocHost((void **) &matrixA_h, M * K * sizeof(__half));
    cudaMallocHost((void **) &matrixB_h, K * N * sizeof(__half));

    float *matrixA_device, *matrixB_device, *matrixC_device;
    __half *matrixA_device_h, *matrixB_device_h;
    cudaMalloc((void **) &matrixA_device, M * K * sizeof(float));
    cudaMalloc((void **) &matrixB_device, K * N * sizeof(float));
    cudaMalloc((void **) &matrixC_device, M * N * sizeof(float));
    cudaMalloc((void **) &matrixA_device_h, M * K * sizeof(__half));
    cudaMalloc((void **) &matrixB_device_h, K * N * sizeof(__half));

    random_matrix<float>(matrixA, M, K, -1.0f, 1.0f);
    for (int i = 0; i < M * K; i++)
    {
        matrixA_h[i] = __float2half_rn(matrixA[i]);
    }

    random_matrix<float>(matrixB, N, K, -1.0f, 1.0f);
    for (int i = 0; i < K * N; i++)
    {
        matrixB_h[i] = __float2half_rn(matrixB[i]);
    }

    cudaMemcpy(matrixA_device, matrixA, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(matrixB_device, matrixB, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(matrixA_device_h, matrixA_h, M * K * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(matrixB_device_h, matrixB_h, K * N * sizeof(__half), cudaMemcpyHostToDevice);

    run_test("mma_kernel_1_0", mma_kernel_1_0,
             matrixA, matrixB, matrixC, matrixC_cpu,
             matrixA_h, matrixB_h,
             matrixA_device_h, matrixB_device_h, matrixC_device,
             M, N, K);

    run_test("mma_kernel_stmatrix", mma_kernel_stmatrix,
             matrixA, matrixB, matrixC, matrixC_cpu,
             matrixA_h, matrixB_h,
             matrixA_device_h, matrixB_device_h, matrixC_device,
             M, N, K);

    cudaFree(matrixA_device);
    cudaFree(matrixB_device);
    cudaFree(matrixC_device);
    cudaFree(matrixA_device_h);
    cudaFree(matrixB_device_h);

    cudaFreeHost(matrixA);
    cudaFreeHost(matrixB);
    cudaFreeHost(matrixC);
    cudaFreeHost(matrixC_cpu);
    cudaFreeHost(matrixA_h);
    cudaFreeHost(matrixB_h);

    return 0;
}
