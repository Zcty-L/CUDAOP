#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include "../../ptx_utils.cuh"

#define CUDA_CHECK(val) { \
    cudaError_t err = val; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d : %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

struct GmmaDescriptor {
    uint64_t desc;
    __device__ static GmmaDescriptor make(const void* ptr, int leading, int stride, int layout) {
        GmmaDescriptor d;
        uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
        d.desc = ((uint64_t)((addr >> 4) & 0x3FFF)) |
                 ((uint64_t)((leading >> 4) & 0x3FFF) << 16) |
                 ((uint64_t)((stride >> 4) & 0x3FFF) << 32) |
                 ((uint64_t)(layout & 0x3) << 62);
        return d;
    }
};

__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }

__device__ __forceinline__ void wgmma_mma_tf32_m64n128k8_ss(float* acc, uint64_t da, uint64_t db, int scale_d) {
    // Adopting Golden Sample pattern: Use predicate for scale_d
    // And using trailing -1, -1 for scale_a, scale_b as per TF32 ISA supplement
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47, "
        " %48, %49, %50, %51, %52, %53, %54, %55, "
        " %56, %57, %58, %59, %60, %61, %62, %63}, "
        "%64, %65, p, -1, -1;\n}\n"
        : "+f"(acc[0]),  "+f"(acc[1]),  "+f"(acc[2]),  "+f"(acc[3]),
          "+f"(acc[4]),  "+f"(acc[5]),  "+f"(acc[6]),  "+f"(acc[7]),
          "+f"(acc[8]),  "+f"(acc[9]),  "+f"(acc[10]), "+f"(acc[11]),
          "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15]),
          "+f"(acc[16]), "+f"(acc[17]), "+f"(acc[18]), "+f"(acc[19]),
          "+f"(acc[20]), "+f"(acc[21]), "+f"(acc[22]), "+f"(acc[23]),
          "+f"(acc[24]), "+f"(acc[25]), "+f"(acc[26]), "+f"(acc[27]),
          "+f"(acc[28]), "+f"(acc[29]), "+f"(acc[30]), "+f"(acc[31]),
          "+f"(acc[32]), "+f"(acc[33]), "+f"(acc[34]), "+f"(acc[35]),
          "+f"(acc[36]), "+f"(acc[37]), "+f"(acc[38]), "+f"(acc[39]),
          "+f"(acc[40]), "+f"(acc[41]), "+f"(acc[42]), "+f"(acc[43]),
          "+f"(acc[44]), "+f"(acc[45]), "+f"(acc[46]), "+f"(acc[47]),
          "+f"(acc[48]), "+f"(acc[49]), "+f"(acc[50]), "+f"(acc[51]),
          "+f"(acc[52]), "+f"(acc[53]), "+f"(acc[54]), "+f"(acc[55]),
          "+f"(acc[56]), "+f"(acc[57]), "+f"(acc[58]), "+f"(acc[59]),
          "+f"(acc[60]), "+f"(acc[61]), "+f"(acc[62]), "+f"(acc[63])
        : "l"(da), "l"(db), "r"(scale_d)
    );
}

__global__ __launch_bounds__(128)
void wgmma_tf32_gemm_kernel(const float* A, const float* B, float* C) {
    int tid = threadIdx.x;
    extern __shared__ float smem[];
    float* sA = smem;           // 64 * 8
    float* sB = smem + 64 * 8;  // 8 * 128

    // Load smem (Linear for now, TODO: Swizzle)
    for (int i = tid; i < 64 * 8; i += 128) sA[i] = A[i];
    for (int i = tid; i < 8 * 128; i += 128) sB[i] = B[i];
    
    __syncthreads();
    // MANDATORY FENCE after smem loads
    asm volatile("fence.proxy.async;\n" ::: "memory");
    __syncwarp();

    float acc[64];
    #pragma unroll
    for (int i = 0; i < 64; i++) acc[i] = 0.0f;

    wgmma_fence();
    uint64_t da = GmmaDescriptor::make(sA, 32, 256, 0).desc; // leading=32B, stride=256B
    uint64_t db = GmmaDescriptor::make(sB, 32, 256, 0).desc;
    wgmma_mma_tf32_m64n128k8_ss(acc, da, db, 1);
    wgmma_commit();
    wgmma_wait();

    // Write back using the derived mapping for m64n128k8
    int W = tid / 32, L = tid % 32;
    for (int k = 0; k < 16; k++) {
        for (int j = 0; j < 2; j++) {
            for (int i = 0; i < 2; i++) {
                int r = 4 * k + 2 * j + i;
                int row = (L / 4) + 8 * j + 16 * W;
                int col = (L % 4) * 2 + 8 * k + i;
                if (row < 64 && col < 128) C[row * 128 + col] = acc[r];
            }
        }
    }
}

void cpu_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) sum += A[i * K + k] * B[j * K + k]; // B is Col-major
            C[i * N + j] = sum;
        }
    }
}

int main() {
    const int M = 64, N = 128, K = 8;
    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N), h_C_ref(M * N, 0.0f);
    for (int i = 0; i < M * K; ++i) h_A[i] = (float)(i % 5) - 2.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = (float)(i % 7) - 3.0f;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * 4));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * 4));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * 4));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, M * N * 4));

    size_t smem_size = (M * K + K * N) * 4;
    CUDA_CHECK(cudaFuncSetAttribute(wgmma_tf32_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    wgmma_tf32_gemm_kernel<<<1, 128, smem_size>>>(d_A, d_B, d_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * 4, cudaMemcpyDeviceToHost));
    cpu_matmul(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);

    double max_err = 0;
    for (int i = 0; i < M * N; ++i) max_err = std::max(max_err, (double)std::abs(h_C[i] - h_C_ref[i]));
    std::cout << "Max Error: " << max_err << std::endl;
    if (max_err < 1e-1) std::cout << "Verification PASSED" << std::endl;
    else {
        std::cout << "Verification FAILED" << std::endl;
        for(int i=0; i<5; ++i) printf("C[%d]: %f, Ref: %f\n", i, h_C[i], h_C_ref[i]);
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
