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

__device__ __forceinline__ void wgmma_mma_tf32_m64n128k8_reg_a(float* accum, float* a_reg, uint64_t descB) {
    // Exact PTX ISA compliance for m64n128k8.f32.tf32.tf32
    // d: 64 regs, a: 4 regs, b-desc: 1 reg, scale-d: imm, scale-a: imm, scale-b: imm
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47, "
        " %48, %49, %50, %51, %52, %53, %54, %55, "
        " %56, %57, %58, %59, %60, %61, %62, %63}, "
        "{%64, %65, %66, %67}, "
        "%68, 1, 1, 1;\n"
        : "+f"(accum[0]),  "+f"(accum[1]),  "+f"(accum[2]),  "+f"(accum[3]),
          "+f"(accum[4]),  "+f"(accum[5]),  "+f"(accum[6]),  "+f"(accum[7]),
          "+f"(accum[8]),  "+f"(accum[9]),  "+f"(accum[10]), "+f"(accum[11]),
          "+f"(accum[12]), "+f"(accum[13]), "+f"(accum[14]), "+f"(accum[15]),
          "+f"(accum[16]), "+f"(accum[17]), "+f"(accum[18]), "+f"(accum[19]),
          "+f"(accum[20]), "+f"(accum[21]), "+f"(accum[22]), "+f"(accum[23]),
          "+f"(accum[24]), "+f"(accum[25]), "+f"(accum[26]), "+f"(accum[27]),
          "+f"(accum[28]), "+f"(accum[29]), "+f"(accum[30]), "+f"(accum[31]),
          "+f"(accum[32]), "+f"(accum[33]), "+f"(accum[34]), "+f"(accum[35]),
          "+f"(accum[36]), "+f"(accum[37]), "+f"(accum[38]), "+f"(accum[39]),
          "+f"(accum[40]), "+f"(accum[41]), "+f"(accum[42]), "+f"(accum[43]),
          "+f"(accum[44]), "+f"(accum[45]), "+f"(accum[46]), "+f"(accum[47]),
          "+f"(accum[48]), "+f"(accum[49]), "+f"(accum[50]), "+f"(accum[51]),
          "+f"(accum[52]), "+f"(accum[53]), "+f"(accum[54]), "+f"(accum[55]),
          "+f"(accum[56]), "+f"(accum[57]), "+f"(accum[58]), "+f"(accum[59]),
          "+f"(accum[60]), "+f"(accum[61]), "+f"(accum[62]), "+f"(accum[63])
        : "f"(a_reg[0]), "f"(a_reg[1]), "f"(a_reg[2]), "f"(a_reg[3]),
          "l"(descB)
    );
}

__global__ void wgmma_tf32_reg_kernel(const float* A, const float* B, float* C) {
    int T = threadIdx.x;
    
    // Matrix A in Registers (64x8)
    float a_reg[4];
    int m_base = (T % 8) + 8 * (T / 16);
    int k_base = 4 * ((T % 16) / 8);
    for(int i=0; i<4; ++i) a_reg[i] = A[m_base * 8 + k_base + i];

    // Matrix B in Shared Memory (8x128)
    extern __shared__ float smem_b[];
    for (int i = T; i < 8 * 128; i += 128) smem_b[i] = B[i];
    __syncthreads();

    float accum[64] = {0.0f};
    
    // Simple descriptor for B (LBO=32, SBO=256)
    uint32_t smem_b_ptr = ptx::smem_u32addr(smem_b);
    uint64_t descB = 0;
    descB |= (uint64_t)(smem_b_ptr >> 4) << 0;
    descB |= (uint64_t)(32 >> 4) << 16;
    descB |= (uint64_t)(256 >> 4) << 32;

    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    wgmma_mma_tf32_m64n128k8_reg_a(accum, a_reg, descB);
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    __syncthreads();

    // Mapping Formula:
    // m = floor(L / 4) + 8*j + 16*W
    // n = (L % 4) * 2 + 8*k + i
    // reg = 4*k + 2*j + i
    int W = T / 32;
    int L = T % 32;
    for (int k = 0; k < 16; k++) {
        for (int j = 0; j < 2; j++) {
            for (int i = 0; i < 2; i++) {
                int reg_idx = 4 * k + 2 * j + i;
                int row = (L / 4) + 8 * j + 16 * W;
                int col = (L % 4) * 2 + 8 * k + i;
                if (row < 64 && col < 128) C[row * 128 + col] = accum[reg_idx];
            }
        }
    }
}

int main() {
    const int M = 64, N = 128, K = 8;
    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 1.0f);
    std::vector<float> h_C(M * N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, M * N * sizeof(float));

    wgmma_tf32_reg_kernel<<<1, 128, K * N * sizeof(float)>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Result at (0,0): " << h_C[0] << " (Expected: 8.0)" << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
