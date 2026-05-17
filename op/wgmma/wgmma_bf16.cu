#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

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

__device__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }

__device__ __forceinline__ void wgmma_fence_operand(float& reg) {
    asm volatile("" : "+f"(reg) :: "memory");
}

__device__ void wgmma_m64n64k16_bf16(float* acc, uint64_t da, uint64_t db, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
        "%32,%33,p,1,1,0,0;\n}\n"
        : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3]),"+f"(acc[4]),"+f"(acc[5]),"+f"(acc[6]),"+f"(acc[7]),
          "+f"(acc[8]),"+f"(acc[9]),"+f"(acc[10]),"+f"(acc[11]),"+f"(acc[12]),"+f"(acc[13]),"+f"(acc[14]),"+f"(acc[15]),
          "+f"(acc[16]),"+f"(acc[17]),"+f"(acc[18]),"+f"(acc[19]),"+f"(acc[20]),"+f"(acc[21]),"+f"(acc[22]),"+f"(acc[23]),
          "+f"(acc[24]),"+f"(acc[25]),"+f"(acc[26]), "+f"(acc[27]),"+f"(acc[28]),"+f"(acc[29]),"+f"(acc[30]),"+f"(acc[31])
        : "l"(da), "l"(db), "r"(scale_d));
}

// Output register to matrix coordinate mapping for m64n64
__device__ void get_coord(int tid, int reg, int& row, int& col) {
    int t0 = tid % 4, t1 = (tid / 4) % 8, t2 = tid / 32;
    int r0 = reg % 2, r1 = (reg / 2) % 2, r2 = reg / 4;
    int lin = t0 * 128 + t1 * 1 + t2 * 16 + r0 * 64 + r1 * 8 + r2 * 512;
    row = lin % 64; 
    col = lin / 64;
}

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 16;
constexpr int WGMMA_STRIDE = 8 * BLOCK_K * sizeof(__nv_bfloat16); // 256 bytes

__global__ __launch_bounds__(128)
void wgmma_bf16_gemm_kernel(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C, int M, int N, int K) {
    __shared__ __align__(1024) __nv_bfloat16 sA[BLOCK_M * BLOCK_K];
    __shared__ __align__(1024) __nv_bfloat16 sB[BLOCK_N * BLOCK_K];
    
    int tid = threadIdx.x;
    
    // Load A[64][16] and B[16][64] (B is stored as B_t[64][16] for WGMMA)
    for (int i = tid; i < BLOCK_M * BLOCK_K; i += 128) sA[i] = A[i];
    for (int i = tid; i < BLOCK_N * BLOCK_K; i += 128) sB[i] = B[i];
    
    __syncthreads();
    asm volatile("fence.proxy.async;\n" ::: "memory");
    __syncwarp();

    float acc[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) acc[i] = 0.0f;

    wgmma_fence();
    // Layout 0: INTERLEAVE (None swizzle)
    uint64_t da = GmmaDescriptor::make(sA, 32, 256, 0).desc; // leading=16*2=32B, stride=8*16*2=256B
    uint64_t db = GmmaDescriptor::make(sB, 32, 256, 0).desc;
    wgmma_m64n64k16_bf16(acc, da, db, 1);
    wgmma_commit();
    
    #pragma unroll
    for (int i = 0; i < 32; i++) wgmma_fence_operand(acc[i]);
    wgmma_wait();

    // Write back
    #pragma unroll
    for (int r = 0; r < 32; r++) {
        int lm, ln;
        get_coord(tid, r, lm, ln);
        if (lm < M && ln < N) {
            C[lm * N + ln] = acc[r];
        }
    }
}

int main() {
    const int M = 64, N = 64, K = 16;
    std::vector<__nv_bfloat16> h_A(M * K), h_B(K * N);
    std::vector<float> h_C(M * N), h_C_ref(M * N, 0.0f);

    for (int i = 0; i < M * K; ++i) h_A[i] = __float2bfloat16(1.0f);
    for (int i = 0; i < K * N; ++i) h_B[i] = __float2bfloat16(1.0f);

    __nv_bfloat16 *d_A, *d_B;
    float *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * 2));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * 2));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * 4));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, M * N * 4));

    size_t smem = (BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K) * 2;
    CUDA_CHECK(cudaFuncSetAttribute(wgmma_bf16_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    wgmma_bf16_gemm_kernel<<<1, 128, smem>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * 4, cudaMemcpyDeviceToHost));

    std::cout << "Top-left result: " << h_C[0] << " (Expected: 16.0)" << std::endl;
    if (std::abs(h_C[0] - 16.0f) < 0.1f) std::cout << "BF16 PASSED" << std::endl;
    else std::cout << "BF16 FAILED" << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
