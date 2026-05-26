#include <iostream>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include <type_traits>

/**
 * QKAttnU8Kernel: Processes bit-packed spikes.
 * 
 * q, k, output dimensions: [H, D, N] (each element is uint8_t containing T steps)
 */
__global__ void QKAttnU8Kernel(const uint8_t * __restrict__ q, const uint8_t * __restrict__ k, uint8_t * __restrict__ output, int h, int d, int n)
{
    int n_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int h_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (n_idx >= n || h_idx >= h)
        return;

    uint8_t has_one = 0;
    uint8_t has_two = 0;
    
    int base_idx = h_idx * d * n + n_idx;

    for (int d_idx = 0; d_idx < d; ++d_idx)
    {
        uint8_t q_val = q[base_idx + d_idx * n];
        has_two |= (has_one & q_val);
        has_one |= q_val;
    }

    uint8_t attn_mask = has_two;

    for (int d_idx = 0; d_idx < d; ++d_idx)
    {
        output[base_idx + d_idx * n] = k[base_idx + d_idx * n] & attn_mask;
    }
}

/**
 * QKAttnU8Kernel_Break: Optimized version with early exit.
 * If all time steps in t_mask have already seen >= 2 spikes, we stop traversing D.
 */
__global__ void QKAttnU8Kernel_Break(const uint8_t * __restrict__ q, const uint8_t * __restrict__ k, uint8_t * __restrict__ output, int h, int d, int n, uint8_t t_mask)
{
    int n_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int h_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (n_idx >= n || h_idx >= h)
        return;

    uint8_t has_one = 0;
    uint8_t has_two = 0;
    
    int base_idx = h_idx * d * n + n_idx;

    for (int d_idx = 0; d_idx < d; ++d_idx)
    {
        uint8_t q_val = q[base_idx + d_idx * n];
        has_two |= (has_one & q_val);
        has_one |= q_val;

        // Early exit optimization
        if (has_two == t_mask) break;
    }

    uint8_t attn_mask = has_two;

    for (int d_idx = 0; d_idx < d; ++d_idx)
    {
        output[base_idx + d_idx * n] = k[base_idx + d_idx * n] & attn_mask;
    }
}

/**
 * QKAttnU8Kernel_v2: Vectorized version.
 * Each thread handles 4 'n' indices at once.
 */
__global__ void QKAttnU8Kernel_v2(const uint8_t * __restrict__ q, const uint8_t * __restrict__ k, uint8_t * __restrict__ output, int h, int d, int n)
{
    int n_base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int h_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (n_base >= n || h_idx >= h)
        return;

    uint32_t has_one = 0;
    uint32_t has_two = 0;
    
    for (int d_idx = 0; d_idx < d; ++d_idx)
    {
        uint32_t q_val = 0;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            if (n_base + i < n) {
                q_val |= ((uint32_t)q[h_idx * d * n + d_idx * n + n_base + i] << (i * 8));
            }
        }

        has_two |= (has_one & q_val);
        has_one |= q_val;
    }

    uint32_t attn_mask = has_two;

    for (int d_idx = 0; d_idx < d; ++d_idx)
    {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            if (n_base + i < n) {
                uint8_t m = (uint8_t)((attn_mask >> (i * 8)) & 0xFF);
                output[h_idx * d * n + d_idx * n + n_base + i] = k[h_idx * d * n + d_idx * n + n_base + i] & m;
            }
        }
    }
}

// CPU Reference
void qk_attn_u8_cpu_ref(const uint8_t* q, const uint8_t* k, uint8_t* output, int h, int d, int n, int t_steps)
{
    for (int hi = 0; hi < h; ++hi) {
        for (int ni = 0; ni < n; ++ni) {
            uint8_t attn_mask = 0;
            for (int t = 0; t < t_steps; ++t) {
                int count = 0;
                for (int di = 0; di < d; ++di) {
                    uint8_t val = q[hi * d * n + di * n + ni];
                    if ((val >> t) & 1) count++;
                }
                if (count > 1) attn_mask |= (1 << t);
            }
            
            for (int di = 0; di < d; ++di) {
                int idx = hi * d * n + di * n + ni;
                output[idx] = k[idx] & attn_mask;
            }
        }
    }
}

void test_qk_attn_u8(int h, int d, int n, int t_steps)
{
    printf("--- [U8_Packed] H=%d C(d)=%d N=%d T=%d ---\n", h, d, n, t_steps);
    
    size_t numel = (size_t)h * d * n;
    size_t bytes = numel * sizeof(uint8_t);
    
    uint8_t *h_q = new uint8_t[numel];
    uint8_t *h_k = new uint8_t[numel];
    uint8_t *h_out = new uint8_t[numel];
    uint8_t *h_ref = new uint8_t[numel];
    
    srand(42);
    uint8_t t_mask = (1 << t_steps) - 1;
    for (size_t i = 0; i < numel; ++i) {
        uint8_t val = 0;
        for (int t = 0; t < t_steps; t++) {
            if ((rand() % 100) < 10) val |= (1 << t); // 10% spike density
        }
        h_q[i] = val;
        h_k[i] = (uint8_t)(rand() & t_mask);
    }
    
    uint8_t *d_q, *d_k, *d_out;
    cudaMalloc(&d_q, bytes);
    cudaMalloc(&d_k, bytes);
    cudaMalloc(&d_out, bytes);
    
    cudaMemcpy(d_q, h_q, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k, bytes, cudaMemcpyHostToDevice);
    
    qk_attn_u8_cpu_ref(h_q, h_k, h_ref, h, d, n, t_steps);

    auto run_kernel = [&](const char* name, auto func, int elements_per_thread) {
        cudaMemset(d_out, 0, bytes);
        
        dim3 blockDim(32, 8);
        dim3 gridDim(((n + elements_per_thread - 1) / elements_per_thread + blockDim.x - 1) / blockDim.x, 
                     (h + blockDim.y - 1) / blockDim.y);
        
        auto launch = [&]() {
            if constexpr (std::is_same_v<decltype(func), void(*)(const uint8_t*, const uint8_t*, uint8_t*, int, int, int)>) {
                func<<<gridDim, blockDim>>>(d_q, d_k, d_out, h, d, n);
            } else {
                func<<<gridDim, blockDim>>>(d_q, d_k, d_out, h, d, n, t_mask);
            }
        };

        launch();
        cudaDeviceSynchronize();
        cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

        int errors = 0;
        for (size_t i = 0; i < numel; ++i) {
            if (h_out[i] != h_ref[i]) {
                errors++;
            }
        }

        constexpr int WARMUP = 10;
        constexpr int ITERS = 100;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        for (int i = 0; i < WARMUP; ++i) launch();
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < ITERS; ++i) launch();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("  %-10s: %s (%d errors), Avg Time: %.6f ms\n", name, (errors == 0 ? "PASS" : "FAIL"), errors, ms / ITERS);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    };

    run_kernel("v1", QKAttnU8Kernel, 1);
    run_kernel("Break", QKAttnU8Kernel_Break, 1);
    run_kernel("v2 (x4)", QKAttnU8Kernel_v2, 4);
    
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_out);
    delete[] h_q;
    delete[] h_k;
    delete[] h_out;
    delete[] h_ref;
}

int main()
{
    printf("=== QKAttn U8 Packed Spikes Tests ===\n");
    
    test_qk_attn_u8(8, 64, 128, 4);
    test_qk_attn_u8(16, 128, 1024, 4);
    test_qk_attn_u8(1, 16, 7, 4);
    test_qk_attn_u8(32, 256, 512, 8);

    printf("=== All tests complete ===\n");
    return 0;
}