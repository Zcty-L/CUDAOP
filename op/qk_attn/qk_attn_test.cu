#include <iostream>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// --- Kernels from qk_attn.cu ---

__global__ void QKAttnKernel(const int *q, const int *k, int *output, int t, int h, int d, int n)
{
    int n_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int h_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int t_idx = blockIdx.z;

    if (n_idx >= n || h_idx >= h || t_idx >= t)
        return;

    int sum = 0;
    int base_idx = t_idx * h * d * n + h_idx * d * n + n_idx;

    for (int d_idx = 0; d_idx < d; ++d_idx)
    {
        sum += q[base_idx + d_idx * n];

        if (sum > 1)
        {
            break;
        }
    }

    if (sum == 0)
    {
        for (int d_idx = 0; d_idx < d; ++d_idx)
        {
            output[base_idx + d_idx * n] = 0;
        }
    }
    else
    {
        for (int d_idx = 0; d_idx < d; ++d_idx)
        {
            output[base_idx + d_idx * n] = k[base_idx + d_idx * n];
        }
    }
}

__global__ void QKAttnUint16Kernel(
        const uint16_t *q, const uint16_t *k, uint16_t *output, int t, int h, int d, int n)
{
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int h_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int t_idx = blockIdx.z;

    const int n0 = pair_idx * 2;
    if (n0 >= n || h_idx >= h || t_idx >= t)
        return;

    const int n_idx = n0;
    const bool has_pair = (n0 + 1) < n;

    ushort2 sum = {0, 0};
    int base_idx = t_idx * h * d * n + h_idx * d * n + n_idx;

    for (int d_idx = 0; d_idx < d; ++d_idx)
    {
        if (sum.x == 0)
        {
            sum.x += q[base_idx + d_idx * n];
        }
        if (has_pair && sum.y == 0)
        {
            sum.y += q[base_idx + d_idx * n + 1];
        }
        if (sum.x > 0 && (!has_pair || sum.y > 0))
        {
            break;
        }
    }

    if (sum.x == 0)
    {
        for (int d_idx = 0; d_idx < d; ++d_idx)
        {
            output[base_idx + d_idx * n] = 0;
        }
    }
    else
    {
        for (int d_idx = 0; d_idx < d; ++d_idx)
        {
            output[base_idx + d_idx * n] = k[base_idx + d_idx * n];
        }
    }
    if (has_pair)
    {
        if (sum.y == 0)
        {
            for (int d_idx = 0; d_idx < d; ++d_idx)
            {
                output[base_idx + d_idx * n + 1] = 0;
            }
        }
        else
        {
            for (int d_idx = 0; d_idx < d; ++d_idx)
            {
                output[base_idx + d_idx * n + 1] = k[base_idx + d_idx * n + 1];
            }
        }
    }
}

// --- CPU Reference ---

template<typename T>
void qk_attn_cpu_ref(const T* q, const T* k, T* output, int t, int h, int d, int n)
{
    for (int ti = 0; ti < t; ++ti) {
        for (int hi = 0; hi < h; ++hi) {
            for (int ni = 0; ni < n; ++ni) {
                int sum = 0;
                int base_idx = ti * h * d * n + hi * d * n + ni;
                
                // q_sum = sum(q, dim=-2)
                for (int di = 0; di < d; ++di) {
                    sum += (int)q[base_idx + di * n];
                }
                
                // attn = q_sum > 1 ? 1 : 0
                int attn = (sum > 1) ? 1 : 0;
                
                // output = attn * k
                for (int di = 0; di < d; ++di) {
                    output[base_idx + di * n] = attn ? k[base_idx + di * n] : (T)0;
                }
            }
        }
    }
}

// --- Test Infrastructure ---

template<typename T>
void test_qk_attn(int t, int h, int d, int n, const char* label)
{
    printf("--- [%s] T=%d H=%d C(d)=%d N=%d ---\n", label, t, h, d, n);
    
    size_t numel = (size_t)t * h * d * n;
    size_t bytes = numel * sizeof(T);
    
    T *h_q = new T[numel];
    T *h_k = new T[numel];
    T *h_out = new T[numel];
    T *h_ref = new T[numel];
    
    srand(42);
    for (size_t i = 0; i < numel; ++i) {
        h_q[i] = (T)(rand() % 3); // 0, 1, 2 for spiking activity
        h_k[i] = (T)(rand() % 10);
    }
    
    T *d_q, *d_k, *d_out;
    cudaMalloc(&d_q, bytes);
    cudaMalloc(&d_k, bytes);
    cudaMalloc(&d_out, bytes);
    
    cudaMemcpy(d_q, h_q, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k, bytes, cudaMemcpyHostToDevice);
    
    dim3 blockDim(32, 8);
    
    if (sizeof(T) == sizeof(int)) {
        dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (h + blockDim.y - 1) / blockDim.y, t);
        QKAttnKernel<<<(dim3)gridDim, blockDim>>>((const int*)d_q, (const int*)d_k, (int*)d_out, t, h, d, n);
    } else {
        int nPairs = (n + 1) / 2;
        dim3 gridDim((nPairs + blockDim.x - 1) / blockDim.x, (h + blockDim.y - 1) / blockDim.y, t);
        QKAttnUint16Kernel<<<gridDim, blockDim>>>((const uint16_t*)d_q, (const uint16_t*)d_k, (uint16_t*)d_out, t, h, d, n);
    }
    
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    
    qk_attn_cpu_ref(h_q, h_k, h_ref, t, h, d, n);
    
    int errors = 0;
    for (size_t i = 0; i < numel; ++i) {
        if ((int)h_out[i] != (int)h_ref[i]) {
            if (errors < 5) printf("  Error at [%zu]: gpu=%d, cpu=%d\n", i, (int)h_out[i], (int)h_ref[i]);
            errors++;
        }
    }
    
    // Benchmark
    constexpr int WARMUP = 10;
    constexpr int ITERS = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < ITERS + WARMUP; ++i) {
        if (i == WARMUP) cudaEventRecord(start);
        if (sizeof(T) == sizeof(int)) {
            dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (h + blockDim.y - 1) / blockDim.y, t);
            QKAttnKernel<<<gridDim, blockDim>>>((const int*)d_q, (const int*)d_k, (int*)d_out, t, h, d, n);
        } else {
            int nPairs = (n + 1) / 2;
            dim3 gridDim((nPairs + blockDim.x - 1) / blockDim.x, (h + blockDim.y - 1) / blockDim.y, t);
            QKAttnUint16Kernel<<<gridDim, blockDim>>>((const uint16_t*)d_q, (const uint16_t*)d_k, (uint16_t*)d_out, t, h, d, n);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("  Result: %s (%d errors), Avg Time: %.6f ms\n", (errors == 0 ? "PASS" : "FAIL"), errors, ms / ITERS);
    
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_out);
    delete[] h_q;
    delete[] h_k;
    delete[] h_out;
    delete[] h_ref;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main()
{
    printf("=== QKAttn Kernel Tests ===\n");
    
    // int32 tests
    test_qk_attn<int>(4, 8, 64, 128, "INT32_Small");
    test_qk_attn<int>(1, 16, 128, 1024, "INT32_Large");
    
    // uint16 tests
    test_qk_attn<uint16_t>(4, 8, 64, 128, "UINT16_Small");
    test_qk_attn<uint16_t>(1, 16, 128, 1024, "UINT16_Large");
    test_qk_attn<uint16_t>(1, 1, 16, 7, "UINT16_Odd_N");

    printf("=== All tests complete ===\n");
    return 0;
}