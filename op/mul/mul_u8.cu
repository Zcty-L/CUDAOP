#include <iostream>
#include <vector>
#include <cstdint>
#include <type_traits>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// CPU Reference
template <typename T>
void mul_u8_cpu_ref(const uint8_t* hx, const uint8_t* xt, uint8_t* ht, T* h_float, int N, int t_step)
{
    for (int i = 0; i < N; ++i) {
        uint8_t val = (hx[i] & 1) & (xt[i] & 1);
        ht[i] |= (val << t_step);
        h_float[i] = static_cast<T>(val);
    }
}

template <typename T>
__global__ void MulU8Kernel_v2(const uint8_t * __restrict__ hx, 
                               const uint8_t * __restrict__ xt, 
                               uint8_t * __restrict__ ht, 
                               T * __restrict__ h_float, 
                               int N, int t_step)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base_idx = idx * 4;

    if (base_idx >= N) return;

    // Handle full float4/half2 vector if aligned and enough elements
    if (base_idx + 3 < N) {
        uchar4 hx_val = *reinterpret_cast<const uchar4*>(&hx[base_idx]);
        uchar4 xt_val = *reinterpret_cast<const uchar4*>(&xt[base_idx]);
        
        uchar4 ht_val = *reinterpret_cast<uchar4*>(&ht[base_idx]);
        
        uint8_t v0 = (hx_val.x & 1) & (xt_val.x & 1);
        uint8_t v1 = (hx_val.y & 1) & (xt_val.y & 1);
        uint8_t v2 = (hx_val.z & 1) & (xt_val.z & 1);
        uint8_t v3 = (hx_val.w & 1) & (xt_val.w & 1);

        ht_val.x |= (v0 << t_step);
        ht_val.y |= (v1 << t_step);
        ht_val.z |= (v2 << t_step);
        ht_val.w |= (v3 << t_step);

        *reinterpret_cast<uchar4*>(&ht[base_idx]) = ht_val;

        if constexpr (std::is_same_v<T, float>) {
            float4 f_val;
            f_val.x = static_cast<float>(v0);
            f_val.y = static_cast<float>(v1);
            f_val.z = static_cast<float>(v2);
            f_val.w = static_cast<float>(v3);
            *reinterpret_cast<float4*>(&h_float[base_idx]) = f_val;
        } else if constexpr (std::is_same_v<T, half>) {
            half2 h_val1 = __floats2half2_rn(static_cast<float>(v0), static_cast<float>(v1));
            half2 h_val2 = __floats2half2_rn(static_cast<float>(v2), static_cast<float>(v3));
            *reinterpret_cast<half2*>(&h_float[base_idx]) = h_val1;
            *reinterpret_cast<half2*>(&h_float[base_idx + 2]) = h_val2;
        }
    } else {
        // Remainder handling
        for (int i = 0; i < 4; ++i) {
            if (base_idx + i < N) {
                uint8_t v = (hx[base_idx + i] & 1) & (xt[base_idx + i] & 1);
                ht[base_idx + i] |= (v << t_step);
                h_float[base_idx + i] = static_cast<T>(v);
            }
        }
    }
}

template <typename T>
void test_mul_u8(int c, int h, int w, int t_step)
{
    size_t N = (size_t)c * h * w;
    printf("--- [mul_u8] Shape [%d, %d, %d] N=%zu T_step=%d, Type=%s ---\n", 
            c, h, w, N, t_step, std::is_same_v<T, float> ? "float" : "half");

    size_t bytes_u8 = N * sizeof(uint8_t);
    size_t bytes_T = N * sizeof(T);

    uint8_t *h_hx = new uint8_t[N];
    uint8_t *h_xt = new uint8_t[N];
    uint8_t *h_ht = new uint8_t[N];
    T *h_hfloat = new T[N];
    uint8_t *ref_ht = new uint8_t[N];
    T *ref_hfloat = new T[N];

    srand(42);
    for (size_t i = 0; i < N; ++i) {
        h_hx[i] = rand() % 256;
        h_xt[i] = rand() % 256;
        h_ht[i] = rand() % 256;
        ref_ht[i] = h_ht[i];
    }

    uint8_t *d_hx, *d_xt, *d_ht;
    T *d_hfloat;
    cudaMalloc(&d_hx, bytes_u8);
    cudaMalloc(&d_xt, bytes_u8);
    cudaMalloc(&d_ht, bytes_u8);
    cudaMalloc(&d_hfloat, bytes_T);

    cudaMemcpy(d_hx, h_hx, bytes_u8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_xt, h_xt, bytes_u8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ht, h_ht, bytes_u8, cudaMemcpyHostToDevice);

    mul_u8_cpu_ref<T>(h_hx, h_xt, ref_ht, ref_hfloat, N, t_step);

    dim3 blockDim(256);
    dim3 gridDim(((N + 3) / 4 + blockDim.x - 1) / blockDim.x);

    auto launch = [&]() {
        MulU8Kernel_v2<T><<<gridDim, blockDim>>>(d_hx, d_xt, d_ht, d_hfloat, N, t_step);
    };

    launch();
    cudaDeviceSynchronize();

    cudaMemcpy(h_ht, d_ht, bytes_u8, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_hfloat, d_hfloat, bytes_T, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (size_t i = 0; i < N; ++i) {
        if (h_ht[i] != ref_ht[i]) errors++;
        float gpu_f = std::is_same_v<T, half> ? __half2float(h_hfloat[i]) : static_cast<float>(h_hfloat[i]);
        float cpu_f = std::is_same_v<T, half> ? __half2float(ref_hfloat[i]) : static_cast<float>(ref_hfloat[i]);
        if (std::abs(gpu_f - cpu_f) > 1e-5) errors++;
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
    printf("  %s (%d errors), Avg Time: %.6f ms\n", (errors == 0 ? "PASS" : "FAIL"), errors, ms / ITERS);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_hx); cudaFree(d_xt); cudaFree(d_ht); cudaFree(d_hfloat);
    delete[] h_hx; delete[] h_xt; delete[] h_ht; delete[] h_hfloat;
    delete[] ref_ht; delete[] ref_hfloat;
}

int main()
{
    printf("=== mul_u8 kernel tests ===\n");
    test_mul_u8<float>(128, 64, 64, 3); // exact multiple of 4
    test_mul_u8<half>(128, 64, 64, 1);  // exact multiple of 4
    test_mul_u8<float>(1, 15, 7, 2);    // not multiple of 4
    test_mul_u8<half>(3, 33, 11, 4);    // not multiple of 4
    printf("=== All tests complete ===\n");
    return 0;
}
