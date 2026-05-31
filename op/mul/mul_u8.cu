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
        uint8_t val = hx[i] * xt[i]; // Inputs are 0 or 1
        ht[i] |= (val << (t_step + 1));
        if (t_step == 0) {
            h_float[i] = static_cast<T>(val);
        } else {
            if constexpr (std::is_same_v<T, float>) {
                h_float[i] += static_cast<T>(val);
            } else {
                h_float[i] = __hadd(h_float[i], __float2half(static_cast<float>(val)));
            }
        }
    }
}

// v2: Simplified vectorized version (x4 elements per thread)
// Assumes N % 4 == 0 and aligned
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

    uchar4 hx_val = *reinterpret_cast<const uchar4*>(&hx[base_idx]);
    uchar4 xt_val = *reinterpret_cast<const uchar4*>(&xt[base_idx]);
    uchar4 ht_val = *reinterpret_cast<uchar4*>(&ht[base_idx]);
    
    uint8_t v0 = hx_val.x & xt_val.x;
    uint8_t v1 = hx_val.y & xt_val.y;
    uint8_t v2 = hx_val.z & xt_val.z;
    uint8_t v3 = hx_val.w & xt_val.w;

    // Logic: ht |= (hx * xt) << (t + 1)
    ht_val.x |= (v0 << (t_step + 1));
    ht_val.y |= (v1 << (t_step + 1));
    ht_val.z |= (v2 << (t_step + 1));
    ht_val.w |= (v3 << (t_step + 1));

    *reinterpret_cast<uchar4*>(&ht[base_idx]) = ht_val;

    // Logic: if t==0, h_float = val; else h_float += val
    if (t_step == 0) {
        if constexpr (std::is_same_v<T, float>) {
            float4 f_out;
            f_out.x = static_cast<float>(v0);
            f_out.y = static_cast<float>(v1);
            f_out.z = static_cast<float>(v2);
            f_out.w = static_cast<float>(v3);
            *reinterpret_cast<float4*>(&h_float[base_idx]) = f_out;
        } else if constexpr (std::is_same_v<T, half>) {
            half2 h_val1 = __floats2half2_rn(static_cast<float>(v0), static_cast<float>(v1));
            half2 h_val2 = __floats2half2_rn(static_cast<float>(v2), static_cast<float>(v3));
            *reinterpret_cast<half2*>(&h_float[base_idx]) = h_val1;
            *reinterpret_cast<half2*>(&h_float[base_idx + 2]) = h_val2;
        }
    } else {
        if constexpr (std::is_same_v<T, float>) {
            float4 f_val = *reinterpret_cast<const float4*>(&h_float[base_idx]);
            f_val.x += static_cast<float>(v0);
            f_val.y += static_cast<float>(v1);
            f_val.z += static_cast<float>(v2);
            f_val.w += static_cast<float>(v3);
            *reinterpret_cast<float4*>(&h_float[base_idx]) = f_val;
        } else if constexpr (std::is_same_v<T, half>) {
            half2 h_val1 = *reinterpret_cast<const half2*>(&h_float[base_idx]);
            half2 h_val2 = *reinterpret_cast<const half2*>(&h_float[base_idx + 2]);
            h_val1 = __hadd2(h_val1, __floats2half2_rn(static_cast<float>(v0), static_cast<float>(v1)));
            h_val2 = __hadd2(h_val2, __floats2half2_rn(static_cast<float>(v2), static_cast<float>(v3)));
            *reinterpret_cast<half2*>(&h_float[base_idx]) = h_val1;
            *reinterpret_cast<half2*>(&h_float[base_idx + 2]) = h_val2;
        }
    }
}

// v3: Optimized version for N % 16 == 0 (x16 elements per thread)
template <typename T>
__global__ void MulU8Kernel_v3(const uint8_t * __restrict__ hx, 
                               const uint8_t * __restrict__ xt, 
                               uint8_t * __restrict__ ht, 
                               T * __restrict__ h_float, 
                               int N, int t_step)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base_idx = idx * 16;

    if (base_idx >= N) return;

    uint4 hx_vec = reinterpret_cast<const uint4*>(hx)[idx];
    uint4 xt_vec = reinterpret_cast<const uint4*>(xt)[idx];
    uint4 ht_vec = reinterpret_cast<uint4*>(ht)[idx];

    uint32_t r0 = hx_vec.x & xt_vec.x;
    uint32_t r1 = hx_vec.y & xt_vec.y;
    uint32_t r2 = hx_vec.z & xt_vec.z;
    uint32_t r3 = hx_vec.w & xt_vec.w;

    // Logic: ht |= (hx * xt) << (t + 1)
    ht_vec.x |= (r0 << (t_step + 1));
    ht_vec.y |= (r1 << (t_step + 1));
    ht_vec.z |= (r2 << (t_step + 1));
    ht_vec.w |= (r3 << (t_step + 1));
    reinterpret_cast<uint4*>(ht)[idx] = ht_vec;

    uint32_t results[4] = {r0, r1, r2, r3};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t res = results[i];
        if (t_step == 0) {
            if constexpr (std::is_same_v<T, float>) {
                float4 f_out;
                f_out.x = static_cast<float>(res & 0xFF);
                f_out.y = static_cast<float>((res >> 8) & 0xFF);
                f_out.z = static_cast<float>((res >> 16) & 0xFF);
                f_out.w = static_cast<float>((res >> 24) & 0xFF);
                reinterpret_cast<float4*>(h_float)[idx * 4 + i] = f_out;
            } else if constexpr (std::is_same_v<T, half>) {
                half2 h01 = __floats2half2_rn(static_cast<float>(res & 0xFF), static_cast<float>((res >> 8) & 0xFF));
                half2 h23 = __floats2half2_rn(static_cast<float>((res >> 16) & 0xFF), static_cast<float>((res >> 24) & 0xFF));
                reinterpret_cast<half2*>(h_float)[idx * 8 + i * 2] = h01;
                reinterpret_cast<half2*>(h_float)[idx * 8 + i * 2 + 1] = h23;
            }
        } else {
            if constexpr (std::is_same_v<T, float>) {
                float4 f_out = reinterpret_cast<const float4*>(h_float)[idx * 4 + i];
                f_out.x += static_cast<float>(res & 0xFF);
                f_out.y += static_cast<float>((res >> 8) & 0xFF);
                f_out.z += static_cast<float>((res >> 16) & 0xFF);
                f_out.w += static_cast<float>((res >> 24) & 0xFF);
                reinterpret_cast<float4*>(h_float)[idx * 4 + i] = f_out;
            } else if constexpr (std::is_same_v<T, half>) {
                half2 h01 = reinterpret_cast<const half2*>(h_float)[idx * 8 + i * 2];
                half2 h23 = reinterpret_cast<const half2*>(h_float)[idx * 8 + i * 2 + 1];
                h01 = __hadd2(h01, __floats2half2_rn(static_cast<float>(res & 0xFF), static_cast<float>((res >> 8) & 0xFF)));
                h23 = __hadd2(h23, __floats2half2_rn(static_cast<float>((res >> 16) & 0xFF), static_cast<float>((res >> 24) & 0xFF)));
                reinterpret_cast<half2*>(h_float)[idx * 8 + i * 2] = h01;
                reinterpret_cast<half2*>(h_float)[idx * 8 + i * 2 + 1] = h23;
            }
        }
    }
}

template <typename T>
void test_mul_u8_sequence(int c, int h, int w, int total_t_steps)
{
    size_t N = (size_t)c * h * w;
    printf("--- [mul_u8 Revised Logic] Shape [%d, %d, %d] N=%zu Total_T=%d, Type=%s ---\n", 
            c, h, w, N, total_t_steps, std::is_same_v<T, float> ? "float" : "half");

    size_t bytes_u8 = N * sizeof(uint8_t);
    size_t bytes_T = N * sizeof(T);

    uint8_t *h_hx_base = new uint8_t[N];
    uint8_t *h_xt_base = new uint8_t[N];
    uint8_t *h_ht_init = new uint8_t[N];
    
    srand(42);
    for (size_t i = 0; i < N; ++i) {
        h_hx_base[i] = rand() % 2; 
        h_xt_base[i] = rand() % 2; 
        h_ht_init[i] = 0; 
    }

    uint8_t *d_hx, *d_xt, *d_ht;
    T *d_hfloat;
    cudaMalloc(&d_hx, bytes_u8);
    cudaMalloc(&d_xt, bytes_u8);
    cudaMalloc(&d_ht, bytes_u8);
    cudaMalloc(&d_hfloat, bytes_T);

    cudaMemcpy(d_hx, h_hx_base, bytes_u8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_xt, h_xt_base, bytes_u8, cudaMemcpyHostToDevice);

    auto run_seq_test = [&](const char* name, auto kernel, int elements_per_thread) {
        cudaMemcpy(d_ht, h_ht_init, bytes_u8, cudaMemcpyHostToDevice);
        cudaMemset(d_hfloat, 0, bytes_T);
        
        std::vector<uint8_t> h_ht_ref(N, 0);
        std::vector<T> h_hfloat_ref(N);
        std::vector<T> h_hfloat_gpu(N);
        std::vector<uint8_t> h_ht_gpu(N);

        dim3 blockDim(256);
        dim3 gridDim(((N + elements_per_thread - 1) / elements_per_thread + blockDim.x - 1) / blockDim.x);

        float total_ms = 0;
        int total_errors = 0;

        for (int t = 0; t < total_t_steps; ++t) {
            mul_u8_cpu_ref<T>(h_hx_base, h_xt_base, h_ht_ref.data(), h_hfloat_ref.data(), N, t);

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            
            kernel<<<gridDim, blockDim>>>(d_hx, d_xt, d_ht, d_hfloat, N, t);
            
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            total_ms += ms;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            cudaMemcpy(h_ht_gpu.data(), d_ht, bytes_u8, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_hfloat_gpu.data(), d_hfloat, bytes_T, cudaMemcpyDeviceToHost);

            int step_errors = 0;
            for (size_t i = 0; i < N; ++i) {
                if (h_ht_gpu[i] != h_ht_ref[i]) step_errors++;
                float gpu_f = std::is_same_v<T, half> ? __half2float(h_hfloat_gpu[i]) : static_cast<float>(h_hfloat_gpu[i]);
                float cpu_f = std::is_same_v<T, half> ? __half2float(h_hfloat_ref[i]) : static_cast<float>(h_hfloat_ref[i]);
                if (std::abs(gpu_f - cpu_f) > 1e-5) step_errors++;
            }
            total_errors += step_errors;
        }

        printf("  %-10s: %s (total %d errors), Avg Step Time: %.6f ms\n", 
               name, (total_errors == 0 ? "PASS" : "FAIL"), total_errors, total_ms / total_t_steps);
    };

    run_seq_test("v2 (x4)", MulU8Kernel_v2<T>, 4);
    if (N % 16 == 0) {
        run_seq_test("v3 (x16)", MulU8Kernel_v3<T>, 16);
    }

    cudaFree(d_hx); cudaFree(d_xt); cudaFree(d_ht); cudaFree(d_hfloat);
    delete[] h_hx_base; delete[] h_xt_base; delete[] h_ht_init;
}

int main()
{
    printf("=== mul_u8 0/1 Input Comprehensive Benchmark (Revised Logic) ===\n");
    
    std::vector<int> channels = {96, 192, 256, 384, 512};
    std::vector<int> spatial = {10, 20, 40, 80};

    for (int c : channels) {
        for (int s : spatial) {
            test_mul_u8_sequence<float>(c, s, s, 4);
            test_mul_u8_sequence<half>(c, s, s, 4);
            printf("\n");
        }
    }
    
    printf("=== All benchmarks complete ===\n");
    return 0;
}
