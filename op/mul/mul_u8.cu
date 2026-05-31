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
