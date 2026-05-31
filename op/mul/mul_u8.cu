#include <iostream>
#include <vector>
#include <cstdint>
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
