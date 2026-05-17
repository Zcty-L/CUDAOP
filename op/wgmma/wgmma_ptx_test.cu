#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ void wgmma_try_packed_a(float* accum, uint4 a_packed, uint64_t descB) {
    // VARIANT: Using a single packed 128-bit register for A
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
        "%64, %65, 1, 1, 1;\n"
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
        : "r"(a_packed.x), // Note: PTX usually treats uint4 as a sequence of regs
          "l"(descB)
    );
}

__global__ void test_kernel(const float* A, const float* B, float* C) {
    uint4 a = {0,0,0,0};
    float accum[64] = {0.f};
    wgmma_try_packed_a(accum, a, 0);
}

int main() {
    test_kernel<<<1, 128>>>(nullptr, nullptr, nullptr);
    return 0;
}
