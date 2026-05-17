#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

typedef struct
{
    uint32_t in_h;
    uint32_t in_w;
    uint32_t in_ch;
    uint32_t inHW;
    uint32_t inChKhKw;
    uint32_t inBatchNumel;
    uint32_t out_ch;
    uint32_t out_h;
    uint32_t out_w;
    uint32_t outHW;
    uint32_t outBatchNumel;
    uint32_t Kh;
    uint32_t Kw;
    uint32_t KhKw;
    uint32_t Sh;
    uint32_t Sw;
    uint32_t Ph;
    uint32_t Pw;

    uint32_t k_tiles;
    uint32_t first_k_tile;

    size_t kernelWeightsCount;
    size_t biasWeightsCount;
} Conv2DParam;

__device__ __forceinline__ uint32_t smem_u32addr(const void *smem_ptr)
{
    uint32_t addr;
    asm ("{.reg .u64 u64addr;\n"
        " cvta.to.shared.u64 u64addr, %1;\n"
        " cvt.u32.u64 %0, u64addr;}\n"
        : "=r"(addr)
        : "l"(smem_ptr)
    );

    return addr;
}

__device__ __forceinline__ void ldg_nc_0(uint16_t &reg, const void *ptr, bool guard)
{
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @!p mov.u16 %0, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && __CUDA_ARCH__ >= 750
        " @p ld.global.nc.L2::128B.u16 %0, [%1];\n"
#else
        " @p ld.global.nc.u16 %0, [%1];\n"
#endif
        "}"
        : "=h"(reg)
        : "l"(ptr), "r"((int) guard)
    );
}

__device__ __forceinline__ void ldg16_nc_0(half &reg, const void *ptr, bool guard)
{
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @!p mov.b16 %0, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && __CUDA_ARCH__ >= 750
        " @p ld.global.nc.L2::128B.b16 %0, [%1];\n"
#else
        " @p ld.global.nc.b16 %0, [%1];\n"
#endif
        "}"
        : "=h"(*reinterpret_cast<unsigned short *>(&reg))
        : "l"(ptr), "r"((int) guard)
    );
}

__device__ __forceinline__ void ldg16_nc(half &reg, const void *ptr, bool guard)
{
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && __CUDA_ARCH__ >= 750
        " @p ld.global.nc.L2::128B.b16 %0, [%1];\n"
#else
        " @p ld.global.nc.b16 %0, [%1];\n"
#endif
        "}"
        : "=h"(*reinterpret_cast<unsigned short *>(&reg))
        : "l"(ptr), "r"((int) guard)
    );
}

__device__ __forceinline__ void ldg16_nc(uint16_t &reg, const void *ptr, bool guard)
{
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && __CUDA_ARCH__ >= 750
        " @p ld.global.nc.L2::128B.b16 %0, [%1];\n"
#else
        " @p ld.global.nc.b16 %0, [%1];\n"
#endif
        "}"
        : "=h"(reg)
        : "l"(ptr), "r"((int) guard)
    );
}

__device__ __forceinline__ void ldg16_bit_or_low(uint32_t &reg, const void *ptr, bool guard)
{
    asm volatile (
        "{ .reg .pred p;\n"
        "  .reg .u32 t; \n"
        "  setp.ne.b32 p, %2, 0;         \n"
        "  mov.u32 t, 0;                 \n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && __CUDA_ARCH__ >= 750
        "  @p ld.global.nc.L2::128B.u16 t, [%1];  \n"
#else
        "  @p ld.global.nc.u16 t, [%1];  \n"
#endif
        "  setp.ne.u32 p, t, 0;          \n"
        "  @p  mov.b32 %0, 0x0000FFFF;   \n"
        "  @!p mov.b32 %0, 0;            \n"
        "}"
        : "+r"(reg)
        : "l"(ptr), "r"((int) guard));
}

__device__ __forceinline__ void ldg16_bit_or_low(uint32_t &reg, const void *ptr)
{
    asm volatile (
        "{ .reg .pred p;\n"
        "  .reg .u32 t; \n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && __CUDA_ARCH__ >= 750
        "  ld.global.nc.L2::128B.u16 t, [%1];  \n"
#else
        "  ld.global.nc.u16 t, [%1];   \n"
#endif
        "  setp.ne.u32 p, t, 0;        \n"
        "  @p  mov.b32 %0, 0x0000FFFF; \n"
        "  @!p mov.b32 %0, 0;          \n"
        "}"
        : "+r"(reg)
        : "l"(ptr));
}

__device__ __forceinline__ void ldg16_bit_or_low(uint32_t &reg, const void *ptr, int bit)
{
    asm volatile (
        "{ .reg .pred p;\n"
        "  .reg .u32 t; \n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && __CUDA_ARCH__ >= 750
        "  ld.global.nc.L2::128B.u16 t, [%1];  \n"
#else
        "  ld.global.nc.u16 t, [%1];  \n"
#endif
        "  shr.u32 t, t, %2;             \n" // t >>= bit
        "  and.b32 t, t, 0x1;            \n" // t = t & 0x1
        "  setp.ne.u32 p, t, 0;          \n"
        "  @p  mov.b32 %0, 0x0000FFFF;   \n"
        "  @!p mov.b32 %0, 0;            \n"
        "}"
        : "+r"(reg)
        : "l"(ptr), "r"(bit));
}

__device__ __forceinline__ void ldg16_bit_or_low(uint32_t &reg, const void *ptr, bool guard, int bit)
{
    asm volatile (
        "{ .reg .pred p;\n"
        "  .reg .u32 t; \n"
        "  setp.ne.b32 p, %2, 0;         \n"
        "  mov.u32 t, 0;                 \n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && __CUDA_ARCH__ >= 750
        "  @p ld.global.nc.L2::128B.u16 t, [%1];  \n"
#else
        "  @p ld.global.nc.u16 t, [%1];  \n"
#endif
        "  shr.u32 t, t, %3;             \n" // t >>= bit
        "  and.b32 t, t, 0x1;            \n" // t = t & 0x1
        "  setp.ne.u32 p, t, 0;          \n"
        "  @p  mov.b32 %0, 0x0000FFFF;   \n"
        "  @!p mov.b32 %0, 0;            \n"
        "}"
        : "+r"(reg)
        : "l"(ptr), "r"((int) guard), "r"(bit));
}

__device__ __forceinline__ void ldg16_bit_or_high(uint32_t &reg, const void *ptr, bool guard)
{
    asm volatile (
        "{ .reg .pred p;\n"
        "  .reg .u32 t;\n"
        "  setp.ne.b32 p, %2, 0;         \n"
        "  mov.u32 t, 0;                 \n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && __CUDA_ARCH__ >= 750
        "  @p ld.global.nc.L2::128B.u16 t, [%1];  \n"
#else
        "  @p ld.global.nc.u16 t, [%1];  \n"
#endif
        "  setp.ne.u32 p, t, 0;          \n"
        "  @p or.b32 %0, %0, 0xFFFF0000; \n"
        "}"
        : "+r"(reg)
        : "l"(ptr), "r"((int) guard));
}

__device__ __forceinline__ void ldg16_bit_or_high(uint32_t &reg, const void *ptr)
{
    asm volatile (
        "{ .reg .pred p;\n"
        "  .reg .u32 t;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && __CUDA_ARCH__ >= 750
        "  ld.global.nc.L2::128B.u16 t, [%1];  \n"
#else
        "  ld.global.nc.u16 t, [%1];  \n"
#endif
        "  setp.ne.u32 p, t, 0;          \n"
        "  @p or.b32 %0, %0, 0xFFFF0000; \n"
        "}"
        : "+r"(reg)
        : "l"(ptr));
}

__device__ __forceinline__ void ldg16_bit_or_high(uint32_t &reg, const void *ptr, int bit)
{
    asm volatile (
        "{ .reg .pred p;\n"
        "  .reg .u32 t;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && __CUDA_ARCH__ >= 750
        " ld.global.nc.L2::128B.u16 t, [%1];  \n"
#else
        " ld.global.nc.u16 t, [%1];  \n"
#endif
        "  shr.u32 t, t, %2;             \n" // t >>= bit
        "  and.b32 t, t, 0x1;            \n" // t = t & 0x1
        "  setp.ne.u32 p, t, 0;          \n"
        "  @p or.b32 %0, %0, 0xFFFF0000; \n"
        "}"
        : "+r"(reg)
        : "l"(ptr), "r"(bit));
}

__device__ __forceinline__ void ldg16_bit_or_high(uint32_t &reg, const void *ptr, bool guard, int bit)
{
    asm volatile (
        "{ .reg .pred p;\n"
        "  .reg .u32 t;\n"
        "  setp.ne.b32 p, %2, 0;         \n"
        "  mov.u32 t, 0;                 \n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && __CUDA_ARCH__ >= 750
        "  @p ld.global.nc.L2::128B.u16 t, [%1];  \n"
#else
        "  @p ld.global.nc.u16 t, [%1];  \n"
#endif
        "  shr.u32 t, t, %3;             \n" // t >>= bit
        "  and.b32 t, t, 0x1;            \n" // t = t & 0x1
        "  setp.ne.u32 p, t, 0;          \n"
        "  @p or.b32 %0, %0, 0xFFFF0000; \n"
        "}"
        : "+r"(reg)
        : "l"(ptr), "r"((int) guard), "r"(bit));
}

__device__ __forceinline__ void ldg32_nc(uint32_t &reg, const void *ptr, bool guard)
{
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && __CUDA_ARCH__ >= 750
        " @p ld.global.nc.L2::128B.b32 %0, [%1];\n"
#else
        " @p ld.global.nc.b32 %0, [%1];\n"
#endif
        "}"
        : "=r"(reg)
        : "l"(ptr), "r"((int) guard)
    );
}

__device__ __forceinline__ void ldg32_nc(float &reg, const void *ptr, bool guard)
{
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && __CUDA_ARCH__ >= 750
        " @p ld.global.nc.L2::128B.f32 %0, [%1];\n"
#else
        " @p ld.global.nc.f32 %0, [%1];\n"
#endif
        "}"
        : "=f"(reg)
        : "l"(ptr), "r"((int) guard)
    );
}

__device__ __forceinline__ void ldg32_nc(half2 &reg, const void *ptr, bool guard)
{
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && __CUDA_ARCH__ >= 750
        " @p ld.global.nc.L2::128B.b32 %0, [%1];\n"
#else
        " @p ld.global.nc.b32 %0, [%1];\n"
#endif
        "}"
        : "=r"(*reinterpret_cast<unsigned *>(&reg))
        : "l"(ptr), "r"((int) guard)
    );
}

__device__ __forceinline__ void ldg32_nc_0(float &reg, const void *ptr, bool guard)
{
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @!p mov.b32 %0, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && __CUDA_ARCH__ >= 750
        " @p ld.global.nc.L2::128B.f32 %0, [%1];\n"
#else
        " @p ld.global.nc.f32 %0, [%1];\n"
#endif
        "}"
        : "=f"(reg)
        : "l"(ptr), "r"((int) guard)
    );
}

__device__ __forceinline__ void ldg32_nc_0(uint32_t &reg, const void *ptr, bool guard)
{
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @!p mov.b32 %0, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && __CUDA_ARCH__ >= 750
        " @p ld.global.nc.L2::128B.b32 %0, [%1];\n"
#else
        " @p ld.global.nc.b32 %0, [%1];\n"
#endif
        "}"
        : "=r"(reg)
        : "l"(ptr), "r"((int) guard)
    );
}

__device__ __forceinline__ void ldg32_nc_ninf(float &reg, const void *ptr, bool guard)
{
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @!p mov.b32 %0, 0xff800000;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && __CUDA_ARCH__ >= 750
        " @p ld.global.nc.L2::128B.f32 %0, [%1];\n"
#else
        " @p ld.global.nc.f32 %0, [%1];\n"
#endif
        "}"
        : "=f"(reg)
        : "l"(ptr), "r"((int) guard)
    );
}

__device__ __forceinline__ void ldg32_nc_0(half2 &reg, const void *ptr, bool guard)
{
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @!p mov.b32 %0, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && __CUDA_ARCH__ >= 750
        " @p ld.global.nc.L2::128B.b32 %0, [%1];\n"
#else
        " @p ld.global.nc.b32 %0, [%1];\n"
#endif
        "}"
        : "=r"(*reinterpret_cast<unsigned *>(&reg))
        : "l"(ptr), "r"((int) guard)
    );
}

__device__ __forceinline__ void ldg64_nc(float &reg0, float &reg1, const void *ptr)
{
    asm volatile (
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && __CUDA_ARCH__ >= 750
        " ld.global.nc.L2::128B.v2.f32 {%0,%1}, [%2];\n"
#else
        " ld.global.nc.v2.f32 {%0,%1}, [%2];\n"
#endif
        : "=f"(reg0), "=f"(reg1)
        : "l"(ptr)
    );
}

__device__ __forceinline__ void stg16(const half &reg, void *ptr, bool guard)
{
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @p st.global.b16 [%0], %1;}\n"
        : : "l"(ptr), "h"(*reinterpret_cast<const unsigned short *>(&reg)), "r"((int) guard)
    );
}

__device__ __forceinline__ void stg32(const float &reg, void *ptr, bool guard)
{
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @p st.global.f32 [%0], %1;}\n"
        : : "l"(ptr), "f"(reg), "r"((int) guard)
    );
}

__device__ __forceinline__ void lds128(float &reg0, float &reg1, float &reg2, float &reg3, const uint32_t &addr)
{
    asm volatile (
        "ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
        : "=f"(reg0), "=f"(reg1), "=f"(reg2), "=f"(reg3)
        : "r"(addr)
    );
}

__device__ __forceinline__ void
lds128(uint32_t &reg0, uint32_t &reg1, uint32_t &reg2, uint32_t &reg3, const uint32_t &addr)
{
    asm volatile (
        "ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(reg0), "=r"(reg1), "=r"(reg2), "=r"(reg3)
        : "r"(addr)
    );
}

__device__ __forceinline__ void
lds128(half2 &reg0, half2 &reg1, half2 &reg2, half2 &reg3, const uint32_t &addr)
{
    asm volatile (
        "ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(*reinterpret_cast<unsigned *>(&reg0)),
        "=r"(*reinterpret_cast<unsigned *>(&reg1)),
        "=r"(*reinterpret_cast<unsigned *>(&reg2)),
        "=r"(*reinterpret_cast<unsigned *>(&reg3))
        : "r"(addr)
    );
}

__device__ __forceinline__ void
lds128(ushort2 &reg0, ushort2 &reg1, ushort2 &reg2, ushort2 &reg3, const uint32_t &addr)
{
    asm volatile (
        "ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(*reinterpret_cast<unsigned *>(&reg0)),
        "=r"(*reinterpret_cast<unsigned *>(&reg1)),
        "=r"(*reinterpret_cast<unsigned *>(&reg2)),
        "=r"(*reinterpret_cast<unsigned *>(&reg3))
        : "r"(addr)
    );
}

__device__ __forceinline__ void sts32(const float &reg, const uint32_t &addr)
{
    asm volatile (
        "st.shared.f32 [%0], %1;\n"
        : : "r"(addr), "f"(reg)
    );
}

__device__ __forceinline__ void sts32(const uint32_t &reg, const uint32_t &addr)
{
    asm volatile (
        "st.shared.b32 [%0], %1;\n"
        : : "r"(addr), "r"(reg)
    );
}

__device__ __forceinline__ void sts32(const half2 &reg, const uint32_t &addr)
{
    asm volatile (
        "st.shared.b32 [%0], %1;\n"
        : : "r"(addr), "r"(*reinterpret_cast<const unsigned *>(&reg))
    );
}

__device__ __forceinline__ void sts32(const ushort2 &reg, const uint32_t &addr)
{
    asm volatile (
        "st.shared.b32 [%0], %1;\n"
        : : "r"(addr), "r"(*reinterpret_cast<const unsigned *>(&reg))
    );
}

__device__ __forceinline__ void sts64(const float &reg0, const float &reg1, const uint32_t &addr)
{
    asm volatile (
        "st.shared.v2.f32 [%0], {%1, %2};\n"
        : : "r"(addr), "f"(reg0), "f"(reg1)
    );
}

__device__ __forceinline__ void sts64(const half2 &reg0, const half2 &reg1, const uint32_t &addr)
{
    asm volatile (
        "st.shared.v2.b32 [%0], {%1, %2};\n"
        : : "r"(addr),
        "r"(*reinterpret_cast<const unsigned *>(&reg0)),
        "r"(*reinterpret_cast<const unsigned *>(&reg1))
    );
}

__device__ __forceinline__ void
sts128(const uint32_t &reg0, const uint32_t &reg1, const uint32_t &reg2, const uint32_t &reg3, const uint32_t &addr)
{
    asm volatile (
        "st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n"
        : : "r"(addr), "r"(reg0), "r"(reg1), "r"(reg2), "r"(reg3)
    );
}

__device__ __forceinline__ void
sts128(const ushort2 &reg0, const ushort2 &reg1, const ushort2 &reg2, const ushort2 &reg3, const uint32_t &addr)
{
    asm volatile (
        "st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n"
        : : "r"(addr),
        "r"(*reinterpret_cast<const unsigned *>(&reg0)),
        "r"(*reinterpret_cast<const unsigned *>(&reg1)),
        "r"(*reinterpret_cast<const unsigned *>(&reg2)),
        "r"(*reinterpret_cast<const unsigned *>(&reg3))
    );
}

__device__ __forceinline__ void
sts128(const float &reg0, const float &reg1, const float &reg2, const float &reg3, const uint32_t &addr)
{
    asm volatile (
        "st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n"
        : : "r"(addr), "f"(reg0), "f"(reg1), "f"(reg2), "f"(reg3)
    );
}

__device__ __forceinline__ void
sts128(const half2 &reg0, const half2 &reg1, const half2 &reg2, const half2 &reg3, const uint32_t &addr)
{
    asm volatile (
        "st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n"
        : : "r"(addr),
        "r"(*reinterpret_cast<const unsigned *>(&reg0)),
        "r"(*reinterpret_cast<const unsigned *>(&reg1)),
        "r"(*reinterpret_cast<const unsigned *>(&reg2)),
        "r"(*reinterpret_cast<const unsigned *>(&reg3))
    );
}

__device__ __forceinline__ void add_f32(float &a, float &b, int guard)
{
    asm volatile(
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;   \n"
        " @p add.f32 %0, %0, %1;  \n"
        "}"
        : "+f"(a)
        : "f"(b), "r"(guard)
    );
}

__device__ __forceinline__ void add_masked_half2(half2 &y, half2 &x, uint32_t mask)
{
    asm volatile(
        "{ .reg .b32 t;           \n"
        "  and.b32 t, %1, %2;     \n"
        "  add.f16x2 %0, %3, t;   \n"
        "}"
        : "=r"(*reinterpret_cast<uint32_t *>(&y))
        : "r"(*reinterpret_cast<uint32_t *>(&x)), "r"(mask), "r"(*reinterpret_cast<uint32_t *>(&y)));
}
