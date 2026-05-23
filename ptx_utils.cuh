#ifndef PTX_UTILS_CUH
#define PTX_UTILS_CUH

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace ptx
{

__device__ __forceinline__ uint32_t smem_u32addr(const void *ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// =============================================================================
// Standard Non-Coherent Loads (ldg_nc)
// =============================================================================

template <typename T>
__device__ __forceinline__ void ldg16_nc(T &reg, const void *ptr, bool guard)
{
    static_assert(sizeof(T) == 2, "ldg16_nc requires 2-byte type");
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @p ld.global.nc.b16 %0, [%1];\n"
        "}"
        : "=h"(*reinterpret_cast<unsigned short*>(&reg))
        : "l"(ptr), "r"((int)guard)
    );
}

template <typename T>
__device__ __forceinline__ void ldg32_nc(T &reg, const void *ptr, bool guard)
{
    static_assert(sizeof(T) == 4, "ldg32_nc requires 4-byte type");
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        #if __CUDA_ARCH__ >= 750
        " @p ld.global.nc.L2::128B.b32 %0, [%1];\n"
        #else
        " @p ld.global.nc.b32 %0, [%1];\n"
        #endif
        "}"
        : "=r"(*reinterpret_cast<unsigned*>(&reg))
        : "l"(ptr), "r"((int)guard)
    );
}

template <typename T>
__device__ __forceinline__ void ldg64_nc(T &reg, const void *ptr, bool guard)
{
    static_assert(sizeof(T) == 8, "ldg64_nc requires 8-byte type");
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @p ld.global.nc.b64 %0, [%1];\n"
        "}"
        : "=l"(*reinterpret_cast<unsigned long long*>(&reg))
        : "l"(ptr), "r"((int)guard)
    );
}

template <typename T>
__device__ __forceinline__ void ldg128_nc(T &reg, const void *ptr, bool guard)
{
    static_assert(sizeof(T) == 16, "ldg128_nc requires 16-byte type");
    unsigned *r = reinterpret_cast<unsigned*>(&reg);
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %4, 0;\n"
        " @p ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%5];\n"
        "}"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"((int)guard), "l"(ptr)
    );
}

// =============================================================================
// Non-Coherent Loads with Zero-Fill (ldg_nc_0)
// =============================================================================

template <typename T>
__device__ __forceinline__ void ldg16_nc_0(T &reg, const void *ptr, bool guard)
{
    static_assert(sizeof(T) == 2, "ldg16_nc_0 requires 2-byte type");
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @!p mov.b16 %0, 0;\n"
        " @p ld.global.nc.b16 %0, [%1];\n"
        "}"
        : "=h"(*reinterpret_cast<unsigned short*>(&reg))
        : "l"(ptr), "r"((int)guard)
    );
}

template <typename T>
__device__ __forceinline__ void ldg32_nc_0(T &reg, const void *ptr, bool guard)
{
    static_assert(sizeof(T) == 4, "ldg32_nc_0 requires 4-byte type");
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @!p mov.b32 %0, 0;\n"
        #if __CUDA_ARCH__ >= 750
        " @p ld.global.nc.L2::128B.b32 %0, [%1];\n"
        #else
        " @p ld.global.nc.b32 %0, [%1];\n"
        #endif
        "}"
        : "=r"(*reinterpret_cast<unsigned*>(&reg))
        : "l"(ptr), "r"((int)guard)
    );
}

template <typename T>
__device__ __forceinline__ void ldg64_nc_0(T &reg, const void *ptr, bool guard)
{
    static_assert(sizeof(T) == 8, "ldg64_nc_0 requires 8-byte type");
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @!p mov.b64 %0, 0;\n"
        " @p ld.global.nc.b64 %0, [%1];\n"
        "}"
        : "=l"(*reinterpret_cast<unsigned long long*>(&reg))
        : "l"(ptr), "r"((int)guard)
    );
}

template <typename T>
__device__ __forceinline__ void ldg128_nc_0(T &reg, const void *ptr, bool guard)
{
    static_assert(sizeof(T) == 16, "ldg128_nc_0 requires 16-byte type");
    unsigned *r = reinterpret_cast<unsigned*>(&reg);
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %4, 0;\n"
        " @!p mov.b32 %0, 0;\n"
        " @!p mov.b32 %1, 0;\n"
        " @!p mov.b32 %2, 0;\n"
        " @!p mov.b32 %3, 0;\n"
        " @p ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%5];\n"
        "}"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"((int)guard), "l"(ptr)
    );
}

// =============================================================================
// Shared Memory Loads (lds)
// =============================================================================

template <typename T>
__device__ __forceinline__ void lds32(T &reg0, const uint32_t &addr)
{
    static_assert(sizeof(T) == 4, "lds32 requires 4-byte type");
    asm volatile (
        "ld.shared.b32 %0, [%1];"
        : "=r"(*reinterpret_cast<unsigned*>(&reg0))
        : "r"(addr)
    );
}

template <typename T>
__device__ __forceinline__ void lds64(T &reg0, T &reg1, const uint32_t &addr)
{
    static_assert(sizeof(T) == 4, "lds64 registers must be 4-byte types");
    asm volatile (
        "ld.shared.v2.b32 {%0, %1}, [%2];"
        : "=r"(*reinterpret_cast<unsigned*>(&reg0)), 
          "=r"(*reinterpret_cast<unsigned*>(&reg1))
        : "r"(addr)
    );
}

template <typename T>
__device__ __forceinline__ void lds128(T &reg0, T &reg1, T &reg2, T &reg3, const uint32_t &addr)
{
    static_assert(sizeof(T) == 4, "lds128 registers must be 4-byte types");
    asm volatile (
        "ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
        : "=r"(*reinterpret_cast<unsigned*>(&reg0)), 
          "=r"(*reinterpret_cast<unsigned*>(&reg1)), 
          "=r"(*reinterpret_cast<unsigned*>(&reg2)), 
          "=r"(*reinterpret_cast<unsigned*>(&reg3))
        : "r"(addr)
    );
}

// =============================================================================
// Shared Memory Stores (sts)
// =============================================================================

template <typename T>
__device__ __forceinline__ void sts32(const T &reg0, const uint32_t &addr)
{
    static_assert(sizeof(T) == 4, "sts32 requires 4-byte type");
    asm volatile (
        "st.shared.b32 [%0], %1;"
        : : "r"(addr), "r"(*reinterpret_cast<const unsigned*>(&reg0))
    );
}

template <typename T>
__device__ __forceinline__ void sts64(const T &reg0, const T &reg1, const uint32_t &addr)
{
    static_assert(sizeof(T) == 4, "sts64 registers must be 4-byte types");
    asm volatile (
        "st.shared.v2.b32 [%0], {%1, %2};"
        : : "r"(addr), 
            "r"(*reinterpret_cast<const unsigned*>(&reg0)), 
            "r"(*reinterpret_cast<const unsigned*>(&reg1))
    );
}

template <typename T>
__device__ __forceinline__ void sts128(const T &reg0, const T &reg1, const T &reg2, const T &reg3, const uint32_t &addr)
{
    static_assert(sizeof(T) == 4, "sts128 registers must be 4-byte types");
    asm volatile (
        "st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
        : : "r"(addr), 
            "r"(*reinterpret_cast<const unsigned*>(&reg0)), 
            "r"(*reinterpret_cast<const unsigned*>(&reg1)), 
            "r"(*reinterpret_cast<const unsigned*>(&reg2)), 
            "r"(*reinterpret_cast<const unsigned*>(&reg3))
    );
}

// =============================================================================
// Global Memory Stores (stg)
// =============================================================================

template <typename T>
__device__ __forceinline__ void stg8(const T &reg, void *ptr, bool guard)
{
    static_assert(sizeof(T) == 1, "stg8 requires 1-byte type");
    unsigned v = (unsigned)(*reinterpret_cast<const unsigned char*>(&reg));
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @p st.global.u8 [%0], %1;\n"
        "}"
        : : "l"(ptr), "r"(v), "r"((int)guard)
    );
}

template <typename T>
__device__ __forceinline__ void stg16(const T &reg, void *ptr, bool guard)
{
    static_assert(sizeof(T) == 2, "stg16 requires 2-byte type");
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @p st.global.b16 [%0], %1;\n"
        "}"
        : : "l"(ptr), "h"(*reinterpret_cast<const unsigned short*>(&reg)), "r"((int)guard)
    );
}

template <typename T>
__device__ __forceinline__ void stg32(const T &reg, void *ptr, bool guard)
{
    static_assert(sizeof(T) == 4, "stg32 requires 4-byte type");
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @p st.global.b32 [%0], %1;\n"
        "}"
        : : "l"(ptr), "r"(*reinterpret_cast<const unsigned*>(&reg)), "r"((int)guard)
    );
}

template <typename T>
__device__ __forceinline__ void stg64(const T &reg0, const T &reg1, void *ptr, bool guard)
{
    static_assert(sizeof(T) == 4, "stg64 registers must be 4-byte types");
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %3, 0;\n"
        " @p st.global.v2.b32 [%0], {%1, %2};\n"
        "}"
        : : "l"(ptr), "r"(*reinterpret_cast<const unsigned*>(&reg0)), 
            "r"(*reinterpret_cast<const unsigned*>(&reg1)), "r"((int)guard)
    );
}

template <typename T>
__device__ __forceinline__ void stg128(const T &reg0, const T &reg1, const T &reg2, const T &reg3, void *ptr, bool guard)
{
    static_assert(sizeof(T) == 4, "stg128 registers must be 4-byte types");
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %5, 0;\n"
        " @p st.global.v4.b32 [%0], {%1, %2, %3, %4};\n"
        "}"
        : : "l"(ptr), "r"(*reinterpret_cast<const unsigned*>(&reg0)), 
            "r"(*reinterpret_cast<const unsigned*>(&reg1)), 
            "r"(*reinterpret_cast<const unsigned*>(&reg2)), 
            "r"(*reinterpret_cast<const unsigned*>(&reg3)), "r"((int)guard)
    );
}

// =============================================================================
// Dispatchers (Global Loads)
// =============================================================================

template <typename T>
__device__ __forceinline__ void ldg_nc(T &reg, const void *ptr, bool guard = true)
{
    if constexpr (sizeof(T) == 2) { ldg16_nc(reg, ptr, guard); }
    else if constexpr (sizeof(T) == 4) { ldg32_nc(reg, ptr, guard); }
    else if constexpr (sizeof(T) == 8) { ldg64_nc(reg, ptr, guard); }
    else if constexpr (sizeof(T) == 16) { ldg128_nc(reg, ptr, guard); }
}

template <typename T>
__device__ __forceinline__ void ldg_nc_0(T &reg, const void *ptr, bool guard = true)
{
    if constexpr (sizeof(T) == 2) { ldg16_nc_0(reg, ptr, guard); }
    else if constexpr (sizeof(T) == 4) { ldg32_nc_0(reg, ptr, guard); }
    else if constexpr (sizeof(T) == 8) { ldg64_nc_0(reg, ptr, guard); }
    else if constexpr (sizeof(T) == 16) { ldg128_nc_0(reg, ptr, guard); }
}


// Predicated fp16x2 add: @p add.f16x2 acc, acc, w
__device__ __forceinline__ void add_f16x2(__half2 &a, const __half2 &b, int guard)
{
    asm volatile(
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @p add.f16x2 %0, %0, %1;\n"
        "}"
        : "+r"(*reinterpret_cast<unsigned *>(&a))
        : "r"(*reinterpret_cast<const unsigned *>(&b)), "r"(guard)
    );
}

} // namespace ptx

#endif
