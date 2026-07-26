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
// Clock Counter
// =============================================================================
//
// Instruction: mov.u64 from %clock64
// Source: NVIDIA PTX ISA, special registers
// https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-clock64
// Purpose: read the per-SM cycle counter for device-side latency measurement.

__device__ __forceinline__ uint64_t read_clock64()
{
    uint64_t value;
    asm volatile (
        "mov.u64 %0, %%clock64;"
        : "=l"(value)
        :
        : "memory"
    );
    return value;
}

// =============================================================================
// Global Memory Prefetch
// =============================================================================
//
// Instruction: prefetch.global.L1
// Source: NVIDIA PTX ISA, data movement and conversion instructions
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
// Purpose: issue a cache prefetch for a global memory address without allocating
// a destination register.

__device__ __forceinline__ void prefetch_global_l1(
    const void *ptr,
    bool guard)
{
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %1, 0;\n"
        " @p prefetch.global.L1 [%0];\n"
        "}"
        :
        : "l"(ptr), "r"((int)guard)
    );
}

// =============================================================================
// Standard Non-Coherent Loads (ldg_nc)
// =============================================================================

// Instruction: ld.global.cg.b32
// Source: NVIDIA PTX ISA, cache operators
// https://docs.nvidia.com/cuda/parallel-thread-execution/#cache-operators
// Purpose: load a 32-bit value through the global-memory path while bypassing
// L1 and caching the line only in L2.

template <typename T>
__device__ __forceinline__ void ldg32_cg(T &reg, const void *ptr)
{
    static_assert(sizeof(T) == 4, "ldg32_cg requires 4-byte type");
    asm volatile (
        "ld.global.cg.b32 %0, [%1];"
        : "=r"(*reinterpret_cast<unsigned*>(&reg))
        : "l"(ptr)
        : "memory"
    );
}

// Instruction: ld.global.cs.v4.b32
// Source: NVIDIA PTX ISA, cache operators
// https://docs.nvidia.com/cuda/parallel-thread-execution/#cache-operators
// Purpose: issue a 128-bit streaming global load with evict-first cache policy.

template <typename T>
__device__ __forceinline__ void ldg128_cs(
    T &reg0,
    T &reg1,
    T &reg2,
    T &reg3,
    const void *ptr)
{
    static_assert(sizeof(T) == 4, "ldg128_cs registers must be 4-byte types");
    asm volatile (
        "ld.global.cs.v4.b32 {%0, %1, %2, %3}, [%4];"
        : "=r"(*reinterpret_cast<unsigned*>(&reg0)),
          "=r"(*reinterpret_cast<unsigned*>(&reg1)),
          "=r"(*reinterpret_cast<unsigned*>(&reg2)),
          "=r"(*reinterpret_cast<unsigned*>(&reg3))
        : "l"(ptr)
        : "memory"
    );
}

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

// Instruction: ld.global.nc.b64
// Source: NVIDIA PTX ISA, data movement and conversion instructions
// https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-ld
// Purpose: issue an unguarded 64-bit non-coherent load for dependent
// L1/TEX-cache pointer chasing.

template <typename T>
__device__ __forceinline__ void ldg64_nc(T &reg, const void *ptr)
{
    static_assert(sizeof(T) == 8, "ldg64_nc requires 8-byte type");
    asm volatile (
        "ld.global.nc.b64 %0, [%1];"
        : "=l"(*reinterpret_cast<unsigned long long*>(&reg))
        : "l"(ptr)
        : "memory"
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

// Instruction: ld.global.nc.v4.b32
// Source: NVIDIA PTX ISA, data movement and conversion instructions
// https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-ld
// Purpose: issue an unguarded 128-bit non-coherent load for L1/TEX-cache
// throughput measurement.

template <typename T>
__device__ __forceinline__ void ldg128_nc(T &reg, const void *ptr)
{
    static_assert(sizeof(T) == 16, "ldg128_nc requires 16-byte type");
    unsigned *values = reinterpret_cast<unsigned*>(&reg);
    asm volatile (
        "ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
        : "=r"(values[0]),
          "=r"(values[1]),
          "=r"(values[2]),
          "=r"(values[3])
        : "l"(ptr)
        : "memory"
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

// Instruction: st.global.cs.v4.b32
// Source: NVIDIA PTX ISA, cache operators
// https://docs.nvidia.com/cuda/parallel-thread-execution/#cache-operators
// Purpose: issue a 128-bit streaming global store with evict-first cache policy.

template <typename T>
__device__ __forceinline__ void stg128_cs(
    const T &reg0,
    const T &reg1,
    const T &reg2,
    const T &reg3,
    void *ptr)
{
    static_assert(sizeof(T) == 4, "stg128_cs registers must be 4-byte types");
    asm volatile (
        "st.global.cs.v4.b32 [%4], {%0, %1, %2, %3};"
        :
        : "r"(*reinterpret_cast<const unsigned*>(&reg0)),
          "r"(*reinterpret_cast<const unsigned*>(&reg1)),
          "r"(*reinterpret_cast<const unsigned*>(&reg2)),
          "r"(*reinterpret_cast<const unsigned*>(&reg3)),
          "l"(ptr)
        : "memory"
    );
}

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


// Instruction: add.f32 with predicate guard
// Source: NVIDIA PTX ISA, floating-point arithmetic instructions
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
// Purpose: conditionally accumulate one FP32 weight for a binary spike.
__device__ __forceinline__ void add_f32(float &a, const float &b, int guard)
{
    asm volatile(
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @p add.f32 %0, %0, %1;\n"
        "}"
        : "+f"(a)
        : "f"(b), "r"(guard)
    );
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
