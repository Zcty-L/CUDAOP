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
// Tensor Memory Accelerator and Transaction Barriers
// =============================================================================

// Instruction: mbarrier.init.shared::cta.b64
// Source: NVIDIA PTX ISA, mbarrier.init
// https://docs.nvidia.com/cuda/parallel-thread-execution/
// Purpose: initialize a CTA shared-memory barrier used to track an asynchronous
// TMA transaction.

__device__ __forceinline__ void mbarrier_init(
    uint64_t *barrier,
    uint32_t arrive_count)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const uint32_t barrier_address = smem_u32addr(barrier);
    asm volatile (
        "mbarrier.init.shared::cta.b64 [%0], %1;"
        :
        : "r"(barrier_address), "r"(arrive_count)
        : "memory"
    );
#endif
}

// Instruction: fence.proxy.async.shared::cta
// Source: NVIDIA PTX ISA, fence.proxy.async
// https://docs.nvidia.com/cuda/parallel-thread-execution/
// Purpose: make generic shared-memory writes, including mbarrier
// initialization, visible to the asynchronous TMA proxy.

__device__ __forceinline__ void fence_proxy_async_shared_cta()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile (
        "fence.proxy.async.shared::cta;"
        :
        :
        : "memory"
    );
#endif
}

// Instruction: mbarrier.arrive.expect_tx.shared::cta.b64
// Source: NVIDIA PTX ISA, mbarrier.arrive.expect_tx
// https://docs.nvidia.com/cuda/parallel-thread-execution/
// Purpose: arrive at a transaction barrier and declare the byte count that a
// following TMA operation must complete.

__device__ __forceinline__ void mbarrier_arrive_expect_tx(
    uint64_t *barrier,
    uint32_t transaction_bytes)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const uint32_t barrier_address = smem_u32addr(barrier);
    asm volatile (
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :
        : "r"(barrier_address), "r"(transaction_bytes)
        : "memory"
    );
#endif
}

// Instruction: mbarrier.try_wait.parity.shared::cta.b64
// Source: NVIDIA PTX ISA, mbarrier.try_wait
// https://docs.nvidia.com/cuda/parallel-thread-execution/
// Purpose: wait until all arrivals and asynchronous transaction bytes for the
// requested barrier phase have completed.

__device__ __forceinline__ void mbarrier_wait_parity(
    uint64_t *barrier,
    uint32_t phase)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const uint32_t barrier_address = smem_u32addr(barrier);
    uint32_t wait_complete = 0;

    do
    {
        asm volatile (
            "{.reg .pred complete;\n"
            " mbarrier.try_wait.parity.shared::cta.b64 "
            "complete, [%1], %2;\n"
            " selp.b32 %0, 1, 0, complete;\n"
            "}"
            : "=r"(wait_complete)
            : "r"(barrier_address), "r"(phase)
            : "memory"
        );
    }
    while (wait_complete == 0);
#endif
}

// Instruction: mbarrier.inval.shared::cta.b64
// Source: NVIDIA PTX ISA, mbarrier.inval
// https://docs.nvidia.com/cuda/parallel-thread-execution/
// Purpose: invalidate a shared-memory barrier after its final phase completes.

__device__ __forceinline__ void mbarrier_invalidate(uint64_t *barrier)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const uint32_t barrier_address = smem_u32addr(barrier);
    asm volatile (
        "mbarrier.inval.shared::cta.b64 [%0];"
        :
        : "r"(barrier_address)
        : "memory"
    );
#endif
}

// Instruction:
// cp.async.bulk.tensor.1d.shared::cta.global.tile
//     .mbarrier::complete_tx::bytes
// Source: NVIDIA PTX ISA, cp.async.bulk.tensor
// https://docs.nvidia.com/cuda/parallel-thread-execution/
// Purpose: issue one asynchronous 1D Tensor Map copy from global memory into
// CTA shared memory and report the completed bytes to an mbarrier.

__device__ __forceinline__ void tma_load_1d(
    void *shared_destination,
    const void *tensor_map,
    int32_t coordinate,
    uint64_t *barrier)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const uint32_t destination_address = smem_u32addr(shared_destination);
    const uint32_t barrier_address = smem_u32addr(barrier);
    asm volatile (
        "cp.async.bulk.tensor.1d.shared::cta.global.tile"
        ".mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2}], [%3];"
        :
        : "r"(destination_address),
          "l"(tensor_map),
          "r"(coordinate),
          "r"(barrier_address)
        : "memory"
    );
#endif
}

// Instruction:
// cp.async.bulk.tensor.2d.shared::cta.global.tile
//     .mbarrier::complete_tx::bytes
// Source: NVIDIA PTX ISA, cp.async.bulk.tensor
// https://docs.nvidia.com/cuda/parallel-thread-execution/
// Purpose: issue one asynchronous 2D Tensor Map copy from global memory into
// CTA shared memory, including an optional Tensor Map swizzle.

__device__ __forceinline__ void tma_load_2d(
    void *shared_destination,
    const void *tensor_map,
    int32_t coordinate_x,
    int32_t coordinate_y,
    uint64_t *barrier)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const uint32_t destination_address =
        smem_u32addr(shared_destination);
    const uint32_t barrier_address = smem_u32addr(barrier);
    asm volatile (
        "cp.async.bulk.tensor.2d.shared::cta.global.tile"
        ".mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :
        : "r"(destination_address),
          "l"(tensor_map),
          "r"(coordinate_x),
          "r"(coordinate_y),
          "r"(barrier_address)
        : "memory"
    );
#endif
}

// Instruction: cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group
// Source: NVIDIA PTX ISA, cp.async.bulk.tensor
// https://docs.nvidia.com/cuda/parallel-thread-execution/
// Purpose: copy a 2D tile from shared memory back to the global-memory region
// described by a Tensor Map, undoing the Tensor Map swizzle.

__device__ __forceinline__ void tma_store_2d(
    const void *tensor_map,
    int32_t coordinate_x,
    int32_t coordinate_y,
    const void *shared_source)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const uint32_t source_address = smem_u32addr(shared_source);
    asm volatile (
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group "
        "[%0, {%1, %2}], [%3];"
        :
        : "l"(tensor_map),
          "r"(coordinate_x),
          "r"(coordinate_y),
          "r"(source_address)
        : "memory"
    );
#endif
}

// Instructions: cp.async.bulk.commit_group and
// cp.async.bulk.wait_group.read 0
// Source: NVIDIA PTX ISA, asynchronous copy groups
// https://docs.nvidia.com/cuda/parallel-thread-execution/
// Purpose: commit a TMA shared-to-global store and wait until the asynchronous
// proxy has finished reading its shared-memory source.

__device__ __forceinline__ void tma_store_wait_read()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile (
        "cp.async.bulk.commit_group;"
        :
        :
    );
    asm volatile (
        "cp.async.bulk.wait_group.read 0;"
        :
        :
        : "memory"
    );
#endif
}

// =============================================================================
// Thread Block Cluster and Distributed Shared Memory
// =============================================================================

// Instructions: mov.u32 from cluster special registers
// Source: NVIDIA PTX ISA, clusterid and cluster_ctaid special registers
// https://docs.nvidia.com/cuda/parallel-thread-execution/
// Purpose: expose the cluster identity, local CTA identity, cluster shape, and
// flattened CTA rank for explicit cluster-launch validation.

__device__ __forceinline__ uint32_t cluster_id_x()
{
    uint32_t value = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile (
        "mov.u32 %0, %%clusterid.x;"
        : "=r"(value)
    );
#endif
    return value;
}

__device__ __forceinline__ uint32_t cluster_cta_id_x()
{
    uint32_t value = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile (
        "mov.u32 %0, %%cluster_ctaid.x;"
        : "=r"(value)
    );
#endif
    return value;
}

__device__ __forceinline__ uint32_t cluster_cta_count_x()
{
    uint32_t value = 1;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile (
        "mov.u32 %0, %%cluster_nctaid.x;"
        : "=r"(value)
    );
#endif
    return value;
}

__device__ __forceinline__ uint32_t cluster_cta_rank()
{
    uint32_t value = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile (
        "mov.u32 %0, %%cluster_ctarank;"
        : "=r"(value)
    );
#endif
    return value;
}

// Instructions: barrier.cluster.arrive.release and
// barrier.cluster.wait.acquire
// Source: NVIDIA PTX ISA, barrier.cluster
// https://docs.nvidia.com/cuda/parallel-thread-execution/
// Purpose: synchronize all CTAs in an explicitly launched cluster while
// providing release/acquire ordering for DSM communication.

__device__ __forceinline__ void cluster_sync()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile (
        "barrier.cluster.arrive.release;"
        :
        :
        : "memory"
    );
    asm volatile (
        "barrier.cluster.wait.acquire;"
        :
        :
        : "memory"
    );
#endif
}

// Instruction: mapa.shared::cluster.u32
// Source: NVIDIA PTX ISA, mapa
// https://docs.nvidia.com/cuda/parallel-thread-execution/
// Purpose: map a local shared-memory address to the corresponding DSM address
// owned by another CTA rank in the same cluster.

__device__ __forceinline__ uint32_t map_shared_rank(
    const void *local_shared_pointer,
    uint32_t target_rank)
{
    const uint32_t local_address = smem_u32addr(local_shared_pointer);
    uint32_t remote_address = local_address;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile (
        "mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(remote_address)
        : "r"(local_address), "r"(target_rank)
        : "memory"
    );
#endif
    return remote_address;
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

// Instruction: ld.shared::cluster.b32
// Source: NVIDIA PTX ISA, ld with shared::cluster state space
// https://docs.nvidia.com/cuda/parallel-thread-execution/
// Purpose: load a 32-bit value from a local or remote DSM address produced by
// mapa.shared::cluster.

template <typename T>
__device__ __forceinline__ void lds32_cluster(
    T &reg0,
    const uint32_t &address)
{
    static_assert(sizeof(T) == 4, "lds32_cluster requires 4-byte type");
    asm volatile (
        "ld.shared::cluster.b32 %0, [%1];"
        : "=r"(*reinterpret_cast<unsigned *>(&reg0))
        : "r"(address)
        : "memory"
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
