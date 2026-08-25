#ifndef CUDAOP_TRANSPOSE_TRANSPOSE_H
#define CUDAOP_TRANSPOSE_TRANSPOSE_H

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace cudaop
{

enum class TransposeDataType : uint32_t
{
    kFloat32,
    kBfloat16,
    kFloat8E4M3,
    kFloat4E2M1
};

enum class TransposeSharedMemoryLayout : uint32_t
{
    kPadded,
    kSwizzled
};

// FP4 matrices use CUDA NVFP4-compatible row-wise packing: element 2k is in
// the low nibble and element 2k + 1 is in the high nibble. Every row occupies
// ceil(columns / 2) bytes, and the unused high nibble of an odd-width row is 0.
size_t transpose_storage_bytes(
    uint32_t rows,
    uint32_t columns,
    TransposeDataType data_type);

const char *transpose_data_type_name(TransposeDataType data_type);

const char *transpose_layout_name(
    TransposeSharedMemoryLayout shared_memory_layout);

// Transposes a row-major rows x columns matrix into a row-major
// columns x rows matrix. Input and output must not alias.
cudaError_t transpose_cuda(
    const void *input,
    void *output,
    uint32_t rows,
    uint32_t columns,
    TransposeDataType data_type,
    TransposeSharedMemoryLayout shared_memory_layout,
    cudaStream_t stream = nullptr);

// Uses one 32-bit global-memory pack per thread: two BF16 values or four FP8
// E4M3 values. Other data types return cudaErrorInvalidValue. Unaligned rows
// and partial edge packs are handled by a scalar bit-preserving path.
cudaError_t transpose_vectorized_cuda(
    const void *input,
    void *output,
    uint32_t rows,
    uint32_t columns,
    TransposeDataType data_type,
    TransposeSharedMemoryLayout shared_memory_layout,
    cudaStream_t stream = nullptr);

}  // namespace cudaop

#endif
