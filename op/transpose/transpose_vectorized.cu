#include "op/transpose/transpose.h"

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace cudaop
{
namespace
{

constexpr int kTileDimension = 32;
constexpr int kBlockRows = 8;
constexpr int kPackBytes = 4;

enum class KernelSharedMemoryLayout
{
    kPadded,
    kSwizzled
};

// These structures are bit containers. No floating-point conversion is
// performed when a pack moves between global memory and shared memory.
struct alignas(kPackBytes) Bfloat16x2Pack
{
    uint32_t storage;
};

struct alignas(kPackBytes) Float8E4M3x4Pack
{
    uint32_t storage;
};

static_assert(sizeof(Bfloat16x2Pack) == kPackBytes);
static_assert(alignof(Bfloat16x2Pack) == kPackBytes);
static_assert(sizeof(Float8E4M3x4Pack) == kPackBytes);
static_assert(alignof(Float8E4M3x4Pack) == kPackBytes);
static_assert(sizeof(__nv_bfloat162) == kPackBytes);
static_assert(sizeof(__nv_fp8x4_e4m3) == kPackBytes);

template <typename Pack>
__device__ __forceinline__ uint32_t load_aligned_pack(const uint8_t *address)
{
    return reinterpret_cast<const Pack *>(address)->storage;
}

template <typename Pack>
__device__ __forceinline__ void store_aligned_pack(uint8_t *address, uint32_t storage)
{
    reinterpret_cast<Pack *>(address)->storage = storage;
}

template <int ElementBytes>
__device__ __forceinline__ uint32_t load_element_bits(const uint8_t *address)
{
    uint32_t value = 0;
#pragma unroll
    for (int byte = 0; byte < ElementBytes; ++byte)
    {
        value |= static_cast<uint32_t>(address[byte]) << (byte * 8);
    }
    return value;
}

template <int ElementBytes>
__device__ __forceinline__ void store_element_bits(uint8_t *address, uint32_t value)
{
#pragma unroll
    for (int byte = 0; byte < ElementBytes; ++byte)
    {
        address[byte] = static_cast<uint8_t>(value >> (byte * 8));
    }
}

template <
    int VectorWidth,
    KernelSharedMemoryLayout Layout>
__device__ __forceinline__ int shared_word_column(
    int row,
    int logical_word)
{
    if constexpr (Layout == KernelSharedMemoryLayout::kPadded)
    {
        return logical_word;
    }
    else
    {
        return logical_word ^ (row / VectorWidth);
    }
}

template <
    typename Pack,
    int ElementBytes,
    KernelSharedMemoryLayout Layout>
__global__ void transpose_vectorized_kernel(
    const uint8_t *__restrict__ input,
    uint8_t *__restrict__ output,
    uint32_t rows,
    uint32_t columns)
{
    constexpr int vector_width = kPackBytes / ElementBytes; // 4 / ElementBytes
    constexpr int words_per_row = kTileDimension / vector_width;
    constexpr int shared_words = Layout == KernelSharedMemoryLayout::kPadded ? words_per_row + 1 : words_per_row;
    constexpr int element_bits = ElementBytes * 8;
    constexpr uint32_t element_mask = (1U << element_bits) - 1U;

    static_assert(kPackBytes % ElementBytes == 0);
    static_assert(kTileDimension % vector_width == 0);
    static_assert((words_per_row & (words_per_row - 1)) == 0);

    __shared__ __align__(128) uint32_t tile[kTileDimension][shared_words];

    const int local_word = static_cast<int>(threadIdx.x);
    const int local_row_base = static_cast<int>(threadIdx.y);
    const uint32_t input_column_base = blockIdx.x * kTileDimension + local_word * vector_width;

    constexpr int input_offset_limit =
        Layout == KernelSharedMemoryLayout::kPadded ? kTileDimension / vector_width : kTileDimension;
    constexpr int input_offset_step =
        Layout == KernelSharedMemoryLayout::kPadded ? kBlockRows / vector_width : kBlockRows;

#pragma unroll
    for (int offset = 0; offset < input_offset_limit; offset += input_offset_step)
    {
        int local_row = local_row_base + offset;
        if constexpr (Layout == KernelSharedMemoryLayout::kPadded)
        {
            // A warp contains VectorWidth logical rows. Visiting rows from
            // separate row subgroups makes their padded word ranges occupy
            // disjoint bank ranges while retaining a conventional padded
            // row-major shared-memory layout.
            constexpr int rows_per_subgroup = kTileDimension / vector_width;
            const int subgroup = local_row_base % vector_width;
            const int row_in_subgroup = local_row_base / vector_width + offset;
            local_row = subgroup * rows_per_subgroup + row_in_subgroup;
        }
        const uint32_t input_row = blockIdx.y * kTileDimension + local_row;
        uint32_t packed = 0;

        if (input_row < rows && input_column_base < columns)
        {
            const size_t input_element_offset = static_cast<size_t>(input_row) * columns + input_column_base;
            const uint8_t *source = input + input_element_offset * ElementBytes;
            const bool full_pack = columns - input_column_base >= vector_width;
            const bool aligned = (reinterpret_cast<uintptr_t>(source) & (alignof(Pack) - 1U)) == 0;

            if (full_pack && aligned)
            {
                packed = load_aligned_pack<Pack>(source);
            }
            else
            {
#pragma unroll
                for (int element = 0; element < vector_width; ++element)
                {
                    if (static_cast<uint32_t>(element) < columns - input_column_base)
                    {
                        packed |= load_element_bits<ElementBytes>(source + element * ElementBytes) << (element * element_bits);
                    }
                }
            }
        }

        const int physical_word = shared_word_column<vector_width, Layout>(local_row, local_word);
        tile[local_row][physical_word] = packed;
    }

    __syncthreads();

    const uint32_t output_column_base = blockIdx.y * kTileDimension + local_word * vector_width;

#pragma unroll
    for (int offset = 0; offset < kTileDimension; offset += kBlockRows)
    {
        const int local_output_row = local_row_base + offset;
        const uint32_t output_row = blockIdx.x * kTileDimension + local_output_row;

        if (output_row < columns && output_column_base < rows)
        {
            const int logical_word = local_output_row / vector_width;
            const int element_in_word = local_output_row % vector_width;
            uint32_t packed = 0;

#pragma unroll
            for (int element = 0; element < vector_width; ++element)
            {
                if (static_cast<uint32_t>(element) < rows - output_column_base)
                {
                    const int source_row = local_word * vector_width + element;
                    const int physical_word = shared_word_column<vector_width, Layout>(source_row, logical_word);
                    const uint32_t source = tile[source_row][physical_word];
                    const uint32_t value = (source >> (element_in_word * element_bits)) & element_mask;
                    packed |= value << (element * element_bits);
                }
            }

            const size_t output_element_offset = static_cast<size_t>(output_row) * rows + output_column_base;
            uint8_t *destination = output + output_element_offset * ElementBytes;
            const bool full_pack = rows - output_column_base >= vector_width;
            const bool aligned = (reinterpret_cast<uintptr_t>(destination) & (alignof(Pack) - 1U)) == 0;

            if (full_pack && aligned)
            {
                store_aligned_pack<Pack>(destination, packed);
            }
            else
            {
#pragma unroll
                for (int element = 0; element < vector_width; ++element)
                {
                    if (static_cast<uint32_t>(element) < rows - output_column_base)
                    {
                        store_element_bits<ElementBytes>(destination + element * ElementBytes, packed >> (element * element_bits));
                    }
                }
            }
        }
    }
}

template <typename Pack, int ElementBytes>
cudaError_t launch_transpose_vectorized(
    const void *input,
    void *output,
    uint32_t rows,
    uint32_t columns,
    TransposeSharedMemoryLayout shared_memory_layout,
    cudaStream_t stream)
{
    constexpr int vector_width = kPackBytes / ElementBytes;
    constexpr int words_per_row = kTileDimension / vector_width;
    const dim3 block(words_per_row, kBlockRows);
    const dim3 grid(
        (columns - 1U) / kTileDimension + 1U,
        (rows - 1U) / kTileDimension + 1U);
    const auto *typed_input = static_cast<const uint8_t *>(input);
    auto *typed_output = static_cast<uint8_t *>(output);

    switch (shared_memory_layout)
    {
        case TransposeSharedMemoryLayout::kPadded:
            transpose_vectorized_kernel<
                Pack,
                ElementBytes,
                KernelSharedMemoryLayout::kPadded>
                <<<grid, block, 0, stream>>>(
                    typed_input,
                    typed_output,
                    rows,
                    columns);
            break;
        case TransposeSharedMemoryLayout::kSwizzled:
            transpose_vectorized_kernel<
                Pack,
                ElementBytes,
                KernelSharedMemoryLayout::kSwizzled>
                <<<grid, block, 0, stream>>>(
                    typed_input,
                    typed_output,
                    rows,
                    columns);
            break;
        default:
            return cudaErrorInvalidValue;
    }

    return cudaGetLastError();
}

}  // namespace

cudaError_t transpose_vectorized_cuda(
    const void *input,
    void *output,
    uint32_t rows,
    uint32_t columns,
    TransposeDataType data_type,
    TransposeSharedMemoryLayout shared_memory_layout,
    cudaStream_t stream)
{
    if (input == nullptr || output == nullptr || input == output ||
        rows == 0 || columns == 0 ||
        transpose_storage_bytes(rows, columns, data_type) == 0 ||
        transpose_storage_bytes(columns, rows, data_type) == 0)
    {
        return cudaErrorInvalidValue;
    }

    if (shared_memory_layout != TransposeSharedMemoryLayout::kPadded &&
        shared_memory_layout != TransposeSharedMemoryLayout::kSwizzled)
    {
        return cudaErrorInvalidValue;
    }

    switch (data_type)
    {
        case TransposeDataType::kBfloat16:
            return launch_transpose_vectorized<Bfloat16x2Pack, 2>(
                input,
                output,
                rows,
                columns,
                shared_memory_layout,
                stream);
        case TransposeDataType::kFloat8E4M3:
            return launch_transpose_vectorized<Float8E4M3x4Pack, 1>(
                input,
                output,
                rows,
                columns,
                shared_memory_layout,
                stream);
        default:
            return cudaErrorInvalidValue;
    }
}

}  // namespace cudaop
