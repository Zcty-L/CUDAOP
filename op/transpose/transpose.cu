#include "op/transpose/transpose.h"

#include <cstddef>
#include <cstdint>
#include <limits>

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace cudaop
{
namespace
{

constexpr int kTileDimension = 32;
constexpr int kBlockRows = 8;
constexpr int kSharedMemoryBankBytes = 4;

enum class KernelSharedMemoryLayout
{
    kPadded,
    kSwizzled
};

static_assert(sizeof(float) == 4, "float must use a 32-bit container");
static_assert(
    sizeof(__nv_bfloat16) == 2,
    "bfloat16 must use a 16-bit container");
static_assert(
    sizeof(__nv_fp8_e4m3) == 1,
    "FP8 E4M3 must use an 8-bit container");
static_assert(
    sizeof(__nv_fp4x2_e2m1) == 1,
    "two packed FP4 E2M1 values must use one byte");

template <typename Element, KernelSharedMemoryLayout Layout>
__device__ __forceinline__ int shared_column(int row, int column)
{
    static_assert(
        sizeof(Element) <= kSharedMemoryBankBytes,
        "transpose supports element containers no wider than one bank");
    static_assert(
        kSharedMemoryBankBytes % sizeof(Element) == 0,
        "element size must divide the shared-memory bank width");

    if constexpr (Layout == KernelSharedMemoryLayout::kPadded)
    {
        return column;
    }
    else
    {
        constexpr int elements_per_bank = kSharedMemoryBankBytes / sizeof(Element); // 4/2 = 2
        const int logical_bank = column / elements_per_bank;     // column / 2
        const int element_in_bank = column % elements_per_bank;  // column % 2
        // A compact row contains 32 / elements_per_bank bank words. Its
        // natural row-stride contribution supplies the low log2(E) row bits
        // to the bank index, where E is elements_per_bank. XOR the remaining
        // row bits into the bank-within-row so a column read spans all banks.
        const int physical_bank = logical_bank ^ (row / elements_per_bank);
        return physical_bank * elements_per_bank + element_in_bank;
    }
}

template <typename Element, KernelSharedMemoryLayout Layout>
__global__ void transpose_kernel(
    const Element *__restrict__ input,
    Element *__restrict__ output,
    uint32_t rows,
    uint32_t columns)
{
    constexpr int elements_per_bank = kSharedMemoryBankBytes / sizeof(Element);
    constexpr int shared_columns =
        Layout == KernelSharedMemoryLayout::kPadded ? kTileDimension + elements_per_bank : kTileDimension;

    __shared__ __align__(128) Element tile[kTileDimension][shared_columns]; // [32][shared_columns]

    const int local_column = static_cast<int>(threadIdx.x);
    const int local_row_base = static_cast<int>(threadIdx.y);
    const uint32_t input_column = blockIdx.x * kTileDimension + threadIdx.x;

#pragma unroll
    for (int offset = 0; offset < kTileDimension; offset += kBlockRows) // += 8
    {
        const int local_row = local_row_base + offset;
        const uint32_t input_row = blockIdx.y * kTileDimension + local_row;
        if (input_row < rows && input_column < columns)
        {
            tile[local_row][shared_column<Element, Layout>(local_row, local_column)] = input[static_cast<size_t>(input_row) * columns + input_column];
        }
    }

    __syncthreads();

    const uint32_t output_column = blockIdx.y * kTileDimension + threadIdx.x;

#pragma unroll
    for (int offset = 0; offset < kTileDimension; offset += kBlockRows)
    {
        const int local_output_row = local_row_base + offset;
        const uint32_t output_row = blockIdx.x * kTileDimension + local_output_row;
        if (output_row < columns && output_column < rows)
        {
            output[static_cast<size_t>(output_row) * rows + output_column] = tile[local_column][shared_column<Element, Layout>(local_column, local_output_row)];
        }
    }
}

__device__ __forceinline__ uint8_t load_fp4(
    const uint8_t *input,
    uint32_t row,
    uint32_t column,
    size_t row_stride)
{
    const uint8_t packed = input[static_cast<size_t>(row) * row_stride + column / 2U];
    const uint32_t shift = (column & 1U) * 4U;
    return static_cast<uint8_t>((packed >> shift) & 0x0FU);
}

template <KernelSharedMemoryLayout Layout>
__global__ void transpose_fp4_kernel(
    const uint8_t *__restrict__ input,
    uint8_t *__restrict__ output,
    uint32_t rows,
    uint32_t columns)
{
    constexpr int elements_per_bank = kSharedMemoryBankBytes;
    constexpr int shared_columns =
        Layout == KernelSharedMemoryLayout::kPadded ?
        kTileDimension + elements_per_bank :
        kTileDimension;

    // One byte per logical FP4 value is used only inside shared memory. Global
    // memory remains densely packed at two E2M1 values per byte.
    __shared__ __align__(128)
        uint8_t tile[kTileDimension][shared_columns];

    const int local_column = static_cast<int>(threadIdx.x);
    const int local_row_base = static_cast<int>(threadIdx.y);
    const uint32_t input_column =
        blockIdx.x * kTileDimension + threadIdx.x;
    const size_t input_row_stride =
        (static_cast<size_t>(columns) + 1U) / 2U;

#pragma unroll
    for (int offset = 0; offset < kTileDimension; offset += kBlockRows)
    {
        const int local_row = local_row_base + offset;
        const uint32_t input_row =
            blockIdx.y * kTileDimension + local_row;
        if (input_row < rows && input_column < columns)
        {
            tile[local_row][shared_column<uint8_t, Layout>(
                local_row,
                local_column)] = load_fp4(
                    input,
                    input_row,
                    input_column,
                    input_row_stride);
        }
    }

    __syncthreads();

    // One thread owns an entire output byte so that the two nibble writes
    // cannot race. blockIdx.y starts at a multiple of 32 logical columns and
    // therefore also at a packed-byte boundary.
    if (threadIdx.x < kTileDimension / 2)
    {
        const int first_local_input_row =
            static_cast<int>(threadIdx.x) * 2;
        const uint32_t first_output_column =
            blockIdx.y * kTileDimension + first_local_input_row;
        const size_t output_row_stride =
            (static_cast<size_t>(rows) + 1U) / 2U;

#pragma unroll
        for (int offset = 0;
             offset < kTileDimension;
             offset += kBlockRows)
        {
            const int local_output_row = local_row_base + offset;
            const uint32_t output_row =
                blockIdx.x * kTileDimension + local_output_row;
            if (output_row < columns && first_output_column < rows)
            {
                const uint8_t low =
                    tile[first_local_input_row]
                        [shared_column<uint8_t, Layout>(
                            first_local_input_row,
                            local_output_row)] &
                    0x0FU;
                uint8_t high = 0;
                if (first_output_column + 1U < rows)
                {
                    const int second_local_input_row =
                        first_local_input_row + 1;
                    high =
                        tile[second_local_input_row]
                            [shared_column<uint8_t, Layout>(
                                second_local_input_row,
                                local_output_row)] &
                        0x0FU;
                }

                output[static_cast<size_t>(output_row) *
                           output_row_stride +
                       first_output_column / 2U] =
                    static_cast<uint8_t>(low | (high << 4U));
            }
        }
    }
}

template <typename Element>
cudaError_t launch_transpose(
    const void *input,
    void *output,
    dim3 grid,
    dim3 block,
    uint32_t rows,
    uint32_t columns,
    TransposeSharedMemoryLayout shared_memory_layout,
    cudaStream_t stream)
{
    const auto *typed_input = static_cast<const Element *>(input);
    auto *typed_output = static_cast<Element *>(output);

    switch (shared_memory_layout)
    {
        case TransposeSharedMemoryLayout::kPadded:
            transpose_kernel<Element, KernelSharedMemoryLayout::kPadded><<<grid, block, 0, stream>>>(typed_input, typed_output, rows, columns);
            break;
        case TransposeSharedMemoryLayout::kSwizzled:
            transpose_kernel<Element, KernelSharedMemoryLayout::kSwizzled><<<grid, block, 0, stream>>>(typed_input, typed_output, rows, columns);
            break;
        default:
            return cudaErrorInvalidValue;
    }

    return cudaGetLastError();
}

cudaError_t launch_transpose_fp4(
    const void *input,
    void *output,
    dim3 grid,
    dim3 block,
    uint32_t rows,
    uint32_t columns,
    TransposeSharedMemoryLayout shared_memory_layout,
    cudaStream_t stream)
{
    const auto *typed_input = static_cast<const uint8_t *>(input);
    auto *typed_output = static_cast<uint8_t *>(output);

    switch (shared_memory_layout)
    {
        case TransposeSharedMemoryLayout::kPadded:
            transpose_fp4_kernel<KernelSharedMemoryLayout::kPadded>
                <<<grid, block, 0, stream>>>(
                    typed_input,
                    typed_output,
                    rows,
                    columns);
            break;
        case TransposeSharedMemoryLayout::kSwizzled:
            transpose_fp4_kernel<KernelSharedMemoryLayout::kSwizzled>
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

bool checked_multiply(size_t left, size_t right, size_t *result)
{
    if (right != 0 && left > std::numeric_limits<size_t>::max() / right)
    {
        return false;
    }

    *result = left * right;
    return true;
}

}  // namespace

size_t transpose_storage_bytes(
    uint32_t rows,
    uint32_t columns,
    TransposeDataType data_type)
{
    if (rows == 0 || columns == 0)
    {
        return 0;
    }

    size_t row_bytes = 0;
    switch (data_type)
    {
        case TransposeDataType::kFloat32:
            if (!checked_multiply(columns, sizeof(float), &row_bytes))
            {
                return 0;
            }
            break;
        case TransposeDataType::kBfloat16:
            if (!checked_multiply(
                    columns,
                    sizeof(__nv_bfloat16),
                    &row_bytes))
            {
                return 0;
            }
            break;
        case TransposeDataType::kFloat8E4M3:
            row_bytes = columns;
            break;
        case TransposeDataType::kFloat4E2M1:
            row_bytes = (static_cast<size_t>(columns) + 1U) / 2U;
            break;
        default:
            return 0;
    }

    size_t storage_bytes = 0;
    if (!checked_multiply(rows, row_bytes, &storage_bytes))
    {
        return 0;
    }
    return storage_bytes;
}

const char *transpose_data_type_name(TransposeDataType data_type)
{
    switch (data_type)
    {
        case TransposeDataType::kFloat32:
            return "float";
        case TransposeDataType::kBfloat16:
            return "bf16";
        case TransposeDataType::kFloat8E4M3:
            return "fp8-e4m3";
        case TransposeDataType::kFloat4E2M1:
            return "fp4-e2m1";
        default:
            return "unknown";
    }
}

const char *transpose_layout_name(
    TransposeSharedMemoryLayout shared_memory_layout)
{
    switch (shared_memory_layout)
    {
        case TransposeSharedMemoryLayout::kPadded:
            return "pad";
        case TransposeSharedMemoryLayout::kSwizzled:
            return "swizzle";
        default:
            return "unknown";
    }
}

cudaError_t transpose_cuda(
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

    const dim3 block(kTileDimension, kBlockRows); // 32 x 8 threads per block
    const dim3 grid((columns - 1U) / kTileDimension + 1U, (rows - 1U) / kTileDimension + 1U);

    switch (data_type)
    {
        case TransposeDataType::kFloat32:
            return launch_transpose<float>(
                input,
                output,
                grid,
                block,
                rows,
                columns,
                shared_memory_layout,
                stream);
        case TransposeDataType::kBfloat16:
            return launch_transpose<__nv_bfloat16>(
                input,
                output,
                grid,
                block,
                rows,
                columns,
                shared_memory_layout,
                stream);
        case TransposeDataType::kFloat8E4M3:
            return launch_transpose<__nv_fp8_e4m3>(
                input,
                output,
                grid,
                block,
                rows,
                columns,
                shared_memory_layout,
                stream);
        case TransposeDataType::kFloat4E2M1:
            return launch_transpose_fp4(
                input,
                output,
                grid,
                block,
                rows,
                columns,
                shared_memory_layout,
                stream);
        default:
            return cudaErrorInvalidValue;
    }
}

}  // namespace cudaop
