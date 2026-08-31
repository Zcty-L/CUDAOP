#pragma once

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include <cute/tensor.hpp>

namespace cudaop::grouped_gemm
{

namespace fused_lora_detail
{

constexpr int kRank = 16;
constexpr int kBlockM = 32;
constexpr int kDownBlockN = 32;
constexpr int kDownBlockK = 32;
constexpr int kDownPipelineStages = 2;
constexpr int kUpBlockN = 128;
constexpr int kBgradBlockM = 32;
constexpr int kBgradBlockN = 128;
constexpr int kBgradBlockK = 32;
constexpr int kThreads = 128;

using Element = cute::bfloat16_t;

__device__ inline int find_expert(
    int tile_index,
    const int32_t* cumulative_tiles,
    int experts)
{
    int low = 0;
    int high = experts;
    while (low < high)
    {
        const int middle = (low + high) / 2;
        if (cumulative_tiles[middle] <= tile_index)
        {
            low = middle + 1;
        }
        else
        {
            high = middle;
        }
    }
    return low;
}

__global__ __launch_bounds__(kThreads) void fused_down_up_kernel(
    const Element* input,
    const Element* down_weight,
    const Element* up_weight_transposed,
    Element* hidden,
    Element* output,
    const int32_t* cumulative_tiles,
    const int32_t* token_offsets,
    const int32_t* token_counts,
    int experts,
    int input_size,
    int output_size)
{
    using namespace cute;

    const int tile_index = static_cast<int>(blockIdx.x);
    const int expert = find_expert(
        tile_index,
        cumulative_tiles,
        experts);
    const int previous_tiles =
        expert == 0 ? 0 : cumulative_tiles[expert - 1];
    const int local_tile = tile_index - previous_tiles;
    const int token_offset = token_offsets[expert];
    const int row_start = token_offset + local_tile * kBlockM;
    const int valid_rows = min(
        kBlockM,
        token_counts[expert] - local_tile * kBlockM);

    const auto swizzle_64 = composition(
        Swizzle<3, 3, 3>{},
        cute::Layout<
            cute::Shape<_8, cute::Shape<_8, _8>>,
            cute::Stride<_8, cute::Stride<_1, _64>>>{});
    const auto swizzle_32 = composition(
        Swizzle<2, 3, 3>{},
        cute::Layout<
            cute::Shape<_8, _32>,
            cute::Stride<_32, _1>>{});
    const auto swizzle_16 = composition(
        Swizzle<1, 3, 3>{},
        cute::Layout<
            cute::Shape<_8, _16>,
            cute::Stride<_16, _1>>{});
    const auto down_a_layout = tile_to_shape(
        swizzle_32,
        make_shape(
            Int<kBlockM>{},
            Int<kDownBlockK>{},
            Int<kDownPipelineStages>{}));
    const auto down_b_layout = tile_to_shape(
        swizzle_32,
        make_shape(
            Int<kDownBlockN>{},
            Int<kDownBlockK>{},
            Int<kDownPipelineStages>{}));
    const auto down_c_layout = make_layout(
        make_shape(Int<kBlockM>{}, Int<kDownBlockN>{}),
        make_stride(Int<kDownBlockN>{}, Int<1>{}));
    const auto up_a_layout = tile_to_shape(
        swizzle_16,
        make_shape(Int<kBlockM>{}, Int<kRank>{}));
    const auto up_b_layout = tile_to_shape(
        swizzle_16,
        make_shape(Int<kUpBlockN>{}, Int<kRank>{}));
    const auto up_c_layout = tile_to_shape(
        swizzle_64,
        make_shape(Int<kBlockM>{}, Int<kUpBlockN>{}));

    const auto tiled_mma = make_tiled_mma(
        SM80_16x8x16_F32BF16BF16F32_TN{},
        cute::Layout<cute::Shape<_2, _2>>{},
        cute::Tile<_32, _32, _16>{});
    const auto smem_to_register =
        Copy_Atom<SM75_U32x4_LDSM_N, Element>{};
    const auto global_to_shared = Copy_Atom<
        SM80_CP_ASYNC_CACHEALWAYS<uint128_t>,
        Element>{};

    extern __shared__ char shared_memory_bytes[];
    Element* shared_memory =
        reinterpret_cast<Element*>(shared_memory_bytes);
    Tensor shared_down_a = make_tensor(
        make_smem_ptr(shared_memory),
        down_a_layout);
    Tensor shared_down_b = make_tensor(
        make_smem_ptr(
            shared_memory + cosize(down_a_layout)),
        down_b_layout);
    Tensor shared_down_c = make_tensor(
        make_smem_ptr(shared_memory),
        down_c_layout);

    ThrMMA down_thread_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor down_thread_c = down_thread_mma.partition_C(shared_down_c);
    Tensor down_accumulator =
        down_thread_mma.make_fragment_C(down_thread_c);
    clear(down_accumulator);

    TiledCopy down_copy_a = make_tiled_copy_A(
        smem_to_register,
        tiled_mma);
    ThrCopy down_thread_copy_a =
        down_copy_a.get_slice(threadIdx.x);
    Tensor down_register_a =
        down_thread_mma.partition_fragment_A(
            shared_down_a(_, _, Int<0>{}));
    Tensor down_thread_shared_a =
        down_thread_copy_a.partition_S(shared_down_a);
    Tensor down_thread_register_a =
        down_thread_copy_a.retile_D(down_register_a);

    TiledCopy down_copy_b = make_tiled_copy_B(
        smem_to_register,
        tiled_mma);
    ThrCopy down_thread_copy_b =
        down_copy_b.get_slice(threadIdx.x);
    Tensor down_register_b =
        down_thread_mma.partition_fragment_B(
            shared_down_b(_, _, Int<0>{}));
    Tensor down_thread_shared_b =
        down_thread_copy_b.partition_S(shared_down_b);
    Tensor down_thread_register_b =
        down_thread_copy_b.retile_D(down_register_b);

    auto load_down_tile = [&](int reduction_start, int stage)
    {
        const int vector_index = static_cast<int>(threadIdx.x);
        const int a_row = vector_index / (kDownBlockK / 8);
        const int a_reduction = vector_index % (kDownBlockK / 8) * 8;
        const int a_global_reduction = reduction_start + a_reduction;
        if (a_row < valid_rows && input_size % 8 == 0 &&
            a_global_reduction + 7 < input_size)
        {
            Tensor global_vector = make_tensor(
                make_gmem_ptr(
                    input +
                    (row_start + a_row) * input_size +
                    a_global_reduction),
                make_shape(Int<8>{}));
            Tensor shared_vector = make_tensor(
                make_smem_ptr(
                    &shared_down_a(
                        a_row,
                        a_reduction,
                        stage)),
                make_shape(Int<8>{}));
            copy(global_to_shared, global_vector, shared_vector);
        }
        else
        {
            CUTE_UNROLL
            for (int offset = 0; offset < 8; ++offset)
            {
                const int global_reduction =
                    a_global_reduction + offset;
                Element value = Element(0.0F);
                if (a_row < valid_rows &&
                    global_reduction < input_size)
                {
                    value = input[
                        (row_start + a_row) * input_size +
                        global_reduction];
                }
                shared_down_a(
                    a_row,
                    a_reduction + offset,
                    stage) = value;
            }
        }

        const int b_rank = vector_index / (kDownBlockK / 8);
        const int b_reduction = vector_index % (kDownBlockK / 8) * 8;
        const int b_global_reduction = reduction_start + b_reduction;
        if (b_rank < kRank && input_size % 8 == 0 &&
            b_global_reduction + 7 < input_size)
        {
            Tensor global_vector = make_tensor(
                make_gmem_ptr(
                    down_weight +
                    (expert * kRank + b_rank) * input_size +
                    b_global_reduction),
                make_shape(Int<8>{}));
            Tensor shared_vector = make_tensor(
                make_smem_ptr(
                    &shared_down_b(
                        b_rank,
                        b_reduction,
                        stage)),
                make_shape(Int<8>{}));
            copy(global_to_shared, global_vector, shared_vector);
        }
        else
        {
            CUTE_UNROLL
            for (int offset = 0; offset < 8; ++offset)
            {
                const int global_reduction =
                    b_global_reduction + offset;
                Element value = Element(0.0F);
                if (b_rank < kRank &&
                    global_reduction < input_size)
                {
                    value = down_weight[
                        (expert * kRank + b_rank) * input_size +
                        global_reduction];
                }
                shared_down_b(
                    b_rank,
                    b_reduction + offset,
                    stage) = value;
            }
        }
        cp_async_fence();
    };

    int read_stage = 0;
    load_down_tile(0, read_stage);
    for (int reduction_start = 0;
         reduction_start < input_size;
         reduction_start += kDownBlockK)
    {
        cp_async_wait<0>();
        __syncthreads();
        const int write_stage = 1 - read_stage;
        if (reduction_start + kDownBlockK < input_size)
        {
            load_down_tile(
                reduction_start + kDownBlockK,
                write_stage);
        }
        CUTE_UNROLL
        for (int k_block = 0;
             k_block < size<2>(down_register_a);
             ++k_block)
        {
            copy(
                smem_to_register,
                down_thread_shared_a(
                    _, _, k_block, read_stage),
                down_thread_register_a(_, _, k_block));
            copy(
                smem_to_register,
                down_thread_shared_b(
                    _, _, k_block, read_stage),
                down_thread_register_b(_, _, k_block));
            gemm(
                tiled_mma,
                down_register_a(_, _, k_block),
                down_register_b(_, _, k_block),
                down_accumulator);
        }
        read_stage = write_stage;
    }
    cp_async_wait<0>();
    __syncthreads();

    Tensor down_output_fragment =
        make_fragment_like<Element>(down_accumulator);
    CUTE_UNROLL
    for (int index = 0; index < size(down_accumulator); ++index)
    {
        down_output_fragment(index) = Element(
            static_cast<float>(down_accumulator(index)));
    }

    copy(down_output_fragment, down_thread_c);
    __syncthreads();

    // shared_up_a 位于 up epilogue tile 之后，因此循环处理所有输出列时
    // 都不会被覆盖。hidden 同时写回 GMEM 供反向使用，但第二个 GEMM
    // 只从这份 SMEM 副本读取。
    Tensor shared_up_a = make_tensor(
        make_smem_ptr(
            shared_memory + cosize(up_c_layout)),
        up_a_layout);
    for (int index = static_cast<int>(threadIdx.x);
         index < kBlockM * kRank;
         index += kThreads)
    {
        const int row = index / kRank;
        const int rank = index % kRank;
        const Element value = shared_down_c(row, rank);
        shared_up_a(row, rank) = value;
        if (row < valid_rows)
        {
            hidden[(row_start + row) * kRank + rank] = value;
        }
    }
    __syncthreads();

    Tensor shared_up_b = make_tensor(
        make_smem_ptr(shared_memory),
        up_b_layout);
    Tensor shared_up_c = make_tensor(
        make_smem_ptr(shared_memory),
        up_c_layout);
    TiledCopy up_copy_a = make_tiled_copy_A(
        smem_to_register,
        tiled_mma);
    ThrCopy up_thread_copy_a = up_copy_a.get_slice(threadIdx.x);
    TiledCopy up_copy_b = make_tiled_copy_B(
        smem_to_register,
        tiled_mma);
    ThrCopy up_thread_copy_b = up_copy_b.get_slice(threadIdx.x);
    const auto shared_to_global = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, Element>{},
        cute::Layout<
            cute::Shape<_16, _8>,
            cute::Stride<_8, _1>>{},
        cute::Layout<cute::Shape<_1, _8>>{});
    ThrCopy thread_shared_to_global =
        shared_to_global.get_slice(threadIdx.x);
    Tensor thread_epilogue_shared =
        thread_shared_to_global.partition_S(shared_up_c);

    for (int column_start = 0;
         column_start < output_size;
         column_start += kUpBlockN)
    {
        for (int vector_index = static_cast<int>(threadIdx.x);
             vector_index < kUpBlockN * kRank / 8;
             vector_index += kThreads)
        {
            const int column = vector_index / (kRank / 8);
            const int rank = vector_index % (kRank / 8) * 8;
            const int global_column = column_start + column;
            if (global_column < output_size)
            {
                Tensor global_vector = make_tensor(
                    make_gmem_ptr(
                        up_weight_transposed +
                        (expert * output_size + global_column) *
                            kRank +
                        rank),
                    make_shape(Int<8>{}));
                Tensor shared_vector = make_tensor(
                    make_smem_ptr(
                        &shared_up_b(column, rank)),
                    make_shape(Int<8>{}));
                copy(
                    global_to_shared,
                    global_vector,
                    shared_vector);
            }
            else
            {
                CUTE_UNROLL
                for (int offset = 0; offset < 8; ++offset)
                {
                    shared_up_b(column, rank + offset) =
                        Element(0.0F);
                }
            }
        }
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        ThrMMA up_thread_mma = tiled_mma.get_slice(threadIdx.x);
        Tensor up_thread_c = up_thread_mma.partition_C(shared_up_c);
        Tensor up_accumulator =
            up_thread_mma.make_fragment_C(up_thread_c);
        clear(up_accumulator);
        Tensor up_register_a =
            up_thread_mma.partition_fragment_A(shared_up_a);
        Tensor up_register_b =
            up_thread_mma.partition_fragment_B(shared_up_b);
        Tensor up_thread_shared_a =
            up_thread_copy_a.partition_S(shared_up_a);
        Tensor up_thread_register_a =
            up_thread_copy_a.retile_D(up_register_a);
        Tensor up_thread_shared_b =
            up_thread_copy_b.partition_S(shared_up_b);
        Tensor up_thread_register_b =
            up_thread_copy_b.retile_D(up_register_b);

        CUTE_UNROLL
        for (int k_block = 0;
             k_block < size<2>(up_register_a);
             ++k_block)
        {
            copy(
                smem_to_register,
                up_thread_shared_a(_, _, k_block),
                up_thread_register_a(_, _, k_block));
            copy(
                smem_to_register,
                up_thread_shared_b(_, _, k_block),
                up_thread_register_b(_, _, k_block));
            gemm(
                tiled_mma,
                up_register_a(_, _, k_block),
                up_register_b(_, _, k_block),
                up_accumulator);
        }
        __syncthreads();

        Tensor up_output_fragment =
            make_fragment_like<Element>(up_accumulator);
        CUTE_UNROLL
        for (int index = 0; index < size(up_accumulator); ++index)
        {
            up_output_fragment(index) = Element(
                static_cast<float>(up_accumulator(index)));
        }
        copy(up_output_fragment, up_thread_c);
        __syncthreads();

        if (valid_rows == kBlockM &&
            column_start + kUpBlockN <= output_size &&
            output_size % 8 == 0)
        {
            Tensor global_output = make_tensor(
                make_gmem_ptr(
                    output + row_start * output_size + column_start),
                make_shape(Int<kBlockM>{}, Int<kUpBlockN>{}),
                make_stride(output_size, Int<1>{}));
            Tensor thread_epilogue_global =
                thread_shared_to_global.partition_D(global_output);
            copy(
                shared_to_global,
                thread_epilogue_shared,
                thread_epilogue_global);
        }
        else
        {
            for (int index = static_cast<int>(threadIdx.x);
                 index < kBlockM * kUpBlockN;
                 index += kThreads)
            {
                const int row = index / kUpBlockN;
                const int column = index % kUpBlockN;
                const int global_column = column_start + column;
                if (row < valid_rows && global_column < output_size)
                {
                    output[
                        (row_start + row) * output_size +
                        global_column] = shared_up_c(row, column);
                }
            }
        }
        __syncthreads();
    }
}

__global__ __launch_bounds__(kThreads) void lora_bgrad_kernel(
    const Element* lhs,
    const Element* rhs,
    Element* output,
    const int32_t* token_offsets,
    const int32_t* token_counts,
    int output_size)
{
    using namespace cute;

    const int expert = static_cast<int>(blockIdx.x);
    const int column_start =
        static_cast<int>(blockIdx.y) * kBgradBlockN;
    const int token_offset = token_offsets[expert];
    const int token_count = token_counts[expert];

    const auto swizzle_64 = composition(
        Swizzle<3, 3, 3>{},
        cute::Layout<
            cute::Shape<_8, cute::Shape<_8, _8>>,
            cute::Stride<_8, cute::Stride<_1, _64>>>{});
    const auto shared_a_layout = make_layout(
        make_shape(Int<kBgradBlockM>{}, Int<kBgradBlockK>{}),
        make_stride(Int<kBgradBlockK>{}, Int<1>{}));
    const auto shared_b_layout = make_layout(
        make_shape(Int<kBgradBlockN>{}, Int<kBgradBlockK>{}),
        make_stride(Int<kBgradBlockK>{}, Int<1>{}));
    const auto shared_c_layout = tile_to_shape(
        swizzle_64,
        make_shape(Int<kBgradBlockM>{}, Int<kBgradBlockN>{}));
    const auto tiled_mma = make_tiled_mma(
        SM80_16x8x16_F32BF16BF16F32_TN{},
        cute::Layout<cute::Shape<_2, _2>>{},
        cute::Tile<_32, _32, _16>{});
    const auto smem_to_register =
        Copy_Atom<SM75_U32x4_LDSM_N, Element>{};

    extern __shared__ char shared_memory_bytes[];
    Element* shared_memory =
        reinterpret_cast<Element*>(shared_memory_bytes);
    Tensor shared_a = make_tensor(
        make_smem_ptr(shared_memory),
        shared_a_layout);
    Tensor shared_b = make_tensor(
        make_smem_ptr(shared_memory + cosize(shared_a_layout)),
        shared_b_layout);
    Tensor shared_c = make_tensor(
        make_smem_ptr(shared_memory),
        shared_c_layout);

    ThrMMA thread_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor thread_c = thread_mma.partition_C(shared_c);
    Tensor accumulator = thread_mma.make_fragment_C(thread_c);
    clear(accumulator);

    TiledCopy copy_a = make_tiled_copy_A(
        smem_to_register,
        tiled_mma);
    ThrCopy thread_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor register_a = thread_mma.partition_fragment_A(shared_a);
    Tensor thread_shared_a = thread_copy_a.partition_S(shared_a);
    Tensor thread_register_a = thread_copy_a.retile_D(register_a);

    TiledCopy copy_b = make_tiled_copy_B(
        smem_to_register,
        tiled_mma);
    ThrCopy thread_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor register_b = thread_mma.partition_fragment_B(shared_b);
    Tensor thread_shared_b = thread_copy_b.partition_S(shared_b);
    Tensor thread_register_b = thread_copy_b.retile_D(register_b);

    for (int token_start = 0;
         token_start < token_count;
         token_start += kBgradBlockK)
    {
        for (int index = static_cast<int>(threadIdx.x);
             index < kBgradBlockM * kBgradBlockK;
             index += kThreads)
        {
            const int rank = index / kBgradBlockK;
            const int token = index % kBgradBlockK;
            Element value = Element(0.0F);
            if (rank < kRank && token_start + token < token_count)
            {
                value = lhs[
                    (token_offset + token_start + token) * kRank +
                    rank];
            }
            shared_a(rank, token) = value;
        }
        for (int index = static_cast<int>(threadIdx.x);
             index < kBgradBlockN * kBgradBlockK;
             index += kThreads)
        {
            const int column = index / kBgradBlockK;
            const int token = index % kBgradBlockK;
            const int global_column = column_start + column;
            Element value = Element(0.0F);
            if (token_start + token < token_count &&
                global_column < output_size)
            {
                value = rhs[
                    (token_offset + token_start + token) * output_size +
                    global_column];
            }
            shared_b(column, token) = value;
        }
        __syncthreads();

        CUTE_UNROLL
        for (int k_block = 0;
             k_block < size<2>(register_a);
             ++k_block)
        {
            copy(
                smem_to_register,
                thread_shared_a(_, _, k_block),
                thread_register_a(_, _, k_block));
            copy(
                smem_to_register,
                thread_shared_b(_, _, k_block),
                thread_register_b(_, _, k_block));
            gemm(
                tiled_mma,
                register_a(_, _, k_block),
                register_b(_, _, k_block),
                accumulator);
        }
        __syncthreads();
    }

    Tensor output_fragment = make_fragment_like<Element>(accumulator);
    CUTE_UNROLL
    for (int index = 0; index < size(accumulator); ++index)
    {
        output_fragment(index) = Element(
            static_cast<float>(accumulator(index)));
    }
    copy(output_fragment, thread_c);
    __syncthreads();

    for (int index = static_cast<int>(threadIdx.x);
         index < kRank * kBgradBlockN;
         index += kThreads)
    {
        const int rank = index / kBgradBlockN;
        const int column = index % kBgradBlockN;
        const int global_column = column_start + column;
        if (global_column < output_size)
        {
            output[
                (expert * kRank + rank) * output_size +
                global_column] = shared_c(rank, column);
        }
    }
}

inline void check_int32_metadata(
    const torch::Tensor& tensor,
    const torch::Tensor& input,
    int64_t experts,
    const char* name)
{
    TORCH_CHECK(tensor.is_cuda(), name, " 必须是 CUDA Tensor");
    TORCH_CHECK(
        tensor.device() == input.device(),
        name,
        " 必须与输入位于同一设备");
    TORCH_CHECK(
        tensor.scalar_type() == torch::kInt32,
        name,
        " 必须使用 int32");
    TORCH_CHECK(
        tensor.dim() == 1 && tensor.numel() == experts,
        name,
        " 必须是一维且长度等于 expert 数");
    TORCH_CHECK(tensor.is_contiguous(), name, " 必须连续");
}

}  // namespace fused_lora_detail

inline std::vector<torch::Tensor> fused_lora_forward(
    torch::Tensor input,
    torch::Tensor down_weight,
    torch::Tensor up_weight_transposed,
    torch::Tensor cumulative_tiles,
    torch::Tensor token_offsets,
    torch::Tensor token_counts,
    int64_t total_tiles)
{
    using namespace fused_lora_detail;

    TORCH_CHECK(input.is_cuda(), "input 必须是 CUDA Tensor");
    TORCH_CHECK(down_weight.is_cuda(), "down_weight 必须是 CUDA Tensor");
    TORCH_CHECK(
        up_weight_transposed.is_cuda(),
        "up_weight_transposed 必须是 CUDA Tensor");
    TORCH_CHECK(input.dim() == 2, "input 必须是二维 Tensor");
    TORCH_CHECK(
        down_weight.dim() == 3 && down_weight.size(1) == kRank,
        "down_weight 形状必须是 [E, 16, D]");
    TORCH_CHECK(
        up_weight_transposed.dim() == 3 &&
            up_weight_transposed.size(2) == kRank,
        "up_weight_transposed 形状必须是 [E, I, 16]");
    TORCH_CHECK(
        input.size(1) == down_weight.size(2),
        "input 与 down_weight 的收缩维度不匹配");
    TORCH_CHECK(
        down_weight.size(0) == up_weight_transposed.size(0),
        "down_weight 与 up_weight_transposed 的 expert 数不匹配");
    TORCH_CHECK(
        input.device() == down_weight.device() &&
            input.device() == up_weight_transposed.device(),
        "输入和权重必须位于同一 CUDA 设备");
    TORCH_CHECK(
        input.scalar_type() == torch::kBFloat16 &&
            down_weight.scalar_type() == torch::kBFloat16 &&
            up_weight_transposed.scalar_type() == torch::kBFloat16,
        "输入和权重必须使用 bfloat16");
    TORCH_CHECK(
        input.is_contiguous() && down_weight.is_contiguous() &&
            up_weight_transposed.is_contiguous(),
        "输入和权重必须连续");

    const int64_t experts = down_weight.size(0);
    TORCH_CHECK(experts > 0, "expert 数必须大于零");
    check_int32_metadata(
        cumulative_tiles,
        input,
        experts,
        "cumulative_tiles");
    check_int32_metadata(
        token_offsets,
        input,
        experts,
        "token_offsets");
    check_int32_metadata(
        token_counts,
        input,
        experts,
        "token_counts");
    TORCH_CHECK(total_tiles >= 0, "total_tiles 不能为负数");
    TORCH_CHECK(
        input.size(0) <= INT32_MAX &&
            input.size(1) <= INT32_MAX &&
            up_weight_transposed.size(1) <= INT32_MAX &&
            experts <= INT32_MAX && total_tiles <= INT32_MAX,
        "当前 CUTLASS fusion 只支持 int32 范围内的形状");

    torch::Tensor hidden = torch::empty(
        {input.size(0), kRank},
        input.options());
    torch::Tensor output = torch::empty(
        {input.size(0), up_weight_transposed.size(1)},
        input.options());
    if (input.size(0) == 0)
    {
        TORCH_CHECK(
            total_tiles == 0,
            "空输入的 total_tiles 必须为零");
        return {hidden, output};
    }
    TORCH_CHECK(total_tiles > 0, "非空输入的 total_tiles 必须大于零");

    constexpr int down_shared_elements =
        kDownPipelineStages *
        (kBlockM * kDownBlockK +
         kDownBlockN * kDownBlockK);
    constexpr int up_shared_elements =
        kBlockM * kUpBlockN + kBlockM * kRank;
    constexpr int shared_elements =
        down_shared_elements > up_shared_elements
        ? down_shared_elements
        : up_shared_elements;
    constexpr int shared_bytes =
        shared_elements * static_cast<int>(sizeof(Element));
    const auto kernel = fused_down_up_kernel;
    check_cuda(
        cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_bytes),
        "设置 fused LoRA 动态共享内存");
    kernel<<<
        static_cast<unsigned int>(total_tiles),
        kThreads,
        shared_bytes,
        c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const Element*>(input.data_ptr()),
        reinterpret_cast<const Element*>(down_weight.data_ptr()),
        reinterpret_cast<const Element*>(
            up_weight_transposed.data_ptr()),
        reinterpret_cast<Element*>(hidden.data_ptr()),
        reinterpret_cast<Element*>(output.data_ptr()),
        cumulative_tiles.data_ptr<int32_t>(),
        token_offsets.data_ptr<int32_t>(),
        token_counts.data_ptr<int32_t>(),
        static_cast<int>(experts),
        static_cast<int>(input.size(1)),
        static_cast<int>(up_weight_transposed.size(1)));
    check_cuda(cudaGetLastError(), "启动 fused LoRA kernel");
    return {hidden, output};
}

inline std::vector<torch::Tensor> fused_lora_backward(
    torch::Tensor grad_output,
    torch::Tensor up_weight,
    torch::Tensor down_weight_transposed,
    torch::Tensor cumulative_tiles,
    torch::Tensor token_offsets,
    torch::Tensor token_counts,
    int64_t total_tiles)
{
    // 反向输入梯度与前向 down/up 具有相同的两级 GEMM 结构：
    // grad_output @ up_weight.T 得到 grad_hidden，再用
    // grad_hidden @ down_weight 得到 grad_input。因此复用同一个 CuTe
    // kernel，并将 grad_hidden 写回一次供权重梯度计算使用。
    return fused_lora_forward(
        grad_output,
        up_weight,
        down_weight_transposed,
        cumulative_tiles,
        token_offsets,
        token_counts,
        total_tiles);
}

inline torch::Tensor lora_bgrad(
    torch::Tensor lhs,
    torch::Tensor rhs,
    torch::Tensor token_offsets,
    torch::Tensor token_counts)
{
    using namespace fused_lora_detail;

    TORCH_CHECK(lhs.is_cuda(), "lhs 必须是 CUDA Tensor");
    TORCH_CHECK(rhs.is_cuda(), "rhs 必须是 CUDA Tensor");
    TORCH_CHECK(
        lhs.device() == rhs.device(),
        "lhs 和 rhs 必须位于同一 CUDA 设备");
    TORCH_CHECK(
        lhs.dim() == 2 && lhs.size(1) == kRank,
        "lhs 形状必须是 [N, 16]");
    TORCH_CHECK(
        rhs.dim() == 2 && rhs.size(0) == lhs.size(0),
        "rhs 形状必须是 [N, K]");
    TORCH_CHECK(
        lhs.scalar_type() == torch::kBFloat16 &&
            rhs.scalar_type() == torch::kBFloat16,
        "lhs 和 rhs 必须使用 bfloat16");
    TORCH_CHECK(
        lhs.is_contiguous() && rhs.is_contiguous(),
        "lhs 和 rhs 必须连续");

    const int64_t experts = token_counts.numel();
    TORCH_CHECK(experts > 0, "expert 数必须大于零");
    TORCH_CHECK(rhs.size(1) > 0, "rhs 的列数必须大于零");
    check_int32_metadata(
        token_offsets,
        lhs,
        experts,
        "token_offsets");
    check_int32_metadata(
        token_counts,
        lhs,
        experts,
        "token_counts");
    TORCH_CHECK(
        experts <= INT32_MAX && rhs.size(1) <= INT32_MAX,
        "当前 CUTLASS bgrad 只支持 int32 范围内的形状");

    torch::Tensor output = torch::empty(
        {experts, kRank, rhs.size(1)},
        lhs.options());
    constexpr int shared_elements =
        kBgradBlockM * kBgradBlockK +
        kBgradBlockN * kBgradBlockK;
    constexpr int shared_bytes =
        shared_elements * static_cast<int>(sizeof(Element));
    const auto kernel = lora_bgrad_kernel;
    const int64_t column_tiles =
        (rhs.size(1) + kBgradBlockN - 1) / kBgradBlockN;
    const dim3 grid(
        static_cast<unsigned int>(experts),
        static_cast<unsigned int>(column_tiles));
    kernel<<<
        grid,
        kThreads,
        shared_bytes,
        c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const Element*>(lhs.data_ptr()),
        reinterpret_cast<const Element*>(rhs.data_ptr()),
        reinterpret_cast<Element*>(output.data_ptr()),
        token_offsets.data_ptr<int32_t>(),
        token_counts.data_ptr<int32_t>(),
        static_cast<int>(rhs.size(1)));
    check_cuda(cudaGetLastError(), "启动 LoRA bgrad kernel");
    return output;
}

}  // namespace cudaop::grouped_gemm
