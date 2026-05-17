#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ uint32_t smem_u32addr(const void *smem_ptr) {
    uint32_t addr;
    asm ("{.reg .u64 u64addr;\n"
         " cvta.to.shared.u64 u64addr, %1;\n"
         " cvt.u32.u64 %0, u64addr;}\n"
         : "=r"(addr) : "l"(smem_ptr));
    return addr;
}

/**
 * SNN Conv 权重加载 — 无边界情况 (C_out 是 64 整数倍)
 *
 * Weights GMEM: [in_features, lda] float, row-major
 *   lda = c_out (16B 对齐, c_out 为 4 的倍数)
 * SMEM tile:    [16, 64] float = 4KB
 * 线程:         256, 每线程 1 次 cp.async (float4)
 * 映射:         row_f4 = tid/16, col_f4 = tid%16
 *               → 16×16 = 256 覆盖 [16,64] SMEM ✓
 *
 * Grid: blockIdx.x 沿 C_out 拆分
 * Loop: 每 block 沿 in_features 迭代 in_features/16 次
 */
__global__ void test_weight_load_full(
    const float* __restrict__ weights,
    float* __restrict__ output,
    int in_features,
    int c_out
) {
    constexpr int K_CHUNK = 16;
    constexpr int C_OUT_TILE = 64;
    constexpr int COPIES_PER_THREAD = (K_CHUNK * C_OUT_TILE) / 256;

    int c_out_start = blockIdx.x * C_OUT_TILE;
    __shared__ __align__(128) float smem[K_CHUNK][C_OUT_TILE];

    const int tid = threadIdx.x;

    for (int k = 0; k < in_features; k += K_CHUNK) {
        int row_f4 = tid / (C_OUT_TILE / 4);
        int col_f4 = tid % (C_OUT_TILE / 4);
        int col    = col_f4 * 4;

        uint32_t s_ptr = smem_u32addr(&smem[row_f4][col]);

        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16;\n"
            :: "r"(s_ptr),
               "l"(&weights[(k + row_f4) * c_out + (c_out_start + col)])
        );

        asm volatile("cp.async.commit_group;\n" :::);
        asm volatile("cp.async.wait_group 0;\n" :::);
        __syncthreads();

#pragma unroll
        for (int i = 0; i < COPIES_PER_THREAD; i++) {
            int linear = tid + i * 256;
            int r = linear / C_OUT_TILE;
            int c = linear % C_OUT_TILE;
            output[(k + r) * c_out + (c_out_start + c)] = smem[r][c];
        }
        __syncthreads();
    }
}

/**
 * SNN Conv 权重加载 — 带边界处理 (使用 cp.async src_size)
 *
 * 边界 block (valid_cols < 64):
 *   - 每线程固定覆盖 1 个 float4 SMEM 槽
 *   - src_size = min(4, max(0, valid_cols - col)) * sizeof(float)
 *     · src_size = 16: 4 floats 全部有效, 完整 copy
 *     · 0 < src_size < 16: 部分有效, 硬件零填 float4 槽内剩余字节
 *     · src_size = 0: NOP, 该 SMEM 槽不在输出范围内, 计算中不会被访问
 *   - commit + wait + __syncthreads
 *
 * 无需预清零: src_size=0 的槽在计算中不会被访问 (padding C_out),
 *   其值不影响结果, 下一个 K-iteration 会被有效数据覆盖或仍为 don't-care
 */
__global__ void test_weight_load_boundary(
    const float* __restrict__ weights,
    float* __restrict__ output,
    int in_features,
    int c_out,
    int padded_c_out
) {
    constexpr int K_CHUNK = 16;
    constexpr int C_OUT_TILE = 64;
    constexpr int COPIES_PER_THREAD = (K_CHUNK * C_OUT_TILE) / 256;

    int c_out_start = blockIdx.x * C_OUT_TILE;
    int valid_cols = c_out_start < c_out ? min(C_OUT_TILE, c_out - c_out_start) : 0;

    __shared__ __align__(128) float smem[K_CHUNK][C_OUT_TILE];

    const int tid = threadIdx.x;

    for (int k = 0; k < in_features; k += K_CHUNK) {
        int row_f4 = tid / (C_OUT_TILE / 4);
        int col_f4 = tid % (C_OUT_TILE / 4);
        int col    = col_f4 * 4;

        int cols_in_f4 = min(4, max(0, valid_cols - col));
        int src_size   = cols_in_f4 * (int)sizeof(float);

        uint32_t s_ptr = smem_u32addr(&smem[row_f4][col]);

        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
            :: "r"(s_ptr),
               "l"(&weights[(k + row_f4) * c_out + (c_out_start + col)]),
               "r"(src_size)
        );

        asm volatile("cp.async.commit_group;\n" :::);
        asm volatile("cp.async.wait_group 0;\n" :::);
        __syncthreads();

        // SMEM → output (padded 宽度, 避免 partial block OOB)
#pragma unroll
        for (int i = 0; i < COPIES_PER_THREAD; i++) {
            int linear = tid + i * 256;
            int r = linear / C_OUT_TILE;
            int c = linear % C_OUT_TILE;
            output[(k + r) * padded_c_out + (c_out_start + c)] = smem[r][c];
        }
        __syncthreads();
    }
}

/**
 * SNN Conv 权重加载 — 处理 in_features + C_out 双边界
 *
 * 在 K 维 (in_features) 和 N 维 (C_out) 均可非整数倍:
 *   valid_rows = min(K_CHUNK, in_features - k)
 *   valid_cols = min(C_OUT_TILE, c_out - c_out_start)
 *
 * 每线程固定覆盖 1 个 (row_f4, col_f4) float4 SMEM 槽:
 *   - row_f4 >= valid_rows: src_size=0 (整行 padding, NOP)
 *   - row_f4 <  valid_rows: src_size = min(4, max(0, valid_cols-col)) * 4
 */
__global__ void test_weight_load_general(
    const float* __restrict__ weights,
    float* __restrict__ output,
    int in_features,
    int padded_in_features,  // pad 到 K_CHUNK 整数倍, 避免 output OOB
    int c_out,
    int padded_c_out         // pad 到 C_OUT_TILE 整数倍
) {
    constexpr int K_CHUNK = 16;
    constexpr int C_OUT_TILE = 64;
    constexpr int COPIES_PER_THREAD = (K_CHUNK * C_OUT_TILE) / 256;

    int c_out_start = blockIdx.x * C_OUT_TILE;
    int valid_cols = c_out_start < c_out ? min(C_OUT_TILE, c_out - c_out_start) : 0;

    __shared__ __align__(128) float smem[K_CHUNK][C_OUT_TILE];

    const int tid = threadIdx.x;

    for (int k = 0; k < in_features; k += K_CHUNK) {
        int valid_rows = min(K_CHUNK, in_features - k);

        int row_f4 = tid / (C_OUT_TILE / 4);
        int col_f4 = tid % (C_OUT_TILE / 4);
        int col    = col_f4 * 4;

        // src_size 综合考虑 row 和 col 边界
        int cols_in_f4 = min(4, max(0, valid_cols - col));
        int src_size   = (row_f4 < valid_rows) ? cols_in_f4 * (int)sizeof(float) : 0;

        uint32_t s_ptr = smem_u32addr(&smem[row_f4][col]);

        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
            :: "r"(s_ptr),
               "l"(&weights[(k + row_f4) * c_out + (c_out_start + col)]),
               "r"(src_size)
        );

        asm volatile("cp.async.commit_group;\n" :::);
        asm volatile("cp.async.wait_group 0;\n" :::);
        __syncthreads();

        // SMEM → output (padded 维度)
#pragma unroll
        for (int i = 0; i < COPIES_PER_THREAD; i++) {
            int linear = tid + i * 256;
            int r = linear / C_OUT_TILE;
            int c = linear % C_OUT_TILE;
            output[(k + r) * padded_c_out + (c_out_start + c)] = smem[r][c];
        }
        __syncthreads();
    }
}

int main() {
    // ============ Test 1: 无边界 (in_features=576, C_out=128) ============
    {
        const int in_features = 576;  // 64 * 3 * 3
        const int c_out       = 128;

        std::cout << "\n=== Test 1: Full Tiles (in_features=" << in_features
                  << ", C_out=" << c_out << ") ===" << std::endl;

        std::vector<float> h_weights(in_features * c_out);
        for (int r = 0; r < in_features; r++)
            for (int c = 0; c < c_out; c++)
                h_weights[r * c_out + c] = (float)(r * 1000 + c + 1);

        float *d_weights, *d_output;
        cudaMalloc(&d_weights, in_features * c_out * sizeof(float));
        cudaMalloc(&d_output, in_features * c_out * sizeof(float));
        cudaMemcpy(d_weights, h_weights.data(),
                   in_features * c_out * sizeof(float), cudaMemcpyHostToDevice);

        test_weight_load_full<<<c_out / 64, 256>>>(
            d_weights, d_output, in_features, c_out);

        std::vector<float> h_output(in_features * c_out);
        cudaMemcpy(h_output.data(), d_output,
                   in_features * c_out * sizeof(float), cudaMemcpyDeviceToHost);

        int errors = 0;
        for (int r = 0; r < in_features; r++)
            for (int c = 0; c < c_out; c++)
                if (h_output[r * c_out + c] != h_weights[r * c_out + c])
                    errors++;
        std::cout << "  " << (errors == 0 ? "PASSED!" : "FAILED: " + std::to_string(errors) + " errors")
                  << std::endl;
        cudaFree(d_weights);
        cudaFree(d_output);
    }

    // ============ Test 2: C_out 边界 (in_features=576, C_out=72) ============
    {
        const int in_features  = 576;
        const int c_out        = 72;
        const int padded_c_out = 128;

        std::cout << "\n=== Test 2: C_out Boundary (in_features=" << in_features
                  << ", C_out=" << c_out << ") ===" << std::endl;
        std::cout << "Block 0: C_out[0:64]   full" << std::endl;
        std::cout << "Block 1: C_out[64:72]  8 valid cols" << std::endl;

        std::vector<float> h_weights(in_features * c_out);
        for (int r = 0; r < in_features; r++)
            for (int c = 0; c < c_out; c++)
                h_weights[r * c_out + c] = (float)(r * 1000 + c + 1);

        int grid_blocks = (c_out + 63) / 64;

        float *d_weights, *d_output;
        cudaMalloc(&d_weights, in_features * c_out * sizeof(float));
        cudaMalloc(&d_output, in_features * padded_c_out * sizeof(float));
        cudaMemcpy(d_weights, h_weights.data(),
                   in_features * c_out * sizeof(float), cudaMemcpyHostToDevice);

        test_weight_load_boundary<<<grid_blocks, 256>>>(
            d_weights, d_output, in_features, c_out, padded_c_out);

        std::vector<float> h_output(in_features * padded_c_out);
        cudaMemcpy(h_output.data(), d_output,
                   in_features * padded_c_out * sizeof(float), cudaMemcpyDeviceToHost);

        int errors = 0;
        for (int r = 0; r < in_features; r++)
            for (int c = 0; c < c_out; c++)
                if (h_output[r * padded_c_out + c] != h_weights[r * c_out + c])
                    errors++;
        std::cout << "  " << (errors == 0 ? "PASSED!" : "FAILED: " + std::to_string(errors) + " errors")
                  << std::endl;
        cudaFree(d_weights);
        cudaFree(d_output);
    }

    // ============ Test 3: in_features + C_out 双边界 (in_features=8, C_out=72) ============
    {
        const int in_features         = 8;   // < K_CHUNK=16, 单次迭代仅 8 有效行
        const int padded_in_features  = 16;
        const int c_out               = 72;
        const int padded_c_out        = 128;

        std::cout << "\n=== Test 3: in_features + C_out Dual Boundary ===" << std::endl;
        std::cout << "K-iteration: only " << in_features << "/16 rows valid, 8 rows padding" << std::endl;
        std::cout << "Block 0: C_out[0:64]   full" << std::endl;
        std::cout << "Block 1: C_out[64:72]  8 valid cols" << std::endl;

        std::vector<float> h_weights(in_features * c_out);
        for (int r = 0; r < in_features; r++)
            for (int c = 0; c < c_out; c++)
                h_weights[r * c_out + c] = (float)(r * 1000 + c + 1);

        int grid_blocks = (c_out + 63) / 64;

        float *d_weights, *d_output;
        cudaMalloc(&d_weights, in_features * c_out * sizeof(float));
        cudaMalloc(&d_output, padded_in_features * padded_c_out * sizeof(float));
        cudaMemcpy(d_weights, h_weights.data(),
                   in_features * c_out * sizeof(float), cudaMemcpyHostToDevice);

        test_weight_load_general<<<grid_blocks, 256>>>(
            d_weights, d_output, in_features, padded_in_features, c_out, padded_c_out);

        std::vector<float> h_output(padded_in_features * padded_c_out);
        cudaMemcpy(h_output.data(), d_output,
                   padded_in_features * padded_c_out * sizeof(float), cudaMemcpyDeviceToHost);

        int errors = 0;
        for (int r = 0; r < in_features; r++)
            for (int c = 0; c < c_out; c++)
                if (h_output[r * padded_c_out + c] != h_weights[r * c_out + c])
                    errors++;
        std::cout << "  " << (errors == 0 ? "PASSED!" : "FAILED: " + std::to_string(errors) + " errors")
                  << std::endl;
        cudaFree(d_weights);
        cudaFree(d_output);
    }

    return 0;
}
