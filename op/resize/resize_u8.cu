#include <iostream>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>

// Original version
__global__ void resize_nearest_u8_kernel_v1(const uint8_t* input, uint8_t* output, int C, int in_H, int in_W) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_W = in_W * 2;
    int out_H = in_H * 2;
    int total_out = C * out_H * out_W;
    
    if (idx >= total_out) return;

    int ow = idx % out_W;
    int oh = (idx / out_W) % out_H;
    int c = idx / (out_W * out_H);

    int iw = ow / 2;
    int ih = oh / 2;

    int in_idx = c * (in_H * in_W) + ih * in_W + iw;
    output[idx] = input[in_idx];
}

// Optimized version: Process multiple pixels per thread
// Each thread handles 4 output pixels in a row (if possible)
__global__ void resize_nearest_u8_kernel_v2(const uint8_t* __restrict__ input, uint8_t* __restrict__ output, int C, int in_H, int in_W) 
{
    int out_W = in_W * 2;
    int out_H = in_H * 2;
    int total_out = C * out_H * out_W;
    
    // Each thread processes 4 output elements
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx >= total_out) return;

    // We can use uint32_t to write 4 bytes at once
    uint32_t out_val = 0;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int cur_idx = idx + i;
        if (cur_idx < total_out) {
            int ow = cur_idx % out_W;
            int oh = (cur_idx / out_W) % out_H;
            int c = cur_idx / (out_W * out_H);

            int iw = ow / 2;
            int ih = oh / 2;

            int in_idx = c * (in_H * in_W) + ih * in_W + iw;
            uint8_t val = input[in_idx];
            out_val |= ((uint32_t)val << (i * 8));
        }
    }

    if (idx + 3 < total_out) {
        *((uint32_t*)&output[idx]) = out_val;
    } else {
        for (int i = 0; i < 4; i++) {
            if (idx + i < total_out) {
                output[idx + i] = (uint8_t)((out_val >> (i * 8)) & 0xFF);
            }
        }
    }
}

// v3: Even more aggressive, process 16 pixels (uint4) per thread
__global__ void resize_nearest_u8_kernel_v3(const uint8_t* __restrict__ input, uint8_t* __restrict__ output, int C, int in_H, int in_W) 
{
    int out_W = in_W * 2;
    int out_H = in_H * 2;
    int total_out = C * out_H * out_W;
    
    // Each thread processes 16 output elements
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
    if (idx >= total_out) return;

    uint4 out_val;
    uint32_t* ptr = (uint32_t*)&out_val;

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        uint32_t word = 0;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int cur_idx = idx + j * 4 + i;
            if (cur_idx < total_out) {
                int ow = cur_idx % out_W;
                int oh = (cur_idx / out_W) % out_H;
                int c = cur_idx / (out_W * out_H);

                int iw = ow / 2;
                int ih = oh / 2;

                int in_idx = c * (in_H * in_W) + ih * in_W + iw;
                uint8_t val = input[in_idx];
                word |= ((uint32_t)val << (i * 8));
            }
        }
        ptr[j] = word;
    }

    if (idx + 15 < total_out) {
        *((uint4*)&output[idx]) = out_val;
    } else {
        uint8_t* out_ptr = (uint8_t*)&out_val;
        for (int i = 0; i < 16; i++) {
            if (idx + i < total_out) {
                output[idx + i] = out_ptr[i];
            }
        }
    }
}

// v4: Optimized kernel using __byte_perm and 2D/3D grid mapping
// Now with alignment checks to avoid segfaults on odd dimensions
__global__ void resize_nearest_u8_kernel_v4(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int C, int in_H, int in_W)
{
    int in_x8 = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    int in_y  = blockIdx.y * blockDim.y + threadIdx.y;
    int c     = blockIdx.z;

    if (in_x8 >= in_W || in_y >= in_H || c >= C) return;

    int out_W    = in_W * 2;
    int out_H    = in_H * 2;
    int in_base  = c * in_H * in_W + in_y * in_W + in_x8;
    int out_base = c * out_H * out_W + in_y * 2 * out_W + in_x8 * 2;

    int remaining = in_W - in_x8;
    
    // Safety check for vector alignment
    bool can_vec8 = (((size_t)(input + in_base) & 7) == 0) && 
                    (((size_t)(output + out_base) & 15) == 0) && 
                    ((out_W & 15) == 0);

    if (remaining >= 8 && can_vec8) {
        uint2 iv = *((const uint2*)(input + in_base));
        uint4 row;
        row.x = __byte_perm(iv.x, iv.x, 0x1100u);
        row.y = __byte_perm(iv.x, iv.x, 0x3322u);
        row.z = __byte_perm(iv.y, iv.y, 0x1100u);
        row.w = __byte_perm(iv.y, iv.y, 0x3322u);
        *((uint4*)(output + out_base))         = row;
        *((uint4*)(output + out_base + out_W)) = row;
    } else if (remaining >= 4 && (((size_t)(input + in_base) & 3) == 0) && 
               (((size_t)(output + out_base) & 7) == 0) && ((out_W & 7) == 0)) {
        uint32_t iv4 = *((const uint32_t*)(input + in_base));
        uint32_t w0 = __byte_perm(iv4, iv4, 0x1100u);
        uint32_t w1 = __byte_perm(iv4, iv4, 0x3322u);
        *((uint32_t*)(output + out_base))             = w0;
        *((uint32_t*)(output + out_base + 4))         = w1;
        *((uint32_t*)(output + out_base + out_W))     = w0;
        *((uint32_t*)(output + out_base + out_W + 4)) = w1;
        for (int i = 4; i < remaining; i++) {
            uint8_t v = input[in_base + i];
            uint16_t p = (uint16_t)v | ((uint16_t)v << 8);
            *((uint16_t*)(output + out_base + i * 2))         = p;
            *((uint16_t*)(output + out_base + out_W + i * 2)) = p;
        }
    } else {
        for (int i = 0; i < remaining; i++) {
            uint8_t v = input[in_base + i];
            uint16_t p = (uint16_t)v | ((uint16_t)v << 8);
            // out_base + i*2 is always 2-byte aligned if in_x8 is even (which it is, multiple of 8)
            *((uint16_t*)(output + out_base + i * 2))         = p;
            *((uint16_t*)(output + out_base + out_W + i * 2)) = p;
        }
    }
}

// v5: Optimized for 4-byte alignment (Standard for most CV tasks)
// Each thread handles 4 input pixels -> 8 output pixels (2 rows)
__global__ void resize_nearest_u8_kernel_v5(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int C, int in_H, int in_W)
{
    int in_x4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int in_y  = blockIdx.y * blockDim.y + threadIdx.y;
    int c     = blockIdx.z;

    if (in_x4 >= in_W || in_y >= in_H || c >= C) return;

    int out_W    = in_W * 2;
    int out_H    = in_H * 2;
    int in_base  = c * in_H * in_W + in_y * in_W + in_x4;
    int out_base = c * out_H * out_W + in_y * 2 * out_W + in_x4 * 2;

    int remaining = in_W - in_x4;
    
    // Check for 4-byte alignment (Standard uint32_t)
    bool can_vec4 = (((size_t)(input + in_base) & 3) == 0) && 
                    (((size_t)(output + out_base) & 3) == 0) && 
                    ((out_W & 3) == 0);

    if (remaining >= 4 && can_vec4) {
        uint32_t iv = *((const uint32_t*)(input + in_base));
        uint32_t w0 = __byte_perm(iv, iv, 0x1100u);
        uint32_t w1 = __byte_perm(iv, iv, 0x3322u);
        
        // Write two rows
        *((uint32_t*)(output + out_base))             = w0;
        *((uint32_t*)(output + out_base + 4))         = w1;
        *((uint32_t*)(output + out_base + out_W))     = w0;
        *((uint32_t*)(output + out_base + out_W + 4)) = w1;
    } else {
        // Scalar fallback for very small remainders or unaligned
        for (int i = 0; i < remaining && i < 4; i++) {
            uint8_t v = input[in_base + i];
            uint16_t p = (uint16_t)v | ((uint16_t)v << 8);
            *((uint16_t*)(output + out_base + i * 2))         = p;
            *((uint16_t*)(output + out_base + out_W + i * 2)) = p;
        }
    }
}

// v6: 3D grid, 8 input pixels per thread.
//     Reads 8 input bytes as uint2, expands to uint4 per row via __byte_perm.
//     Writes 2×uint4 (16 bytes per output row × 2 rows = 32 bytes total).
//     Block: (32, 8), Grid: (ceil(in_W/256), ceil(in_H/8), C)
__global__ void resize_nearest_u8_kernel_v6(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int C, int in_H, int in_W)
{
    int in_x8 = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    int in_y  = blockIdx.y * blockDim.y + threadIdx.y;
    int c     = blockIdx.z;

    if (in_x8 >= in_W || in_y >= in_H) return;

    int out_W    = in_W * 2;
    int out_H    = in_H * 2;
    int in_base  = c * in_H * in_W + in_y * in_W + in_x8;
    int out_base = c * out_H * out_W + in_y * 2 * out_W + in_x8 * 2;

    int remaining = in_W - in_x8;
    if (remaining >= 8) {
        uint2 iv = *((const uint2*)(input + in_base));
        uint4 row;
        row.x = __byte_perm(iv.x, iv.x, 0x1100u);
        row.y = __byte_perm(iv.x, iv.x, 0x3322u);
        row.z = __byte_perm(iv.y, iv.y, 0x1100u);
        row.w = __byte_perm(iv.y, iv.y, 0x3322u);
        *((uint4*)(output + out_base))         = row;
        *((uint4*)(output + out_base + out_W)) = row;
    } else if (remaining >= 4) {
        uint32_t iv4 = *((const uint32_t*)(input + in_base));
        uint32_t w0 = __byte_perm(iv4, iv4, 0x1100u);
        uint32_t w1 = __byte_perm(iv4, iv4, 0x3322u);
        *((uint32_t*)(output + out_base))             = w0;
        *((uint32_t*)(output + out_base + 4))         = w1;
        *((uint32_t*)(output + out_base + out_W))     = w0;
        *((uint32_t*)(output + out_base + out_W + 4)) = w1;
        for (int i = 4; i < remaining; i++) {
            uint8_t v = input[in_base + i];
            uint16_t p = (uint16_t)v | ((uint16_t)v << 8);
            *((uint16_t*)(output + out_base + i * 2))         = p;
            *((uint16_t*)(output + out_base + out_W + i * 2)) = p;
        }
    } else {
        for (int i = 0; i < remaining; i++) {
            uint8_t v = input[in_base + i];
            uint16_t p = (uint16_t)v | ((uint16_t)v << 8);
            *((uint16_t*)(output + out_base + i * 2))         = p;
            *((uint16_t*)(output + out_base + out_W + i * 2)) = p;
        }
    }
}

// CPU reference
static void resize_nearest_cpu_ref(const uint8_t* input, uint8_t* output, int C, int in_H, int in_W) 
{
    int out_H = in_H * 2;
    int out_W = in_W * 2;
    
    for (int c = 0; c < C; c++) {
        for (int oh = 0; oh < out_H; oh++) {
            for (int ow = 0; ow < out_W; ow++) {
                int ih = oh / 2;
                int iw = ow / 2;
                
                int out_idx = c * (out_H * out_W) + oh * out_W + ow;
                int in_idx = c * (in_H * in_W) + ih * in_W + iw;
                
                output[out_idx] = input[in_idx];
            }
        }
    }
}

// Test infrastructure
void resize_u8_test(int C, int in_H, int in_W, const char* label)
{
    int out_H = in_H * 2;
    int out_W = in_W * 2;
    
    printf("--- [%s] C=%d in_H=%d in_W=%d -> out_H=%d out_W=%d ---\n",
           label, C, in_H, in_W, out_H, out_W);

    size_t input_sz = (size_t)C * in_H * in_W;
    size_t output_sz = (size_t)C * out_H * out_W;

    uint8_t *h_inputs = new uint8_t[input_sz];
    uint8_t *h_outputs = new uint8_t[output_sz];
    uint8_t *h_ref = new uint8_t[output_sz];

    srand(42);
    for (size_t i = 0; i < input_sz; i++) {
        h_inputs[i] = (uint8_t)(rand() & 255);
    }

    uint8_t *d_inputs, *d_outputs;
    cudaMalloc(&d_inputs, input_sz * sizeof(uint8_t));
    cudaMalloc(&d_outputs, output_sz * sizeof(uint8_t));

    cudaMemcpy(d_inputs, h_inputs, input_sz * sizeof(uint8_t), cudaMemcpyHostToDevice);
    resize_nearest_cpu_ref(h_inputs, h_ref, C, in_H, in_W);

    auto run_test = [&](const char* name, auto kernel, int elements_per_thread) {
        cudaMemset(d_outputs, 0, output_sz * sizeof(uint8_t));
        int threads = 256;
        int blocks = (output_sz / elements_per_thread + threads - 1) / threads;

        kernel<<<blocks, threads>>>(d_inputs, d_outputs, C, in_H, in_W);
        cudaDeviceSynchronize();
        cudaMemcpy(h_outputs, d_outputs, output_sz * sizeof(uint8_t), cudaMemcpyDeviceToHost);

        int errors = 0;
        for (size_t i = 0; i < output_sz; i++) {
            if (h_outputs[i] != h_ref[i]) {
                if (errors < 5) printf("  %s Error[%zu]: gpu=%d cpu=%d\n", name, i, h_outputs[i], h_ref[i]);
                errors++;
            }
        }

        constexpr int WARMUP_ITERS = 20;
        constexpr int BENCH_ITERS  = 100;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        for (int i = 0; i < WARMUP_ITERS; i++) kernel<<<blocks, threads>>>(d_inputs, d_outputs, C, in_H, in_W);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < BENCH_ITERS; i++) kernel<<<blocks, threads>>>(d_inputs, d_outputs, C, in_H, in_W);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        printf("  %-4s: %s (%d errors) avg_ms=%.6f\n", name, (errors == 0 ? "PASS" : "FAIL"), errors, ms / BENCH_ITERS);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    };

    auto run_test_v4 = [&](const char* name) {
        cudaMemset(d_outputs, 0, output_sz * sizeof(uint8_t));
        dim3 block(32, 8);
        dim3 grid(((in_W + 7) / 8 + block.x - 1) / block.x, (in_H + block.y - 1) / block.y, C);

        resize_nearest_u8_kernel_v4<<<grid, block>>>(d_inputs, d_outputs, C, in_H, in_W);
        cudaDeviceSynchronize();
        cudaMemcpy(h_outputs, d_outputs, output_sz * sizeof(uint8_t), cudaMemcpyDeviceToHost);

        int errors = 0;
        for (size_t i = 0; i < output_sz; i++) {
            if (h_outputs[i] != h_ref[i]) {
                if (errors < 5) printf("  %s Error[%zu]: gpu=%d cpu=%d\n", name, i, h_outputs[i], h_ref[i]);
                errors++;
            }
        }

        constexpr int WARMUP_ITERS = 20;
        constexpr int BENCH_ITERS  = 100;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        for (int i = 0; i < WARMUP_ITERS; i++) resize_nearest_u8_kernel_v4<<<grid, block>>>(d_inputs, d_outputs, C, in_H, in_W);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < BENCH_ITERS; i++) resize_nearest_u8_kernel_v4<<<grid, block>>>(d_inputs, d_outputs, C, in_H, in_W);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        printf("  %-4s: %s (%d errors) avg_ms=%.6f\n", name, (errors == 0 ? "PASS" : "FAIL"), errors, ms / BENCH_ITERS);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    };

    auto run_test_v6 = [&](const char* name) {
        cudaMemset(d_outputs, 0, output_sz * sizeof(uint8_t));
        dim3 block(32, 8);
        dim3 grid((in_W + 255) / 256, (in_H + 7) / 8, C);

        resize_nearest_u8_kernel_v6<<<grid, block>>>(d_inputs, d_outputs, C, in_H, in_W);
        cudaDeviceSynchronize();
        cudaMemcpy(h_outputs, d_outputs, output_sz * sizeof(uint8_t), cudaMemcpyDeviceToHost);

        int errors = 0;
        for (size_t i = 0; i < output_sz; i++) {
            if (h_outputs[i] != h_ref[i]) {
                if (errors < 5) printf("  %s Error[%zu]: gpu=%d cpu=%d\n", name, i, h_outputs[i], h_ref[i]);
                errors++;
            }
        }

        constexpr int WARMUP_ITERS = 20;
        constexpr int BENCH_ITERS  = 100;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        for (int i = 0; i < WARMUP_ITERS; i++) resize_nearest_u8_kernel_v6<<<grid, block>>>(d_inputs, d_outputs, C, in_H, in_W);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < BENCH_ITERS; i++) resize_nearest_u8_kernel_v6<<<grid, block>>>(d_inputs, d_outputs, C, in_H, in_W);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        printf("  %-4s: %s (%d errors) avg_ms=%.6f\n", name, (errors == 0 ? "PASS" : "FAIL"), errors, ms / BENCH_ITERS);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    };

    auto run_test_v5 = [&](const char* name) {
        cudaMemset(d_outputs, 0, output_sz * sizeof(uint8_t));
        dim3 block(32, 8);
        dim3 grid(((in_W + 3) / 4 + block.x - 1) / block.x, (in_H + block.y - 1) / block.y, C);

        resize_nearest_u8_kernel_v5<<<grid, block>>>(d_inputs, d_outputs, C, in_H, in_W);
        cudaDeviceSynchronize();
        cudaMemcpy(h_outputs, d_outputs, output_sz * sizeof(uint8_t), cudaMemcpyDeviceToHost);

        int errors = 0;
        for (size_t i = 0; i < output_sz; i++) {
            if (h_outputs[i] != h_ref[i]) {
                if (errors < 5) printf("  %s Error[%zu]: gpu=%d cpu=%d\n", name, i, h_outputs[i], h_ref[i]);
                errors++;
            }
        }

        constexpr int WARMUP_ITERS = 20;
        constexpr int BENCH_ITERS  = 100;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        for (int i = 0; i < WARMUP_ITERS; i++) resize_nearest_u8_kernel_v5<<<grid, block>>>(d_inputs, d_outputs, C, in_H, in_W);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < BENCH_ITERS; i++) resize_nearest_u8_kernel_v5<<<grid, block>>>(d_inputs, d_outputs, C, in_H, in_W);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        printf("  %-4s: %s (%d errors) avg_ms=%.6f\n", name, (errors == 0 ? "PASS" : "FAIL"), errors, ms / BENCH_ITERS);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    };

    run_test("v1", resize_nearest_u8_kernel_v1, 1);
    run_test("v2", resize_nearest_u8_kernel_v2, 4);
    run_test("v3", resize_nearest_u8_kernel_v3, 16);
    run_test_v4("v4");
    run_test_v5("v5");
    // v6 requires 8-byte alignment (in_W must be multiple of 8)
    if (in_W % 8 == 0) {
        run_test_v6("v6");
    } else {
        printf("  v6  : SKIP (in_W=%d not multiple of 8)\n", in_W);
    }

    cudaFree(d_inputs);
    cudaFree(d_outputs);
    delete[] h_inputs;
    delete[] h_outputs;
    delete[] h_ref;
}

int main()
{
    printf("\n=== Simplified resize_nearest_u8_kernel tests ===\n");

    std::vector<int> channels = {64, 96, 128, 256, 384, 512};
    
    for (int c : channels) {
        // 10x10: multiple of 2, out_W=20 (multiple of 4) -> triggers 4-byte logic
        resize_u8_test(c, 10, 10, "10x10");
        // 20x20: multiple of 4, out_W=40 (multiple of 8) -> triggers 8-byte logic
        resize_u8_test(c, 20, 20, "20x20");
    }

    printf("\n=== All tests complete ===\n");
    return 0;
}