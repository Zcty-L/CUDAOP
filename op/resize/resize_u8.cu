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

    run_test("v1", resize_nearest_u8_kernel_v1, 1);
    run_test("v2", resize_nearest_u8_kernel_v2, 4);
    run_test("v3", resize_nearest_u8_kernel_v3, 16);

    cudaFree(d_inputs);
    cudaFree(d_outputs);
    delete[] h_inputs;
    delete[] h_outputs;
    delete[] h_ref;
}

int main()
{
    printf("\n=== resize_nearest_u8_kernel tests ===\n");

    resize_u8_test(1, 40, 40, "1x40x40");
    resize_u8_test(64, 40, 40, "64x40x40");
    resize_u8_test(128, 80, 80, "128x80x80");
    resize_u8_test(32, 21, 21, "odd_dims");

    printf("\n=== All tests complete ===\n");
    return 0;
}