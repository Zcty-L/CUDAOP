#include <iostream>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>

/**
 * thread_map.txt verification
 *
 * Within a 32-thread warp:
 *   mma_tid_x = lane_id / 16 * 2 + lane_id % 2   (0..3, 4 columns)
 *   mma_tid_y = lane_id % 16 / 2                  (0..7, 8 rows)
 *
 * Grid layout (8 rows × 4 cols = 32 threads):
 *     mma_tid_x -->
 *    0    1    2    3      ← mma_tid_x
 *  0  0    1   16   17
 *  1  2    3   18   19
 *  2  4    5   20   21
 *  3  6    7   22   23     values are lane_id
 *  4  8    9   24   25
 *  5 10   11   26   27
 *  6 12   13   28   29
 *  7 14   15   30   31
 *  ↑ mma_tid_y
 */

// Kernel 0: verify thread_map.txt mapping
// Each thread stores its (mma_tid_x, mma_tid_y)
__global__ void verify_thread_map(uint32_t *output) {
    const int lane_id = threadIdx.x % 32;
    uint32_t mma_tid_x = lane_id / 16 * 2 + lane_id % 2;
    uint32_t mma_tid_y = lane_id % 16 / 2;
    output[lane_id * 2 + 0] = mma_tid_x;
    output[lane_id * 2 + 1] = mma_tid_y;
}

// Kernel 1: SMEM 16 floats = 8 float2, broadcast by mma_tid_y
// SMEM layout: [8 (mma_tid_y)] float2
// 4 threads share same mma_tid_y → read same float2 (broadcast)
// Bank analysis: 8 unique float2 → banks 0..15, no conflict → 1 transaction
__global__ void smem_float2_load(float *output) {
    __shared__ __align__(128) float smem[16];

    const int lane_id = threadIdx.x % 32;
    const int mma_tid_x = lane_id / 16 * 2 + lane_id % 2;  // 0..3
    const int mma_tid_y = lane_id % 16 / 2;                 // 0..7

    // Init SMEM: 32 threads cover 16 positions
    for (int i = threadIdx.x; i < 16; i += blockDim.x) {
        smem[i] = (float)(i + 1);
    }
    __syncthreads();

    // Read float2 at smem[mma_tid_y * 2]
    // 4 threads with same mma_tid_y all read the same float2 → broadcast
    int offset = mma_tid_y * 2;
    float2 val = *reinterpret_cast<float2 *>(&smem[offset]);

    output[lane_id * 2 + 0] = __float2uint_rn(val.x);
    output[lane_id * 2 + 1] = __float2uint_rn(val.y);
}


// Kernel 2: SMEM 32 uint8 = 4 × uint64_t, broadcast by mma_tid_x
// mma_tid_x = 0..3 → offset = mma_tid_x * 8 (bytes)
// 8 threads per mma_tid_x → broadcast same uint64
// Bank: 4 uint64 × 2 banks = 8 banks (0-7), all distinct → 0 conflicts
__global__ void smem_uint64_load(uint64_t *output) {
    __shared__ __align__(128) uint8_t smem[32];

    const int lane_id = threadIdx.x % 32;
    const int mma_tid_x = lane_id / 16 * 2 + lane_id % 2;  // 0..3
    const int mma_tid_y = lane_id % 16 / 2;                 // 0..7

    // Init SMEM: 32 threads cover 32 positions
    for (int i = threadIdx.x; i < 32; i += blockDim.x) {
        smem[i] = (uint8_t)(i + 1);
    }
    __syncthreads();

    // Read uint64 at smem[mma_tid_x * 8]
    // 4 unique uint64, each broadcast to 8 threads (same mma_tid_x, different mma_tid_y)
    int offset = mma_tid_x * 8;
    uint64_t val = *reinterpret_cast<uint64_t *>(&smem[offset]);

    output[lane_id] = val;
}


int main() {
    // ===== Test 0: verify thread_map.txt mapping =====
    {
        std::cout << "\n=== Test 0: verify thread_map.txt mapping ===" << std::endl;

        uint32_t *d_output;
        cudaMalloc(&d_output, 32 * 2 * sizeof(uint32_t));
        verify_thread_map<<<1, 32>>>(d_output);

        std::vector<uint32_t> h_output(32 * 2);
        cudaMemcpy(h_output.data(), d_output, 32 * 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // Print as 8×4 grid
        std::cout << "  mma_tid_x →   0   1   2   3" << std::endl;
        for (int y = 0; y < 8; y++) {
            std::cout << "  mma_tid_y=" << y << "  ";
            for (int x = 0; x < 4; x++) {
                int lane_found = -1;
                for (int lane = 0; lane < 32; lane++) {
                    if (h_output[lane * 2] == (uint32_t)x && h_output[lane * 2 + 1] == (uint32_t)y) {
                        lane_found = lane;
                        break;
                    }
                }
                if (lane_found >= 0)
                    std::cout << " " << lane_found << "  ";
                else
                    std::cout << " ?  ";
            }
            std::cout << std::endl;
        }

        bool ok = true;
        for (int lane = 0; lane < 32; lane++) {
            uint32_t expected_x = lane / 16 * 2 + lane % 2;
            uint32_t expected_y = (lane % 16) / 2;
            if (h_output[lane * 2] != expected_x || h_output[lane * 2 + 1] != expected_y) {
                std::cout << "  MISMATCH lane " << lane << ": got("
                          << h_output[lane * 2] << "," << h_output[lane * 2 + 1]
                          << ") expected(" << expected_x << "," << expected_y << ")" << std::endl;
                ok = false;
            }
        }
        std::cout << "  " << (ok ? "PASSED!" : "FAILED") << std::endl;

        cudaFree(d_output);
    }

    // ===== Test 1: SMEM float2 load (16 floats = 8 float2) =====
    {
        std::cout << "\n=== Test 1: SMEM float2 load, 16 floats ===" << std::endl;

        float *d_output;
        cudaMalloc(&d_output, 32 * 2 * sizeof(float));
        smem_float2_load<<<1, 32>>>(d_output);

        std::vector<float> h_output(32 * 2);
        cudaMemcpy(h_output.data(), d_output, 32 * 2 * sizeof(float), cudaMemcpyDeviceToHost);

        // Verify: thread reads smem[mma_tid_y * 2] as float2
        bool ok = true;
        for (int lane = 0; lane < 32; lane++) {
            int ty = (lane % 16) / 2;  // mma_tid_y
            int offset = ty * 2;
            float expected_x = (float)(offset + 1);
            float expected_y = (float)(offset + 2);
            if (h_output[lane * 2] != expected_x || h_output[lane * 2 + 1] != expected_y) {
                std::cout << "  MISMATCH lane " << lane << " (mma_tid_y=" << ty
                          << "): got(" << h_output[lane * 2] << "," << h_output[lane * 2 + 1]
                          << ") expected(" << expected_x << "," << expected_y << ")" << std::endl;
                ok = false;
            }
        }
        std::cout << "  " << (ok ? "PASSED!" : "FAILED") << std::endl;

        cudaFree(d_output);
    }

    // ===== Test 2: SMEM uint64 load (32 uint8, broadcast by mma_tid_x) =====
    {
        std::cout << "\n=== Test 2: SMEM uint64 load, 32 uint8, mma_tid_x index ===" << std::endl;

        uint64_t *d_output;
        cudaMalloc(&d_output, 32 * sizeof(uint64_t));
        smem_uint64_load<<<1, 32>>>(d_output);

        std::vector<uint64_t> h_output(32);
        cudaMemcpy(h_output.data(), d_output, 32 * sizeof(uint64_t), cudaMemcpyDeviceToHost);

        // Verify: thread reads smem[mma_tid_x * 8] as uint64 (little-endian)
        bool ok = true;
        for (int lane = 0; lane < 32; lane++) {
            int tx = lane / 16 * 2 + lane % 2;  // mma_tid_x
            int offset = tx * 8;                  // byte offset
            // Build expected: 8 bytes [offset+1, offset+2, ..., offset+8] packed LE
            uint64_t expected = 0;
            for (int b = 0; b < 8; b++) {
                expected |= (uint64_t)(offset + b + 1) << (b * 8);
            }
            if (h_output[lane] != expected) {
                std::cout << "  MISMATCH lane " << lane << " (mma_tid_x=" << tx
                          << "): got=0x" << std::hex << h_output[lane] << std::dec
                          << " expected=0x" << std::hex << expected << std::dec
                          << std::endl;
                ok = false;
            }
        }
        std::cout << "  " << (ok ? "PASSED!" : "FAILED") << std::endl;

        cudaFree(d_output);
    }

    return 0;
}
