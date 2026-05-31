#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iomanip>

#include "../../cuda_utils.cuh"
#include "../../ptx_utils.cuh"

// =============================================================================
// snn_conv2d_erf_64x64_k16_u8
// =============================================================================

static void pad_weights_sn(
    const float *src, float *dst,
    int in_features, int C_out,
    int in_features_padded, int C_out_padded)
{
    for (int k = 0; k < in_features_padded; k++)
    {
        for (int m = 0; m < C_out_padded; m++)
        {
            dst[k * C_out_padded + m] =
                (k < in_features && m < C_out) ? src[k * C_out + m] : 0.f;
        }
    }
}

template <int Kh, int Kw, int Sh, int Sw, int Ph, int Pw>
__global__
void snn_conv2d_erf_64x64_k16_u8(
    const uint8_t * __restrict__ inputs,
    const float   * __restrict__ weights,
    const float   * __restrict__ bias,
    uint8_t       * __restrict__ outputs,
    Conv2DParam param,
    int   out_ch_padded,
    int   shift_z)
{
    constexpr int K_CHUNK  = 16;
    constexpr int M_TILE   = 64;
    constexpr int N_TILE   = 64;
    constexpr int KhKw     = Kh * Kw;

    __shared__ __align__(128) char smem[10 * 1024];

    float   *smemweight[2];
    smemweight[0] = reinterpret_cast<float *>(smem);
    smemweight[1] = reinterpret_cast<float *>(smem + 4 * 1024);

    uint8_t *smeminput[2];
    smeminput[0] = reinterpret_cast<uint8_t *>(smem + 8 * 1024);
    smeminput[1] = reinterpret_cast<uint8_t *>(smem + 9 * 1024);

    float *smembias = reinterpret_cast<float *>(smem);

    const int tid     = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;

    const int mma_tid_x = lane_id / 16 * 2 + lane_id % 2;
    const int mma_tid_y = lane_id % 16 / 2;

    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;

    const int thread_m_base = warp_m * 16 + mma_tid_y * 2;
    const int thread_n_base = warp_n * 32 + mma_tid_x * 8;

    const int m_tile_base = blockIdx.y * M_TILE;
    const int n_tile_base = blockIdx.x * N_TILE;

    if (tid < M_TILE)
    {
        const int m_global = m_tile_base + tid;
        smembias[tid] = (m_global < (int)param.out_ch) ? bias[m_global] : 0.f;
    }
    __syncthreads();

    float output_frag[2][8];
#pragma unroll
    for (int i = 0; i < 2; i++)
    {
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            output_frag[i][j] = smembias[thread_m_base + i];
        }
    }

    __syncthreads();

    const int in_features = param.inChKhKw;
    const int k_iters     = in_features / K_CHUNK;

    auto load_weight = [&](int k_iter, int buf)
    {
        const int k_base = k_iter * K_CHUNK;
        const int w_row  = tid / 16;
        const int w_col  = (tid % 16) * 4;
        uint32_t smem_ptr = ptx::smem_u32addr(
            &smemweight[buf][w_row * M_TILE + w_col]);
        const float *src = &weights[(size_t)(k_base + w_row) * out_ch_padded
                                    + m_tile_base + w_col];
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 16;\n"
            :: "r"(smem_ptr), "l"(src)
        );
    };

    auto load_input = [&](int k_iter, int buf)
    {
        const int k_base   = k_iter * K_CHUNK;
        const int i_k      = tid / 16;
        const int i_n4     = tid % 16;
        const int global_k = k_base + i_k;

        uint32_t packed = 0;
        if (global_k < in_features)
        {
            if constexpr (KhKw == 1)
            {
                int c_idx = global_k;
#pragma unroll
                for (int b = 0; b < 4; b++)
                {
                    int n_idx = n_tile_base + i_n4 * 4 + b;
                    if (n_idx < (int)param.outHW)
                    {
                        int oh = n_idx / (int)param.out_w;
                        int ow = n_idx % (int)param.out_w;
                        int ih = oh * (int)param.Sh - (int)param.Ph;
                        int iw = ow * (int)param.Sw - (int)param.Pw;
                        packed |= (uint32_t)inputs[(size_t)c_idx * param.inHW
                                                   + ih * param.in_w + iw]
                                  << (b * 8);
                    }
                }
            }
            else
            {
                int c_idx = global_k / KhKw;
                int ky    = global_k % KhKw / (int)param.Kw;
                int kx    = global_k % KhKw % (int)param.Kw;
#pragma unroll
                for (int b = 0; b < 4; b++)
                {
                    int n_idx = n_tile_base + i_n4 * 4 + b;
                    if (n_idx < (int)param.outHW)
                    {
                        int oh = n_idx / (int)param.out_w;
                        int ow = n_idx % (int)param.out_w;
                        int ih = oh * (int)param.Sh - (int)param.Ph + ky;
                        int iw = ow * (int)param.Sw - (int)param.Pw + kx;
                        if (ih >= 0 && ih < (int)param.in_h &&
                            iw >= 0 && iw < (int)param.in_w)
                        {
                            packed |= (uint32_t)inputs[(size_t)c_idx * param.inHW
                                                       + ih * param.in_w + iw]
                                      << (b * 8);
                        }
                    }
                }
            }
        }
        uint32_t smeminput_ptr = ptx::smem_u32addr(
            &reinterpret_cast<uint32_t *>(smeminput[buf])[i_k * (N_TILE / 4) + i_n4]);
        ptx::sts32(packed, smeminput_ptr);
    };

    if (k_iters > 0)
    {
        load_weight(0, 0);
        load_input(0, 0);
        asm volatile("cp.async.commit_group;\n" :::);
    }

    for (int k_iter = 0; k_iter < k_iters; k_iter++)
    {
        const int cur  = k_iter & 1;
        const int next = cur ^ 1;

        if (k_iter + 1 < k_iters)
        {
            load_weight(k_iter + 1, next);
            load_input(k_iter + 1, next);
            asm volatile("cp.async.commit_group;\n" :::);
        }

        asm volatile("cp.async.wait_group 1;\n" :::);
        __syncthreads();

#pragma unroll
        for (int k = 0; k < K_CHUNK; k++)
        {
            float    weight_frag[2];
            uint32_t input_frag_lo, input_frag_hi;

            uint32_t w_addr = ptx::smem_u32addr(
                &smemweight[cur][k * M_TILE + thread_m_base]);
            ptx::lds64(weight_frag[0], weight_frag[1], w_addr);

            uint32_t i_addr = ptx::smem_u32addr(
                &smeminput[cur][k * N_TILE + thread_n_base]);
            ptx::lds64(input_frag_lo, input_frag_hi, i_addr);

#pragma unroll
            for (int i = 0; i < 2; i++)
            {
#pragma unroll
                for (int j = 0; j < 8; j++)
                {
                    uint32_t word  = (j < 4) ? input_frag_lo : input_frag_hi;
                    int      shift = (j % 4) * 8 + shift_z;
                    int      spike = (word >> shift) & 1;
                    add_f32(output_frag[i][j], weight_frag[i], spike);
                }
            }
        }

        __syncthreads();
    }

    asm volatile("cp.async.wait_all;\n" :::);

    uint8_t packed_out[2][8];

#pragma unroll
    for (int i = 0; i < 2; i++)
    {
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            int spike = (output_frag[i][j] > 0.f) ? 1 : 0;
            packed_out[i][j] = (uint8_t)spike;
        }
    }

    uint8_t *warp_smem8 = reinterpret_cast<uint8_t *>(smem) + warp_id * 512;

    const int warp_m_global = m_tile_base + warp_m * 16;
    const int warp_n_global = n_tile_base + warp_n * 32;
    const int n_global      = warp_n_global + lane_id;
    const bool n_valid      = (n_global < (int)param.outHW);

#pragma unroll
    for (int i = 0; i < 2; i++)
    {
        const int smem_m = mma_tid_y * 2 + i;
        uint32_t lo = (uint32_t)packed_out[i][0]
                    | ((uint32_t)packed_out[i][1] << 8)
                    | ((uint32_t)packed_out[i][2] << 16)
                    | ((uint32_t)packed_out[i][3] << 24);
        uint32_t hi = (uint32_t)packed_out[i][4]
                    | ((uint32_t)packed_out[i][5] << 8)
                    | ((uint32_t)packed_out[i][6] << 16)
                    | ((uint32_t)packed_out[i][7] << 24);
        uint32_t addr_lo = ptx::smem_u32addr(
            &warp_smem8[smem_m * 32 + mma_tid_x * 8]);
        ptx::sts32(lo, addr_lo);
        ptx::sts32(hi, addr_lo + 4);
    }

    __syncthreads();

#pragma unroll
    for (int m_row = 0; m_row < 16; m_row++)
    {
        const int  m_global = warp_m_global + m_row;
        const bool m_valid  = (m_global < (int)param.out_ch);
        const uint8_t val   = warp_smem8[lane_id + m_row * 32];
        ptx::stg8(val,
                  outputs + (size_t)m_global * param.outHW + n_global,
                  m_valid && n_valid);
    }
}

// =============================================================================
// Launch wrappers
// =============================================================================

template <int Kh, int Kw, int Sh, int Sw, int Ph, int Pw>
void snn_conv2d_erf_u8_launch(
    const uint8_t *d_inputs, const float *d_weights_padded,
    const float *d_bias, uint8_t *d_outputs,
    Conv2DParam &param, int out_ch_padded, int shift_z)
{
    dim3 block(256);
    dim3 grid(
        ((int)param.outHW  + 63) / 64,
        ((int)param.out_ch + 63) / 64,
        1
    );
    snn_conv2d_erf_64x64_k16_u8<Kh, Kw, Sh, Sw, Ph, Pw>
        <<<grid, block>>>(d_inputs, d_weights_padded, d_bias, d_outputs,
                          param, out_ch_padded, shift_z);
}

void snn_conv2d_erf_u8_1x1_s1_launch(
    const uint8_t *d_in, const float *d_w, const float *d_bias, uint8_t *d_out,
    Conv2DParam &param, int shift_z, int out_ch_padded)
{
    snn_conv2d_erf_u8_launch<1,1,1,1,0,0>(d_in,d_w,d_bias,d_out,param,out_ch_padded,shift_z);
}

void snn_conv2d_erf_u8_3x3_s1_launch(
    const uint8_t *d_in, const float *d_w, const float *d_bias, uint8_t *d_out,
    Conv2DParam &param, int shift_z, int out_ch_padded)
{
    snn_conv2d_erf_u8_launch<3,3,1,1,1,1>(d_in,d_w,d_bias,d_out,param,out_ch_padded,shift_z);
}

void snn_conv2d_erf_u8_3x3_s2_launch(
    const uint8_t *d_in, const float *d_w, const float *d_bias, uint8_t *d_out,
    Conv2DParam &param, int shift_z, int out_ch_padded)
{
    snn_conv2d_erf_u8_launch<3,3,2,2,1,1>(d_in,d_w,d_bias,d_out,param,out_ch_padded,shift_z);
}

// =============================================================================
// CPU reference
// =============================================================================

template <int Kh, int Kw, int Sh, int Sw, int Ph, int Pw>
static void snn_conv2d_erf_cpu_ref(
    const uint8_t *inputs,
    const float   *weights,
    const float   *bias,
    uint8_t       *outputs,
    int shift_z, int C_in, int H, int W, int C_out)
{
    constexpr int KhKw = Kh * Kw;
    int H_out = (H + 2 * Ph - Kh) / Sh + 1;
    int W_out = (W + 2 * Pw - Kw) / Sw + 1;

    memset(outputs, 0, (size_t)C_out * H_out * W_out);

    for (int m = 0; m < C_out; m++)
    {
        for (int oh = 0; oh < H_out; oh++)
        {
            for (int ow = 0; ow < W_out; ow++)
            {
                float sum = bias[m];
                for (int c = 0; c < C_in; c++)
                {
                    for (int ky = 0; ky < Kh; ky++)
                    {
                        for (int kx = 0; kx < Kw; kx++)
                        {
                            int ih = oh * Sh - Ph + ky;
                            int iw = ow * Sw - Pw + kx;
                            if (ih < 0 || ih >= H || iw < 0 || iw >= W)
                            {
                                continue;
                            }
                            uint8_t pk = inputs[c * H * W + ih * W + iw];
                            if ((pk >> shift_z) & 1)
                            {
                                sum += weights[(c * KhKw + ky * Kw + kx) * C_out + m];
                            }
                        }
                    }
                }
                
                int spike = (sum > 0.f) ? 1 : 0;
                outputs[m * H_out * W_out + oh * W_out + ow] = (uint8_t)spike;
            }
        }
    }
}

// =============================================================================
// Test function
// =============================================================================

template <int Kh, int Kw, int Sh, int Sw, int Ph, int Pw>
void snn_conv2d_erf_u8_test(int C_in, int H, int W, int C_out,
                            int shift_z, const char *label)
{
    constexpr int KhKw    = Kh * Kw;
    constexpr int K_CHUNK = 16;

    int H_out          = (H + 2 * Ph - Kh) / Sh + 1;
    int W_out          = (W + 2 * Pw - Kw) / Sw + 1;
    int in_features    = C_in * KhKw;
    int in_feat_padded = (in_features + K_CHUNK - 1) / K_CHUNK * K_CHUNK;
    int C_out_padded   = (C_out + 63) / 64 * 64;

    std::cout << "  [" << label << "] shift_z=" << shift_z 
              << " C_in=" << C_in << " H=" << H << " C_out=" << C_out 
              << " -> H_out=" << H_out << "  ";

    size_t input_sz   = (size_t)C_in * H * W;
    size_t weight_sz  = (size_t)in_features * C_out;
    size_t weightP_sz = (size_t)in_feat_padded * C_out_padded;
    size_t bias_sz    = (size_t)C_out;
    size_t output_sz  = (size_t)C_out * H_out * W_out;

    uint8_t *h_inputs   = new uint8_t[input_sz];
    float   *h_weights  = new float[weight_sz];
    float   *h_weightsP = new float[weightP_sz];
    float   *h_bias     = new float[bias_sz];
    uint8_t *h_outputs  = new uint8_t[output_sz];
    uint8_t *h_ref      = new uint8_t[output_sz];

    srand(42);
    for (size_t i = 0; i < input_sz; i++)
    {
        uint8_t packed = 0;
        for (int t = 0; t < 8; t++)
        {
            if (rand() & 1)
            {
                packed |= (1u << t);
            }
        }
        h_inputs[i] = packed;
    }
    for (size_t i = 0; i < weight_sz; i++)
    {
        h_weights[i] = (float)(rand() & 255) / 128.f - 1.f;
    }
    for (size_t i = 0; i < bias_sz; i++)
    {
        h_bias[i] = (float)((int)(i % 17) - 8) / 16.f;
    }

    pad_weights_sn(h_weights, h_weightsP, in_features, C_out,
                   in_feat_padded, C_out_padded);

    uint8_t *d_inputs;
    float   *d_weightsP, *d_bias;
    uint8_t *d_outputs;
    cudaMalloc(&d_inputs,   input_sz   * sizeof(uint8_t));
    cudaMalloc(&d_weightsP, weightP_sz * sizeof(float));
    cudaMalloc(&d_bias,     bias_sz    * sizeof(float));
    cudaMalloc(&d_outputs,  output_sz  * sizeof(uint8_t));
    cudaMemset(d_outputs, 0, output_sz * sizeof(uint8_t));
    cudaMemcpy(d_inputs,   h_inputs,   input_sz   * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weightsP, h_weightsP, weightP_sz * sizeof(float),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias,     h_bias,     bias_sz    * sizeof(float),   cudaMemcpyHostToDevice);

    Conv2DParam param;
    param.in_h = H; param.in_w = W; param.inHW = H * W;
    param.inChKhKw = in_feat_padded; param.inBatchNumel = C_in * H * W;
    param.out_ch = C_out; param.out_w = W_out;
    param.outHW = H_out * W_out; param.outBatchNumel = C_out * H_out * W_out;
    param.Kh = Kh; param.Kw = Kw; param.KhKw = KhKw;
    param.Sh = Sh; param.Sw = Sw; param.Ph = Ph; param.Pw = Pw;

    snn_conv2d_erf_u8_launch<Kh, Kw, Sh, Sw, Ph, Pw>(
        d_inputs, d_weightsP, d_bias, d_outputs, param, C_out_padded, shift_z);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "  CUDA error: " << cudaGetErrorString(err) << std::endl;
        goto cleanup;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_outputs, d_outputs, output_sz * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    snn_conv2d_erf_cpu_ref<Kh, Kw, Sh, Sw, Ph, Pw>(
        h_inputs, h_weights, h_bias, h_ref, shift_z, C_in, H, W, C_out);

    {
        int errors = 0;
        for (size_t i = 0; i < output_sz; i++)
        {
            if (h_outputs[i] != h_ref[i])
            {
                errors++;
            }
        }
        if (errors == 0)
        {
            std::cout << "PASSED!" << std::endl;
        }
        else
        {
            std::cout << "FAILED (" << errors << " errors)" << std::endl;
        }
    }

cleanup:
    cudaFree(d_inputs); cudaFree(d_weightsP); cudaFree(d_bias); cudaFree(d_outputs);
    delete[] h_inputs; delete[] h_weights; delete[] h_weightsP;
    delete[] h_bias; delete[] h_outputs; delete[] h_ref;
}

int main()
{
    std::cout << std::endl << "=== snn_conv2d_erf_64x64_k16_u8 correctness tests ===" << std::endl;

    std::cout << std::endl << "--- 1x1, s=1, p=0 ---" << std::endl;
    snn_conv2d_erf_u8_test<1,1,1,1,0,0>( 64,  80, 80,  64, 0, "base_z0");
    snn_conv2d_erf_u8_test<1,1,1,1,0,0>( 64,  80, 80,  64, 3, "base_z3");
    snn_conv2d_erf_u8_test<1,1,1,1,0,0>(128,  40, 40, 128, 1, "C128_z1");
    snn_conv2d_erf_u8_test<1,1,1,1,0,0>( 64,  80, 80,  48, 2, "C_out boundary");

    std::cout << std::endl << "--- 3x3, s=1, p=1 ---" << std::endl;
    snn_conv2d_erf_u8_test<3,3,1,1,1,1>( 64,  80, 80,  64, 0, "base_z0");
    snn_conv2d_erf_u8_test<3,3,1,1,1,1>( 32,  40, 40,  32, 2, "small_z2");
    snn_conv2d_erf_u8_test<3,3,1,1,1,1>( 32,  43, 43,  48, 1, "boundary_z1");

    std::cout << std::endl << "--- 3x3, s=2, p=1 ---" << std::endl;
    snn_conv2d_erf_u8_test<3,3,2,2,1,1>( 64,  80, 80,  64, 0, "H_out=40_z0");
    snn_conv2d_erf_u8_test<3,3,2,2,1,1>( 32,  80, 80,  64, 3, "C expand_z3");
    snn_conv2d_erf_u8_test<3,3,2,2,1,1>( 16,  40, 40,  32, 1, "small H=40_z1");
    snn_conv2d_erf_u8_test<3,3,2,2,1,1>( 32,  43, 43, 128, 2, "boundary_z2");

    std::cout << std::endl << "=== Done ===" << std::endl;
    return 0;
}
