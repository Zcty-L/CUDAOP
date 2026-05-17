#include "cpu_ops.h"
#include <iostream>
#include <vector>
#include <cstdint>

/**
 * Standard 2D Convolution on CPU (NCHW layout)
 * 
 * @param input  Input tensor [N, C, H, W]
 * @param weight Weight tensor [K, C, R, S]
 * @param bias   Bias tensor [K]
 * @param output Output tensor [N, K, Oh, Ow]
 */
void conv2d_cpu(const float* input, const float* weight, const float* bias, float* output,
                int N, int C, int H, int W, 
                int K, int R, int S, 
                int stride_h, int stride_w, 
                int pad_h, int pad_w) {
    int Oh = (H + 2 * pad_h - R) / stride_h + 1;
    int Ow = (W + 2 * pad_w - S) / stride_w + 1;

    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int oh = 0; oh < Oh; ++oh) {
                for (int ow = 0; ow < Ow; ++ow) {
                    float sum = 0.0f;
                    for (int c = 0; c < C; ++c) {
                        for (int r = 0; r < R; ++r) {
                            for (int s = 0; s < S; ++s) {
                                int ih = oh * stride_h - pad_h + r;
                                int iw = ow * stride_w - pad_w + s;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    sum += input[((n * C + c) * H + ih) * W + iw] * 
                                           weight[((k * C + c) * R + r) * S + s];
                                }
                            }
                        }
                    }
                    output[((n * K + k) * Oh + oh) * Ow + ow] = sum + (bias ? bias[k] : 0.0f);
                }
            }
        }
    }
}

/**
 * Depthwise 2D Convolution on CPU (NCHW layout)
 * Each input channel is convolved with its own set of filters.
 * Usually C_in == C_out.
 * 
 * @param input  Input tensor [N, C, H, W]
 * @param weight Weight tensor [C, R, S] (assuming 1 filter per channel)
 * @param bias   Bias tensor [C]
 * @param output Output tensor [N, C, Oh, Ow]
 */
void dwconv2d_cpu(const float* input, const float* weight, const float* bias, float* output,
                  int N, int C, int H, int W, 
                  int R, int S, 
                  int stride_h, int stride_w, 
                  int pad_h, int pad_w) {
    int Oh = (H + 2 * pad_h - R) / stride_h + 1;
    int Ow = (W + 2 * pad_w - S) / stride_w + 1;

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < Oh; ++oh) {
                for (int ow = 0; ow < Ow; ++ow) {
                    float sum = 0.0f;
                    for (int r = 0; r < R; ++r) {
                        for (int s = 0; s < S; ++s) {
                            int ih = oh * stride_h - pad_h + r;
                            int iw = ow * stride_w - pad_w + s;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                sum += input[((n * C + c) * H + ih) * W + iw] * 
                                       weight[(c * R + r) * S + s];
                            }
                        }
                    }
                    output[((n * C + c) * Oh + oh) * Ow + ow] = sum + (bias ? bias[c] : 0.0f);
                }
            }
        }
    }
}

/**
 * Grouped 2D Convolution on CPU (NCHW layout)
 * 
 * @param input  Input tensor [N, C, H, W]
 * @param weight Weight tensor [K, C/groups, R, S]
 * @param bias   Bias tensor [K]
 * @param output Output tensor [N, K, Oh, Ow]
 * @param groups Number of groups
 */
/**
 * GEMM on CPU: C = A x B^T
 * A[M x K] row-major, B[N x K] row-major, C[M x N] row-major.
 */
void gemm_mma_cpu(const float* A, const float* B, float* C,
                  int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (int p = 0; p < K; ++p)
            {
                sum += A[i * K + p] * B[j * K + p];
            }
            C[i * N + j] = sum;
        }
    }
}

void grouped_conv2d_cpu(const float* input, const float* weight, const float* bias, float* output,
                        int N, int C, int H, int W, 
                        int K, int R, int S, 
                        int stride_h, int stride_w, 
                        int pad_h, int pad_w,
                        int groups) {
    int Oh = (H + 2 * pad_h - R) / stride_h + 1;
    int Ow = (W + 2 * pad_w - S) / stride_w + 1;
    int C_per_group = C / groups;
    int K_per_group = K / groups;

    for (int n = 0; n < N; ++n) {
        for (int g = 0; groups > g; ++g) {
            for (int k = 0; k < K_per_group; ++k) {
                int out_k = g * K_per_group + k;
                for (int oh = 0; oh < Oh; ++oh) {
                    for (int ow = 0; ow < Ow; ++ow) {
                        float sum = 0.0f;
                        for (int c = 0; c < C_per_group; ++c) {
                            int in_c = g * C_per_group + c;
                            for (int r = 0; r < R; ++r) {
                                for (int s = 0; s < S; ++s) {
                                    int ih = oh * stride_h - pad_h + r;
                                    int iw = ow * stride_w - pad_w + s;
                                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                        sum += input[((n * C + in_c) * H + ih) * W + iw] * 
                                               weight[((out_k * C_per_group + c) * R + r) * S + s];
                                    }
                                }
                            }
                        }
                        output[((n * K + out_k) * Oh + oh) * Ow + ow] = sum + (bias ? bias[out_k] : 0.0f);
                    }
                }
            }
        }
    }
}
