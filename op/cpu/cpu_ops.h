#ifndef CPU_OPS_H
#define CPU_OPS_H

#include <cstdint>

/**
 * Standard 2D Convolution on CPU (NCHW layout)
 */
void conv2d_cpu(const float* input, const float* weight, const float* bias, float* output,
                int N, int C, int H, int W, 
                int K, int R, int S, 
                int stride_h, int stride_w, 
                int pad_h, int pad_w);

/**
 * Depthwise 2D Convolution on CPU (NCHW layout)
 */
void dwconv2d_cpu(const float* input, const float* weight, const float* bias, float* output,
                  int N, int C, int H, int W, 
                  int R, int S, 
                  int stride_h, int stride_w, 
                  int pad_h, int pad_w);

/**
 * Grouped 2D Convolution on CPU (NCHW layout)
 */
void grouped_conv2d_cpu(const float* input, const float* weight, const float* bias, float* output,
                        int N, int C, int H, int W, 
                        int K, int R, int S, 
                        int stride_h, int stride_w, 
                        int pad_h, int pad_w,
                        int groups);

/**
 * GEMM on CPU: C = A x B^T
 * A is M x K row-major, B is N x K row-major, C is M x N row-major.
 */
void gemm_mma_cpu(const float* A, const float* B, float* C,
                  int M, int N, int K);

#endif // CPU_OPS_H
