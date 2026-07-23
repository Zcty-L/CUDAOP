#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cudnn.h>

typedef struct
{
    uint32_t in_h;
    uint32_t in_w;
    uint32_t in_ch;
    uint32_t inHW;
    uint32_t inChKhKw;
    uint32_t inBatchNumel;
    uint32_t out_ch;
    uint32_t out_h;
    uint32_t out_w;
    uint32_t outHW;
    uint32_t outBatchNumel;
    uint32_t Kh;
    uint32_t Kw;
    uint32_t KhKw;
    uint32_t Sh;
    uint32_t Sw;
    uint32_t Ph;
    uint32_t Pw;

    uint32_t k_tiles;
    uint32_t first_k_tile;

    size_t kernelWeightsCount;
    size_t biasWeightsCount;
} Conv2DParam;

typedef struct
{
    uint32_t batch_size;
    uint32_t input_size;
    uint32_t hidden_size;
    uint32_t output_size;
    uint32_t warmup_iterations;
    uint32_t benchmark_iterations;
} NcclMlpParam;

