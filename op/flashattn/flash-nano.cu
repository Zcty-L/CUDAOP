#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>
#include "flash-nano.cuh"

using namespace nvcuda;

#define WARPS     4
#define WARP_SIZE 32
#define tileQ     16
#define tileKV    128
#define D         16

// #define use_cudnn


__global__ void flashattn_nano_kernel(
        __half *__restrict__ Q,
        __half *__restrict__ K,
        __half *__restrict__ V,
        __half *__restrict__ O,
        float *__restrict__ S,
        const int N,
        const float softmax_scale
)
{
    __shared__ __align__ (64)
    char smem[16 * 1024]; // 16 * 16 + 16 * 128 + 16 * 128
    __half *sQ = reinterpret_cast<__half *>(smem); // tileQ  * D = 16 * 16
    __half *sKV = sQ + tileQ * D;                  // tileKV * D = 128 * 16
    float *sP = reinterpret_cast<float *>(sKV + tileKV * D);    // tileQ * tileKV = 16 * 128
    float *sO = sP + tileQ * tileKV;       // tileQ * D = 16 * 16
    float *sRowMax = sO + tileQ * D;       // 16
    float *sRowSum = sRowMax + tileQ;      // 16

    __half2 *sQ_2 = reinterpret_cast<__half2 *>(sQ);
    __half2 *sKV_2 = reinterpret_cast<__half2 *>(sKV);
    float2 *sO_float2_ptr = reinterpret_cast<float2 *>(sO);
    float4 *sP_float4_ptr = reinterpret_cast<float4 *>(sP);

    const int tid = threadIdx.x;
    const int row = tid / 8;
    const int col = tid % 8;

    __half2 *Q_ldg_ptr = reinterpret_cast<__half2 *>(Q + blockIdx.z * N * D + blockIdx.x * tileQ * D);
    __half2 *K_ldg_ptr = reinterpret_cast<__half2 *>(K + blockIdx.z * N * D);
    __half2 *V_ldg_ptr = reinterpret_cast<__half2 *>(V + blockIdx.z * N * D);
    __half2 *O_stg_ptr = reinterpret_cast<__half2 *>(O + blockIdx.z * N * D + blockIdx.x * tileQ * D);

    if (tid < tileQ)
    {
        sRowMax[tid] = -INFINITY;
        sRowSum[tid] = 0.0f;
    }

    // Zero out sO
    sO[tid] = 0.0f;
    sO[tid + 128] = 0.0f;

    // ldg --> sts Q
    sQ_2[tid] = Q_ldg_ptr[tid];
    __syncthreads();

    // lds Q
    const __half *q_tile_ptr = sQ;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> q_frag;
    wmma::load_matrix_sync(q_frag, q_tile_ptr, D);

    int k_tiles = (N + tileKV - 1) / tileKV;

    for (int kt = 0; kt < k_tiles; kt++)
    {
        // ldg --> sts K
#pragma unroll
        for (int i = 0; i < 8; i++)
        {
            sKV_2[i * 128 + tid] = K_ldg_ptr[i * 128 + tid];
        }
        __syncthreads();

        // S = Q @ K^T
        for (int KV_Stride = 0; KV_Stride < tileKV; KV_Stride += 16)
        {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag;
            wmma::fill_fragment(s_frag, 0.0f);

            const __half *k_tile_ptr = sKV + KV_Stride * D;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> k_frag;

            wmma::load_matrix_sync(k_frag, k_tile_ptr, D);
            wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);

#pragma unroll
            for (int i = 0; i < s_frag.num_elements; ++i)
            {
                s_frag.x[i] *= softmax_scale;
            }

            wmma::store_matrix_sync(sP + KV_Stride, s_frag, tileKV, wmma::mem_row_major);
        }

        // =============================================================== test S
        float *S_stg_ptr = S + blockIdx.z * N * N + blockIdx.x * tileQ * N + kt * tileKV + tid;
#pragma unroll
        for (int i = 0; i < 16; i++)
        {
            S_stg_ptr[i * N] = sP[i * 128 + tid];
        }
        __syncthreads();
        // =============================================================== end test S

        // Online Softmax
        float thread_max = -INFINITY;
        float thread_sum = 0.0f;

#pragma unroll
        for (int i = 0; i < 32; i += 8)
        {
            float4 val = sP_float4_ptr[row * 32 + col + i];
            thread_max = fmaxf(thread_max, fmaxf(fmaxf(val.x, val.y), fmaxf(val.z, val.w)));
        }

#pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1)
        {
            thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, offset));
        }

        const float old_max = sRowMax[row];
        const float new_max = fmaxf(old_max, thread_max);
        const float exp_diff = __expf(old_max - new_max);
        __syncthreads();

#pragma unroll
        for (int i = 0; i < 32; i += 8)
        {
            const int idx = row * 32 + col + i;
            float4 val = sP_float4_ptr[idx];
            float4 sum4;
            sum4.x = __expf(val.x - new_max);
            sum4.y = __expf(val.y - new_max);
            sum4.z = __expf(val.z - new_max);
            sum4.w = __expf(val.w - new_max);
            sP_float4_ptr[idx] = sum4;

            thread_sum += (sum4.x + sum4.y + sum4.z + sum4.w);
        }

        for (int offset = 4; offset >= 1; offset /= 2)
        {
            thread_sum += __shfl_xor_sync(0xffffffff, thread_sum, offset);
        }

        if (col == 0)
        {
            const float old_sum = sRowSum[row];
            const float new_sum = exp_diff * old_sum + thread_sum;
            sRowSum[row] = new_sum;
            sRowMax[row] = new_max;
        }
        __syncthreads();

        // Convert sP to half precision  16x128
        float2 *sP_float2_ptr = reinterpret_cast<float2 *>(sP);
        __half2 *sP_half2_ptr = reinterpret_cast<__half2 *>(sP);
#pragma unroll
        for (int i = 0; i < 8; i++)
        {
            float2 val = sP_float2_ptr[i * 128 + tid];
            __half2 h2 = __float22half2_rn(val);
            sP_half2_ptr[i * 128 + tid] = h2;
        }
        __syncthreads();

        // ldg --> sts V
#pragma unroll
        for (int i = 0; i < 8; i++)
        {
            sKV_2[i * 128 + tid] = V_ldg_ptr[i * 128 + tid];
        }

        // O = P @ V | 16 * 16
        // out = out * torch.exp(old_max - new_max) + oi
        {
            float2 val = sO_float2_ptr[tid];
            val.x = val.x * exp_diff;
            val.y = val.y * exp_diff;
            sO_float2_ptr[tid] = val;
        }
        __syncthreads();

        __half *sP_half_ptr = reinterpret_cast<__half *>(sP);

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        load_matrix_sync(c_frag, sO, D, wmma::mem_row_major);

        for (int KV_Stride = 0; KV_Stride < tileKV; KV_Stride += 16)
        {
            const __half *a_tile_ptr = sP_half_ptr + KV_Stride;
            const __half *b_tile_ptr = sKV + KV_Stride * D;

            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;

            wmma::load_matrix_sync(a_frag, a_tile_ptr, tileKV);
            wmma::load_matrix_sync(b_frag, b_tile_ptr, D);

            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(sO, c_frag, D, wmma::mem_row_major);

        // ================== next kv_tile
        K_ldg_ptr += tileKV * D / 2;
        V_ldg_ptr += tileKV * D / 2;
    }

    // out / l
    const float _sum = sRowSum[row];

    float2 val = sO_float2_ptr[tid];
    val.x = val.x / _sum;
    val.y = val.y / _sum;

    // stg O | 16 * 16
    __half2 h2 = __float22half2_rn(val);
    O_stg_ptr[tid] = h2;

    // // if (blockIdx.x == 0 && tid == 0)
    // if (blockIdx.x == 0 && tid < tileQ)
    // {
    //     // printf("Idx: %3d  \n", tid);
    //     // printf("Idx: %3d | MAX: %.6lf  \n", tid, sRowMax[tid]);
    //     // printf("Idx: %3d | SUM: %.6lf  \n", tid, sRowSum[tid]);
    //     printf("Row: %3d | MAX: %.6lf | SUM: %.6lf  \n", tid, sRowMax[tid], sRowSum[tid]);
    // }
}


void flashattn_nano_launch(
        void *Q,
        void *K,
        void *V,
        void *O,
        void *S,
        const int B,
        const int H,
        const int N,
        const float softmax_scale
)
{
    int bx = (N + tileQ - 1) / tileQ;
    int by = 1;
    int bz = B * H;

    dim3 grid(bx, by, bz);

    printf("%d %d %d\n", bx, by, bz);

    flashattn_nano_kernel<<<grid, 32 * 4>>>(
            (__half *) Q, (__half *) K, (__half *) V, (__half *) O, (float *) S, N, softmax_scale);
}

void flashattn_nano()
{
    const int B = 1;   // batch size
    const int H = 1;   // head num
    const int N = 128 * 8;  // seq len
    const int d = D;
    const float softmax_scale = 0.5f;

    const int typeSize = sizeof(half);

    half *Q_h, *K_h, *V_h, *O_h;
    half *Q_d, *K_d, *V_d, *O_d;

    float *S_h, *S_cpu, *P_cpu, *O_cpu;
    float *S_d;

    cudaMallocHost((void **) &Q_h, B * H * N * D * typeSize);
    cudaMallocHost((void **) &K_h, B * H * N * D * typeSize);
    cudaMallocHost((void **) &V_h, B * H * N * D * typeSize);
    cudaMallocHost((void **) &O_h, B * H * N * D * typeSize);
    cudaMallocHost((void **) &S_h, N * N * sizeof(float));
    cudaMallocHost((void **) &S_cpu, N * N * sizeof(float));
    cudaMallocHost((void **) &P_cpu, N * N * sizeof(float));
    cudaMallocHost((void **) &O_cpu, B * H * N * D * sizeof(float));

    cudaMalloc((void **) &Q_d, B * H * N * D * typeSize);
    cudaMalloc((void **) &K_d, B * H * N * D * typeSize);
    cudaMalloc((void **) &V_d, B * H * N * D * typeSize);
    cudaMalloc((void **) &O_d, B * H * N * D * typeSize);
    cudaMalloc((void **) &S_d, N * N * sizeof(float));

    for (int i = 0; i < B * H * N * D; i++)
    {
        // Q_h[i] = __float2half(1.0f);
        // K_h[i] = __float2half(1.0f);
        // V_h[i] = __float2half(1.0f);

        Q_h[i] = __float2half((rand() & 255) / 255.0f);
        K_h[i] = __float2half((rand() & 255) / 255.0f);
        V_h[i] = __float2half((rand() & 255) / 255.0f);
    }

    cudaMemcpy(Q_d, Q_h, B * H * N * D * typeSize, cudaMemcpyHostToDevice);
    cudaMemcpy(K_d, K_h, B * H * N * D * typeSize, cudaMemcpyHostToDevice);
    cudaMemcpy(V_d, V_h, B * H * N * D * typeSize, cudaMemcpyHostToDevice);

    printf("===================  CPU   ===================\n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < D; k++)
            {
                sum += __half2float(Q_h[i * D + k]) * __half2float(K_h[j * D + k]);
            }
            S_cpu[i * N + j] = sum * softmax_scale;
        }
    }
    // printf("%.4lf %.4lf\n", S_h[0], S_cpu[0]);
    // printf("%.4lf %.4lf\n", S_h[1], S_cpu[1]);
    // printf("%.4lf %.4lf\n", S_h[2], S_cpu[2]);
    // printf("%.4lf %.4lf\n", S_h[3], S_cpu[3]);

    for (int r = 0; r < N; r++)
    {
        float max_value = -INFINITY;
        for (int c = 0; c < N; c++)
        {
            max_value = std::max(max_value, S_cpu[r * N + c]);
        }

        float sum_value = 0;
        for (int c = 0; c < N; c++)
        {
            sum_value += exp(S_cpu[r * N + c] - max_value);
        }

        for (int c = 0; c < N; c++)
        {
            P_cpu[r * N + c] = exp(S_cpu[r * N + c] - max_value) / sum_value;
            // P_cpu[r * N + c] = exp(S_cpu[r * N + c] - max_value);
        }

        // if (r < 16)
        // // if (r > 15 && r < 32)
        // {
        //     printf("Row: %3d | MAX: %.6lf | SUM: %.6lf\n", r, max_value, sum_value);
        // }
    }
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < D; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
            {
                sum += P_cpu[i * N + k] * __half2float(V_h[k * D + j]);
            }
            O_cpu[i * D + j] = sum;

            // if (i * N + j <= 16)
            // {
            //     printf("A %d %.4lf %.4lf\n", i * N + j, O_cpu[i * N + j], sum);
            // }
        }
    }
    // printf("%.4lf %.4lf\n", O_cpu[0], S_cpu[0]);
    // printf("%.4lf %.4lf\n", O_cpu[1], S_cpu[1]);
    // printf("%.4lf %.4lf\n", O_cpu[2], S_cpu[2]);
    // printf("%.4lf %.4lf\n", O_cpu[3], S_cpu[3]);

    printf("===================  GPU 1  ===================\n");
    flashattn_nano_launch(Q_d, K_d, V_d, O_d, S_d, B, H, N, softmax_scale);
    cudaMemcpy(S_h, S_d, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(O_h, O_d, B * H * N * d * sizeof(__half), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N * N; i++)
    {
        if (abs(S_cpu[i] - S_h[i]) > 0.0001f)
        {
            printf("Error: %d, gpu:%lf, cpu:%lf | %lf\n", i, S_h[i], S_cpu[i], S_h[i] - S_cpu[i]);
            break;
        }
    }
    printf("S %.4lf %.4lf\n", S_h[0], S_cpu[0]);
    printf("S %.4lf %.4lf\n", S_h[1], S_cpu[1]);
    printf("S %.4lf %.4lf\n", S_h[2], S_cpu[2]);
    printf("S %.4lf %.4lf\n", S_h[3], S_cpu[3]);

    printf("===================  GPU 2  ===================\n");
    for (int i = 0; i < N * D; i++)
    {
        if (abs(O_cpu[i] - __half2float(O_h[i])) > 0.1f)
        {
            printf("Error: %d, gpu:%lf, cpu:%lf | %lf\n",
                   i, __half2float(O_h[i]), O_cpu[i], __half2float(O_h[i]) - O_cpu[i]);
            break;
        }
    }
    printf("O %.4lf %.4lf\n", __half2float(O_h[0]), O_cpu[0]);
    printf("O %.4lf %.4lf\n", __half2float(O_h[1]), O_cpu[1]);
    printf("O %.4lf %.4lf\n", __half2float(O_h[2]), O_cpu[2]);
    printf("O %.4lf %.4lf\n", __half2float(O_h[3]), O_cpu[3]);
    printf("O %.4lf %.4lf\n", __half2float(O_h[16]), O_cpu[16]);
    printf("O %.4lf %.4lf\n", __half2float(O_h[17]), O_cpu[17]);

#ifdef use_cudnn
    printf("=================== cuBLAS ===================\n");
    cublasHandle_t blas_handle;
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta = 0;

    // S^T
    cublasSgemm(blas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                N, N, d, &alpha, Q_d, d, K_d, d, &beta, S_d, N
    );

    cudaMemcpy(S_h, S_d, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%.4lf ", S_h[i * N + j]);
        }
        printf("\n");
    }
#endif

    cudaFree(Q_d);
    cudaFree(K_d);
    cudaFree(V_d);
    cudaFree(O_d);
    cudaFree(S_d);

    cudaFreeHost(Q_h);
    cudaFreeHost(K_h);
    cudaFreeHost(V_h);
    cudaFreeHost(O_h);
    cudaFreeHost(S_h);
    cudaFreeHost(S_cpu);
    cudaFreeHost(P_cpu);
    cudaFreeHost(O_cpu);
}