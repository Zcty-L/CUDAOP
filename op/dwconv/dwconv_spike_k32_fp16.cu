#define DWCONV_SPIKE_KERNEL_CAPACITY 32
#define DWCONV_SPIKE_BOUNDARY_KH 4
#define DWCONV_SPIKE_BOUNDARY_KW 8
#define DWCONV_SPIKE_KERNEL_NAME dwconv_spike_4x32x256_fp16_kernel
#define DWCONV_SPIKE_TARGET_NAME "dwconv_spike_k32_fp16"

#include "dwconv_spike_fp16_common.cuh"
