#define DWCONV_SPIKE_KERNEL_CAPACITY 128
#define DWCONV_SPIKE_BOUNDARY_KH 8
#define DWCONV_SPIKE_BOUNDARY_KW 16
#define DWCONV_SPIKE_KERNEL_NAME dwconv_spike_4x128x256_fp16_kernel
#define DWCONV_SPIKE_TARGET_NAME "dwconv_spike_k128_fp16"

#include "dwconv_spike_fp16_common.cuh"
