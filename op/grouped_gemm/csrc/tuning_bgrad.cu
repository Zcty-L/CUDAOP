#include "grouped_gemm.cuh"
#include "tuning_bindings.h"
#include "tuning_configs.cuh"

namespace cudaop::grouped_gemm
{
namespace
{

template <typename ShapeConfig>
void bind_bgrad(
    pybind11::module_& module,
    const char* name)
{
    module.def(name, &run<ShapeConfig, true, false>, name);
}

using K8M64N128S2 = TuningShapeConfig<
    64, 128, 16, 32, 64, 16, 8, 2>;
using K8M64N128S3 = TuningShapeConfig<
    64, 128, 16, 32, 64, 16, 8, 3>;
using K8M64N128S4 = TuningShapeConfig<
    64, 128, 16, 32, 64, 16, 8, 4>;
using K8M64N128S5 = TuningShapeConfig<
    64, 128, 16, 32, 64, 16, 8, 5>;
using K8M64N64S3 = TuningShapeConfig<
    64, 64, 16, 32, 32, 16, 8, 3>;
using K8M64N64S4 = TuningShapeConfig<
    64, 64, 16, 32, 32, 16, 8, 4>;
using K8M64N64S5 = TuningShapeConfig<
    64, 64, 16, 32, 32, 16, 8, 5>;
using K8M32N128S2 = TuningShapeConfig<
    32, 128, 16, 32, 64, 16, 8, 2>;
using K8M32N128S3 = TuningShapeConfig<
    32, 128, 16, 32, 64, 16, 8, 3>;
using K8M32N128S4 = TuningShapeConfig<
    32, 128, 16, 32, 64, 16, 8, 4>;
using K8M32N128S5 = TuningShapeConfig<
    32, 128, 16, 32, 64, 16, 8, 5>;
using K8M32N64S3 = TuningShapeConfig<
    32, 64, 16, 32, 32, 16, 8, 3>;
using K8M32N64S4 = TuningShapeConfig<
    32, 64, 16, 32, 32, 16, 8, 4>;
using K16M64N128S2 = TuningShapeConfig<
    64, 128, 32, 32, 64, 32, 16, 2>;
using K16M64N128S3 = TuningShapeConfig<
    64, 128, 32, 32, 64, 32, 16, 3>;

}  // namespace

void bind_bgrad_variants(pybind11::module_& module)
{
    bind_bgrad<K8M64N128S2>(
        module,
        "bgrad_tb64x128x16_w32x64x16_i8_s2");
    bind_bgrad<K8M64N128S3>(
        module,
        "bgrad_tb64x128x16_w32x64x16_i8_s3");
    bind_bgrad<K8M64N128S4>(
        module,
        "bgrad_tb64x128x16_w32x64x16_i8_s4");
    bind_bgrad<K8M64N128S5>(
        module,
        "bgrad_tb64x128x16_w32x64x16_i8_s5");
    bind_bgrad<K8M64N64S3>(
        module,
        "bgrad_tb64x64x16_w32x32x16_i8_s3");
    bind_bgrad<K8M64N64S4>(
        module,
        "bgrad_tb64x64x16_w32x32x16_i8_s4");
    bind_bgrad<K8M64N64S5>(
        module,
        "bgrad_tb64x64x16_w32x32x16_i8_s5");
    bind_bgrad<K8M32N128S2>(
        module,
        "bgrad_tb32x128x16_w32x64x16_i8_s2");
    bind_bgrad<K8M32N128S3>(
        module,
        "bgrad_tb32x128x16_w32x64x16_i8_s3");
    bind_bgrad<K8M32N128S4>(
        module,
        "bgrad_tb32x128x16_w32x64x16_i8_s4");
    bind_bgrad<K8M32N128S5>(
        module,
        "bgrad_tb32x128x16_w32x64x16_i8_s5");
    bind_bgrad<K8M32N64S3>(
        module,
        "bgrad_tb32x64x16_w32x32x16_i8_s3");
    bind_bgrad<K8M32N64S4>(
        module,
        "bgrad_tb32x64x16_w32x32x16_i8_s4");
    bind_bgrad<K16M64N128S2>(
        module,
        "bgrad_tb64x128x32_w32x64x32_i16_s2");
    bind_bgrad<K16M64N128S3>(
        module,
        "bgrad_tb64x128x32_w32x64x32_i16_s3");
}

}  // namespace cudaop::grouped_gemm
