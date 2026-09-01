#include "grouped_gemm.cuh"
#include "tuning_bindings.h"
#include "tuning_configs.cuh"

namespace cudaop::grouped_gemm
{
namespace
{

template <typename ShapeConfig>
void bind_down(
    pybind11::module_& module,
    const char* name)
{
    module.def(name, &run<ShapeConfig, false, true>, name);
}

using K8M128N64S2 = TuningShapeConfig<
    128, 64, 16, 64, 32, 16, 8, 2>;
using K8M128N64S3 = TuningShapeConfig<
    128, 64, 16, 64, 32, 16, 8, 3>;
using K8M128N64S4 = TuningShapeConfig<
    128, 64, 16, 64, 32, 16, 8, 4>;
using K8M128N64S5 = TuningShapeConfig<
    128, 64, 16, 64, 32, 16, 8, 5>;
using K8M64N64S3 = TuningShapeConfig<
    64, 64, 16, 32, 32, 16, 8, 3>;
using K8M64N64S4 = TuningShapeConfig<
    64, 64, 16, 32, 32, 16, 8, 4>;
using K8M64N64S5 = TuningShapeConfig<
    64, 64, 16, 32, 32, 16, 8, 5>;
using K8M128N32S2 = TuningShapeConfig<
    128, 32, 16, 64, 32, 16, 8, 2>;
using K8M128N32S3 = TuningShapeConfig<
    128, 32, 16, 64, 32, 16, 8, 3>;
using K8M128N32S4 = TuningShapeConfig<
    128, 32, 16, 64, 32, 16, 8, 4>;
using K8M128N32S5 = TuningShapeConfig<
    128, 32, 16, 64, 32, 16, 8, 5>;
using K8M64N32S3 = TuningShapeConfig<
    64, 32, 16, 32, 32, 16, 8, 3>;
using K8M64N32S4 = TuningShapeConfig<
    64, 32, 16, 32, 32, 16, 8, 4>;
using K16M128N64S2 = TuningShapeConfig<
    128, 64, 32, 64, 32, 32, 16, 2>;
using K16M128N64S3 = TuningShapeConfig<
    128, 64, 32, 64, 32, 32, 16, 3>;

}  // namespace

void bind_down_variants(pybind11::module_& module)
{
    bind_down<K8M128N64S2>(
        module,
        "down_tb128x64x16_w64x32x16_i8_s2");
    bind_down<K8M128N64S3>(
        module,
        "down_tb128x64x16_w64x32x16_i8_s3");
    bind_down<K8M128N64S4>(
        module,
        "down_tb128x64x16_w64x32x16_i8_s4");
    bind_down<K8M128N64S5>(
        module,
        "down_tb128x64x16_w64x32x16_i8_s5");
    bind_down<K8M64N64S3>(
        module,
        "down_tb64x64x16_w32x32x16_i8_s3");
    bind_down<K8M64N64S4>(
        module,
        "down_tb64x64x16_w32x32x16_i8_s4");
    bind_down<K8M64N64S5>(
        module,
        "down_tb64x64x16_w32x32x16_i8_s5");
    bind_down<K8M128N32S2>(
        module,
        "down_tb128x32x16_w64x32x16_i8_s2");
    bind_down<K8M128N32S3>(
        module,
        "down_tb128x32x16_w64x32x16_i8_s3");
    bind_down<K8M128N32S4>(
        module,
        "down_tb128x32x16_w64x32x16_i8_s4");
    bind_down<K8M128N32S5>(
        module,
        "down_tb128x32x16_w64x32x16_i8_s5");
    bind_down<K8M64N32S3>(
        module,
        "down_tb64x32x16_w32x32x16_i8_s3");
    bind_down<K8M64N32S4>(
        module,
        "down_tb64x32x16_w32x32x16_i8_s4");
    bind_down<K16M128N64S2>(
        module,
        "down_tb128x64x32_w64x32x32_i16_s2");
    bind_down<K16M128N64S3>(
        module,
        "down_tb128x64x32_w64x32x32_i16_s3");
}

}  // namespace cudaop::grouped_gemm
