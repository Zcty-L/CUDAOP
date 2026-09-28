#include "grouped_gemm.cuh"
#include "tuning_bindings.h"
#include "tuning_configs.cuh"

namespace cudaop::grouped_gemm
{
namespace
{

template <typename ShapeConfig>
void bind_up(
    pybind11::module_& module,
    const char* name)
{
    module.def(name, &run<ShapeConfig, false, false>, name);
}

using K8SquareS2 = TuningShapeConfig<
    128, 128, 16, 64, 64, 16, 8, 2>;
using K8SquareS3 = TuningShapeConfig<
    128, 128, 16, 64, 64, 16, 8, 3>;
using K8SquareS5 = TuningShapeConfig<
    128, 128, 16, 64, 64, 16, 8, 5>;
using K16SquareS2 = TuningShapeConfig<
    128, 128, 32, 64, 64, 32, 16, 2>;
using K16SquareS3 = TuningShapeConfig<
    128, 128, 32, 64, 64, 32, 16, 3>;
using K8M64N128S2 = TuningShapeConfig<
    64, 128, 16, 32, 64, 16, 8, 2>;
using K8M64N128S3 = TuningShapeConfig<
    64, 128, 16, 32, 64, 16, 8, 3>;
using K8M64N128S4 = TuningShapeConfig<
    64, 128, 16, 32, 64, 16, 8, 4>;
using K16M64N128S2 = TuningShapeConfig<
    64, 128, 32, 32, 64, 32, 16, 2>;
using K8SquareW32N64S3 = TuningShapeConfig<
    128, 128, 16, 32, 64, 16, 8, 3>;
using K8SquareW32N64S4 = TuningShapeConfig<
    128, 128, 16, 32, 64, 16, 8, 4>;
using K8SquareW64N32S3 = TuningShapeConfig<
    128, 128, 16, 64, 32, 16, 8, 3>;
using K8SquareW64N32S4 = TuningShapeConfig<
    128, 128, 16, 64, 32, 16, 8, 4>;
using K8M128N64S3 = TuningShapeConfig<
    128, 64, 16, 64, 32, 16, 8, 3>;
using K8M128N64S4 = TuningShapeConfig<
    128, 64, 16, 64, 32, 16, 8, 4>;
using K8M64N64S3 = TuningShapeConfig<
    64, 64, 16, 32, 32, 16, 8, 3>;
using K8M64N64S4 = TuningShapeConfig<
    64, 64, 16, 32, 32, 16, 8, 4>;

}  // namespace

void bind_up_variants(pybind11::module_& module)
{
    bind_up<K8SquareS2>(
        module,
        "up_tb128x128x16_w64x64x16_i8_s2");
    bind_up<K8SquareS3>(
        module,
        "up_tb128x128x16_w64x64x16_i8_s3");
    bind_up<K8SquareS5>(
        module,
        "up_tb128x128x16_w64x64x16_i8_s5");
    bind_up<K16SquareS2>(
        module,
        "up_tb128x128x32_w64x64x32_i16_s2");
    bind_up<K16SquareS3>(
        module,
        "up_tb128x128x32_w64x64x32_i16_s3");
    bind_up<K8M64N128S2>(
        module,
        "up_tb64x128x16_w32x64x16_i8_s2");
    bind_up<K8M64N128S3>(
        module,
        "up_tb64x128x16_w32x64x16_i8_s3");
    bind_up<K8M64N128S4>(
        module,
        "up_tb64x128x16_w32x64x16_i8_s4");
    bind_up<K16M64N128S2>(
        module,
        "up_tb64x128x32_w32x64x32_i16_s2");
    bind_up<K8SquareW32N64S3>(
        module,
        "up_tb128x128x16_w32x64x16_i8_s3");
    bind_up<K8SquareW32N64S4>(
        module,
        "up_tb128x128x16_w32x64x16_i8_s4");
    bind_up<K8SquareW64N32S3>(
        module,
        "up_tb128x128x16_w64x32x16_i8_s3");
    bind_up<K8SquareW64N32S4>(
        module,
        "up_tb128x128x16_w64x32x16_i8_s4");
    bind_up<K8M128N64S3>(
        module,
        "up_tb128x64x16_w64x32x16_i8_s3");
    bind_up<K8M128N64S4>(
        module,
        "up_tb128x64x16_w64x32x16_i8_s4");
    bind_up<K8M64N64S3>(
        module,
        "up_tb64x64x16_w32x32x16_i8_s3");
    bind_up<K8M64N64S4>(
        module,
        "up_tb64x64x16_w32x32x16_i8_s4");
}

}  // namespace cudaop::grouped_gemm
