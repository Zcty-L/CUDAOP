#include "grouped_gemm.cuh"

#include <torch/extension.h>

namespace cudaop::grouped_gemm
{

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
    module.def(
        "grouped_gemm",
        &grouped_gemm,
        "CUTLASS BF16 grouped GEMM");
}

}  // namespace cudaop::grouped_gemm
