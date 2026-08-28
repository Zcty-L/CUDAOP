#include "grouped_gemm.cuh"

#include <torch/extension.h>

namespace cudaop::grouped_gemm
{

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
    module.def(
        "grouped_gemm",
        &grouped_gemm,
        "CUTLASS BF16 grouped GEMM with K32/K16 MMA");
    module.def(
        "grouped_gemm_k16",
        &grouped_gemm_k16,
        "CUTLASS BF16 grouped GEMM with K16/K8 MMA");
}

}  // namespace cudaop::grouped_gemm
