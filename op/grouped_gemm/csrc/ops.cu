#include "grouped_gemm.cuh"
#include "fused_lora.cuh"

#include <torch/extension.h>

namespace cudaop::grouped_gemm
{

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
    module.def(
        "grouped_gemm",
        &grouped_gemm,
        "CUTLASS BF16 grouped GEMM");
    module.def(
        "fused_lora_forward",
        &fused_lora_forward,
        "CUTLASS BF16 fused LoRA down/up forward");
    module.def(
        "fused_lora_backward",
        &fused_lora_backward,
        "CUTLASS BF16 fused LoRA input-gradient backward");
    module.def(
        "lora_bgrad",
        &lora_bgrad,
        "CUTLASS BF16 rank-16 LoRA weight gradient");
    module.def(
        "lora_bgrad_grouped",
        &lora_bgrad_grouped,
        "CUTLASS BF16 rank-16 grouped LoRA weight gradient");
}

}  // namespace cudaop::grouped_gemm
