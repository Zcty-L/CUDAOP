#include "tuning_bindings.h"

#include <torch/extension.h>

namespace cudaop::grouped_gemm
{

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
    bind_up_variants(module);
    bind_down_variants(module);
    bind_bgrad_variants(module);
}

}  // namespace cudaop::grouped_gemm
