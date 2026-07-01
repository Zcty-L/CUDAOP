#include "cumsum.h"
#include "histogram.h"
#include "indices.h"
#include "replicate.h"
#include "sort.h"

#include <torch/extension.h>

namespace lora_moe
{

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
    module.def(
        "exclusive_cumsum",
        &exclusive_cumsum,
        "Batched exclusive cumulative sum");
    module.def("histogram", &histogram, "Even-width histogram");
    module.def(
        "inclusive_cumsum",
        &inclusive_cumsum,
        "Batched inclusive cumulative sum");
    module.def(
        "indices",
        &indices,
        "Construct indices for a sparse matrix");
    module.def(
        "replicate_forward",
        &replicate_forward,
        "Replicate a vector dynamically (forward)");
    module.def(
        "replicate_backward",
        &replicate_backward,
        "Replicate a vector dynamically (backward)");
    module.def("sort", &sort, "Key/value radix sort");
}

}  // namespace lora_moe
