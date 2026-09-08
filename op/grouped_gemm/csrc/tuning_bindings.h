#pragma once

#include <pybind11/pybind11.h>

namespace cudaop::grouped_gemm
{

void bind_up_variants(pybind11::module_& module);
void bind_down_variants(pybind11::module_& module);
void bind_bgrad_variants(pybind11::module_& module);

}  // namespace cudaop::grouped_gemm
