#ifndef CUDATEST_NEURON_H
#define CUDATEST_NEURON_H

#include <iostream>
#include <vector>
#include <string>
#include <torch/torch.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor (C++)")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous (C++)")
#define CHECK_FP16(x) TORCH_CHECK(x.scalar_type() == torch::kHalf, #x "must be CUDA and of type Half (C++)")

#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

enum class SurrogateFunc : int
{
    Sigmoid = 0,
    ATan = 1,
};

enum class NeuronType : int
{
    IF = 0,
    LIF = 1,
};

enum class ResetType : int
{
    SoftReset = 0,
    HardReset = 1,
};


#endif
