#ifndef CUDATEST_IF_CUH
#define CUDATEST_IF_CUH

#include <iostream>
#include <vector>
#include <string>

#include <cuda_fp16.h>
#include <cuda_fp16.hpp>

#include <torch/torch.h>
#include <c10/util/Half.h>

#include "neuron_common.h"


std::vector<torch::Tensor> IFNodeFPTTFLOATLaunch(
        const torch::Tensor &inputs, float v_th, float v_reset, ResetType resetType, int threads);

std::vector<torch::Tensor> IFNodeFPTTHALFLaunch(
        const torch::Tensor &inputs, float v_th, float v_reset, ResetType resetType, int threads);

std::vector<torch::Tensor> IFNodeBPTTSigmoidFLOATLaunch(
        const torch::Tensor &grad_spike_seq, const torch::Tensor &h_seq, const torch::Tensor &grad_v_seq,
        float v_th, float v_reset, float alpha, ResetType resetType, bool detach_reset, int threads);

std::vector<torch::Tensor> IFNodeBPTTSigmoidHALFLaunch(
        const torch::Tensor &grad_spike_seq, const torch::Tensor &h_seq, const torch::Tensor &grad_v_seq,
        float v_th, float v_reset, float alpha, ResetType resetType, bool detach_reset, int threads);

std::vector<torch::Tensor> IFNodeBPTTATanFLOATLaunch(
        const torch::Tensor &grad_spike_seq, const torch::Tensor &h_seq, const torch::Tensor &grad_v_seq,
        float v_th, float v_reset, float alpha, ResetType resetType, bool detach_reset, int threads);

std::vector<torch::Tensor> IFNodeBPTTATanHALFLaunch(
        const torch::Tensor &grad_spike_seq, const torch::Tensor &h_seq, const torch::Tensor &grad_v_seq,
        float v_th, float v_reset, float alpha, ResetType resetType, bool detach_reset, int threads);


void if_main();


#endif
