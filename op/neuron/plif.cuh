#ifndef CUDATEST_PLIF_CUH
#define CUDATEST_PLIF_CUH

#include <iostream>
#include <vector>
#include <string>

#include <cuda_fp16.h>

#include <torch/torch.h>
#include <c10/util/Half.h>

#include "neuron_common.h"


std::vector<torch::Tensor> PLIFNodeBPTTSigmoidFLOATLaunch(
        const torch::Tensor &grad_spike_seq, const torch::Tensor &h_seq, const torch::Tensor &grad_v_seq,
        const torch::Tensor &v_seq, float v_th, float v_reset, float tau, float alpha,
        bool decay_input, ResetType resetType, bool detach_reset, int threads);

std::vector<torch::Tensor> PLIFNodeBPTTSigmoidHALFLaunch(
        const torch::Tensor &grad_spike_seq, const torch::Tensor &h_seq, const torch::Tensor &grad_v_seq,
        const torch::Tensor &v_seq, float v_th, float v_reset, float tau, float alpha,
        bool decay_input, ResetType resetType, bool detach_reset, int threads);


        
std::vector<torch::Tensor> PLIFNodeBPTTATanFLOATLaunch(
        const torch::Tensor &grad_spike_seq, const torch::Tensor &h_seq, const torch::Tensor &grad_v_seq,
        const torch::Tensor &v_seq, const float v_th, const float v_reset, const float tau, const float alpha,
        const bool decay_input, ResetType resetType, const bool detach_reset, const int threads);

std::vector<torch::Tensor> PLIFNodeBPTTATanHALFLaunch(
        const torch::Tensor &grad_spike_seq, const torch::Tensor &h_seq, const torch::Tensor &grad_v_seq,
        const torch::Tensor &v_seq, const float v_th, const float v_reset, const float tau, const float alpha,
        const bool decay_input, ResetType resetType, const bool detach_reset, const int threads);



void plif_main();

#endif