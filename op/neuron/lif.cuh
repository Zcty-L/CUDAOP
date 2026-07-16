#ifndef CUDATEST_LIF_CUH
#define CUDATEST_LIF_CUH

#include <iostream>
#include <vector>
#include <string>

#include <cuda_fp16.h>

#include <torch/torch.h>
#include <c10/util/Half.h>

#include "neuron_common.h"


std::vector<torch::Tensor> LIFNodeFPTTFLOATLaunch(
        const torch::Tensor &inputs, float v_th, float v_reset, float tau,
        ResetType resetType, bool decay_input, int threads);

std::vector<torch::Tensor> LIFNodeFPTTHALFLaunch(
        const torch::Tensor &inputs, float v_th, float v_reset, float tau,
        ResetType resetType, bool decay_input, int threads);

std::vector<torch::Tensor> LIFNodeBPTTSigmoidFLOATLaunch(
        const torch::Tensor &grad_spike_seq, const torch::Tensor &h_seq, const torch::Tensor &grad_v_seq,
        float v_th, float v_reset, float tau, float alpha, bool decay_input, bool detach_reset,
        ResetType resetType, int threads);

std::vector<torch::Tensor> LIFNodeBPTTSigmoidHALFLaunch(
        const torch::Tensor &grad_spike_seq, const torch::Tensor &h_seq, const torch::Tensor &grad_v_seq,
        float v_th, float v_reset, float tau, float alpha, bool decay_input, bool detach_reset,
        ResetType resetType, int threads);

std::vector<torch::Tensor> LIFNodeBPTTATanFLOATLaunch(
        const torch::Tensor &grad_spike_seq, const torch::Tensor &h_seq, const torch::Tensor &grad_v_seq,
        float v_th, float v_reset, float tau, float alpha, bool decay_input, bool detach_reset,
        ResetType resetType, int threads);

std::vector<torch::Tensor> LIFNodeBPTTATanHALFLaunch(
        const torch::Tensor &grad_spike_seq, const torch::Tensor &h_seq, const torch::Tensor &grad_v_seq,
        float v_th, float v_reset, float tau, float alpha, bool decay_input, bool detach_reset,
        ResetType resetType, int threads);

void lif_main();

#endif
