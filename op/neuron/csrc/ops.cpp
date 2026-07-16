#include "if.cuh"
#include "lif.cuh"
#include "plif.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace neuron_ops
{

namespace
{

constexpr int kSigmoid = 0;
constexpr int kATan = 1;

using TensorVector = std::vector<torch::Tensor>;

void check_threads(int threads)
{
    TORCH_CHECK(
        threads >= 32 && threads <= 256 && threads % 32 == 0,
        "threads must be a multiple of 32 in [32, 256]");
}

void check_input(const torch::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(
        tensor.scalar_type() == torch::kFloat32 ||
            tensor.scalar_type() == torch::kFloat16,
        name,
        " must have dtype torch.float32 or torch.float16");
    TORCH_CHECK(tensor.dim() >= 1, name, " must have at least one dimension");
    TORCH_CHECK(tensor.size(0) > 0, name, " must have a non-empty time dimension");

    const int64_t elements_per_step = tensor.numel() / tensor.size(0);
    const int64_t vector_width =
        tensor.scalar_type() == torch::kFloat16 ? 8 : 4;
    TORCH_CHECK(
        elements_per_step % vector_width == 0,
        name,
        " must contain a multiple of ",
        vector_width,
        " elements per time step for aligned vector access");
}

void check_compatible(
    const torch::Tensor &reference,
    const torch::Tensor &tensor,
    const char *name)
{
    check_input(tensor, name);
    TORCH_CHECK(
        tensor.device() == reference.device(),
        name,
        " must be on the same device as the reference tensor");
    TORCH_CHECK(
        tensor.scalar_type() == reference.scalar_type(),
        name,
        " must have the same dtype as the reference tensor");
    TORCH_CHECK(
        tensor.sizes() == reference.sizes(),
        name,
        " must have the same shape as the reference tensor");
}

const float *check_args(const torch::Tensor &args, int64_t minimum_size)
{
    TORCH_CHECK(!args.is_cuda(), "args must be a CPU tensor");
    TORCH_CHECK(args.is_contiguous(), "args must be contiguous");
    TORCH_CHECK(args.scalar_type() == torch::kFloat32, "args must be float32");
    TORCH_CHECK(
        args.numel() >= minimum_size,
        "args must contain at least ",
        minimum_size,
        " values");
    return args.data_ptr<float>();
}

void check_surrogate(int surrogate_func_id)
{
    TORCH_CHECK(
        surrogate_func_id == kSigmoid || surrogate_func_id == kATan,
        "surrogate_func_id must be 0 (Sigmoid) or 1 (ATan)");
}

ResetType reset_type(bool hard_reset)
{
    return hard_reset ? ResetType::HardReset : ResetType::SoftReset;
}

TensorVector reorder_forward_outputs(TensorVector outputs)
{
    TORCH_CHECK(outputs.size() == 3, "forward kernel must return three tensors");
    return {outputs[0], outputs[2], outputs[1]};
}

}  // namespace

TensorVector IFNodeForward(
    const torch::Tensor &inputs,
    const torch::Tensor &args,
    bool hard_reset,
    int threads)
{
    check_input(inputs, "inputs");
    check_threads(threads);
    const float *values = check_args(args, 2);
    const c10::cuda::CUDAGuard device_guard(inputs.device());

    TensorVector outputs;
    if (inputs.scalar_type() == torch::kFloat16)
    {
        outputs = IFNodeFPTTHALFLaunch(
            inputs,
            values[0],
            values[1],
            reset_type(hard_reset),
            threads);
    }
    else
    {
        outputs = IFNodeFPTTFLOATLaunch(
            inputs,
            values[0],
            values[1],
            reset_type(hard_reset),
            threads);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return reorder_forward_outputs(std::move(outputs));
}

TensorVector IFNodeBackward(
    const torch::Tensor &grad_spike_seq,
    const torch::Tensor &grad_v_seq,
    const torch::Tensor &h_seq,
    const torch::Tensor &args,
    bool hard_reset,
    bool detach_reset,
    int surrogate_func_id,
    int threads)
{
    check_input(grad_spike_seq, "grad_spike_seq");
    check_compatible(grad_spike_seq, grad_v_seq, "grad_v_seq");
    check_compatible(grad_spike_seq, h_seq, "h_seq");
    check_threads(threads);
    check_surrogate(surrogate_func_id);
    const float *values = check_args(args, 3);
    const c10::cuda::CUDAGuard device_guard(grad_spike_seq.device());

    TensorVector outputs;
    if (grad_spike_seq.scalar_type() == torch::kFloat16)
    {
        if (surrogate_func_id == kATan)
        {
            outputs = IFNodeBPTTATanHALFLaunch(
                grad_spike_seq,
                h_seq,
                grad_v_seq,
                values[0],
                values[1],
                values[2],
                reset_type(hard_reset),
                detach_reset,
                threads);
        }
        else
        {
            outputs = IFNodeBPTTSigmoidHALFLaunch(
                grad_spike_seq,
                h_seq,
                grad_v_seq,
                values[0],
                values[1],
                values[2],
                reset_type(hard_reset),
                detach_reset,
                threads);
        }
    }
    else if (surrogate_func_id == kATan)
    {
        outputs = IFNodeBPTTATanFLOATLaunch(
            grad_spike_seq,
            h_seq,
            grad_v_seq,
            values[0],
            values[1],
            values[2],
            reset_type(hard_reset),
            detach_reset,
            threads);
    }
    else
    {
        outputs = IFNodeBPTTSigmoidFLOATLaunch(
            grad_spike_seq,
            h_seq,
            grad_v_seq,
            values[0],
            values[1],
            values[2],
            reset_type(hard_reset),
            detach_reset,
            threads);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return outputs;
}

TensorVector LIFNodeForward(
    const torch::Tensor &inputs,
    const torch::Tensor &args,
    bool hard_reset,
    bool decay_input,
    int threads)
{
    check_input(inputs, "inputs");
    check_threads(threads);
    const float *values = check_args(args, 3);
    const c10::cuda::CUDAGuard device_guard(inputs.device());

    TensorVector outputs;
    if (inputs.scalar_type() == torch::kFloat16)
    {
        outputs = LIFNodeFPTTHALFLaunch(
            inputs,
            values[0],
            values[1],
            values[2],
            reset_type(hard_reset),
            decay_input,
            threads);
    }
    else
    {
        outputs = LIFNodeFPTTFLOATLaunch(
            inputs,
            values[0],
            values[1],
            values[2],
            reset_type(hard_reset),
            decay_input,
            threads);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return reorder_forward_outputs(std::move(outputs));
}

TensorVector LIFNodeBackward(
    const torch::Tensor &grad_spike_seq,
    const torch::Tensor &grad_v_seq,
    const torch::Tensor &h_seq,
    const torch::Tensor &args,
    bool hard_reset,
    bool decay_input,
    bool detach_reset,
    int surrogate_func_id,
    int threads)
{
    check_input(grad_spike_seq, "grad_spike_seq");
    check_compatible(grad_spike_seq, grad_v_seq, "grad_v_seq");
    check_compatible(grad_spike_seq, h_seq, "h_seq");
    check_threads(threads);
    check_surrogate(surrogate_func_id);
    const float *values = check_args(args, 4);
    const c10::cuda::CUDAGuard device_guard(grad_spike_seq.device());

    TensorVector outputs;
    if (grad_spike_seq.scalar_type() == torch::kFloat16)
    {
        if (surrogate_func_id == kATan)
        {
            outputs = LIFNodeBPTTATanHALFLaunch(
                grad_spike_seq,
                h_seq,
                grad_v_seq,
                values[0],
                values[1],
                values[2],
                values[3],
                decay_input,
                detach_reset,
                reset_type(hard_reset),
                threads);
        }
        else
        {
            outputs = LIFNodeBPTTSigmoidHALFLaunch(
                grad_spike_seq,
                h_seq,
                grad_v_seq,
                values[0],
                values[1],
                values[2],
                values[3],
                decay_input,
                detach_reset,
                reset_type(hard_reset),
                threads);
        }
    }
    else if (surrogate_func_id == kATan)
    {
        outputs = LIFNodeBPTTATanFLOATLaunch(
            grad_spike_seq,
            h_seq,
            grad_v_seq,
            values[0],
            values[1],
            values[2],
            values[3],
            decay_input,
            detach_reset,
            reset_type(hard_reset),
            threads);
    }
    else
    {
        outputs = LIFNodeBPTTSigmoidFLOATLaunch(
            grad_spike_seq,
            h_seq,
            grad_v_seq,
            values[0],
            values[1],
            values[2],
            values[3],
            decay_input,
            detach_reset,
            reset_type(hard_reset),
            threads);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return outputs;
}

TensorVector PLIFNodeForward(
    const torch::Tensor &inputs,
    const torch::Tensor &args,
    bool hard_reset,
    bool decay_input,
    int threads)
{
    return LIFNodeForward(
        inputs,
        args,
        hard_reset,
        decay_input,
        threads);
}

TensorVector PLIFNodeBackward(
    const torch::Tensor &grad_spike_seq,
    const torch::Tensor &grad_v_seq,
    const torch::Tensor &h_seq,
    const torch::Tensor &v_seq,
    const torch::Tensor &args,
    bool hard_reset,
    bool decay_input,
    bool detach_reset,
    int surrogate_func_id,
    int threads)
{
    check_input(grad_spike_seq, "grad_spike_seq");
    check_compatible(grad_spike_seq, grad_v_seq, "grad_v_seq");
    check_compatible(grad_spike_seq, h_seq, "h_seq");
    check_compatible(grad_spike_seq, v_seq, "v_seq");
    check_threads(threads);
    check_surrogate(surrogate_func_id);
    const float *values = check_args(args, 4);
    const c10::cuda::CUDAGuard device_guard(grad_spike_seq.device());

    TensorVector outputs;
    if (grad_spike_seq.scalar_type() == torch::kFloat16)
    {
        if (surrogate_func_id == kATan)
        {
            outputs = PLIFNodeBPTTATanHALFLaunch(
                grad_spike_seq,
                h_seq,
                grad_v_seq,
                v_seq,
                values[0],
                values[1],
                values[2],
                values[3],
                decay_input,
                reset_type(hard_reset),
                detach_reset,
                threads);
        }
        else
        {
            outputs = PLIFNodeBPTTSigmoidHALFLaunch(
                grad_spike_seq,
                h_seq,
                grad_v_seq,
                v_seq,
                values[0],
                values[1],
                values[2],
                values[3],
                decay_input,
                reset_type(hard_reset),
                detach_reset,
                threads);
        }
    }
    else if (surrogate_func_id == kATan)
    {
        outputs = PLIFNodeBPTTATanFLOATLaunch(
            grad_spike_seq,
            h_seq,
            grad_v_seq,
            v_seq,
            values[0],
            values[1],
            values[2],
            values[3],
            decay_input,
            reset_type(hard_reset),
            detach_reset,
            threads);
    }
    else
    {
        outputs = PLIFNodeBPTTSigmoidFLOATLaunch(
            grad_spike_seq,
            h_seq,
            grad_v_seq,
            v_seq,
            values[0],
            values[1],
            values[2],
            values[3],
            decay_input,
            reset_type(hard_reset),
            detach_reset,
            threads);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return outputs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
    namespace py = pybind11;

    module.def(
        "IFNodeForward",
        &IFNodeForward,
        py::arg("inputs"),
        py::arg("args"),
        py::arg("hard_reset"),
        py::arg("threads") = 256);
    module.def(
        "IFNodeBackward",
        &IFNodeBackward,
        py::arg("grad_spike_seq"),
        py::arg("grad_v_seq"),
        py::arg("h_seq"),
        py::arg("args"),
        py::arg("hard_reset"),
        py::arg("detach_reset"),
        py::arg("surrogate_func_id"),
        py::arg("threads") = 256);
    module.def(
        "LIFNodeForward",
        &LIFNodeForward,
        py::arg("inputs"),
        py::arg("args"),
        py::arg("hard_reset"),
        py::arg("decay_input"),
        py::arg("threads") = 256);
    module.def(
        "LIFNodeBackward",
        &LIFNodeBackward,
        py::arg("grad_spike_seq"),
        py::arg("grad_v_seq"),
        py::arg("h_seq"),
        py::arg("args"),
        py::arg("hard_reset"),
        py::arg("decay_input"),
        py::arg("detach_reset"),
        py::arg("surrogate_func_id"),
        py::arg("threads") = 256);
    module.def(
        "PLIFNodeForward",
        &PLIFNodeForward,
        py::arg("inputs"),
        py::arg("args"),
        py::arg("hard_reset"),
        py::arg("decay_input"),
        py::arg("threads") = 256);
    module.def(
        "PLIFNodeBackward",
        &PLIFNodeBackward,
        py::arg("grad_spike_seq"),
        py::arg("grad_v_seq"),
        py::arg("h_seq"),
        py::arg("v_seq"),
        py::arg("args"),
        py::arg("hard_reset"),
        py::arg("decay_input"),
        py::arg("detach_reset"),
        py::arg("surrogate_func_id"),
        py::arg("threads") = 256);
}

}  // namespace neuron_ops
