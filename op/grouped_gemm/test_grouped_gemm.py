"""CUTLASS、PyTorch、Triton 与 cuTile LoRA Grouped GEMM 对比。"""

import logging
import statistics
from collections.abc import Callable

import torch

from cudaop_grouped_gemm import (
    CuTileLoraBgradGrouped,
    CuTileLoraDownGrouped,
    CuTileLoraFusedDownUpGrouped,
    CuTileLoraUpGrouped,
    CutlassLoraBgradGrouped,
    CutlassLoraFusedDownUp,
    CutlassLoraFusedDownUpGrouped,
    LoraBgradGrouped,
    LoraDownGrouped,
    LoraFusedDownUpGrouped,
    LoraUpGrouped,
    cutlass_fused_lora,
    cutile_fused_lora,
    gmm,
    lora_gmm,
    lora_gmm_k16,
    torch_gmm,
    triton_fused_lora,
)


LOGGER = logging.getLogger("cudaop_grouped_gemm_test")

BF16_RTOL = 2e-2
BF16_ATOL = 2e-2
BF16_GRAD_ATOL = 5e-1
TORCH_WEIGHT_GRAD_ATOL = 8.0
WARMUP_ITERATIONS = 20
BENCHMARK_ITERATIONS = 100
SIZES = [128, 157, 97, 100, 111, 129, 138, 101]
SIZES = [i * 20 for i in SIZES]


def reference_down(
    a: torch.Tensor,
    weight: torch.Tensor,
    batch_sizes: torch.Tensor,
) -> torch.Tensor:
    outputs = []
    offset = 0
    for expert, size in enumerate(batch_sizes.tolist()):
        outputs.append(
            a[offset:offset + size] @ weight[expert].transpose(0, 1)
        )
        offset += size
    return torch.cat(outputs, dim=0)


def torch_lora_gmm(
    a: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    sizes: torch.Tensor,
) -> torch.Tensor:
    hidden = torch_gmm(
        a,
        down_weight,
        sizes,
        True,
    )
    return torch_gmm(
        hidden,
        up_weight,
        sizes,
        False,
    )


def reference_up(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    batch_sizes: torch.Tensor,
) -> torch.Tensor:
    outputs = []
    offset = 0
    for expert, size in enumerate(batch_sizes.tolist()):
        outputs.append(
            hidden[offset:offset + size] @ weight[expert]
        )
        offset += size
    return torch.cat(outputs, dim=0)


def reference_lora(
    a: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    batch_sizes: torch.Tensor,
) -> torch.Tensor:
    """使用逐 expert PyTorch matmul 构造可求导参考。"""
    hidden = reference_down(a, down_weight, batch_sizes)
    return reference_up(hidden, up_weight, batch_sizes)


def reference_bgrad(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    batch_sizes: torch.Tensor,
) -> torch.Tensor:
    """使用逐 expert FP32 matmul 构造 LoRA 权重梯度参考。"""
    output = torch.zeros(
        batch_sizes.numel(),
        16,
        rhs.shape[1],
        device=lhs.device,
        dtype=torch.float32,
    )
    offset = 0
    for expert, size in enumerate(batch_sizes.tolist()):
        if size > 0:
            output[expert] = (
                lhs[offset:offset + size].float().transpose(0, 1)
                @ rhs[offset:offset + size].float()
            )
        offset += size
    return output


def max_error(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> float:
    return (actual.float() - expected.float()).abs().max().item()


def cutile_is_supported() -> bool:
    """当前 cuda.tile 工具链只接受 Blackwell 架构目标。"""
    major, _ = torch.cuda.get_device_capability()
    return major >= 10


def run_portable_accuracy() -> None:
    """在 cuda.tile 不支持的 GPU 上验证其余实现。"""
    torch.manual_seed(19)
    sizes = torch.tensor([17, 9, 23, 11])
    tokens = int(sizes.sum())
    experts = sizes.numel()
    hidden_size = 256
    source_a = torch.randn(
        tokens,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    source_down = torch.randn(
        experts,
        16,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    source_up = torch.randn_like(source_down)
    grad_output = torch.randn_like(source_a) * 0.1

    def execute(operation: Callable) -> tuple[torch.Tensor, ...]:
        a = source_a.detach().clone().requires_grad_(True)
        down_weight = (
            source_down.detach().clone().requires_grad_(True)
        )
        up_weight = source_up.detach().clone().requires_grad_(True)
        output = operation(a, down_weight, up_weight, sizes)
        output.backward(grad_output)
        return (
            output.detach(),
            a.grad.detach(),
            down_weight.grad.detach(),
            up_weight.grad.detach(),
        )

    expected = execute(reference_lora)
    implementations = (
        ("CUTLASS K32/K16", lora_gmm),
        ("CUTLASS K16/K8", lora_gmm_k16),
        ("CUTLASS fused", cutlass_fused_lora),
        ("Torch grouped_mm", torch_lora_gmm),
        ("Triton fused", triton_fused_lora),
    )
    names = ("output", "grad input", "grad down", "grad up")
    LOGGER.info(
        "%-18s | %-12s | %14s",
        "implementation",
        "tensor",
        "reference error",
    )
    LOGGER.info("-" * 51)
    for implementation_name, operation in implementations:
        actual = execute(operation)
        for tensor_name, actual_value, expected_value in zip(
            names,
            actual,
            expected,
        ):
            torch.testing.assert_close(
                actual_value,
                expected_value,
                rtol=BF16_RTOL,
                atol=BF16_GRAD_ATOL,
            )
            LOGGER.info(
                "%-18s | %-12s | %14.6f",
                implementation_name,
                tensor_name,
                max_error(actual_value, expected_value),
            )

    tail_sizes = torch.tensor([0, 67, 2, 0, 129])
    tail_tokens = int(tail_sizes.sum())
    tail_input = torch.randn(
        tail_tokens,
        33,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    tail_down = torch.randn(
        5,
        16,
        33,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    tail_up = torch.randn(
        5,
        16,
        130,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    tail_grad = torch.randn(
        tail_tokens,
        130,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1

    def execute_tail(operation: Callable) -> tuple[torch.Tensor, ...]:
        a = tail_input.detach().clone().requires_grad_(True)
        down_weight = tail_down.detach().clone().requires_grad_(True)
        up_weight = tail_up.detach().clone().requires_grad_(True)
        output = operation(
            a,
            down_weight,
            up_weight,
            tail_sizes,
        )
        output.backward(tail_grad)
        return output, a.grad, down_weight.grad, up_weight.grad

    tail_expected = execute_tail(reference_lora)
    tail_actual = execute_tail(cutlass_fused_lora)
    for actual_value, expected_value in zip(
        tail_actual,
        tail_expected,
    ):
        torch.testing.assert_close(
            actual_value,
            expected_value,
            rtol=BF16_RTOL,
            atol=BF16_GRAD_ATOL,
        )

    lhs = torch.randn(
        tail_tokens,
        16,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    rhs = torch.randn(
        tail_tokens,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    expected_bgrad = reference_bgrad(lhs, rhs, tail_sizes)
    cutlass_bgrad = CutlassLoraBgradGrouped(5, hidden_size)(
        lhs,
        rhs,
        tail_sizes,
    )
    triton_bgrad = LoraBgradGrouped(5, hidden_size)(
        lhs,
        rhs,
        tail_sizes,
    )
    for actual_bgrad in (cutlass_bgrad, triton_bgrad):
        torch.testing.assert_close(
            actual_bgrad.float(),
            expected_bgrad,
            rtol=BF16_RTOL,
            atol=BF16_GRAD_ATOL,
        )

    odd_rhs = rhs[:, :33].contiguous()
    odd_expected = reference_bgrad(lhs, odd_rhs, tail_sizes)
    odd_actual = CutlassLoraBgradGrouped(5, 33)(
        lhs,
        odd_rhs,
        tail_sizes,
    )
    torch.testing.assert_close(
        odd_actual.float(),
        odd_expected,
        rtol=BF16_RTOL,
        atol=BF16_GRAD_ATOL,
    )
    LOGGER.info("")
    LOGGER.info(
        (
            "CUTLASS fused 尾块/空 expert 与 bgrad K=256/33 "
            "回归 [PASS]"
        )
    )


def run_accuracy() -> None:
    sizes = torch.tensor(SIZES)
    tokens = int(sizes.sum())
    experts = sizes.numel()
    hidden_size = 256
    a = torch.randn(
        tokens,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    down_weight = torch.randn(
        experts,
        16,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    up_weight = torch.randn_like(down_weight)
    down = LoraDownGrouped(down_weight)
    up = LoraUpGrouped(up_weight)
    fused = LoraFusedDownUpGrouped(down_weight, up_weight)
    cutile_down = CuTileLoraDownGrouped(down_weight)
    cutile_up = CuTileLoraUpGrouped(up_weight)
    cutile_fused = CuTileLoraFusedDownUpGrouped(
        down_weight,
        up_weight,
    )
    cutlass_fused = CutlassLoraFusedDownUpGrouped(
        down_weight,
        up_weight,
    )

    expected_hidden = reference_down(
        a.float(),
        down_weight.float(),
        sizes,
    )
    cutlass_hidden = gmm(a, down_weight, sizes, True)
    torch_hidden = torch_gmm(a, down_weight, sizes, True)
    triton_hidden = down(a, sizes)
    cutile_hidden = cutile_down(a, sizes)
    for actual in (
        cutlass_hidden,
        torch_hidden,
        triton_hidden,
        cutile_hidden,
    ):
        torch.testing.assert_close(
            actual.float(),
            expected_hidden,
            rtol=BF16_RTOL,
            atol=BF16_ATOL,
        )

    expected_output = reference_up(
        triton_hidden.float(),
        up_weight.float(),
        sizes,
    )
    cutlass_output = gmm(triton_hidden, up_weight, sizes, False)
    torch_output = torch_gmm(
        triton_hidden,
        up_weight,
        sizes,
        False,
    )
    triton_output = up(triton_hidden, sizes)
    cutile_output = cutile_up(cutile_hidden, sizes)
    fused_hidden, fused_output = fused(a, sizes)
    cutile_fused_hidden, cutile_fused_output = cutile_fused(
        a,
        sizes,
    )
    cutlass_fused_hidden, cutlass_fused_output = cutlass_fused(
        a,
        sizes,
    )
    for actual in (
        cutlass_output,
        torch_output,
        triton_output,
        cutile_output,
    ):
        torch.testing.assert_close(
            actual.float(),
            expected_output,
            rtol=BF16_RTOL,
            atol=BF16_ATOL,
        )
    torch.testing.assert_close(
        fused_hidden,
        triton_hidden,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        fused_output,
        triton_output,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        cutile_fused_hidden,
        cutile_hidden,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        cutile_fused_output,
        cutile_output,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        cutlass_fused_hidden.float(),
        expected_hidden,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    )
    expected_cutlass_fused_output = reference_up(
        cutlass_fused_hidden.float(),
        up_weight.float(),
        sizes,
    )
    torch.testing.assert_close(
        cutlass_fused_output.float(),
        expected_cutlass_fused_output,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    )

    LOGGER.info(
        (
            "%-8s | %-14s | %14s | %14s | "
            "%14s | %14s | %14s"
        ),
        "stage",
        "output shape",
        "CUTLASS error",
        "Torch error",
        "Triton error",
        "cuTile error",
        "CUTLASS/Triton",
    )
    LOGGER.info("-" * 117)
    accuracy_rows = (
        (
            "down",
            triton_hidden,
            expected_hidden,
            cutlass_hidden,
            torch_hidden,
            cutile_hidden,
        ),
        (
            "up",
            triton_output,
            expected_output,
            cutlass_output,
            torch_output,
            cutile_output,
        ),
    )
    for (
        stage,
        triton_value,
        expected,
        cutlass_value,
        torch_value,
        cutile_value,
    ) in (
        accuracy_rows
    ):
        LOGGER.info(
            (
                "%-8s | %-14s | %14.6f | %14.6f | "
                "%14.6f | %14.6f | %14.6f"
            ),
            stage,
            str(tuple(triton_value.shape)),
            max_error(cutlass_value, expected),
            max_error(torch_value, expected),
            max_error(triton_value, expected),
            max_error(cutile_value, expected),
            max_error(cutlass_value, triton_value),
        )

    LOGGER.info("")
    LOGGER.info(
        "%-14s | %-14s | %18s",
        "fused output",
        "shape",
        "separate diff",
    )
    LOGGER.info("-" * 53)
    LOGGER.info(
        "%-14s | %-14s | %18.6f",
        "saved hidden",
        str(tuple(fused_hidden.shape)),
        max_error(fused_hidden, triton_hidden),
    )
    LOGGER.info(
        "%-14s | %-14s | %18.6f",
        "final output",
        str(tuple(fused_output.shape)),
        max_error(fused_output, triton_output),
    )
    LOGGER.info(
        "%-14s | %-14s | %18.6f",
        "cuTile output",
        str(tuple(cutile_fused_output.shape)),
        max_error(cutile_fused_output, cutile_output),
    )
    LOGGER.info(
        "%-14s | %-14s | %18.6f",
        "CUTLASS output",
        str(tuple(cutlass_fused_output.shape)),
        max_error(
            cutlass_fused_output,
            expected_cutlass_fused_output,
        ),
    )

    tail_sizes = torch.tensor([0, 67, 2, 0, 129])
    tail_input = torch.randn(
        int(tail_sizes.sum()),
        33,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    tail_down = torch.randn(
        5,
        16,
        33,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    tail_up = torch.randn(
        5,
        16,
        130,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    tail_fused = CutlassLoraFusedDownUpGrouped(
        tail_down,
        tail_up,
    )
    tail_hidden, tail_output = tail_fused(tail_input, tail_sizes)
    tail_expected_hidden = reference_down(
        tail_input.float(),
        tail_down.float(),
        tail_sizes,
    )
    tail_expected_output = reference_up(
        tail_hidden.float(),
        tail_up.float(),
        tail_sizes,
    )
    torch.testing.assert_close(
        tail_hidden.float(),
        tail_expected_hidden,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    )
    torch.testing.assert_close(
        tail_output.float(),
        tail_expected_output,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    )
    LOGGER.info("")
    LOGGER.info(
        (
            "CUTLASS fusion 尾块：sizes=%s D=33 I=130 "
            "hidden_error=%.6f output_error=%.6f [PASS]"
        ),
        tail_sizes.tolist(),
        max_error(tail_hidden, tail_expected_hidden),
        max_error(tail_output, tail_expected_output),
    )


def run_backward_accuracy() -> None:
    sizes = torch.tensor(SIZES)
    tokens = int(sizes.sum())
    experts = sizes.numel()
    hidden_size = 256
    source_a = torch.randn(
        tokens,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    source_down = torch.randn(
        experts,
        16,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    source_up = torch.randn_like(source_down)
    grad_output = torch.randn_like(source_a)

    def execute(operation):
        a = source_a.detach().clone().requires_grad_(True)
        down_weight = (
            source_down.detach().clone().requires_grad_(True)
        )
        up_weight = (
            source_up.detach().clone().requires_grad_(True)
        )
        output = operation(
            a,
            down_weight,
            up_weight,
            sizes,
        )
        output.backward(grad_output)
        return (
            output.detach(),
            a.grad.detach(),
            down_weight.grad.detach(),
            up_weight.grad.detach(),
        )

    cutlass_results = execute(lora_gmm)
    cutlass_fused_results = execute(cutlass_fused_lora)
    torch_results = execute(torch_lora_gmm)
    triton_results = execute(triton_fused_lora)
    cutile_results = execute(cutile_fused_lora)
    names = (
        "output",
        "grad input",
        "grad down",
        "grad up",
    )
    for triton_value, cutlass_value in zip(
        triton_results,
        cutlass_results,
    ):
        torch.testing.assert_close(
            triton_value,
            cutlass_value,
            rtol=BF16_RTOL,
            atol=BF16_GRAD_ATOL,
        )
    for fused_value, cutlass_value in zip(
        cutlass_fused_results,
        cutlass_results,
    ):
        torch.testing.assert_close(
            fused_value,
            cutlass_value,
            rtol=BF16_RTOL,
            atol=BF16_GRAD_ATOL,
        )
    for cutile_value, cutlass_value in zip(
        cutile_results,
        cutlass_results,
    ):
        torch.testing.assert_close(
            cutile_value,
            cutlass_value,
            rtol=BF16_RTOL,
            atol=BF16_GRAD_ATOL,
        )
    torch_atols = (
        BF16_GRAD_ATOL,
        BF16_GRAD_ATOL,
        TORCH_WEIGHT_GRAD_ATOL,
        TORCH_WEIGHT_GRAD_ATOL,
    )
    for torch_value, cutlass_value, atol in zip(
        torch_results,
        cutlass_results,
        torch_atols,
    ):
        torch.testing.assert_close(
            torch_value,
            cutlass_value,
            rtol=BF16_RTOL,
            atol=atol,
        )

    LOGGER.info(
        (
            "%-12s | %-18s | %18s | %18s | "
            "%18s | %18s"
        ),
        "tensor",
        "shape",
        "Torch/CUTLASS diff",
        "CUTLASS fused diff",
        "Triton/CUTLASS diff",
        "cuTile/CUTLASS diff",
    )
    LOGGER.info("-" * 119)
    for (
        name,
        torch_value,
        fused_value,
        triton_value,
        cutile_value,
        cutlass_value,
    ) in zip(
        names,
        torch_results,
        cutlass_fused_results,
        triton_results,
        cutile_results,
        cutlass_results,
    ):
        LOGGER.info(
            (
                "%-12s | %-18s | %18.6f | "
                "%18.6f | %18.6f | %18.6f"
            ),
            name,
            str(tuple(triton_value.shape)),
            max_error(torch_value, cutlass_value),
            max_error(fused_value, cutlass_value),
            max_error(triton_value, cutlass_value),
            max_error(cutile_value, cutlass_value),
        )

    empty_sizes = torch.tensor([0, 3, 0, 5])
    empty_tokens = int(empty_sizes.sum())
    empty_input = torch.randn(
        empty_tokens,
        32,
        device="cuda",
        dtype=torch.bfloat16,
    )
    empty_down = torch.randn(
        4,
        16,
        32,
        device="cuda",
        dtype=torch.bfloat16,
    )
    empty_up = torch.randn_like(empty_down)
    empty_grad_output = torch.randn_like(empty_input)

    def execute_empty(operation):
        a = empty_input.detach().clone().requires_grad_(True)
        down_weight = (
            empty_down.detach().clone().requires_grad_(True)
        )
        up_weight = (
            empty_up.detach().clone().requires_grad_(True)
        )
        output = operation(
            a,
            down_weight,
            up_weight,
            empty_sizes,
        )
        output.backward(empty_grad_output)
        return a.grad, down_weight.grad, up_weight.grad

    cutlass_empty = execute_empty(lora_gmm)
    cutlass_fused_empty = execute_empty(cutlass_fused_lora)
    triton_empty = execute_empty(triton_fused_lora)
    cutile_empty = execute_empty(cutile_fused_lora)
    for triton_value, cutlass_value in zip(
        triton_empty,
        cutlass_empty,
    ):
        torch.testing.assert_close(
            triton_value,
            cutlass_value,
            rtol=BF16_RTOL,
            atol=BF16_GRAD_ATOL,
        )
    for fused_value, cutlass_value in zip(
        cutlass_fused_empty,
        cutlass_empty,
    ):
        torch.testing.assert_close(
            fused_value,
            cutlass_value,
            rtol=BF16_RTOL,
            atol=BF16_GRAD_ATOL,
        )
    for cutile_value, cutlass_value in zip(
        cutile_empty,
        cutlass_empty,
    ):
        torch.testing.assert_close(
            cutile_value,
            cutlass_value,
            rtol=BF16_RTOL,
            atol=BF16_GRAD_ATOL,
        )
    for implementation in (
        cutlass_fused_empty,
        triton_empty,
        cutile_empty,
    ):
        for weight_grad in implementation[1:]:
            if torch.count_nonzero(weight_grad[[0, 2]]).item() != 0:
                raise AssertionError("空 expert 的权重梯度必须为零")
    LOGGER.info("")
    LOGGER.info(
        "空 expert 与 K 尾块回归：sizes=%s hidden_size=32 [PASS]",
        empty_sizes.tolist(),
    )

    tail_sizes = torch.tensor([0, 67, 2, 0, 129])
    tail_tokens = int(tail_sizes.sum())
    tail_input = torch.randn(
        tail_tokens,
        33,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    tail_down = torch.randn(
        5,
        16,
        33,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    tail_up = torch.randn(
        5,
        16,
        130,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    tail_grad_output = torch.randn(
        tail_tokens,
        130,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1

    def execute_tail(operation):
        a = tail_input.detach().clone().requires_grad_(True)
        down_weight = tail_down.detach().clone().requires_grad_(True)
        up_weight = tail_up.detach().clone().requires_grad_(True)
        output = operation(
            a,
            down_weight,
            up_weight,
            tail_sizes,
        )
        output.backward(tail_grad_output)
        return output, a.grad, down_weight.grad, up_weight.grad

    tail_reference = execute_tail(reference_lora)
    tail_fused = execute_tail(cutlass_fused_lora)
    for fused_value, reference_value in zip(
        tail_fused,
        tail_reference,
    ):
        torch.testing.assert_close(
            fused_value,
            reference_value,
            rtol=BF16_RTOL,
            atol=BF16_GRAD_ATOL,
        )
    LOGGER.info(
        (
            "CUTLASS fused backward 尾块：sizes=%s D=33 I=130 "
            "grad_input_error=%.6f [PASS]"
        ),
        tail_sizes.tolist(),
        max_error(tail_fused[1], tail_reference[1]),
    )

    cached_input = tail_input.detach().clone().requires_grad_(True)
    cached_down = tail_down.detach().clone().requires_grad_(True)
    cached_up = tail_up.detach().clone().requires_grad_(True)
    cached_caller = CutlassLoraFusedDownUp(
        cached_down,
        cached_up,
    )
    cached_operation = cached_caller.grouped_operation
    cached_output = cached_caller(
        cached_input,
        tail_sizes,
    )
    cached_output.backward(tail_grad_output)
    cached_values = (
        cached_output,
        cached_input.grad,
        cached_down.grad,
        cached_up.grad,
    )
    for cached_value, fused_value in zip(
        cached_values,
        tail_fused,
    ):
        torch.testing.assert_close(
            cached_value,
            fused_value,
            rtol=0.0,
            atol=0.0,
        )

    with torch.no_grad():
        cached_up.add_(
            torch.tensor(
                0.01,
                device=cached_up.device,
                dtype=cached_up.dtype,
            )
        )
    refreshed_output = cached_operation(tail_input, tail_sizes)[1]
    fresh_output = CutlassLoraFusedDownUpGrouped(
        cached_down,
        cached_up,
    )(tail_input, tail_sizes)[1]
    torch.testing.assert_close(
        refreshed_output,
        fresh_output,
        rtol=0.0,
        atol=0.0,
    )

    routed_sizes = tail_sizes.clone()
    cached_operation(tail_input, routed_sizes)
    routed_sizes[1] += 1
    routed_sizes[4] -= 1
    routed_output = cached_operation(tail_input, routed_sizes)[1]
    fresh_routed_output = CutlassLoraFusedDownUpGrouped(
        cached_down,
        cached_up,
    )(tail_input, routed_sizes)[1]
    torch.testing.assert_close(
        routed_output,
        fresh_routed_output,
        rtol=0.0,
        atol=0.0,
    )
    LOGGER.info(
        "CUTLASS packed weight/metadata 缓存失效回归 [PASS]"
    )


def run_bgrad_accuracy() -> None:
    sizes = torch.tensor([0, 67, 2, 0, 129])
    tokens = int(sizes.sum())
    experts = sizes.numel()
    lhs = torch.randn(
        tokens,
        16,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    aligned_rhs = torch.randn(
        tokens,
        256,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    cutlass = CutlassLoraBgradGrouped(experts, 256)
    triton = LoraBgradGrouped(experts, 256)
    cutile = CuTileLoraBgradGrouped(experts, 256)
    expected = reference_bgrad(lhs, aligned_rhs, sizes)
    cutlass_output = cutlass(lhs, aligned_rhs, sizes)
    triton_output = triton(lhs, aligned_rhs, sizes)
    cutile_output = cutile(lhs, aligned_rhs, sizes)
    for actual in (cutlass_output, triton_output, cutile_output):
        torch.testing.assert_close(
            actual.float(),
            expected,
            rtol=BF16_RTOL,
            atol=BF16_GRAD_ATOL,
        )

    odd_rhs = aligned_rhs[:, :33].contiguous()
    odd_expected = reference_bgrad(lhs, odd_rhs, sizes)
    odd_cutlass = CutlassLoraBgradGrouped(experts, 33)(
        lhs,
        odd_rhs,
        sizes,
    )
    torch.testing.assert_close(
        odd_cutlass.float(),
        odd_expected,
        rtol=BF16_RTOL,
        atol=BF16_GRAD_ATOL,
    )
    if torch.count_nonzero(cutlass_output[[0, 3]]).item() != 0:
        raise AssertionError("空 expert 的 bgrad 必须为零")

    empty_sizes = torch.tensor([0, 0, 0])
    empty_output = CutlassLoraBgradGrouped(3, 33)(
        torch.empty(
            0,
            16,
            device="cuda",
            dtype=torch.bfloat16,
        ),
        torch.empty(
            0,
            33,
            device="cuda",
            dtype=torch.bfloat16,
        ),
        empty_sizes,
    )
    if torch.count_nonzero(empty_output).item() != 0:
        raise AssertionError("全空输入的 bgrad 必须为零")

    routed_sizes = torch.tensor([64, 64])
    routed_lhs = lhs[:128]
    routed_rhs = aligned_rhs[:128]
    cached = CutlassLoraBgradGrouped(2, 256)
    cached(routed_lhs, routed_rhs, routed_sizes)
    routed_sizes[0] += 1
    routed_sizes[1] -= 1
    routed_output = cached(routed_lhs, routed_rhs, routed_sizes)
    routed_expected = reference_bgrad(
        routed_lhs,
        routed_rhs,
        routed_sizes,
    )
    torch.testing.assert_close(
        routed_output.float(),
        routed_expected,
        rtol=BF16_RTOL,
        atol=BF16_GRAD_ATOL,
    )

    LOGGER.info(
        "%-12s | %14s | %14s | %14s | %14s",
        "path",
        "shape",
        "CUTLASS error",
        "Triton error",
        "cuTile error",
    )
    LOGGER.info("-" * 83)
    LOGGER.info(
        "%-12s | %14s | %14.6f | %14.6f | %14.6f",
        "aligned",
        str(tuple(cutlass_output.shape)),
        max_error(cutlass_output, expected),
        max_error(triton_output, expected),
        max_error(cutile_output, expected),
    )
    LOGGER.info(
        "%-12s | %14s | %14.6f | %14s | %14s",
        "tail K=33",
        str(tuple(odd_cutlass.shape)),
        max_error(odd_cutlass, odd_expected),
        "-",
        "-",
    )
    LOGGER.info(
        "空 expert、全空输入与 metadata 版本失效回归 [PASS]"
    )


def benchmark_once(operation: Callable[[], None]) -> float:
    for _ in range(WARMUP_ITERATIONS):
        operation()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(BENCHMARK_ITERATIONS):
        operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / BENCHMARK_ITERATIONS


def benchmark(operation: Callable[[], None]) -> float:
    samples = [benchmark_once(operation) for _ in range(5)]
    return statistics.median(samples)


def run_performance() -> None:
    sizes = torch.tensor(SIZES, device="cuda")
    tokens = int(sizes.sum())
    experts = sizes.numel()
    hidden_size = 2048
    a = torch.randn(
        tokens,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    down_weight = torch.randn(
        experts,
        16,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    up_weight = torch.randn_like(down_weight)
    down = LoraDownGrouped(down_weight)
    up = LoraUpGrouped(up_weight)
    fused = LoraFusedDownUpGrouped(down_weight, up_weight)
    cutile_down = CuTileLoraDownGrouped(down_weight)
    cutile_up = CuTileLoraUpGrouped(up_weight)
    cutile_fused = CuTileLoraFusedDownUpGrouped(
        down_weight,
        up_weight,
    )
    cutlass_fused = CutlassLoraFusedDownUpGrouped(
        down_weight,
        up_weight,
    )
    hidden = down(a, sizes)
    cutile_hidden = cutile_down(a, sizes)

    operations = {
        "CUTLASS down": lambda: gmm(
            a,
            down_weight,
            sizes,
            True,
        ),
        "Torch down": lambda: torch_gmm(
            a,
            down_weight,
            sizes,
            True,
        ),
        "Triton down": lambda: down(a, sizes),
        "CUTLASS up": lambda: gmm(
            hidden,
            up_weight,
            sizes,
            False,
        ),
        "Torch up": lambda: torch_gmm(
            hidden,
            up_weight,
            sizes,
            False,
        ),
        "Triton up": lambda: up(hidden, sizes),
        "cuTile down": lambda: cutile_down(a, sizes),
        "cuTile up": lambda: cutile_up(cutile_hidden, sizes),
    }
    timings = {
        name: benchmark(operation)
        for name, operation in operations.items()
    }

    def cutlass_down_up() -> None:
        current = gmm(a, down_weight, sizes, True)
        gmm(current, up_weight, sizes, False)

    def torch_down_up() -> None:
        current = torch_gmm(a, down_weight, sizes, True)
        torch_gmm(current, up_weight, sizes, False)

    def triton_down_up() -> None:
        current = down(a, sizes)
        up(current, sizes)

    timings["CUTLASS total"] = benchmark(cutlass_down_up)
    timings["Torch total"] = benchmark(torch_down_up)
    timings["Triton total"] = benchmark(triton_down_up)
    timings["Triton fused"] = benchmark(lambda: fused(a, sizes))
    timings["CUTLASS fused"] = benchmark(
        lambda: cutlass_fused(a, sizes)
    )

    def cutile_down_up() -> None:
        current = cutile_down(a, sizes)
        cutile_up(current, sizes)

    timings["cuTile total"] = benchmark(cutile_down_up)
    timings["cuTile fused"] = benchmark(
        lambda: cutile_fused(a, sizes)
    )

    def triton_down_rebuild() -> None:
        down.clear_metadata_cache()
        down(a, sizes)

    def triton_up_rebuild() -> None:
        up.clear_metadata_cache()
        up(hidden, sizes)

    def triton_total_rebuild() -> None:
        down.clear_metadata_cache()
        up.clear_metadata_cache()
        current = down(a, sizes)
        up(current, sizes)

    timings["Triton rebuild down"] = benchmark(
        triton_down_rebuild
    )
    timings["Triton rebuild up"] = benchmark(triton_up_rebuild)
    timings["Triton rebuild total"] = benchmark(
        triton_total_rebuild
    )

    def triton_fused_rebuild() -> None:
        fused.clear_metadata_cache()
        fused(a, sizes)

    timings["Triton rebuild fused"] = benchmark(
        triton_fused_rebuild
    )

    LOGGER.info(
        (
            "%-8s | %12s | %12s | %12s | %12s | "
            "%14s | %14s"
        ),
        "stage",
        "CUTLASS(us)",
        "Torch(us)",
        "Triton(us)",
        "cuTile(us)",
        "CUTLASS/Triton",
        "CUTLASS/cuTile",
    )
    LOGGER.info("-" * 111)
    for stage in ("down", "up", "total"):
        cutlass_us = timings[f"CUTLASS {stage}"]
        torch_us = timings[f"Torch {stage}"]
        triton_us = timings[f"Triton {stage}"]
        cutile_us = timings[f"cuTile {stage}"]
        LOGGER.info(
            (
                "%-8s | %12.3f | %12.3f | %12.3f | "
                "%12.3f | %13.3fx | %13.3fx"
            ),
            stage,
            cutlass_us,
            torch_us,
            triton_us,
            cutile_us,
            cutlass_us / triton_us,
            cutlass_us / cutile_us,
        )
    LOGGER.info(
        (
            "%-8s | %12s | %12s | %12.3f | "
            "%12.3f | %13.3fx | %13.3fx"
        ),
        "fused",
        "-",
        "-",
        timings["Triton fused"],
        timings["cuTile fused"],
        timings["Triton total"] / timings["Triton fused"],
        timings["cuTile total"] / timings["cuTile fused"],
    )
    LOGGER.info(
        (
            "speedup：down/up/total=CUTLASS/实现，"
            "fused=各实现 separate/fused；大于 1 表示后者更快"
        )
    )
    LOGGER.info(
        "CUTLASS fused：%.3f us，separate/fused=%.3fx",
        timings["CUTLASS fused"],
        timings["CUTLASS total"] / timings["CUTLASS fused"],
    )
    LOGGER.info(
        (
            "Triton metadata rebuild(us)："
            "down=%.3f up=%.3f total=%.3f fused=%.3f"
        ),
        timings["Triton rebuild down"],
        timings["Triton rebuild up"],
        timings["Triton rebuild total"],
        timings["Triton rebuild fused"],
    )

    cutlass_bgrad = CutlassLoraBgradGrouped(
        experts,
        hidden_size,
    )
    triton_bgrad = LoraBgradGrouped(experts, hidden_size)
    cutile_bgrad = CuTileLoraBgradGrouped(experts, hidden_size)
    cutlass_bgrad_us = benchmark(
        lambda: cutlass_bgrad(hidden, a, sizes)
    )
    triton_bgrad_us = benchmark(
        lambda: triton_bgrad(hidden, a, sizes)
    )
    cutile_bgrad_us = benchmark(
        lambda: cutile_bgrad(hidden, a, sizes)
    )
    LOGGER.info("")
    LOGGER.info(
        "%-18s | %16s | %12s | %10s",
        "bgrad implementation",
        "time",
        "vs Triton",
        "result",
    )
    LOGGER.info("-" * 65)
    for name, elapsed in (
        ("CUTLASS rank16", cutlass_bgrad_us),
        ("Triton", triton_bgrad_us),
        ("cuTile", cutile_bgrad_us),
    ):
        LOGGER.info(
            "%-18s | %13.3f us | %11.3fx | %10s",
            name,
            elapsed,
            triton_bgrad_us / elapsed,
            "pass",
        )

    grad_output = torch.randn_like(a)
    cached_down_weight = down_weight.detach().requires_grad_(True)
    cached_up_weight = up_weight.detach().requires_grad_(True)
    cached_cutlass_fused = CutlassLoraFusedDownUp(
        cached_down_weight,
        cached_up_weight,
    )

    def cutlass_forward_backward() -> None:
        input_value = a.detach().requires_grad_(True)
        down_value = down_weight.detach().requires_grad_(True)
        up_value = up_weight.detach().requires_grad_(True)
        output = lora_gmm(
            input_value,
            down_value,
            up_value,
            sizes,
        )
        output.backward(grad_output)

    def cutlass_fused_forward_backward() -> None:
        input_value = a.detach().requires_grad_(True)
        down_value = down_weight.detach().requires_grad_(True)
        up_value = up_weight.detach().requires_grad_(True)
        output = cutlass_fused_lora(
            input_value,
            down_value,
            up_value,
            sizes,
        )
        output.backward(grad_output)

    def cutlass_cached_forward_backward() -> None:
        cached_down_weight.grad = None
        cached_up_weight.grad = None
        input_value = a.detach().requires_grad_(True)
        output = cached_cutlass_fused(
            input_value,
            sizes,
        )
        output.backward(grad_output)

    def torch_forward_backward() -> None:
        input_value = a.detach().requires_grad_(True)
        down_value = down_weight.detach().requires_grad_(True)
        up_value = up_weight.detach().requires_grad_(True)
        output = torch_lora_gmm(
            input_value,
            down_value,
            up_value,
            sizes,
        )
        output.backward(grad_output)

    def triton_forward_backward() -> None:
        input_value = a.detach().requires_grad_(True)
        down_value = down_weight.detach().requires_grad_(True)
        up_value = up_weight.detach().requires_grad_(True)
        output = triton_fused_lora(
            input_value,
            down_value,
            up_value,
            sizes,
        )
        output.backward(grad_output)

    def cutile_forward_backward() -> None:
        input_value = a.detach().requires_grad_(True)
        down_value = down_weight.detach().requires_grad_(True)
        up_value = up_weight.detach().requires_grad_(True)
        output = cutile_fused_lora(
            input_value,
            down_value,
            up_value,
            sizes,
        )
        output.backward(grad_output)

    cutlass_backward_us = benchmark(cutlass_forward_backward)
    cutlass_fused_backward_us = benchmark(
        cutlass_fused_forward_backward
    )
    cutlass_cached_backward_us = benchmark(
        cutlass_cached_forward_backward
    )
    torch_backward_us = benchmark(torch_forward_backward)
    triton_backward_us = benchmark(triton_forward_backward)
    cutile_backward_us = benchmark(cutile_forward_backward)
    LOGGER.info("")
    LOGGER.info(
        (
            "%-18s | %16s | %18s | %12s | %10s"
        ),
        "implementation",
        "backward ops",
        "forward+backward",
        "speedup",
        "result",
    )
    LOGGER.info("-" * 88)
    LOGGER.info(
        "%-18s | %16d | %15.3f us | %12s | %10s",
        "CUTLASS separate",
        4,
        cutlass_backward_us,
        "-",
        "baseline",
    )
    LOGGER.info(
        "%-18s | %16d | %15.3f us | %11.3fx | %10s",
        "CUTLASS fused",
        3,
        cutlass_fused_backward_us,
        cutlass_backward_us / cutlass_fused_backward_us,
        "pass",
    )
    LOGGER.info(
        "%-18s | %16d | %15.3f us | %11.3fx | %10s",
        "CUTLASS cached",
        3,
        cutlass_cached_backward_us,
        cutlass_backward_us / cutlass_cached_backward_us,
        "pass",
    )
    LOGGER.info(
        "%-18s | %16d | %15.3f us | %11.3fx | %10s",
        "Torch separate",
        4,
        torch_backward_us,
        cutlass_backward_us / torch_backward_us,
        "pass",
    )
    LOGGER.info(
        "%-18s | %16d | %15.3f us | %11.3fx | %10s",
        "Triton fused",
        3,
        triton_backward_us,
        cutlass_backward_us / triton_backward_us,
        "pass",
    )
    LOGGER.info(
        "%-18s | %16d | %15.3f us | %11.3fx | %10s",
        "cuTile fused",
        3,
        cutile_backward_us,
        cutlass_backward_us / cutile_backward_us,
        "pass",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not torch.cuda.is_available():
        raise RuntimeError("测试需要 CUDA GPU")

    LOGGER.info(
        "配置：device=%s dtype=bfloat16 arch=sm_%d%d torch=%s",
        torch.cuda.get_device_name(),
        *torch.cuda.get_device_capability(),
        torch.__version__,
    )
    if not cutile_is_supported():
        LOGGER.info("")
        LOGGER.info(
            (
                "cuda.tile 不支持当前架构，跳过 cuTile 专项；"
                "执行 Torch/CUTLASS/Triton 可移植回归"
            )
        )
        run_portable_accuracy()
        LOGGER.info("")
        LOGGER.info("[SUCCESS] cudaop_grouped_gemm 对比测试通过")
        return
    LOGGER.info("")
    LOGGER.info("阶段：LoRA down/up 分阶段前向精度验证")
    torch.manual_seed(11)
    run_accuracy()
    LOGGER.info("")
    LOGGER.info("阶段：LoRA fused backward 精度验证")
    run_backward_accuracy()
    LOGGER.info("")
    LOGGER.info("阶段：LoRA bgrad 精度与边界验证")
    run_bgrad_accuracy()
    LOGGER.info("")
    LOGGER.info(
        (
            "阶段：LoRA down/up 分阶段端到端性能对比，"
            "hidden_size=2048 warmup=%d iterations=%d"
        ),
        WARMUP_ITERATIONS,
        BENCHMARK_ITERATIONS,
    )
    run_performance()
    LOGGER.info("")
    LOGGER.info("[SUCCESS] cudaop_grouped_gemm 对比测试通过")


if __name__ == "__main__":
    main()
