"""CUTLASS、PyTorch、Triton 与 cuTile LoRA Grouped GEMM 对比。"""

import logging
import os
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
    LoraFusedAgradGrouped,
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
BENCHMARK_SAMPLES = 10
FUSION_WARMUP_ITERATIONS = 100
FUSION_BENCHMARK_ITERATIONS = 1000
FUSION_HIDDEN_SIZE = int(
    os.environ.get("CUDAOP_GROUPED_GEMM_HIDDEN_SIZE", "2048")
)
FUSION_COMPARISON = os.environ.get(
    "CUDAOP_GROUPED_GEMM_FUSION_COMPARISON",
    "cutlass",
).lower()
CLEAR_METADATA_CACHE_VALUE = os.environ.get(
    "CUDAOP_GROUPED_GEMM_CLEAR_METADATA_CACHE",
    "1",
)
CLEAR_METADATA_CACHE = CLEAR_METADATA_CACHE_VALUE == "1"
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


def relative_l2_error(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> float:
    difference_norm = torch.linalg.vector_norm(
        actual.float() - expected.float()
    )
    reference_norm = torch.linalg.vector_norm(expected.float())
    return (difference_norm / reference_norm).item()


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


def benchmark_once(
    operation: Callable[[], None],
    warmup_iterations: int = WARMUP_ITERATIONS,
    benchmark_iterations: int = BENCHMARK_ITERATIONS,
) -> float:
    for _ in range(warmup_iterations):
        operation()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(benchmark_iterations):
        operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / benchmark_iterations


def benchmark(operation: Callable[[], None]) -> float:
    samples = [benchmark_once(operation) for _ in range(5)]
    return statistics.median(samples)


def benchmark_pair(
    separate: Callable[[], None],
    fused: Callable[[], None],
) -> tuple[list[float], list[float]]:
    """交替测量两个实现，降低温度和频率漂移带来的顺序偏差。"""
    samples = {"separate": [], "fused": []}
    operations = {"separate": separate, "fused": fused}
    for sample_index in range(BENCHMARK_SAMPLES):
        order = (
            ("separate", "fused")
            if sample_index % 2 == 0
            else ("fused", "separate")
        )
        for name in order:
            samples[name].append(
                benchmark_once(
                    operations[name],
                    FUSION_WARMUP_ITERATIONS,
                    FUSION_BENCHMARK_ITERATIONS,
                )
            )
    return samples["separate"], samples["fused"]


def benchmark_summary(
    samples: list[float],
) -> tuple[float, float, float, float]:
    median = statistics.median(samples)
    mean = statistics.mean(samples)
    coefficient_of_variation = (
        statistics.pstdev(samples) / mean * 100.0
    )
    return (
        median,
        min(samples),
        max(samples),
        coefficient_of_variation,
    )


def run_cutlass_fusion_performance() -> None:
    """分别对比 CUTLASS fusion 的前向、反向及训练总耗时。"""
    sizes = torch.tensor(SIZES)
    tokens = int(sizes.sum())
    experts = sizes.numel()
    hidden_size = FUSION_HIDDEN_SIZE
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
    cutlass_fused = CutlassLoraFusedDownUpGrouped(
        down_weight,
        up_weight,
    )

    # 前向精度：两次通用 CUTLASS Grouped GEMM 对比一个融合 kernel。
    separate_hidden = gmm(a, down_weight, sizes, True)
    separate_output = gmm(
        separate_hidden,
        up_weight,
        sizes,
        False,
    )
    fused_hidden, fused_output = cutlass_fused(a, sizes)
    torch.testing.assert_close(
        fused_hidden,
        separate_hidden,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    )
    torch.testing.assert_close(
        fused_output,
        separate_output,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    )

    # 反向精度：分别验证输入、down 权重和 up 权重梯度。
    grad_output = torch.randn_like(a)
    accuracy_separate_a = a.detach().requires_grad_(True)
    accuracy_separate_down = down_weight.detach().requires_grad_(True)
    accuracy_separate_up = up_weight.detach().requires_grad_(True)
    accuracy_fused_a = a.detach().requires_grad_(True)
    accuracy_fused_down = down_weight.detach().requires_grad_(True)
    accuracy_fused_up = up_weight.detach().requires_grad_(True)
    accuracy_fused_operation = CutlassLoraFusedDownUp(
        accuracy_fused_down,
        accuracy_fused_up,
    )
    accuracy_separate_output = lora_gmm(
        accuracy_separate_a,
        accuracy_separate_down,
        accuracy_separate_up,
        sizes,
    )
    accuracy_fused_output = accuracy_fused_operation(
        accuracy_fused_a,
        sizes,
    )
    accuracy_separate_gradients = torch.autograd.grad(
        accuracy_separate_output,
        (
            accuracy_separate_a,
            accuracy_separate_down,
            accuracy_separate_up,
        ),
        grad_output,
    )
    accuracy_fused_gradients = torch.autograd.grad(
        accuracy_fused_output,
        (
            accuracy_fused_a,
            accuracy_fused_down,
            accuracy_fused_up,
        ),
        grad_output,
    )
    for fused_gradient, separate_gradient in zip(
        accuracy_fused_gradients,
        accuracy_separate_gradients,
    ):
        torch.testing.assert_close(
            fused_gradient,
            separate_gradient,
            rtol=BF16_RTOL,
            atol=BF16_GRAD_ATOL,
        )
    LOGGER.info(
        (
            "正确性：tokens=%d experts=%d hidden=%d rank=16 "
            "hidden_diff=%.6f output_diff=%.6f [PASS]"
        ),
        tokens,
        experts,
        hidden_size,
        max_error(fused_hidden, separate_hidden),
        max_error(fused_output, separate_output),
    )
    LOGGER.info(
        (
            "反向正确性：grad_input_diff=%.6f "
            "grad_down_diff=%.6f grad_up_diff=%.6f [PASS]"
        ),
        *(
            max_error(fused_gradient, separate_gradient)
            for fused_gradient, separate_gradient in zip(
                accuracy_fused_gradients,
                accuracy_separate_gradients,
            )
        ),
    )

    def cutlass_separate_forward() -> None:
        hidden = gmm(a, down_weight, sizes, True)
        gmm(hidden, up_weight, sizes, False)

    def cutlass_fused_forward() -> None:
        cutlass_fused(a, sizes)

    forward_samples = benchmark_pair(
        cutlass_separate_forward,
        cutlass_fused_forward,
    )

    # 纯反向计时复用已构建的 autograd graph，不包含两条路径的前向。
    backward_separate_a = a.detach().requires_grad_(True)
    backward_separate_down = down_weight.detach().requires_grad_(True)
    backward_separate_up = up_weight.detach().requires_grad_(True)
    backward_fused_a = a.detach().requires_grad_(True)
    backward_fused_down = down_weight.detach().requires_grad_(True)
    backward_fused_up = up_weight.detach().requires_grad_(True)
    backward_fused_operation = CutlassLoraFusedDownUp(
        backward_fused_down,
        backward_fused_up,
    )
    backward_separate_output = lora_gmm(
        backward_separate_a,
        backward_separate_down,
        backward_separate_up,
        sizes,
    )
    backward_fused_output = backward_fused_operation(
        backward_fused_a,
        sizes,
    )

    def cutlass_separate_backward() -> None:
        torch.autograd.grad(
            backward_separate_output,
            (
                backward_separate_a,
                backward_separate_down,
                backward_separate_up,
            ),
            grad_output,
            retain_graph=True,
        )

    def cutlass_fused_backward() -> None:
        torch.autograd.grad(
            backward_fused_output,
            (
                backward_fused_a,
                backward_fused_down,
                backward_fused_up,
            ),
            grad_output,
            retain_graph=True,
        )

    backward_samples = benchmark_pair(
        cutlass_separate_backward,
        cutlass_fused_backward,
    )

    # 训练总耗时包含一次前向、一次反向，不包含权重预打包。
    total_separate_a = a.detach().requires_grad_(True)
    total_separate_down = down_weight.detach().requires_grad_(True)
    total_separate_up = up_weight.detach().requires_grad_(True)
    total_fused_a = a.detach().requires_grad_(True)
    total_fused_down = down_weight.detach().requires_grad_(True)
    total_fused_up = up_weight.detach().requires_grad_(True)
    total_fused_operation = CutlassLoraFusedDownUp(
        total_fused_down,
        total_fused_up,
    )

    def cutlass_separate_total() -> None:
        output = lora_gmm(
            total_separate_a,
            total_separate_down,
            total_separate_up,
            sizes,
        )
        torch.autograd.grad(
            output,
            (
                total_separate_a,
                total_separate_down,
                total_separate_up,
            ),
            grad_output,
        )

    def cutlass_fused_total() -> None:
        output = total_fused_operation(total_fused_a, sizes)
        torch.autograd.grad(
            output,
            (total_fused_a, total_fused_down, total_fused_up),
            grad_output,
        )

    total_samples = benchmark_pair(
        cutlass_separate_total,
        cutlass_fused_total,
    )

    stages = (
        ("forward", 2, 1, *forward_samples),
        ("backward", 4, 3, *backward_samples),
        ("forward+backward", 6, 4, *total_samples),
    )
    LOGGER.info("")
    LOGGER.info(
        (
            "%-18s | %-10s | %7s | %12s | "
            "%23s | %8s | %10s"
        ),
        "stage",
        "CUTLASS",
        "kernels",
        "median(us)",
        "sample min-max(us)",
        "CV",
        "speedup",
    )
    LOGGER.info("-" * 108)
    for (
        stage,
        separate_kernels,
        fused_kernels,
        separate_stage_samples,
        fused_stage_samples,
    ) in stages:
        separate_summary = benchmark_summary(separate_stage_samples)
        fused_summary = benchmark_summary(fused_stage_samples)
        speedup = separate_summary[0] / fused_summary[0]
        for name, kernels, summary in (
            ("separate", separate_kernels, separate_summary),
            ("fused", fused_kernels, fused_summary),
        ):
            median_us, minimum_us, maximum_us, variation = summary
            LOGGER.info(
                (
                    "%-18s | %-10s | %7d | %12.3f | "
                    "%10.3f-%10.3f | %7.3f%% | %9s"
                ),
                stage,
                name,
                kernels,
                median_us,
                minimum_us,
                maximum_us,
                variation,
                "-" if name == "separate" else f"{speedup:.3f}x",
            )
    LOGGER.info(
        (
            "说明：kernels 仅统计 GEMM/fusion 计算 kernel；"
            "计时不包含 fused 权重预打包，包含各入口自身的元数据处理。"
        )
    )


class _PreprocessedTritonLoraFunction(torch.autograd.Function):
    """使用预转置 down 权重的 Triton LoRA Autograd 入口。"""

    @staticmethod
    def forward(
        context,
        a: torch.Tensor,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
        batch_sizes: torch.Tensor,
        operation,
    ) -> torch.Tensor:
        hidden, output = operation.forward_operation(a, batch_sizes)
        context.save_for_backward(a, hidden, batch_sizes)
        context.operation = operation
        return output

    @staticmethod
    def backward(
        context,
        grad_output: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
    ]:
        a, hidden, batch_sizes = context.saved_tensors
        grad_output = grad_output.contiguous()
        operation = context.operation
        grad_hidden, grad_input = operation.backward_input_operation(
            grad_output,
            batch_sizes,
        )
        grad_down_weight = operation.bgrad_operation(
            grad_hidden,
            a,
            batch_sizes,
        )
        grad_up_weight = operation.bgrad_operation(
            hidden,
            grad_output,
            batch_sizes,
        )
        return (
            grad_input,
            grad_down_weight,
            grad_up_weight,
            None,
            None,
        )


class PreprocessedTritonLora:
    """在计时前完成 down 权重转置，并按配置处理 metadata 缓存。"""

    def __init__(
        self,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
    ) -> None:
        self.down_weight = down_weight
        self.up_weight = up_weight
        detached_down_weight = down_weight.detach()
        detached_up_weight = up_weight.detach()
        self.forward_operation = LoraFusedDownUpGrouped(
            detached_down_weight,
            detached_up_weight,
        )
        self.backward_input_operation = LoraFusedAgradGrouped(
            detached_up_weight,
            detached_down_weight,
        )
        self.bgrad_operation = LoraBgradGrouped(
            down_weight.shape[0],
            down_weight.shape[2],
        )

    def __call__(
        self,
        a: torch.Tensor,
        batch_sizes: torch.Tensor,
    ) -> torch.Tensor:
        if CLEAR_METADATA_CACHE:
            self.forward_operation.clear_metadata_cache()
        return _PreprocessedTritonLoraFunction.apply(
            a,
            self.down_weight,
            self.up_weight,
            batch_sizes,
            self,
        )


def run_cutlass_triton_fusion_performance() -> None:
    """对比 CUTLASS 非融合与预处理后的 Triton 融合训练入口。"""
    torch.manual_seed(23)
    sizes = torch.tensor(SIZES)
    tokens = int(sizes.sum())
    experts = sizes.numel()
    hidden_size = FUSION_HIDDEN_SIZE
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
    grad_output = torch.randn_like(a)

    accuracy_cutlass_a = a.detach().requires_grad_(True)
    accuracy_cutlass_down = down_weight.detach().requires_grad_(True)
    accuracy_cutlass_up = up_weight.detach().requires_grad_(True)
    accuracy_triton_a = a.detach().requires_grad_(True)
    accuracy_triton_down = down_weight.detach().requires_grad_(True)
    accuracy_triton_up = up_weight.detach().requires_grad_(True)
    accuracy_triton_operation = PreprocessedTritonLora(
        accuracy_triton_down,
        accuracy_triton_up,
    )
    accuracy_cutlass_output = lora_gmm(
        accuracy_cutlass_a,
        accuracy_cutlass_down,
        accuracy_cutlass_up,
        sizes,
    )
    accuracy_triton_output = accuracy_triton_operation(
        accuracy_triton_a,
        sizes,
    )
    torch.testing.assert_close(
        accuracy_triton_output,
        accuracy_cutlass_output,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    )
    accuracy_cutlass_gradients = torch.autograd.grad(
        accuracy_cutlass_output,
        (
            accuracy_cutlass_a,
            accuracy_cutlass_down,
            accuracy_cutlass_up,
        ),
        grad_output,
    )
    accuracy_triton_gradients = torch.autograd.grad(
        accuracy_triton_output,
        (
            accuracy_triton_a,
            accuracy_triton_down,
            accuracy_triton_up,
        ),
        grad_output,
    )
    for triton_gradient, cutlass_gradient in zip(
        accuracy_triton_gradients,
        accuracy_cutlass_gradients,
    ):
        torch.testing.assert_close(
            triton_gradient,
            cutlass_gradient,
            rtol=BF16_RTOL,
            atol=BF16_GRAD_ATOL,
        )
    LOGGER.info(
        (
            "正确性：tokens=%d experts=%d hidden=%d rank=16 "
            "output_diff=%.6f [PASS]"
        ),
        tokens,
        experts,
        hidden_size,
        max_error(accuracy_triton_output, accuracy_cutlass_output),
    )
    LOGGER.info(
        (
            "反向正确性：grad_input_diff=%.6f "
            "grad_down_diff=%.6f grad_up_diff=%.6f [PASS]"
        ),
        *(
            max_error(triton_gradient, cutlass_gradient)
            for triton_gradient, cutlass_gradient in zip(
                accuracy_triton_gradients,
                accuracy_cutlass_gradients,
            )
        ),
    )
    LOGGER.info(
        (
            "反向相对 L2：grad_input=%.6e "
            "grad_down=%.6e grad_up=%.6e"
        ),
        *(
            relative_l2_error(triton_gradient, cutlass_gradient)
            for triton_gradient, cutlass_gradient in zip(
                accuracy_triton_gradients,
                accuracy_cutlass_gradients,
            )
        ),
    )

    def cutlass_separate_forward() -> None:
        lora_gmm(a, down_weight, up_weight, sizes)

    triton_forward_operation = PreprocessedTritonLora(
        down_weight,
        up_weight,
    )

    def triton_fused_forward() -> None:
        triton_forward_operation(a, sizes)

    forward_samples = benchmark_pair(
        cutlass_separate_forward,
        triton_fused_forward,
    )

    backward_cutlass_a = a.detach().requires_grad_(True)
    backward_cutlass_down = down_weight.detach().requires_grad_(True)
    backward_cutlass_up = up_weight.detach().requires_grad_(True)
    backward_triton_a = a.detach().requires_grad_(True)
    backward_triton_down = down_weight.detach().requires_grad_(True)
    backward_triton_up = up_weight.detach().requires_grad_(True)
    backward_triton_operation = PreprocessedTritonLora(
        backward_triton_down,
        backward_triton_up,
    )
    backward_cutlass_output = lora_gmm(
        backward_cutlass_a,
        backward_cutlass_down,
        backward_cutlass_up,
        sizes,
    )
    backward_triton_output = backward_triton_operation(
        backward_triton_a,
        sizes,
    )

    def cutlass_separate_backward() -> None:
        torch.autograd.grad(
            backward_cutlass_output,
            (
                backward_cutlass_a,
                backward_cutlass_down,
                backward_cutlass_up,
            ),
            grad_output,
            retain_graph=True,
        )

    def triton_fused_backward() -> None:
        torch.autograd.grad(
            backward_triton_output,
            (
                backward_triton_a,
                backward_triton_down,
                backward_triton_up,
            ),
            grad_output,
            retain_graph=True,
        )

    backward_samples = benchmark_pair(
        cutlass_separate_backward,
        triton_fused_backward,
    )

    total_cutlass_a = a.detach().requires_grad_(True)
    total_cutlass_down = down_weight.detach().requires_grad_(True)
    total_cutlass_up = up_weight.detach().requires_grad_(True)
    total_triton_a = a.detach().requires_grad_(True)
    total_triton_down = down_weight.detach().requires_grad_(True)
    total_triton_up = up_weight.detach().requires_grad_(True)
    total_triton_operation = PreprocessedTritonLora(
        total_triton_down,
        total_triton_up,
    )

    def cutlass_separate_total() -> None:
        output = lora_gmm(
            total_cutlass_a,
            total_cutlass_down,
            total_cutlass_up,
            sizes,
        )
        torch.autograd.grad(
            output,
            (total_cutlass_a, total_cutlass_down, total_cutlass_up),
            grad_output,
        )

    def triton_fused_total() -> None:
        output = total_triton_operation(
            total_triton_a,
            sizes,
        )
        torch.autograd.grad(
            output,
            (total_triton_a, total_triton_down, total_triton_up),
            grad_output,
        )

    total_samples = benchmark_pair(
        cutlass_separate_total,
        triton_fused_total,
    )

    stages = (
        ("forward", 2, 1, *forward_samples),
        ("backward", 4, 3, *backward_samples),
        ("forward+backward", 6, 4, *total_samples),
    )
    LOGGER.info("")
    LOGGER.info(
        (
            "%-18s | %-18s | %7s | %12s | "
            "%23s | %8s | %10s"
        ),
        "stage",
        "implementation",
        "kernels",
        "median(us)",
        "sample min-max(us)",
        "CV",
        "speedup",
    )
    LOGGER.info("-" * 116)
    for (
        stage,
        cutlass_kernels,
        triton_kernels,
        cutlass_samples,
        triton_samples,
    ) in stages:
        cutlass_summary = benchmark_summary(cutlass_samples)
        triton_summary = benchmark_summary(triton_samples)
        speedup = cutlass_summary[0] / triton_summary[0]
        for name, kernels, summary in (
            ("CUTLASS separate", cutlass_kernels, cutlass_summary),
            ("Triton fused", triton_kernels, triton_summary),
        ):
            median_us, minimum_us, maximum_us, variation = summary
            LOGGER.info(
                (
                    "%-18s | %-18s | %7d | %12.3f | "
                    "%10.3f-%10.3f | %7.3f%% | %9s"
                ),
                stage,
                name,
                kernels,
                median_us,
                minimum_us,
                maximum_us,
                variation,
                "-" if name == "CUTLASS separate" else f"{speedup:.3f}x",
            )
    LOGGER.info(
        "说明：Triton down 权重转置位于计时外；metadata=%s；"
        "计时包含 Autograd、Tensor 分配和 kernel launch。",
        (
            "每次前向重建"
            if CLEAR_METADATA_CACHE
            else "固定输入信息，warmup 后复用"
        ),
    )


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
    if FUSION_COMPARISON not in ("cutlass", "triton", "all"):
        raise ValueError(
            "CUDAOP_GROUPED_GEMM_FUSION_COMPARISON "
            "必须是 cutlass、triton 或 all"
        )
    if CLEAR_METADATA_CACHE_VALUE not in ("0", "1"):
        raise ValueError(
            "CUDAOP_GROUPED_GEMM_CLEAR_METADATA_CACHE 必须是 0 或 1"
        )

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
        LOGGER.info("")
        LOGGER.info("阶段：LoRA fusion 可移植精度验证")
        run_portable_accuracy()
        if FUSION_COMPARISON in ("cutlass", "all"):
            LOGGER.info("")
            LOGGER.info(
                (
                    "阶段：CUTLASS fusion/非 fusion 性能对比，"
                    "hidden_size=%d warmup=%d iterations=%d samples=%d"
                ),
                FUSION_HIDDEN_SIZE,
                FUSION_WARMUP_ITERATIONS,
                FUSION_BENCHMARK_ITERATIONS,
                BENCHMARK_SAMPLES,
            )
            run_cutlass_fusion_performance()
        if FUSION_COMPARISON in ("triton", "all"):
            LOGGER.info("")
            LOGGER.info(
                (
                    "阶段：CUTLASS 非 fusion/Triton fusion 性能对比，"
                    "hidden_size=%d metadata=%s warmup=%d "
                    "iterations=%d samples=%d"
                ),
                FUSION_HIDDEN_SIZE,
                "rebuild" if CLEAR_METADATA_CACHE else "cached",
                FUSION_WARMUP_ITERATIONS,
                FUSION_BENCHMARK_ITERATIONS,
                BENCHMARK_SAMPLES,
            )
            run_cutlass_triton_fusion_performance()
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
