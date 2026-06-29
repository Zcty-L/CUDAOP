"""本地 CUTLASS Grouped GEMM 的前向与反向精度测试。"""

import torch

import gmm_ops

from debug_utils import (
    LOGGER,
    log_error,
    log_section,
    log_test_start,
    log_test_success,
)


def grouped_reference(
    a: torch.Tensor,
    b: torch.Tensor,
    batch_sizes: torch.Tensor,
) -> torch.Tensor:
    outputs = []
    offset = 0
    for expert, size in enumerate(batch_sizes.tolist()):
        outputs.append(
            a[offset:offset + size] @ b[expert].transpose(0, 1)
        )
        offset += size
    return torch.cat(outputs, dim=0)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUTLASS Grouped GEMM 测试需要 CUDA GPU")

    torch.manual_seed(11)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_sizes = torch.tensor(
        [4, 0, 3, 1, 5, 2, 0, 4],
        dtype=torch.long,
    )
    total_tokens = int(batch_sizes.sum())
    hidden_in = 256
    hidden_out = 32
    log_test_start(
        "CUTLASS Grouped GEMM",
        f"device={torch.cuda.get_device_name()}，dtype={dtype}，"
        f"experts={batch_sizes.numel()}，tokens={total_tokens}，"
        f"K={hidden_in}，N={hidden_out}",
    )

    a = torch.randn(
        total_tokens,
        hidden_in,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    b = torch.randn(
        batch_sizes.numel(),
        hidden_out,
        hidden_in,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    reference_a = a.detach().clone().requires_grad_(True)
    reference_b = b.detach().clone().requires_grad_(True)

    log_section("运行 Grouped GEMM 前向精度测试")
    actual = gmm_ops.gmm(
        a,
        b,
        batch_sizes.to(device),
        trans_b=True,
    )
    expected = grouped_reference(
        reference_a,
        reference_b,
        batch_sizes,
    )
    torch.testing.assert_close(
        actual,
        expected,
        rtol=2e-2,
        atol=2e-2,
    )
    log_error(
        "grouped_gemm",
        "前向输出",
        (actual.float() - expected.float()).abs().max().item(),
    )
    LOGGER.debug("[DEBUG] Grouped GEMM 前向精度通过")

    log_section("运行 Grouped GEMM 输入及权重梯度测试")
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(
        a.grad,
        reference_a.grad,
        rtol=2e-2,
        atol=2e-2,
    )
    torch.testing.assert_close(
        b.grad,
        reference_b.grad,
        rtol=2e-2,
        atol=2e-2,
    )
    log_error(
        "grouped_gemm",
        "输入梯度",
        (
            a.grad.float() - reference_a.grad.float()
        ).abs().max().item(),
    )
    log_error(
        "grouped_gemm",
        "权重梯度",
        (
            b.grad.float() - reference_b.grad.float()
        ).abs().max().item(),
    )
    LOGGER.debug("[DEBUG] Grouped GEMM 反向传播精度通过")
    log_test_success("CUTLASS Grouped GEMM")


if __name__ == "__main__":
    main()
