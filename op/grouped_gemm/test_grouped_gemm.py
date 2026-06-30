"""CUTLASS Grouped GEMM 的前向、反向精度测试。"""

import logging

import torch

from cudaop_grouped_gemm import gmm


LOGGER = logging.getLogger("cudaop_grouped_gemm_test")

BF16_RTOL = 2e-2
BF16_FORWARD_ATOL = 2e-2
BF16_GRAD_ATOL = 4e-2


def reference(
    a: torch.Tensor,
    b: torch.Tensor,
    batch_sizes: torch.Tensor,
    trans_b: bool,
) -> torch.Tensor:
    outputs = []
    offset = 0
    for expert, size in enumerate(batch_sizes.tolist()):
        weight = b[expert].transpose(0, 1) if trans_b else b[expert]
        outputs.append(a[offset:offset + size] @ weight)
        offset += size
    return torch.cat(outputs, dim=0)


def run_case(trans_b: bool) -> None:
    sizes = torch.tensor([128, 157, 97, 100, 111, 129, 138, 101])
    tokens = int(sizes.sum())
    experts = sizes.numel()
    hidden_in = 256
    hidden_out = 32
    weight_shape = (
        (experts, hidden_out, hidden_in)
        if trans_b
        else (experts, hidden_in, hidden_out)
    )

    a = torch.randn(
        tokens,
        hidden_in,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    b = torch.randn(
        weight_shape,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    # 使用 FP32 reference，避免 PyTorch BF16 GEMM 的归约策略影响基准。
    reference_a = a.detach().float().requires_grad_(True)
    reference_b = b.detach().float().requires_grad_(True)

    actual = gmm(a, b, sizes.cuda(), trans_b=trans_b)
    expected = reference(
        reference_a,
        reference_b,
        sizes,
        trans_b,
    )
    torch.testing.assert_close(
        actual.float(),
        expected,
        rtol=BF16_RTOL,
        atol=BF16_FORWARD_ATOL,
    )

    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(
        a.grad.float(),
        reference_a.grad,
        rtol=BF16_RTOL,
        atol=BF16_GRAD_ATOL,
    )
    torch.testing.assert_close(
        b.grad.float(),
        reference_b.grad,
        rtol=BF16_RTOL,
        atol=BF16_GRAD_ATOL,
    )
    output_error = (actual.float() - expected).abs().max().item()
    input_grad_error = (
        a.grad.float() - reference_a.grad
    ).abs().max().item()
    weight_grad_error = (
        b.grad.float() - reference_b.grad
    ).abs().max().item()
    LOGGER.info(
        (
            "trans_b=%-5s output_shape=%-16s "
            "output_error=%.6f input_grad_error=%.6f "
            "weight_grad_error=%.6f"
        ),
        trans_b,
        str(tuple(actual.shape)),
        output_error,
        input_grad_error,
        weight_grad_error,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not torch.cuda.is_available():
        raise RuntimeError("测试需要 CUDA GPU")

    LOGGER.info(
        "配置：device=%s dtype=bfloat16 arch=sm_%d%d",
        torch.cuda.get_device_name(),
        *torch.cuda.get_device_capability(),
    )
    LOGGER.info("")
    LOGGER.info("阶段：Grouped GEMM 前向与反向精度验证")
    torch.manual_seed(11)
    run_case(trans_b=False)
    run_case(trans_b=True)
    LOGGER.info("")
    LOGGER.info("[SUCCESS] cudaop_grouped_gemm 精度测试通过")


if __name__ == "__main__":
    main()
