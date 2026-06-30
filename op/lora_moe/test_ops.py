"""Minimal runtime checks for ``lora_moe_ops``."""

import torch

import lora_moe_ops

from debug_utils import (
    LOGGER,
    log_section,
    log_test_start,
    log_test_success,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required to run lora_moe_ops")

    log_test_start(
        "lora_moe_ops",
        f"device={torch.cuda.get_device_name()}",
    )

    log_section("验证 histogram")
    values = torch.tensor([2, 0, 1, 2, 2], device="cuda", dtype=torch.int32)
    counts = lora_moe_ops.histogram(values, 3)
    expected_counts = torch.tensor([1, 1, 3], device="cuda", dtype=torch.int32)
    torch.testing.assert_close(counts, expected_counts)
    LOGGER.debug("[DEBUG] histogram 精度通过")

    log_section("验证 inclusive_cumsum 和 exclusive_cumsum")
    matrix = torch.tensor(
        [[1, 2, 3], [4, 5, 6]],
        device="cuda",
        dtype=torch.int32,
    )
    inclusive = torch.empty_like(matrix)
    exclusive = torch.empty_like(matrix)
    lora_moe_ops.inclusive_cumsum(matrix, 1, inclusive)
    lora_moe_ops.exclusive_cumsum(matrix, 1, exclusive)
    torch.testing.assert_close(
        inclusive,
        torch.tensor(
            [[1, 3, 6], [4, 9, 15]],
            device="cuda",
            dtype=torch.int32,
        ),
    )
    torch.testing.assert_close(
        exclusive,
        torch.tensor(
            [[0, 1, 3], [0, 4, 9]],
            device="cuda",
            dtype=torch.int32,
        ),
    )
    LOGGER.debug("[DEBUG] cumsum 精度通过")

    log_section("验证 sort")
    keys = torch.tensor(
        [3, 1, 2, 0],
        device="cuda",
        dtype=torch.int32,
    )
    sorted_keys = torch.empty_like(keys)
    sorted_indices = torch.empty_like(keys)
    lora_moe_ops.sort(keys, 2, sorted_keys, sorted_indices)
    torch.testing.assert_close(
        sorted_keys,
        torch.tensor(
            [0, 1, 2, 3],
            device="cuda",
            dtype=torch.int32,
        ),
    )
    torch.testing.assert_close(
        sorted_indices,
        torch.tensor(
            [3, 1, 2, 0],
            device="cuda",
            dtype=torch.int32,
        ),
    )
    LOGGER.debug("[DEBUG] sort 精度通过")

    log_section("验证 replicate_forward 和 replicate_backward")
    bins = torch.tensor(
        [2, 3, 6],
        device="cuda",
        dtype=torch.int32,
    )
    replicate_input = torch.tensor(
        [[10, 20, 30]],
        device="cuda",
        dtype=torch.int32,
    )
    replicated = torch.empty(
        (1, 6),
        device="cuda",
        dtype=torch.int32,
    )
    lora_moe_ops.replicate_forward(
        replicate_input,
        bins,
        replicated,
    )
    torch.testing.assert_close(
        replicated,
        torch.tensor(
            [[10, 10, 20, 30, 30, 30]],
            device="cuda",
            dtype=torch.int32,
        ),
    )

    gradient = torch.ones(
        (1, 6),
        device="cuda",
        dtype=torch.float16,
    )
    reduced = torch.empty(
        (1, 3),
        device="cuda",
        dtype=torch.float16,
    )
    lora_moe_ops.replicate_backward(gradient, bins, reduced)
    torch.testing.assert_close(
        reduced,
        torch.tensor(
            [[2, 1, 3]],
            device="cuda",
            dtype=torch.float16,
        ),
    )
    LOGGER.debug("[DEBUG] replicate 前向及反向精度通过")

    log_section("验证 indices")
    padded_bins = torch.tensor(
        [4, 8],
        device="cuda",
        dtype=torch.int32,
    )
    sparse_indices = torch.empty(
        6,
        device="cuda",
        dtype=torch.int16,
    )
    lora_moe_ops.indices(
        padded_bins,
        4,
        2,
        3,
        sparse_indices,
    )
    torch.testing.assert_close(
        sparse_indices,
        torch.arange(
            6,
            device="cuda",
            dtype=torch.int16,
        ),
    )
    LOGGER.debug("[DEBUG] indices 精度通过")
    log_test_success("lora_moe_ops")


if __name__ == "__main__":
    main()
