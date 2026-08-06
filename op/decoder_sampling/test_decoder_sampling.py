"""decoder_sampling 的 CPU/GPU 无关正确性测试。"""

from __future__ import annotations

import logging
import unittest

import torch

from op.decoder_sampling import (
    SamplingConfig,
    prepare_sampling_logits,
    sample_next_token,
)


LOGGER = logging.getLogger(__name__)


class DecoderSamplingTest(unittest.TestCase):
    """验证惩罚、过滤、随机复现和 batch/padding 行为。"""

    def test_greedy_uses_penalized_logits(self) -> None:
        """重复惩罚应能改变 greedy token。"""
        logits = torch.tensor([3.0, 2.0, 1.0])
        history = torch.tensor([0])
        config = SamplingConfig(do_sample=False, repetition_penalty=2.0)

        result = sample_next_token(logits, history, config)

        self.assertEqual(result.token_ids.item(), 1)
        self.assertEqual(result.token_ids.shape, (1,))

    def test_frequency_and_presence_penalties(self) -> None:
        """频率惩罚按次数计算，存在惩罚只计算一次。"""
        logits = torch.tensor([4.0, 3.0, 2.0])
        history = torch.tensor([0, 0, 1])
        config = SamplingConfig(
            do_sample=False,
            frequency_penalty=0.5,
            presence_penalty=0.25,
        )

        processed = prepare_sampling_logits(logits, history, config)

        expected = torch.tensor([[2.75, 2.25, 2.0]])
        torch.testing.assert_close(processed, expected)

    def test_top_k_keeps_only_k_candidates(self) -> None:
        """top-k 过滤后只能留下最高的 k 个 logits。"""
        logits = torch.tensor([4.0, 3.0, 2.0, 1.0])
        config = SamplingConfig(temperature=1.0, top_k=2, top_p=1.0)

        processed = prepare_sampling_logits(logits, config=config)

        self.assertEqual(torch.isfinite(processed).sum().item(), 2)
        self.assertTrue(torch.isfinite(processed[0, :2]).all())

    def test_top_p_keeps_boundary_token(self) -> None:
        """累计概率首次越过 top-p 的边界 token 必须被保留。"""
        probabilities = torch.tensor([0.50, 0.30, 0.15, 0.05])
        logits = torch.log(probabilities)
        config = SamplingConfig(temperature=1.0, top_k=0, top_p=0.70)

        processed = prepare_sampling_logits(logits, config=config)

        expected_mask = torch.tensor([[True, True, False, False]])
        torch.testing.assert_close(torch.isfinite(processed), expected_mask)

    def test_min_p_is_relative_to_most_likely_token(self) -> None:
        """min-p 阈值应为最大概率与 min_p 的乘积。"""
        probabilities = torch.tensor([0.60, 0.30, 0.08, 0.02])
        logits = torch.log(probabilities)
        config = SamplingConfig(
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            min_p=0.40,
        )

        processed = prepare_sampling_logits(logits, config=config)

        expected_mask = torch.tensor([[True, True, False, False]])
        torch.testing.assert_close(torch.isfinite(processed), expected_mask)

    def test_fixed_generator_reproduces_sample(self) -> None:
        """相同 seed 和输入必须采出相同 token。"""
        logits = torch.tensor([1.0, 1.0, 1.0, 1.0])
        config = SamplingConfig(temperature=1.0, top_k=0, top_p=1.0)
        first_generator = torch.Generator().manual_seed(2026)
        second_generator = torch.Generator().manual_seed(2026)

        first = sample_next_token(
            logits,
            config=config,
            generator=first_generator,
        )
        second = sample_next_token(
            logits,
            config=config,
            generator=second_generator,
        )

        torch.testing.assert_close(first.token_ids, second.token_ids)

    def test_padding_mask_excludes_padding_from_penalty(self) -> None:
        """batch 的 padding token 不应被误计入历史惩罚。"""
        logits = torch.tensor([[3.0, 2.0], [3.0, 2.0]])
        history = torch.tensor([[0, 0], [0, 1]])
        mask = torch.tensor([[True, False], [True, True]])
        config = SamplingConfig(do_sample=False, frequency_penalty=1.0)

        processed = prepare_sampling_logits(
            logits,
            history,
            config,
            previous_token_mask=mask,
        )

        expected = torch.tensor([[2.0, 2.0], [2.0, 1.0]])
        torch.testing.assert_close(processed, expected)

    def test_probabilities_are_normalized(self) -> None:
        """过滤后的最终概率分布每行之和必须为 1。"""
        logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        config = SamplingConfig(temperature=0.7, top_k=2, top_p=0.9)
        generator = torch.Generator().manual_seed(7)

        result = sample_next_token(logits, config=config, generator=generator)

        torch.testing.assert_close(
            result.probabilities.sum(dim=-1),
            torch.ones(2),
        )
        self.assertEqual(result.token_ids.shape, (2,))

    def test_invalid_configuration_is_rejected(self) -> None:
        """非法配置应在执行前报错。"""
        with self.assertRaises(ValueError):
            SamplingConfig(temperature=-0.1)
        with self.assertRaises(ValueError):
            SamplingConfig(top_p=0.0)
        with self.assertRaises(ValueError):
            SamplingConfig(repetition_penalty=float("nan"))
        with self.assertRaises(ValueError):
            SamplingConfig(frequency_penalty=float("inf"))
        with self.assertRaises(TypeError):
            SamplingConfig(top_k=1.5)  # type: ignore[arg-type]


def main() -> None:
    """运行测试并按项目约定输出阶段与成功标记。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    LOGGER.info("配置: PyTorch=%s, device=cpu", torch.__version__)
    LOGGER.info("")
    LOGGER.info("阶段 1/2: 验证 logits 处理与候选过滤")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DecoderSamplingTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)

    LOGGER.info("")
    LOGGER.info("阶段 2/2: 验证采样概率、batch 与随机复现")
    LOGGER.info("关键结果: %d 个测试全部通过", result.testsRun)
    LOGGER.info("")
    LOGGER.info("[SUCCESS] decoder_sampling 精度验证通过")


if __name__ == "__main__":
    main()
