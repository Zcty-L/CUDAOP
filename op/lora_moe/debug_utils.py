"""LoRA-MoE 独立测试脚本共用的日志辅助函数。"""

import logging


logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
)
LOGGER = logging.getLogger("lora_moe_test")


def log_test_start(name: str, configuration: str) -> None:
    LOGGER.info("")
    LOGGER.info("=" * 72)
    LOGGER.info("[TEST] %s", name)
    LOGGER.info("[DEBUG] %s", configuration)


def log_section(message: str) -> None:
    LOGGER.info("")
    LOGGER.debug("[DEBUG] %s", message)


def log_error(
    method: str,
    value_name: str,
    max_absolute_error: float,
) -> None:
    LOGGER.debug(
        "[DEBUG] %s 的%s最大绝对误差：%.6f",
        method,
        value_name,
        max_absolute_error,
    )


def log_test_success(name: str) -> None:
    LOGGER.info("")
    LOGGER.info("[SUCCESS] %s 全部测试运行成功", name)
    LOGGER.info("=" * 72)
    LOGGER.info("")
