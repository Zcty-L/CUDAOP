"""Minimal surrogate configuration used by the standalone neuron kernels."""


class Sigmoid:
    """Sigmoid surrogate parameters compatible with the imported CuPy code."""

    def __init__(self, alpha: float = 4.0) -> None:
        self.alpha = alpha


class ATan:
    """Arctangent surrogate parameters compatible with the imported CuPy code."""

    def __init__(self, alpha: float = 2.0) -> None:
        self.alpha = alpha
