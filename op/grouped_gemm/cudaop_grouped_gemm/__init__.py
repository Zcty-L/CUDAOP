"""CUDAOP Grouped GEMM 的 Python 接口。"""

from cudaop_grouped_gemm.ops import gmm, torch_gmm

__all__ = ["gmm", "torch_gmm"]
