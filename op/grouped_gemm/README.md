# cudaop_grouped_gemm

独立的 CUTLASS BF16 Grouped GEMM Python 包。当前实现由
`op/lora_moe` 中的 SM120 构建版本拆分而来，支持前向和自动求导。

## 构建与测试

```bash
conda activate py311
cmake -S . -B build
cmake --build build --target cudaop_grouped_gemm
cmake --build build --target cudaop_grouped_gemm_test
```

也可以在当前目录原地构建：

```bash
python build.py
python test_grouped_gemm.py
```

## Python 调用

```python
import torch
from cudaop_grouped_gemm import gmm

sizes = torch.tensor([2, 3], device="cuda")
a = torch.randn(5, 256, device="cuda", dtype=torch.bfloat16)
b = torch.randn(2, 32, 256, device="cuda", dtype=torch.bfloat16)
output = gmm(a, b, sizes, trans_b=True)
```
