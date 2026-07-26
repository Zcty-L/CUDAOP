# 两卡 NCCL 张量并行验证

`nccl_mlp_test` 验证以下计算：

```text
Y = GeLU(XA)B
```

其中，`X` 在两张 GPU 上复制，`A` 沿输出维按列切分，`B` 沿输入维按行切分。每张 GPU 依次使用 cuBLAS SGEMM 计算局部的 `XA` 和 `GeLU(XA)B`，最后使用 NCCL AllReduce(SUM) 得到完整且复制到两卡的 `Y`。

## 构建与运行

```bash
cmake -S . -B build
cmake --build build --target nccl_mlp_test -j
./build/nccl_mlp_test
```

也可以通过 CTest 运行：

```bash
ctest --test-dir build -R nccl_mlp_test --output-on-failure
```

测试会输出固定配置、执行阶段、相对单卡完整 cuBLAS 计算的最大误差、两卡输出副本差异、平均端到端耗时和 `[SUCCESS]` 标记。
