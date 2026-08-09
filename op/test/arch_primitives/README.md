# CUDA 架构原语验证

本目录用于独立验证可复用的 CUDA 架构原语，不放置完整算子实现。

当前包含以下微基准：

- `tma_copy.cu`：验证 1D TMA GMEM 到 SMEM 数据搬运、Tensor Map、
  mbarrier、设备侧完成周期和有效载荷吞吐。
- `tma_swizzle.cu`：通过 TMA load/store 闭环验证 None、32B、64B 和
  128B shared-memory swizzle/unswizzle。
- `cluster_launch.cu`：验证显式 thread block cluster 启动、cluster 特殊
  寄存器和 cluster barrier。
- `cluster_dsm.cu`：验证 `mapa.shared::cluster` 远端地址映射、distributed
  shared memory 读取和同步。

完整 GEMM 及其架构特性组合实验放在 `op/linear/`。所有 PTX wrapper
统一维护在 `op/ptx_utils.cuh`，并在注释中注明指令名称、来源和用途。

每个微基准应作为独立目标注册到项目根目录的 `CMakeLists.txt`，测试输出应包含：

- 当前 GPU 和测试配置。
- 主要执行阶段。
- 正确性及关键性能结果。
- `[SUCCESS]` 成功标记。

当前 RTX 5070 Ti Laptop（SM120）可运行 TMA、TMA swizzle、thread block
cluster 和 DSM 实验。TMEM/`tcgen05` 实验需要 SM100/SM101 环境，不在
本机运行目标范围内。

构建并运行全部架构原语测试：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build \
    --target tma_copy tma_swizzle cluster_launch cluster_dsm
ctest --test-dir build \
    -R '^(tma_copy|tma_swizzle|cluster_launch|cluster_dsm)$' \
    --output-on-failure
```

`tma_swizzle` 验证 Tensor Map swizzle/unswizzle 的硬件闭环正确性。shared
memory bank conflict 的性能差异需要后续使用具有相同访问模式的 kernel
和 Nsight Compute 单独测量，不能从这个闭环测试的 kernel latency 推断。
