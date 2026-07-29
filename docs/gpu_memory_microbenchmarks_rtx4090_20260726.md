# RTX 4090 GPU 存储层级微基准测试报告

## 1. 测试概述

本文档汇总 CUDAOP 中以下 8 项 GPU 存储层级微基准：

| 存储层级 | 延迟测试 | 带宽测试 |
|---|---|---|
| Shared Memory | `smem_latency` | `smem_bandwidth` |
| L1/TEX Cache | `l1cache_latency` | `l1cache_bandwidth` |
| L2 Cache | `l2cache_latency` | `l2cache_bandwidth` |
| DRAM | `dram_latency` | `dram_bandwidth` |

延迟测试使用严格依赖的指针追逐，主动消除访存并行性；带宽测试则使用大量独立、合并的访存请求，使对应存储层级尽可能饱和。

本文数据来自 2026-07-26 的一次串行测试。GPU 频率、温度、功耗状态以及其他 GPU 负载都可能使结果发生波动。

## 2. 测试环境

| 项目 | 配置 |
|---|---|
| 测试设备 | GPU 0 |
| GPU | NVIDIA GeForce RTX 4090 |
| Compute Capability | 8.9 |
| SM 数量 | 128 |
| 显存容量 | 24564 MiB |
| L2 容量 | 75497472 B，即 72 MiB |
| CUDA Runtime 报告的 SM 时钟 | 2520 MHz |
| NVIDIA 驱动 | 590.44.01 |
| CUDA Toolkit | 13.1 |
| NVCC | 13.1.80 |
| 构建类型 | Release |
| CUDA 架构 | SM 8.9 |

机器中安装了两张相同的 RTX 4090，本报告中的测试程序使用当前 CUDA 设备 GPU 0。所有测试均串行执行，避免不同微基准之间争抢 GPU 资源。

## 3. 结果总览

### 3.1 延迟

| 存储层级 | 延迟 | 相对 L1 | 测试状态 |
|---|---:|---:|---|
| Shared Memory | 23.92 cycles/load | 0.70× | 通过 |
| L1/TEX Cache | 34.00 cycles/load | 1.00× | 通过 |
| L2 Cache | 273.70 cycles/load | 8.05× | 通过 |
| DRAM | 624.10 cycles/load | 18.36× | 通过 |

### 3.2 带宽

| 存储层级 | 主要结果 | 结果口径 | 测试状态 |
|---|---:|---|---|
| Shared Memory | 108.86 B/cycle/SM | 实测共享存储吞吐 | 通过 |
| Shared Memory | 128.00 B/cycle/SM | 按 32 B/cycle 粒度推断 | 通过 |
| Shared Memory | 35113.88 GB/s | 按实测每 SM 吞吐换算的整卡参考值 | 参考 |
| Shared Memory | 41287.68 GB/s | 按推断每 SM 峰值换算的整卡参考值 | 参考 |
| L1/TEX Cache | 42517.02 GB/s | 整卡聚合请求字节带宽 | 通过 |
| L2 Cache | 5134.35 GB/s | 整卡聚合读取带宽 | 通过 |
| DRAM Read | 929.92 GB/s | 1 GiB 工作集 | 通过 |
| DRAM Write | 959.08 GB/s | 1 GiB 工作集 | 通过 |
| DRAM Copy | 920.35 GB/s | 读取与写入总流量 | 通过 |

Shared Memory 整卡参考值按 128 个 SM 和 2.52 GHz 报告时钟换算：

```text
108.86 B/cycle/SM × 128 SM × 2.52 GHz
= 35113.88 GB/s

128.00 B/cycle/SM × 128 SM × 2.52 GHz
= 41287.68 GB/s
```

这两个整卡数值不是直接测量结果，而是假设全部 SM 同时维持相同吞吐和报告时钟得到的参考值。带宽结果的统计口径并不完全相同：L1 和 L2 使用 kernel 请求的字节数计算整卡聚合带宽，DRAM 使用大工作集的实际传输时间计算。跨层级比较时应同时考虑这些差异。

## 4. 测试方法与结果

### 4.1 Shared Memory 延迟

测试文件：`op/test/smem_latency.cu`

测试方法：

1. 启动一个包含 16 个线程的 block。
2. 每个线程在共享内存中构造指向自身地址的指针。
3. 使用 50 条严格依赖的 `ld.shared.b32` 进行指针追逐。
4. 后一次加载的地址来自前一次加载结果，因此无法利用指令级并行隐藏延迟。
5. 使用 64 位 SM cycle counter 记录计时区间。
6. 正式测量前执行 100 次 warmup，以预热指令缓存。
7. 在 host 端验证所有线程的指针追逐结果。

关键配置与结果：

| 项目 | 数值 |
|---|---:|
| Threads per block | 16 |
| Warmup launches | 100 |
| Dependent loads | 50 |
| Minimum latency | 23.92 cycles/load |
| Average latency | 23.92 cycles/load |
| Maximum latency | 23.92 cycles/load |

SASS 中保留了恰好 50 条依赖 `LDS` 指令。该结果表示共享内存依赖加载延迟，不表示共享内存流水线吞吐。

### 4.2 Shared Memory 带宽

测试文件：`op/test/smem_bandwidth.cu`

测试方法：

1. 启动一个包含 256 个线程的 block。
2. 每个线程执行 512 条 128-bit 共享内存向量存储。
3. 每条存储写入 16 B，总请求流量为 2097152 B。
4. 记录各 warp 的开始和结束 cycle，使用最早开始时间和最晚结束时间计算整个 block 的持续时间。
5. 正式测量前执行 100 次 warmup。
6. 读取共享内存结果，验证存储操作正确完成。

关键配置与结果：

| 项目 | 数值 |
|---|---:|
| Threads per block | 256 |
| Stores per thread | 512 |
| Bytes per store | 16 B |
| Shared memory allocation | 12288 B |
| Total requested bytes | 2097152 B |
| Duration | 19265 cycles |
| Measured bandwidth per SM | 108.86 B/cycle |
| Inferred peak per SM | 128.00 B/cycle |

推断的整卡峰值按以下公式计算：

```text
128 B/cycle/SM × 128 SM × 2.52 GHz
= 41287.68 GB/s
```

`41287.68 GB/s` 是假设所有 SM 同时达到推断峰值且保持 2.52 GHz 时钟得到的外推结果，不是直接测得的整卡共享内存带宽。

### 4.3 L1/TEX Cache 延迟

测试文件：`op/test/l1cache_latency.cu`

测试方法：

1. 使用 4 个线程和一个 32 B 的自引用指针数组。
2. 每个线程通过 `ld.global.nc.b64` 反复加载自己的指针。
3. `.nc` 指令在 Ada SASS 中生成为 `LDG.E.64.CONSTANT`，访问统一 L1/TEX 路径。
4. 每次正式计时前先执行 50 条相同加载，使指针数据驻留 L1。
5. 计时区间内再执行 50 条严格依赖加载。
6. 外部执行 100 次 warmup，以预热指令缓存和 L1。

关键配置与结果：

| 项目 | 数值 |
|---|---:|
| Active threads | 4 |
| Pointer-chain size | 32 B |
| Pre-timing L1 loads | 50 |
| Timed dependent loads | 50 |
| Minimum latency | 34.00 cycles/load |
| Average latency | 34.00 cycles/load |
| Maximum latency | 34.00 cycles/load |

SASS 中共保留 100 条 `LDG.E.64.CONSTANT`：50 条用于预热，50 条位于计时区间。

### 4.4 L1/TEX Cache 带宽

测试文件：`op/test/l1cache_bandwidth.cu`

测试方法：

1. 使用 32 KiB 工作集，使每个 SM 可以将完整数据保留在统一 L1/TEX。
2. 工作集分成 4 个 8 KiB 段，通过运行时 segment mask 轮换访问。
3. 每个线程每轮发出 4 条独立的 128-bit `.nc` 向量加载。
4. 每个 block 重复 64 轮加载，并让全部返回分量参与校验和。
5. 校验和与运行时地址选择可防止 `ptxas` 删除或将重复加载移出循环。
6. grid 包含 8192 个 block，即每个 SM 平均 64 个 block。
7. 正式测试前执行 20 次 warmup，随后执行 100 次计时 launch。

关键配置与结果：

| 项目 | 数值 |
|---|---:|
| L1 working set | 32768 B |
| Working-set segments | 4 |
| Block threads | 128 |
| Independent vector loads | 4 |
| Vector width | 16 B |
| Load repeats | 64 |
| Grid blocks | 8192 |
| Traffic per launch | 4294967296 B |
| Benchmark launches | 100 |
| Elapsed time | 10.10 ms |
| Aggregate L1 bandwidth | 42517.02 GB/s |
| Derived bandwidth per SM | 131.81 B/cycle |

每 SM 的 B/cycle 使用 CUDA Runtime 报告的 2.52 GHz 时钟换算：

```text
42517.02 GB/s ÷ 128 SM ÷ 2.52 GHz
= 131.81 B/cycle/SM
```

该值统计 kernel 请求的有效字节数，并使用静态报告时钟换算，因此只能视为近似吞吐指标。SASS 回跳范围完整覆盖地址计算与 4 条 `LDG.E.128.CONSTANT`，确认 64 轮加载均真实执行。

### 4.5 L2 Cache 延迟

测试文件：`op/test/l2cache_latency.cu`

测试方法：

1. 使用一个 warp 和 1408 B 指针链。
2. 相邻访问之间间隔 128 B，对应独立缓存线。
3. 使用 `ld.global.cg.b32` 绕过 L1，仅允许数据缓存在 L2。
4. 正式测量前执行 100 次 warmup，使全部缓存线驻留 L2。
5. 在计时区间执行 10 条严格依赖加载。
6. 使用 64 位 SM cycle counter 测量延迟。

关键配置与结果：

| 项目 | 数值 |
|---|---:|
| L2 cache | 75497472 B |
| Pointer-chain size | 1408 B |
| Cache-line stride | 128 B |
| Warmup launches | 100 |
| Dependent loads | 10 |
| Minimum latency | 273.70 cycles/load |
| Average latency | 273.70 cycles/load |
| Maximum latency | 273.70 cycles/load |

SASS 计时区间包含恰好 10 条串行 `LDG.E.STRONG.GPU`。

### 4.6 L2 Cache 带宽

测试文件：`op/test/l2cache_bandwidth.cu`

测试方法：

1. 使用 2 MiB 工作集，远小于当前 GPU 的 72 MiB L2。
2. 使用 `.cg` 标量加载绕过 L1。
3. 每线程发出 16 条独立加载，warp 访问保持合并。
4. 通过地址取模让大量 block 反复读取同一个 2 MiB 工作集。
5. 每个 launch 统计 2 GiB 请求流量。
6. 先执行 200 次 warmup，再对 200 次 launch 使用 CUDA Event 计时。

关键配置与结果：

| 项目 | 数值 |
|---|---:|
| Working set | 2097152 B |
| Block threads | 128 |
| Loads per thread | 16 |
| Grid blocks | 262144 |
| Traffic per launch | 2147483648 B |
| Benchmark launches | 200 |
| Elapsed time | 83.65 ms |
| Aggregate L2 bandwidth | 5134.35 GB/s |

SASS 中每线程保留 16 条 `LDG.E.STRONG.GPU`，kernel 使用 21 个寄存器且无寄存器溢出。

### 4.7 DRAM 延迟

测试文件：`op/test/dram_latency.cu`

测试方法：

1. 使用一个 warp 执行 `.cg` 全局加载，以绕过 L1。
2. 使用 1024 B 步长，使连续访问落入不同 L2 缓存线。
3. 在正式测试前执行一次 pointer-chain kernel，预热指令缓存和 TLB。
4. 使用 150994944 B，即 144 MiB 的工作区读取并驱逐 72 MiB L2。
5. L2 驱逐完成后，执行 10 条严格依赖的冷全局加载。
6. 使用 64 位 SM cycle counter 记录延迟。

关键配置与结果：

| 项目 | 数值 |
|---|---:|
| Address stride | 1024 B |
| Dependent loads | 10 |
| L2 cache | 75497472 B |
| L2 flush workspace | 150994944 B |
| Minimum latency | 624.10 cycles/load |
| Average latency | 624.10 cycles/load |
| Maximum latency | 624.10 cycles/load |

延迟中除了 DRAM 响应时间，还包含依赖地址更新、地址翻译和少量计时边界开销。由于只有 10 次加载，单次测试对 GPU 动态状态较敏感。

### 4.8 DRAM 带宽

测试文件：`op/test/dram_bandwidth.cu`

测试方法：

1. 使用 `ld.global.cs.v4.b32` 和 `st.global.cs.v4.b32` 执行 128-bit 流式读写。
2. block 包含 128 个线程，每线程每次处理 16 B。
3. 对 4 MiB 至 1 GiB 工作集进行二倍递增扫描。
4. 每个尺寸执行 100 次 benchmark launch。
5. 每次 launch 的起始地址移动 16 MiB，降低相邻测试对同一缓存区域的重复使用。
6. 分别测试只读、只写和 copy。
7. copy 只复制半个工作集，但统计读取与写入的总流量，因此报告字节数等于完整工作集大小。
8. 最终 DRAM 结果使用 1 GiB 大工作集，避免将小尺寸缓存和写队列吞吐误认为 DRAM 峰值。

完整尺寸扫描结果：

| Size (MiB) | Read (GB/s) | Write (GB/s) | Copy (GB/s) |
|---:|---:|---:|---:|
| 4 | 700.51 | 996.90 | 748.81 |
| 8 | 769.92 | 920.42 | 878.97 |
| 16 | 822.08 | 926.17 | 878.03 |
| 32 | 955.12 | 1723.72 | 881.10 |
| 64 | 873.11 | 914.67 | 959.95 |
| 128 | 883.29 | 911.99 | 882.22 |
| 256 | 888.27 | 910.44 | 880.41 |
| 512 | 903.66 | 959.13 | 920.97 |
| 1024 | 929.92 | 959.08 | 920.35 |

1 GiB 工作集结果：

| 类型 | 带宽 |
|---|---:|
| DRAM-scale read | 929.92 GB/s |
| DRAM-scale write | 959.08 GB/s |
| DRAM-scale copy | 920.35 GB/s |

32 MiB 写入结果达到 1723.72 GB/s，明显受到 L2、写合并和异步写回队列影响，不能作为 DRAM 物理带宽。1 GiB 工作集结果更适合表示当前 GPU 的持续 DRAM 吞吐。

## 5. 层级对比与结论

### 5.1 延迟层级

本轮测试得到清晰的延迟层次：

```text
Shared Memory  23.92 cycles
L1/TEX Cache   34.00 cycles
L2 Cache      273.70 cycles
DRAM          624.10 cycles
```

L2 命中延迟约为 L1 的 8.05 倍，冷 DRAM 延迟约为 L1 的 18.36 倍、L2 的 2.28 倍。对依赖加载、指针追逐和不规则访存算子而言，提高 L1/L2 命中率会直接改善执行时间。

### 5.2 带宽层级

按本轮请求字节统计：

```text
L1/TEX Cache  42517.02 GB/s
L2 Cache       5134.35 GB/s
DRAM Read       929.92 GB/s
```

L1 聚合请求带宽约为 L2 的 8.28 倍，L2 带宽约为 DRAM 读取带宽的 5.52 倍。Shared Memory 推断整卡峰值为 41287.68 GB/s，与 L1 请求带宽处于相近数量级，但两者的测量与推算方法不同，不能仅凭数值判断硬件数据通路完全等价。

### 5.3 延迟不能唯一推导带宽

延迟只描述一个依赖请求完成所需的时间。带宽还取决于请求大小、并发 warp 数、在途事务数量、访存合并效率、缓存分区和流水线吞吐。

例如，本轮 DRAM 延迟为 624.10 cycles/load，但通过全 GPU 大规模并行仍可达到约 929.92 GB/s 的读取带宽。延迟测试主动消除并行，带宽测试则主动制造并行，两者衡量的是不同性质。

## 6. 结果解释限制

1. 所有结果都是微基准结果，不等同于真实应用中的端到端性能。
2. GPU 未锁定核心频率和显存频率，动态 Boost 会造成运行间波动。
3. `cudaDevAttrClockRate` 返回报告时钟，可能与测试期间的瞬时时钟不同。
4. latency 结果包含少量地址更新和计时指令开销。
5. cache bandwidth 按 kernel 请求字节数统计；硬件内部的 sector、cache line、合并和重放可能改变实际物理流量。
6. `.nc`、`.cg` 和 `.cs` 是缓存策略或缓存提示，最终行为还受 GPU 架构影响。
7. L1 是每 SM 私有资源；L1 聚合带宽依赖 block 是否均匀分布到全部 SM。
8. L2 与 DRAM 是整卡共享资源，结果容易受到其他 GPU 任务影响。
9. DRAM copy 带宽统计读取与写入流量之和，不等同于仅按有效复制字节计算的 copy 吞吐。
10. 若用于跨 GPU 对比，应固定驱动、CUDA、功耗上限、时钟、温度和测试顺序，并重复多次报告中位数。

## 7. 构建与复现

配置并构建全部测试：

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --target \
    smem_latency \
    smem_bandwidth \
    l1cache_latency \
    l1cache_bandwidth \
    l2cache_latency \
    l2cache_bandwidth \
    dram_latency \
    dram_bandwidth
```

执行 CTest：

```bash
ctest --test-dir build \
    -R '^(smem|l1cache|l2cache|dram)_(latency|bandwidth)$' \
    --output-on-failure
```

本轮联合回归结果为：

```text
100% tests passed, 0 tests failed out of 8
```
