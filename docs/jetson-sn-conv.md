# Jetson Orin NX SNN Conv Kernel 设计讨论

## 平台参数

| 项目 | 值 |
|------|-----|
| GPU | Ampere (SM 8.7), 1024 CUDA Cores, 16 SMs |
| Tensor Cores | 32 (每SM 2个) — **本设计不使用** |
| 内存 | 16 GB LPDDR5 (CPU/GPU 统一) |
| Max SMEM/Block | 48 KB (默认), 可配置为 100 KB |
| Max Threads/Block | 1024 |
| L1/SMEM 可配 | 128 KB/SM |
| L2 | 256 KB |

## 场景

SNN 脉冲卷积。输入为 T 个时间步的二进制脉冲 (0/1)，通过移位压缩到一个 int 中。Kernel 内展开后与 weights 做标准 Conv2d。

**计算特性**: 脉冲是二进制的，乘法退化为浮点加法 (conditional add: spike=1 则累加 weight，spike=0 则跳过)。

## 输入输出规格

- **输入**: `[1, C, H, W]` uint8 (T bits 打包到 uint8，TRT 已验证支持)
- **Weights**: `[C * Kh * Kw, C_out]` (提前转置，列主序，float)
- **输出**: `[T, C_out, H, W]` float (T 个时间步的结果)
- **T**: 1, 2, 3, 4 (分别构建特化 kernel)
- **Block tile**: 64×64 输出元素/block
- **数据流**: 输入脉冲和 weights 各从 GMEM 读一次到 SMEM，T 个时间步在寄存器中展开
- **核心类型**: CUDA Core only，不用 Tensor Core
- **方式**: 直接卷积索引，**不做 im2col**

## 网络结构 (参考 YOLOv8)

```
Input: [1, C, 160, 160]
  ↓  (第一层 conv, 不管)
  ↓
Scale 1: 160×160  ──── stride=1 convs ────→  (大部分层)
  ↓  3x3/stride=2 (约4处，下采样)
Scale 2: 80×80
  ↓  3x3/stride=2
Scale 3: 40×40
  ↓  3x3/stride=2
Scale 4: 20×20
  ↓  3x3/stride=2
Scale 5: 10×10
```

| 属性 | 值 |
|------|-----|
| 空间尺寸 (H=W) | 160, 80, 40, 20, 10 |
| 下采样层 | ~4 个 3×3/stride=2/padding=1 conv |
| 其余 conv | stride=1, padding=1 (same) 或 1×1/stride=1/padding=0 |
| 第一层 conv | **排除，不处理** |

## 从 conv_info.txt 提取的 Layer 特征

### 总览

conv_info.txt 共 162 个权重条目 (`torch.Size([C_out, C_in, Kh, Kw])`)，排除注释掉的行后:

| 类别 | 数量 | 占比 |
|------|------|------|
| 1×1 Conv | 93 | 57% |
| 3×3 Conv | 48 | 30% |
| Depthwise (C_in=1) 1x1/3x3 | 1 | 0.6% |
| 大 kernel (5×5/7×7/9×9) | 21 | 13% |
| **排除大 kernel 后合计** | **141** | **87%** |

> 大 kernel (21 个) 全部为 depthwise (C_in=1): 5×5 ×4, 7×7 ×8, 9×9 ×9。排除，不在当前 scope。

### Kernel Size 分布 (仅 1×1 + 3×3)

#### 1×1 Conv (93 层, 66% of 141)

1×1 无空间维度，`in_features = C_in`，K 迭代次数 = C_in/16。始终 stride=1, pad=0。

**Channel 模式分类**:

| 模式 | 数量 | 典型 `[C_out, C_in]` | 含义 |
|------|------|---------------------|------|
| 通道不变 | 18 | `[64,64]×4, [144,144]×7, [192,192]×4, [16,16],[32,32],[48,48]` | 残差连接前的 bottleneck |
| 通道扩展 (×2~3) | 18 | `[96,192]×6, [72,144]×4, [192,384]×2, [72,216]×2` | 增加宽度 |
| 通道压缩 (×0.25~0.75) | 47 | `[384,192]×5, [288,144]×5, [192,128]×3, [96,64]×3` | 减少宽度/bottleneck |
| 其他比例 | 10 | `[216,72]×2, [192,64]×2, [144,64]×2, [24,72],[72,24]...` | 各类变换 |

**关键观察**:
- 1×1 中 **无 stride=2**，全部 stride=1
- 通道压缩占 50%，通常是 bottleneck 或输出融合前的降维
- C_in 范围: 8~384 (排除大 kernel 后最大 384)
- C_out 范围: 16~768
- **最常见**: `[144,144,1,1]` (7 次), `[96,192,1,1]` (6 次), `[144,384,1,1]` (5 次)

#### 3×3 Conv (48 层, 34% of 141)

`in_features = C_in × 9`，K 迭代 = C_in×9/16。

**Channel 模式 + stride 推断**:

| 模式 | 数量 | 典型 `[C_out, C_in]` | 推断 stride | 含义 |
|------|------|---------------------|------------|------|
| 通道不变 | 31 (65%) | `[32,32]×13, [64,64]×9, [16,16]×6, [8,8]×2, [48,48]` | **stride=1** | 空间特征提取 (same conv) |
| 通道翻倍 | 9 (19%) | `[32,64]×2, [8,16], [16,32], [64,128], [24,48], [48,96], [96,192], [192,384]` | **stride=2** | 下采样层 |
| 通道减半 | 3 (6%) | `[128,64], [64,32], [128,32]` | stride=1 或 2 | 降维 |
| 通道压缩 (其他) | 3 (6%) | `[144,48]×2, [64,32]` | stride=1 | 降维 |
| Depthwise | 1 | `[72,1]` | stride=1 或 2 | 逐通道 |
| **stride=1 合计** | **~37 (77%)** | — | 通道不变 + 降维 | — |
| **stride=2 合计** | **~9 (19%)** | — | 通道翻倍 | — |

**关键观察**:
- 3×3 stride=1 占 **~77%**，是 3×3 的主体
- 3×3 stride=2 仅 **~9 层** (19%)，全为通道翻倍模式
- C_in 范围: 3~128
- C_out 范围: 8~384
- **最常见**: `[32,32,3,3]` (13 次), `[64,64,3,3]` (9 次), `[16,16,3,3]` (6 次)

### Stride 类型总结

| Stride | Kernel | Padding | 层数 (估) | 用途 |
|--------|--------|---------|-----------|------|
| **1** | 1×1 | 0 | 93 | 通道变换，K迭代极少 (C_in/16) |
| **1** | 3×3 | 1 | ~37 | 空间特征提取，same conv |
| **2** | 3×3 | 1 | ~9 | 下采样 (伴随通道翻倍) |

**对 kernel 设计的启示**:
- **1×1 输入索引极度简化**: Kh=Kw=1 → ky=kx=0 恒成立，偏移计算退化为 `ih=oh*Sh-Ph, iw=ow*Sw-Pw`，stride=1/pad=0 时 `ih=oh, iw=ow`（恒等映射），输入搬运基本无分支
- **3×3 stride=1**: 主体场景，需要感受野索引但无下采样
- **3×3 stride=2**: 仅 9 层，可单独处理或复用 stride=1 路径（仅 stride 参数不同）
- 输入变为 `[8,16,24,32,...,128]` → `KhKw = [1, 9]` → `in_features = [8, 16, 24, ..., 1152]`
- 常见 1×1: 64×1=64, 3×3: 32×9=288, 64×9=576, 128×9=1152 个 in_features

## 核心设计问题

### Q1: 脉冲压缩格式 ✓ 已确定

每个 uint8 打包**一个空间位置上的 T 个时间步**的二进制脉冲：

```
input[c][h][w] = sum_{t=0}^{T-1} spike[t][c][h][w] << t
```

Kernel 内展开: `spike_t = (input >> t) & 1`

**数据类型**: **uint8** (TRT 实测支持，每元素 1B)

优势: 相比 int16 输入 SMEM 再减半，带宽开销仅 1/4 of float。

---

### Q2: 64×64 Output Tile 如何分配到 C_out 和 spatial？

每个 block 计算 **64 (C_out) × 64 (spatial) = 4096 个输出元素**。

参考 `conv2d_mma_k128`: output tile [128,128], grid 按 `(spatial/128, C_out/128)` 划分。类比：

| 方案 | C_out tile | Spatial tile | 适用场景 |
|------|-----------|-------------|---------|
| 默认 | 64 | 64 (如 8×8) | 通用 |
| A | 32 | 128 (如 8×16) | C_out 小、spatial 大 |
| B | 128 | 32 (如 4×8) | C_out 大、spatial 小 |
| C | 1 | 64×64 (如 64×64) | C_in=1 逐通道滤波 |

以默认方案 [64, 64] 为例 (3×3/s=1/p=1, K_chunk=16):

```
SMEM per K-iteration:
  smemweight: [M_tile, K_chunk] = [64, 16] float = 4KB
  smeminput:  [K_chunk, N_tile] = [16, 64] uint8 = 1KB
  Total: 5KB

cp.async: 64×16/4 = 256 float4 → 256线程×1 = 完美 ✓
```

**与参考 kernel 对比**:

| 项目 | conv2d_mma_k128 | SNN Conv (本设计) |
|------|----------------|-------------------|
| Output tile | [128, 128] = 16K | [64, 64] = 4K |
| SMEM weight | [128, 8] = 4KB | [64, 16] = 4KB |
| SMEM input | [8, 128] = 4KB | [16, 64] = 1KB (uint8) |
| 总 SMEM | 8KB | **5KB** |
| 线程 | 256 | **256** |
| cp.async/线程 | 1 | **1** |

SMEM 仅 5KB，48KB 下可放 9 个 block（寄存器会是真正限制）。

---

### Q3: stride 和 padding

基于 conv_info.txt 的 channel 模式推断 stride:

| 类型 | stride | padding | 用途 | 推断方法 | 层数 (估) |
|------|--------|---------|------|---------|-----------|
| 1×1 | 1 | 0 | 通道变换 | 始终 stride=1 | **93** |
| 3×3 same | 1 | 1 | 空间特征提取 | 通道不变 (C_out=C_in) | **~37** |
| 3×3 下采样 | 2 | 1 | 空间减半 + 通道翻倍 | 通道翻倍 (C_out=2×C_in) | **~9** |
| 3×3 depthwise | 1 或 2 | 1 | 逐通道滤波 | C_in=1 | **1** |

> stride 值不在 conv_info.txt 中直接给出，上表为根据网络结构规律推断（通道翻倍伴随空间下采样是标准 CNN 设计模式）。

**1×1 Conv 输入索引简化**: Kh=Kw=1, stride=1, pad=0 → ky=kx=0 恒成立 → `ih = oh, iw = ow`（恒等映射）。输入加载完全无空间分支，仅需 `c_idx = global_k` 的通道选择。

**3×3 stride=2**: 仅 ~9 层 (~6%)，由于 stride=2 时输出 tile 对应的输入感受野范围不同，加载路径需要单开。可以**先实现 stride=1，stride=2 后续处理**（层数少，非主体）。

---

### Q4: 1×1 Conv 是否走特殊路径？

1×1 Conv 本质是 MatMul:
```
Input:  [C_in, H*W]
Weight: [C_in, C_out]
Output: [C_out, H*W]
```
对于 T 个时间步，T 次 MatMul 复用同一输入（不同 bit）。

- 直接 MatMul tile，不做 im2col
- 逐时间步: `if bit_t: acc[c_out] += weight`
- 约 50% 的层是 1×1，**值得特化**

---

### Q5: C_in=1 的特殊处理

C_in=1 时，Conv 退化为每个输出通道对输入做 2D 滤波。in_features = Kh×Kw 最多 9。
- 每个 block 可以覆盖更多 C_out (甚至全部)
- 输入 SMEM 开销极小 (只需 1 通道)
- 适合大 spatial tile

---

### Q6: SMEM 预算 (基于 K_chunk=16, Output tile [64,64])

每次 K-iteration 只加载一个 K_chunk=16 到 SMEM。SMEM layout 参考 `conv2d_mma_k128`:

```
smemweight: [M_tile=64, K_chunk=16] float = 64×16×4B = 4KB
smeminput:  [K_chunk=16, N_tile=64] uint8 = 16×64×1B = 1KB
Total: 5KB per K_chunk
```

**各层 SMEM 预算** (256 线程, K_chunk=16, 3×3 kernel):

| C_out_tile | Kernel | N_tile | Weight SMEM | Input SMEM | 总计 |
|-----------|--------|--------|-------------|-----------|------|
| 64 | 3×3 | 64 | [64,16]×4B=4KB | [16,64]×1B=1KB | **5KB** |
| 32 | 3×3 | 128 | [32,16]×4B=2KB | [16,128]×1B=2KB | **4KB** |
| 128 | 3×3 | 32 | [128,16]×4B=8KB | [16,32]×1B=0.5KB | **8.5KB** |
| 64 | 1×1 | 64 | [64,16]×4B=4KB | [16,64]×1B=1KB | **5KB** |

1×1 conv 与 3×3 在 SMEM 层面一致 (K 维仅 in_features 不同，K_chunk 大小相同)。

**迭代次数**: in_features / K_chunk
- C_in=64, 3×3: 576/16 = **36 次** K-iteration
- C_in=128, 3×3: 1152/16 = **72 次**
- C_in=64, 1×1: 64/16 = **4 次** (极少!)

**关键结论**:
- **SMEM ~5KB** 极轻，48KB default 不是瓶颈
- **1×1 conv 迭代次数极少** (C_in/16)，非常适合此方案
- **3×3 conv 迭代较多** (C_in×9/16)，但每次计算量也大 (条件加法 ×9)
- SMEM 瓶颈消失后，**寄存器成为 occupancy 真正限制** (见 Q10-F)

---

### Q7: 脉冲展开的计算策略

对于 T=1..4，展开后的计算模式：

**方式 1: 逐时间步展开 → 分别累加** (推荐起步方案)
```
// T=4, 输入uint8打包在低4bit
uint8_t packed = input_smem[pos];
for t in 0..T-1:
    if ((packed >> t) & 1):
        for each c_out:
            acc[t][c_out] += weights[in_feat][c_out]
```
优点: 简单直接，T 次循环
缺点: 每个时间步都要遍历一次 weights

**方式 2: 预计算 LUT → 查表累加**
T 个 bit 有 2^T 种组合。预计算每种组合的加权和，然后查表。
对于 T≤4: LUT 只有 16 项。
```
// 预计算: lut[mask][c_out] = sum{weights[pos][c_out] if mask has bit at pos set}
// 推理: out[t][c_out] += lut[1<<t][c_out]  // 提取单个时间步
```
适合多时间步共享权重，但需要 additional SMEM/寄存器放 LUT。

**方式 3: 多路并行累加**
```
// T=4, 展开4路并行
if (packed & 0x1) acc0[c_out] += w;
if (packed & 0x2) acc1[c_out] += w;
if (packed & 0x4) acc2[c_out] += w;
if (packed & 0x8) acc3[c_out] += w;
```
一次读取，4路条件累加。适合 T 固定时手动展开。

**建议**: 方式 1 起步验证功能，再优化到方式 3。方式 2 在 T=3,4 且 C_out 大时可能有优势。

---

### Q8: 时间复杂度估算

以一层为例: C_in=64, C_out=64, H=W=80, 3×3/s=1:
- 输出元素: 64 × 80 × 80 = 409,600
- 每个输出的乘加: 64 × 9 = 576 (实际是条件加法)
- 总操作: 409,600 × 576 = ~236M ops
- 对于 T=4: ×4 = ~944M ops
- Jetson Orin NX: 1024 cores × ~1.2GHz = ~1.2T FLOPS
- 计算时间 ~0.8ms (理想)，实际受带宽约束

GMEM 读取 (T=4, uint8 输入):
- 输入: 64 × 80 × 80 × 1B = 409.6 KB
- Weights: 64 × 9 × 64 × 4B = 147.5 KB
- 总共: ~557 KB (非常小，不是瓶颈)

---

### Q10: 线程数、SMEM Tile 与 Double Buffer 分析

> 参考实现: `op/conv/conv2d_mma_k128.cu` — 256 线程, SMEM `[128,8]` + `[8,128]` = 8KB, K_chunk=8, 输出 tile [128,128]

#### A. 参考 Kernel 设计模式 (conv2d_mma_k128)

```
Block: 256 threads (8 warps)
SMEM:  float smem[8*128*2] = 8KB total
       ├─ smemweight [128, 8] = 4KB  (M_tile=128, K_chunk=8)
       └─ smeminput  [8, 128]  = 4KB  (K_chunk=8, N_tile=128)
Output tile: [128, 128] (M × N)
K loop: for crs in 0..in_features step 8  (144 iterations for C=128, 3×3)

Per iteration:
  1. cp.async: weights [128,8] → 1024 floats → 256 float4 → 256线程×1 ✓
  2. ldg32+sts32: inputs [8,128] → 8 warps 各处理 1 个 K 元素
  3. commit+wait+syncthreads
  4. Compute: warp 级 [32,64] outer product from SMEM
```

关键设计原则:
- **K_chunk 很小 (8)**，不对齐 C_in 边界，类似 GEMM K-tile
- **SMEM 仅 hold 一个 K_chunk**，不是全部 in_features
- **256 线程干净映射**: 权重 float4 数 = 线程数 → 1 cp.async/线程

#### B. SNN Conv 适配分析

我们的输出 tile = 64 (vs 参考的 128×128=16384)。设 M_tile = C_out_tile, N_tile = spatial_tile, M × N = 64。

**SMEM 设计** (单 K_chunk):

```
smemweight: [M_tile, K_chunk] float     // 权重
smeminput:  [K_chunk, N_tile] uint8     // 打包脉冲 (不同于参考的 float)
```

**权重 cp.async 干净映射条件**: `M_tile × K_chunk / 4` 整除 `num_threads`

| M_tile | K_chunk | Weight SMEM | float4 数 | 256线程 | 128线程 | 512线程 |
|--------|---------|-------------|----------|---------|---------|---------|
| 64 | 16 | 4 KB | 256 | **1/线程** ✓ | 2/线程 ✓ | 0.5 ✗ |
| 64 | 32 | 8 KB | 512 | 2/线程 ✓ | 4/线程 ✓ | **1/线程** ✓ |
| 64 | 8 | 2 KB | 128 | 0.5 ✗ | **1/线程** ✓ | 0.25 ✗ |
| 32 | 16 | 2 KB | 128 | 0.5 ✗ | **1/线程** ✓ | 0.25 ✗ |
| 32 | 32 | 4 KB | 256 | **1/线程** ✓ | 2/线程 ✓ | 0.5 ✗ |
| 128 | 16 | 8 KB | 512 | 2/线程 ✓ | 4/线程 ✓ | **1/线程** ✓ |

**核心发现**: `M_tile × K_chunk = 1024` 时 → 256 float4 → 256 线程完美 1:1 映射 (匹配参考 kernel!)

适用组合:
- [64, 16] → 4KB weight SMEM + ~2KB input SMEM = **6KB** ✓ 
- [32, 32] → 4KB weight SMEM + ~4KB input SMEM = **8KB** ✓
- [128, 8] → 4KB weight SMEM + ~1KB input SMEM = **5KB** ✓ (与参考一致)

#### C. 推荐配置

**256 线程, K_chunk=16, M_tile=64** (C_out=64, spatial=1 或 C_out=32, spatial=2, 以此类推):

| 项目 | 值 |
|------|-----|
| 线程数 | **256** (8 warps) |
| smemweight | [64, 16] float = 4KB |
| smeminput | [16, N_tile] uint8 = ~2KB (N_tile≈128) |
| 总 SMEM | **~6KB** |
| cp.async/线程 | **1** (256 float4 / 256 线程) |
| K 迭代次数 | in_features/16, 如 C_in=64 3×3 → 576/16 = 36 次 |

**对比我之前的分析 (128 线程, 18KB weight SMEM)**:

| 方面 | 之前 (错) | 正确 (参考模式) |
|------|----------|----------------|
| Weight SMEM | 18KB (全部 C_in_chunk) | **4KB** (单 K_chunk=16) |
| 线程 | 128 | **256** |
| K_chunk | 72 (对齐 C_in) | **16** (不对齐 C_in) |
| 迭代次数 | C_in/8 = 8 | **in_features/16 = 36** |
| SMEM 占用 | 19.2KB | **~6KB** |

之前错误: 把 C_in_chunk 当作 K_chunk，导致 weight SMEM = C_in_chunk × Kh × Kw × C_out，爆炸。

#### D. 输入加载方式

uint8 输入 ~2KB，不用 cp.async。参考 kernel 对 input 用 ldg32+sts32 (每 warp 负责一个 K 元素):

```
// 对每个 K 元素 k，确定其 (c, ky, kx)
int c  = (crs + k) / (Kh×Kw);
int ky = ((crs + k) % (Kh×Kw)) / Kw;
int kx = ((crs + k) % (Kh×Kw)) % Kw;
// warp k 从 GMEM 加载 input[c][tile_h+ky][tile_w+kx] 的 N_tile 个 uint8
// 写入 smeminput[k][0..N_tile-1]
```

#### E. Double Buffer 评估

参考 kernel 是 **单 buffer** (smemweight + smeminput 是同一 K_chunk 的两个 half，不是 ping-pong)。

扩展到真正的 double buffer (ping-pong):
- SMEM 需要 2 × 6KB = 12KB
- 48KB 下 48/12 = 4 blocks/SM (vs 48/6 = 8 单 buffer)
- 优势: cp.async 加载下一个 K_chunk 与当前计算重叠
- **建议: 先实现单 buffer (与参考一致)**，profile 确认 GMEM 延迟是否瓶颈

#### F. Occupancy 估算

SMEM ~6KB/block, 256 threads/block, ~64 regs/thread (T=4, 每线程 hold ~16 个 partial):

| SMEM 配置 | Blocks/SM (SMEM限) | Blocks/SM (Reg限) | Warps/SM | Occupancy |
|-----------|-------------------|-------------------|----------|-----------|
| 48KB default | 48/6=8 | 65536/(256×64)=4 | 4×8=32 | **32/48=66.7%** |
| 48KB + double buf | 48/12=4 | 同 4 | 4×8=32 | **66.7%** |

**结论**: SMEM ~6KB 极小，occupancy 不是瓶颈。即使 48KB default 也能跑满 4 blocks/SM (寄存器限制)。

---

### Q9: TRT Plugin 集成注意事项 ✓ 已确认

- **uint8 输入**: TRT 实测支持，无问题
- 插件输出 float `[T, C_out, H, W]`，兼容 TRT 后续层

---

## cp.async 验证计划 (第一阶段)

从 cp.async 搬运验证开始，逐步叠加：

1. ~~权重 tile GMEM→SMEM~~ **已完成** ✓ (见下方已完成工作 1)
2. ~~Thread Map 验证~~ **已完成** ✓ (见下方已完成工作 3)
3. ~~64×64 mini kernel~~ **已完成** ✓ (见下方已完成工作 2)
4. **带 stride 的二维搬运** — 从 `[C, H, W]` 中按卷积感受野索引搬运 `[C, tile_h+Kh-1, tile_w+Kw-1]`，支持 stride=1 和 stride=2
5. **输入 + Weights 双通道搬运** — 模拟真实的输入/权重并发加载，验证 SMEM 分配 (已在 mini kernel 中初步验证)
6. **脉冲展开 + 条件累加** — 搬运基础上加入 bit 提取和 float 条件加法
7. **完整 mini kernel** — 搬运 → 展开 → T 路累加 → 写回

---

## 已完成工作

### 1. cp.async 权重加载验证

文件: `op/cp_async/cp_async.cu`

3 个 kernel，均 256 线程、SMEM `[16, 64]` float (4KB)、1 cp.async/线程:

| Kernel | 功能 | 测试参数 | 结果 |
|--------|------|---------|------|
| `test_weight_load_full` | in_features 和 C_out 均为整数倍 | in_feat=576, C_out=128 | PASSED |
| `test_weight_load_boundary` | C_out 边界处理 (src_size) | in_feat=576, C_out=72 | PASSED |
| `test_weight_load_general` | in_features + C_out 双边界 | in_feat=8, C_out=72 | PASSED |

关键设计:
- 线程映射: `row_f4 = tid/16, col_f4 = tid%16` → 16×16=256 覆盖 `[16,64]` float4 SMEM
- 边界处理: 用 `cp.async` 的 `src_size` 操作数控制拷贝字节数 (0=跳过, 16=完整)
- 编译: 48 regs, 4096 bytes smem, 0 spill

### 2. 64×64 Mini Kernel (单 block)

文件: `op/conv/conv2d_k64.cu` — `conv2d_64x64x16_test`

```
SMEM:  smemweight [K_CHUNK=16, M_TILE=64] float (4KB)
       smeminput  [K_CHUNK=16, N_TILE=64] uint8 (1KB)
       Total: 5KB
Output tile: [64, 64] float
Threads: 256 (8 warps)
K 迭代: 1 次 (K=16 全载入)
```

流程: cp.async 加载 weights → 打包加载 uint8 inputs → sync → K-loop { LDS weight_frag[4] + input_frag → 外积 `add_f32` } → 写回

编译: 37 regs, 5120 bytes smem, 0 spill

测试: K=16, M=64, N=64, T=1, 随机数据 → **PASSED (0 errors)**

### 3. Thread Map 验证

文件: `op/test/thread_map_test.cu`

确认 `thread_map.txt` 中一个 warp 内 32 线程的 thread mapping:

```
mma_tid_x = lane_id / 16 * 2 + lane_id % 2  (0..3, 4 列)
mma_tid_y = lane_id % 16 / 2                 (0..7, 8 行)
```

Grid 布局 (8行 × 4列 = 32线程):
```
mma_tid_x →   0   1   2   3
mma_tid_y=0   0   1  16  17
...
mma_tid_y=7  14  15  30  31    (值为 lane_id)
```

| Kernel | SMEM | 索引维度 | 每次读取 | 广播 | Bank Conflicts (NCU) |
|--------|------|---------|---------|------|---------------------|
| `smem_float2_load` | 16 float (64B) | `mma_tid_y` (0..7) | float2 (8B) | 4线程/地址 | **0** |
| `smem_uint64_load` | 32 uint8 (32B) | `mma_tid_x` (0..3) | uint64 (8B) | 8线程/地址 | **0** |

两者对称: mma_tid_y 8值×4线程=32, mma_tid_x 4值×8线程=32，均 0 bank conflicts → 1 SMEM transaction。

在 conv kernel 中 warp 内 `[16M, 32N]` 的线程排布:
```cuda
M_group = mma_tid_y / 2;                  // 0..3
N_group = mma_tid_x * 2 + mma_tid_y % 2;  // 0..7
```

---

## 待确认清单

- [x] 输入数据类型: **uint8** 确认 (TRT 实测通过)
- [x] 线程/Tile 设计: **256线程, K_chunk=16, [M=64,N=64], SMEM 5KB** (见 Q10)
- [x] Thread Map: **mma_tid_x/y 公式已验证**，0 bank conflicts (见已完成工作 3)
- [x] cp.async 权重加载: **full + boundary 均 PASSED** (见已完成工作 1)
- [x] 64×64 mini kernel: **单 block, T=1 外积 PASSED** (见已完成工作 2)
- [ ] C_out/spatial 划分：默认 [64,64]，需根据实际层分布确认 (见 Q2)
- [ ] 是否需要 LUT 方式优化（T=3,4 时）还是先用方式 1 验证？
- [ ] stride=2 的 4 层是否可以最后处理（先用 stride=1 跑通）？
- [ ] Double buffer: 参考 kernel 是单 buffer，先保持一致，profile 后再评
