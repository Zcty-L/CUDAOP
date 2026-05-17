# WGMMA TF32 Sample Implementation Design (M64, N128, K64)

## 1. 目标 (Goal)
在 Blackwell 架构 (RTX 5070 Ti) 上实现一个基础的 WGMMA 示例，使用 TF32 数据类型。重点在于展示 Warp Group (128 线程) 如何协同工作，以及如何构建 WGMMA 描述符 (Descriptor)。

## 2. 核心参数 (Core Parameters)
- **数据类型**: 
  - 输入 (A, B): TF32 (存储在 4 字节 float 中，计算时硬件取精度)。
  - 累加器 (C): FP32 (寄存器)。
- **维度**: 
  - M = 64
  - N = 128
  - K = 64
- **计算粒度**: 使用 `m64n128k16` 指令，迭代 4 次完成 K=64。
- **布局**: 
  - A: Shared Memory, Row-major, 无 Swizzling (TODO: 后续增加 Swizzling 支持)。
  - B: Shared Memory, Column-major (为了 WGMMA 效率，通常 B 设为 Col-major，本示例 A 为 Row, B 为 Col)。

## 3. 架构设计 (Architecture)

### 3.1 Warp Group 协同
- 使用 128 个线程 (4 个 Warp) 作为一个 Warp Group。
- 在 Kernel 中通过 `threadIdx.x < 128` 逻辑来限定参与计算的线程。

### 3.2 内存布局与描述符
- **矩阵 A (64x64)**: 
  - 存储空间: `float smem_a[64 * 64]`
  - LD (Leading Dimension): 64 * 4 = 256 字节。
  - 描述符位域:
    - `addr`: `smem_u32addr(smem_a) >> 4`
    - `ld`: `256 >> 4 = 16`
    - `swizzle`: `0` (None)
    - `layout`: `0` (Row-major)
- **矩阵 B (64x128)**:
  - 存储空间: `float smem_b[64 * 128]`
  - LD: 128 * 4 = 512 字节。
  - 描述符位域:
    - `addr`: `smem_u32addr(smem_b) >> 4`
    - `ld`: `512 >> 4 = 32`
    - `swizzle`: `0` (None)
    - `layout`: `1` (Column-major)

### 3.3 指令序列逻辑
1. **数据加载**: 从 Global Memory 加载到 Shared Memory。
2. **Fence**: `wgmma.fence` 确保数据可见。
3. **计算循环**:
   ```cpp
   for (int k = 0; k < 4; k++) {
       // 更新描述符的地址偏移 (每次 K 推进 16)
       uint64_t descA = make_desc(smem_a_base + k * 16 * sizeof(float));
       uint64_t descB = make_desc(smem_b_base + k * 16 * LD_B);
       wgmma_mma_async(accum, descA, descB);
   }
   ```
4. **同步与写回**: `wgmma.commit_group`, `wgmma.wait_group`, 然后同步写回。

## 4. 关键 PTX 封装 (PTX Wrappers)
- `wgmma_fence()`: 映射到 `wgmma.fence.aligned;`
- `wgmma_commit_group()`: 映射到 `wgmma.commit_group.aligned;`
- `wgmma_wait_group(int n)`: 映射到 `wgmma.wait_group.aligned %0;`
- `wgmma_mma_async_m64n128k16(...)`: 核心计算指令。

## 5. 验证方案 (Validation)
- 在 Host 端使用简单的 CPU 矩阵乘法计算参考值。
- 对比 GPU 计算结果，允许极小的 TF32 舍入误差。

## 6. TODO (后续优化)
- [ ] 引入 128B/256B/512B Swizzling 以消除 Bank Conflict。
- [ ] 尝试从寄存器读取矩阵 B。
- [ ] 使用 `cp.async` 进行数据预取。
