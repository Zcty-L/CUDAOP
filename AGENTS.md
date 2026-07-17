# Project: CUDAOP

这是一个专注于 CUDA 算子优化的项目
所有回答都必须使用中文


# 项目结构

## 文档目录

所有的文档保存目录在 `./docs`


## 核心代码目录

核心代码目录在 `./op/{op_name}`，比如 `./op/conv` 就是卷积算子 Conv 目录


## 配置文件
- 一些参数配置，结构体定义需要在 `./op/config.h`
- PTX 指令**读取**和**添加**统一在 `./op/ptx_utils.cuh`
- `./op/cuda_utils.cuh` **已弃用，不再使用**，新代码禁止引用
- `cmake` 配置：`./CMakeLists.txt`

### PTX 指令规范
- 所有 PTX 指令统一在 `./op/ptx_utils.cuh` 中管理
- 添加 PTX 指令时需在注释中注明：指令名称、来源（文档链接或参考项目）、用途


# 规则

## 代码编写规定
- 禁止使用 **printf**, **print**, 统一使用 **std::cout** (C++ 要求)。输出要考虑对齐
- **{ }** 的使用需要换行
- 较长的代码行需要换行，**禁止**编写很长的代码行


## 编译代码规则
- 代码的编译统一在 `CMakeLists`，只有在临时测试的时候才允许使用 `nvcc`，产生的临时可执行文件的输出目录为 `./build`
- 最终一定需要放到 `CMakeLists` 进行测试验证


## 测试 Debug 输出规范
- 测试需输出配置、主要阶段、关键结果和 `[SUCCESS]` 成功标记
- 不同测试及主要阶段之间需保留空行
- Python 使用 `logging`，C++/CUDA 使用 `std::cout`


## 新增算子工作流
1. 创建 `./op/{name}/` 目录
2. 实现 kernel
3. 在 `./CMakeLists.txt` 注册新算子目标 (仅适用于C++, python构建算子不需要)
4. 添加精度验证和性能测试
5. 确保编译通过并运行测试


## Git 规范
- 修改代码前必须执行 `git status --short --branch` 检查当前分支
- 禁止在 `main` 分支修改文件；如果当前分支是 `main`，立即停止并提醒用户先切换或创建分支
- Commit 前缀: `feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `chore:`
- 分支命名: `feature/{op_name}`


# 可用 Skill

## `$opt-kernel` — CUDA Kernel 优化工作流

> 仅用于需要修改、迭代和优化 CUDA kernel 实现的性能优化任务。
> 调用方式：`$opt-kernel 优化 op/dwconv/example.cu:128 的 dwconv_forward_kernel。`
>
> 触发条件：用户显式调用 `$opt-kernel`，或明确要求优化、调优、迭代修改 kernel 以提升性能。仅出现“性能”“提升”“分析”等词不足以触发。

### 各阶段说明

| 阶段 | 输入 | 输出 |
|------|------|------|
| **0. 预检** | kernel 文件路径 | 仓库、GPU 和测试入口检查 |
| **1. 基线** | 测试配置 | 正确性与性能基线 |
| **2. 分析** | kernel 与基线数据 | 瓶颈分析报告 |
| **3. 规划** | 瓶颈分析报告 | 优先级实验计划 |
| **4. 实验** | 优化计划 | 单变量实验与有效提交 |
| **5. 验证** | 最终实现 | 完整回归与性能对比 |
| **6. 交付** | 验证结果 | 独立优化报告 |

详细步骤见 `.agents/skills/opt-kernel/SKILL.md`。Codex Skill 使用 `$opt-kernel` 显式调用，不使用 Claude 风格的 `/opt_kernel`。
