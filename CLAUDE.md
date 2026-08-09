# CLAUDE.md

本文件为 Claude Code（claude.ai/code）在此代码库中工作时提供指导。

## 项目概述

FlashAttention-4（FA4）——用 Python 编写的快速、内存高效的精确注意力 kernel，基于 CuTeDSL（NVIDIA CUTLASS DSL）。Kernel 在运行时编译为 PTX/CUBIN。目标 GPU 为 Hopper（SM90）和 Blackwell（SM100/SM110）。包名：`flash-attn-4`。

本仓库还包含旧代实现（顶层 `csrc/` 中的 FA2、`hopper/` 中的 FA3），但当前活跃开发集中在 `flash_attn/cute/` 下的 FA4。

## Agent 草稿空间

使用 `agent_space/` 存放项目内的临时草稿，例如实验笔记、性能分析输出、临时复现脚本和实验产物。把它当作一次性工作区，而不是产品代码。

## 构建与安装

```bash
pip install flash-attn-4
# 或开发安装：
pip install -e "flash_attn/cute[dev]"
```

依赖：`nvidia-cutlass-dsl>=4.5.2`、`torch`、`einops`、`apache-tvm-ffi`、`quack-kernels>=0.5.0`。

## 运行测试

```bash
pytest tests/cute/test_flash_attn.py
pytest tests/cute/test_flash_attn.py -k "test_flash_attn_output" -x  # 单个测试
pytest tests/cute/test_flash_attn_varlen.py
pytest tests/cute/test_mask_mod.py
pytest tests/cute/test_score_mod.py
pytest tests/cute/test_block_sparsity.py
```

### 快速两遍测试法

编译主导了测试时间。快速工作流把编译（并行、无需 GPU）与执行（使用缓存二进制）分开：

```bash
# 第一遍：使用 FakeTensorMode 并行编译所有 kernel（不分配 GPU 内存）
FLASH_ATTENTION_FAKE_TENSOR=1 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 pytest -n 64 -x tests/cute/test_flash_attn.py

# 第二遍：使用缓存的已编译 kernel 运行测试
FLASH_ATTENTION_FAKE_TENSOR=0 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 pytest -x tests/cute/test_flash_attn.py
```

- `FLASH_ATTENTION_FAKE_TENSOR=1` — 使用 PyTorch FakeTensorMode 编译 kernel，不分配 GPU 内存、不真正运行。
- `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1` — 在 `/tmp/${USER}/flash_attention_cute_dsl_cache/` 启用持久化磁盘缓存。
- `-n 256` — pytest-xdist 并行 worker（仅在编译一遍中有用）。

测试按 dtype（fp16/bf16）、head 维度（64、96、128）、序列长度、causal/非 causal 以及 MHA/GQA/MQA 参数化。

如果运行测试或基准时出现 OOM 错误，用 `nvidia-smi` 找一块空闲 GPU，并用 `CUDA_VISIBLE_DEVICES=<id>` 选择它。

## 代码检查（Linting）

Pre-commit 对 `flash_attn/cute/` 下的文件使用 ruff。大型 kernel 文件（`flash_bwd.py`、`flash_fwd.py`、`flash_fwd_sm100.py`、`interface.py`）被排除在自动格式化之外。

```bash
ruff check flash_attn/cute/ --fix
ruff format flash_attn/cute/
```

## 代码架构

### 公共 API（`flash_attn/cute/interface.py`）

从 `flash_attn/cute/__init__.py` 导出的两个入口：
- `flash_attn_func(q, k, v, ...)` — 标准注意力
- `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)` — 变长序列

关键参数：`causal`、`window_size_left/right`、`softmax_scale`、`softcap`、`score_mod`、`mask_mod`、`block_sparse_tensors`、`num_splits`、`pack_gqa`、`m_block_size`、`n_block_size`、`num_threads`。

张量布局：`(batch, seqlen, num_heads, head_dim)`，最后一维连续、16 字节对齐。

### 前向 Kernel

- `flash_fwd.py` — `FlashAttentionForwardSm90`：Hopper 前向。无 SplitKV 或 paged KV。
- `flash_fwd_sm100.py` — `FlashAttentionForwardSm100`：Blackwell 前向。完整特性，包括 SplitKV、paged KV cache、持久化 kernel、2CTA 指令。
- `flash_fwd_combine.py` — `FlashAttentionForwardCombine`：合并 SplitKV 的部分结果。

### 反向 Kernel

- `flash_bwd.py` — `FlashAttentionBackwardSm80`：Ampere 反向（基础版）。
- `flash_bwd_sm90.py` — `FlashAttentionBackwardSm90`：Hopper 反向。
- `flash_bwd_sm100.py` — `FlashAttentionBackwardSm100`：Blackwell 反向，支持 2CTA 和 block sparse。
- `flash_bwd_preprocess.py` / `flash_bwd_postprocess.py` — 辅助反向 kernel。

### 核心抽象

- `softmax.py` — 带 row_max/row_sum 追踪的 online softmax，支持 score modifier。
- `mask.py` — `AttentionMask`：causal、局部/滑窗、block sparse、mask_mod 应用。
- `block_info.py` — `BlockInfo`：tile 维度、causal/local masking 的 n/m block 范围计算。
- `seqlen_info.py` — `SeqlenInfoQK`：varlen 的序列长度与偏移追踪。
- `pipeline.py` — `PipelineStateSimple`：流水线加载的循环缓冲区索引/阶段管理。
- `tile_scheduler.py` — tile 调度策略（单 tile、varlen 感知、持久化）。
- `copy_utils.py` — 类型转换拷贝、shared 到 register 的加载、TMA 拷贝原子操作。
- `named_barrier.py` — warp 同步的命名屏障枚举。

### 架构专属辅助

- `hopper_helpers.py` — SM90 warp-group GEMM、shared memory 布局创建、fence/commit/wait。
- `blackwell_helpers.py` — SM100 UMMA 的 GEMM、PTX 优化路径、2CTA 支持。
- `mma_sm100_desc.py` — 硬件 MMA 描述符枚举（格式、饱和、缩放）。

### 其他组件

- `pack_gqa.py` — 为高效 GQA 打包多个 Q head 到每个 KV head。
- `paged_kv.py` — `PagedKVManager`：带 TMA 支持的 paged KV cache。
- `fast_math.py` — exp2 多项式系数、softcap score_mod 创建。
- `utils.py` — 编译缓存键的哈希函数、warp 归约、谓词。
- `cache_utils.py` — JIT 编译缓存管理。
- `cute_dsl_utils.py` — 打补丁的 `cute.compile`，可选择性导出 SASS。

### 编译与缓存

Kernel 是 JIT 编译的。缓存键包含 dtype、head_dim、causal、mask/score_mod 哈希、架构、block 大小。缓存层级：内存 LRU + 可选的磁盘缓存（通过 `get_jit_cache()`）。

环境变量：`CUTE_CUBIN_PATH`（导出 CUBIN/SASS）、`CUTE_DSL_KEEP_PTX=1`（检查 PTX）、`CUTE_DSL_PTXAS_PATH`（自定义 ptxas）。

## 关键模式

- 编译期常量使用 `cutlass.Constexpr[type]` 做 kernel 特化。
- Score/mask modifier 是用户定义的 `@cute.jit` 可调用对象，在编译期注入 kernel。
- 前向执行：加载 Q tile → 循环 K/V block（流水线）→ online softmax 累积 → 存储 O 和 LSE。
- 2CTA 指令（SM100，hdim=128）：集群内两个 CTA 通过 shared mbarrier 协调；tx_count 必须乘以 `cta_group_size`。

## 调试 GPU Kernel

**在为任何在 CuteDSL 源码中不可见的挂起、死锁、非法地址陷阱、Xid 故障、sanitizer 报告或数值不匹配提出根因之前，先阅读 `AI/DEBUG_METHODOLOGY.md` 并遵循其协议**（可证伪预测的纪律、证据层级、修复验证卫生、`agent_space/` 中的假设台账）。

`AI/` 中的战术文档：
- `DEBUG_2CTA.md` — kernel 挂起/死锁调试（printf 二分、pipeline barrier 分析、2CTA 陷阱）。
- `RACECHECK_TMA_HAZARD.md` — 使用 `cp.async.bulk` 时 `compute-sanitizer` 的误报（复现脚本：`racecheck_repro_1d_*.py`）。
- `CLC_TRACE_DEBUG.md` — CLC 调度的可视化（`parse_clc_log.py`）。
- `SASS_MMA_ANALYSIS.md` — 导出 SASS 并分析 HGMMA 指令组合。
- `SM90_BLOCK_SIZE_TUNING.md` — 在 Hopper 上选择 tile 大小/MMA 配置（`sm90_config_search.py`）。
- `SM90_R2P_MASKING_SASS.md` — SM90 前向中 R2P 谓词掩码的 SASS 级分析。
- `VARLEN_PREPROCESS_TILE_BUG.md` — 事后分析：varlen preprocess tile 大小不匹配与填充偏移布局。

关键工具：
- `cute.printf` 配合线程守卫（`tidx % 32 == 0`、`elect_one()`）做定向输出
- `compute-sanitizer --tool=racecheck`（注意原始 TMA 的误报）
- `CUTE_DSL_KEEP_PTX=1` 和 `CUTE_DSL_LINEINFO=1` 用于 PTX 检查与 sanitizer 源码映射
