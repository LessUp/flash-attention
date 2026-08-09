# compute-sanitizer racecheck 在 `cp.async.bulk` 上的误报

## 摘要

当 `cp.async.bulk`（裸地址 TMA）被用于动态循环内跨 warp 的生产者/消费者流水线时，`compute-sanitizer --tool=racecheck` 会报告共享内存竞争误报。同样的模式改用 `cp.async.bulk.tensor`（基于描述符的 TMA）则报告 **零 hazard**。

flash 反向内核的修复方案是：把 LSE/dPsum 的拷贝从 `CopyBulkG2SOp`（`cp.async.bulk`）切换到 `CopyBulkTensorTileG2SOp`（`cp.async.bulk.tensor`），使用 `cpasync.make_tiled_tma_atom`。

## 受影响的代码

`flash_attn/cute/flash_bwd_sm100.py` —— SM100 反向 attention 内核。

只有 **LSE** 和 **dPsum** 缓冲区受影响，因为它们是仅有的、被线程级共享内存读（`lds`）消费的 TMA 加载缓冲区。Q/K/V/dO 由 UMMA 硬件指令消费，不产生线程级 `lds`，因此从不触发 racecheck。

> 讲解：LSE（log-sum-exp，对数求和指数）和 dPsum 是反向传播中 softmax 校正所需的小型中间量，由线程显式从共享内存读出再做逐元素运算；而 Q/K/V/dO 是被 UMMA（Blackwell 的矩阵乘硬件单元）直接从共享内存消费的，sanitizer 无法插桩硬件单元的访问，自然不会报告竞争——这正好解释了为什么误报只出现在 LSE/dPsum 上。

## 根因

racecheck 会对每次共享内存访问插桩，并检查是否存在缺少可识别 happens-before 关系的冲突访问。

**`cp.async.bulk`（裸地址）：** sanitizer 把 smem 写入归因于发起线程（通过 `elect_one` 的 warp 0 线程 0）。当 warp 1 从相同地址发出 `ld.shared.b32` 时，sanitizer 会寻找 happens-before 边。唯一的同步是 warp 1 上的 `mbarrier.try_wait.parity` 配合硬件发来的 `mbarrier::complete_tx::bytes` 完成。sanitizer 不把这在动态循环中建模为跨 warp 的 happens-before。

**`cp.async.bulk.tensor`（TMA 描述符）：** TMA 引擎是一个独立的硬件单元。sanitizer 不把 smem 写入归因于任何线程。没有写者线程就没有竞争对，因此不报告 race。

> 讲解：这就是"误报"的关键。racecheck 基于"指令级 happens-before"模型来推断数据依赖，它不认识"TMA 硬件在屏障完成后才真正落数据"这个异步语义。当写入被归因到某个线程时，它需要一个可识别的同步原语作桥；而对描述符 TMA，写入没有归因线程，分析器干脆跳过——无论哪种情况，报告都反映的是分析模型的盲区，而不是真实的数据竞争。

### 指令对比

| 变体 | PTX | racecheck |
|---------|-----|-----------|
| 裸地址（cta 作用域） | `cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes` | **hazard** |
| 裸地址（cluster 作用域） | `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes` | **hazard** |
| 描述符 1D | `cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes` | 干净 |
| 描述符 2D | `cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes` | 干净 |

### `--racecheck-memcpy-async=no` 没有帮助

这个开关控制更老的 `cp.async`（sm80）指令家族，不是 `cp.async.bulk`。加上 `--racecheck-memcpy-async=no` 后 hazard 依然存在。

## 证明它是误报

1. **数据正确性** —— 所有变体产生逐位一致的结果。
2. **单 warp 测试** —— 一个 warp 在同一个循环里既做 TMA 写又做线程读；在相同的 mbarrier 同步下 racecheck 报告零 hazard。
3. **展开循环** —— 完全展开（`unroll_full=True`）报告零 hazard；racecheck 能在直线代码里跟踪 mbarrier，但无法跨 warp 间的动态分支回边跟踪。
4. **命名屏障** —— 在生产者与消费者 warp 之间每轮迭代加 `bar.sync` 消除 hazard；这个同步是正确的，racecheck 只是需要一个它认识的基元。
5. **描述符 TMA** —— 用相同流水线代码切到 `cp.async.bulk.tensor` 后 hazard 消失；mbarrier 协议是正确的。

## 最小复现

### `AI/`（推荐，更干净）

| 文件 | 拷贝指令 | 结果 |
|------|-----------------|--------|
| `racecheck_repro_1d_bulk.py` | `cp.async.bulk`（裸地址） | **1 error** |
| `racecheck_repro_1d_tensor.py` | `cp.async.bulk.tensor.1d`（TMA 描述符） | **0 hazards** |

两者都是约 75 行的自包含内核：2 个 warp、4 个 block、带 `PipelineTmaAsync` 的 2 阶段双缓冲。流水线协议完全相同——只有拷贝指令不同。

```bash
python AI/racecheck_repro_1d_bulk.py                                              # correctness
CUTE_DSL_LINEINFO=1 compute-sanitizer --tool=racecheck python AI/racecheck_repro_1d_bulk.py   # 1 error
compute-sanitizer --tool=racecheck python AI/racecheck_repro_1d_tensor.py         # 0 hazards
```

### `benchmarks/`（更早，变体更多）

| 文件 | 测试内容 | 结果 |
|------|--------------|--------|
| `racecheck_false_positive_repro.py` | 跨 warp 循环里的 `cp.async.bulk` + mbarrier | 1 error |
| `racecheck_1d_raw_ptx.py` | 内联 PTX `cp.async.bulk.shared::cta.global` | 1 error |
| `racecheck_tma2d_repro.py` | 经 `make_tiled_tma_atom` 的 `cp.async.bulk.tensor.2d` | 0 hazards |
| `racecheck_tma1d_descriptor.py` | 经 `make_tiled_tma_atom` 的 `cp.async.bulk.tensor.1d` | 0 hazards |

## PTX 级分析

为两个 `AI/` 复现脚本导出了 PTX（`CUTE_DSL_KEEP_PTX=1`）。除单条拷贝指令外，生成的代码逐字节相同：

```
# racecheck_repro_1d_bulk.py  (HAZARD)
cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes
    [%r42], [%rd12], %r43, [%r6+-16];

# racecheck_repro_1d_tensor.py  (CLEAN)
cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint
    [%r43], [%rd1, {%r71}], [%r6+-16], %rd8;
```

所有 mbarrier 操作（init、`fence.mbarrier_init.release.cluster`、`arrive.expect_tx`、`try_wait.parity`、`arrive.release`、`fence.proxy.async.shared::cta`、`bar.warp.sync`）完全相同。

### racecheck 错误输出

```
Error: Race reported between Write access at ...+0x430 in racecheck_repro_1d_bulk.py:46
    and Read access at ...+0x770 in racecheck_repro_1d_bulk.py:55 [248 hazards]
    and Read access at ...+0x7a0 in racecheck_repro_1d_bulk.py:55 [248 hazards]
    and Read access at ...+0x7d0 in racecheck_repro_1d_bulk.py:55 [248 hazards]
    and Read access at ...+0x800 in racecheck_repro_1d_bulk.py:55 [248 hazards]
```

- **写**（0x430）= 第 46 行：`cute.copy(atom, src, s, mbar_ptr=...)` —— `cp.async.bulk` 指令
- **读**（0x770–0x800）= 第 55 行：`dst[...] = s[...]` —— 消费者 warp 中的四条 `ld.shared.b32`

## 修复

把 load 函数里的 `copy_stats` 从：

```python
copy_atom_stats = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), Float32)
copy_stats = partial(cute.copy, copy_atom_stats)
```

改为使用 `cpasync.make_tiled_tma_atom` 加 `CopyBulkTensorTileG2SOp` 的描述符 TMA。这会生成 `cp.async.bulk.tensor.1d` 而不是 `cp.async.bulk`，后者 racecheck 不插桩。

流水线协议（mbarrier init、arrive_expect_tx、try_wait_parity、consumer_release）保持不变。

## 备选方案

`flash_attn/cute/flash_bwd_sm100_gmem_fix.py` 含有一个可行但更慢的修复：计算 warp 直接从全局内存读取 LSE/dPsum，完全绕过 TMA smem 流水线。

## 调查时间线

1. 在 `flash_bwd_sm100.py` 的 LSE 和 dPsum 上观察到 2 个 racecheck 错误。Q/K/V/dO 干净。
2. 注意到 Q/K/V/dO 使用 UMMA 消费者（没有线程 `lds`），而 LSE/dPsum 从 smem 做线程级 `autovec_copy`——这解释了为什么只有 LSE/dPsum 触发。
3. 构建了复现该 hazard 的最小 2-warp 流水线内核。
4. 单 warp 版本干净——同样的 mbarrier，同样的地址。
5. 完全展开版本干净——racecheck 能在直线代码内跟踪 mbarrier。
6. 每轮迭代加 `bar.sync` 修复了它——racecheck 需要它在循环回边上能识别的同步。
7. `cp.async.bulk.tensor.2d` 干净——指令不同，流水线相同。
8. `cp.async.bulk.tensor.1d` 干净——问题是裸地址 vs 描述符，与维度无关。
9. 裸内联 PTX `cp.async.bulk.shared::cta.global` 同样触发——不是 CuTe DSL 抽象层的问题。
10. 为两个 `AI/` 复现脚本导出 PTX——确认除拷贝指令外代码逐字节一致。对 `cp.async.bulk`，sanitizer 把 smem 写入归因于发起线程；对 `cp.async.bulk.tensor` 则不归因。
11. 确认 `--racecheck-memcpy-async=no` 不能抑制该 hazard——该开关针对更老的 `cp.async`，不是 `cp.async.bulk`。
