# 在 CUTLASS DSL / 2CTA 内核中调试 GPU 内核挂起（死锁）

> 背景：2CTA（双 CTA）指两个 CTA 组成一个 cluster，协同计算同一个输出 tile——这是利用 Hopper/Blackwell 集群特性、为 FlashAttention 提升计算密度的常见模式。两个 CTA 共享部分 mbarrier 与集群级共享内存语义，也因此引入了下文这些独有的陷阱。本文适用于所有用 CuteDSL 编写的流水线内核，2CTA 部分是集群模式下特有的坑。

## 调试内核挂起的一般方法

### 第 1 步：构建最小复现

把测试用例裁剪到能触发挂起的最小输入：
- batch=1, nheads=1, 触发挂起的最小 seqlen
- 单一配置，无循环，无基准测试
- 加超时或用 `compute-sanitizer` 运行，以便区分挂起与慢执行

### 第 2 步：加 printf 定位挂起

GPU `printf`（`cute.printf`）是主要工具。目标是二分搜索：缩小到哪个 warp、哪个操作被阻塞。

**printf 守卫**——避免打印风暴：
```python
# One thread per warp:
if cute.arch.thread_idx()[0] % 32 == 0:
    cute.printf("...")

# One thread per CTA (elect_one is a context manager, not a bool):
with cute.arch.elect_one():
    cute.printf("...")

# One specific thread:
if tidx == 0:
    cute.printf("...")
```

**策略——从粗到细：**
1. 先在每个 warp 主函数（load、mma、softmax、correction）的入口/出口打印。这告诉你哪个 warp 卡住了。
2. 然后在每个流水线等待（`consumer_wait`、`producer_acquire`）前后加打印。这告诉你哪个屏障卡住了。
3. 再打印屏障索引、phase 和 stage，理解流水线状态。

**打印什么：**
- CTA 索引（`cute.arch.block_idx()[0]`）——多 CTA 调试的关键
- 流水线 stage 索引和 phase
- 循环迭代计数
- `try_wait` 是否成功（使用 `try_wait_token` 参数）

### 第 3 步：识别死锁链

挂起永远是环。流水线内核中的典型链条：

```
MMA waiting for K from load (pipeline_kv full barrier)
  -> Load finished but stuck in producer_tail (waiting for MMA to release empty barrier)
    -> MMA can't release because it's waiting for K
```

一旦看到哪个屏障卡住，就反向追踪：谁应该给它发信号？为什么还没发？

### 第 4 步：系统性地变化问题规模

用不同的序列长度/块数测试，找出规律：

| seqlen | n_blocks | 结果 |
|--------|----------|------|
| 128    | 1        | ?    |
| 256    | 2        | ?    |
| 384    | 3        | ?    |
| 512    | 4        | ?    |

如果挂起与对某个流水线 stage 的访问次数相关（例如 n_blocks <= kv_stages 时正常，stage 绕回时失败），问题很可能出在屏障 tx_count 或 phase 跟踪上。

### 第 5 步：检查屏障字节数（tx_count）

对基于 TMA 的流水线，`arrive_and_expect_tx` 在 mbarrier 上设置期望的事务字节数。如果期望计数与实际到达的字节不匹配，屏障要么：
- 过早触发（期望 < 实际）——导致数据竞争
- 永不触发（期望 > 实际）——导致挂起

在 **2CTA / cluster 模式** 下，两个 CTA 的 TMA 都会给 **同一个** cluster 级 mbarrier 发信号。如果每个 CTA 的 TMA 贡献 N 字节，那么屏障总共收到 2N 字节。tx_count 必须是 `N * cta_group_size`，而不是 `N`。

> 讲解：`cta_group_size` 是 cluster 中的 CTA 数量。2CTA 模式下两个 CTA 共享同一套 mbarrier，因此生产者侧登记的期望字节数必须按整个 cluster 的写入总量来算；少算一半的话，屏障会在实际数据到达前就触发，消费者读到半成品。

**所有 TMA 流水线都需要加倍**——Q、K、V 都是。即使每个 CTA 加载的 Q M-tile 不同，两个 CTA 的 TMA 操作仍然给同一个 cluster 级屏障发信号，所以期望字节数必须把两者都算上。

### 第 6 步：检查 phase / 奇偶跟踪

`mbarrier_try_wait_parity` 使用单个奇偶位（0 或 1）。如果你的流水线状态把 phase 跟踪为单调递增的计数器（0, 1, 2, 3, ...），在把它传给屏障等待前需要做 `phase % 2`。否则硬件会把 phase=2 看成 phase=0，可能导致等待一个已经完成的屏障，或错过一个待完成的屏障。

> 讲解：mbarrier 的完成状态只有一个 bit，靠"每次 arrive 后奇偶翻转"来区分新旧阶段。所以软件里的阶段计数器必须折叠到 0/1 再传给硬件，否则隔一次就误判。

### 第 7 步：警惕"编译器即 bug 源"

如果内核 **加** printf 就能工作、去掉就挂起，那么 printf 充当了 **编译器屏障**。MLIR/LLVM 后端无法穿透像 printf 这样的不透明函数调用做优化，这阻止了有害的指令重排。

出现这些迹象：
- 在正确函数里加一个 `cute.printf("\n")` 就修好了挂起
- PTX fence（`fence_view_async_shared`、`fence_acq_rel_cluster`、`sync_warp`、`fence_proxy`）无法修复——它们影响硬件内存序，不影响编译器调度
- 修复对位置敏感（printf 在某个函数里有效，在另一个里无效）

可能的变通方案：
- 在流水线方法上加 `@dsl_user_op` 装饰器，使其对编译器不透明
- `asm volatile` 屏障（如果 DSL 支持）
- 对比带/不带 printf 生成的 PTX/SASS，找出编译器重排了什么
- 向 CUTLASS DSL / MLIR 流水线提交 bug

> 讲解：这类问题最迷惑人——同样的源码，只是插了条打印就"修好"了挂起。这往往意味着真正的缺陷是"编译器对异步共享内存访问的调度违反了 mbarrier 语义"，即代码生成级 bug，而不是内核逻辑本身的 bug。此时纠结 PTX fence 没有用，因为它们约束的是硬件内存序，约束不了编译器的指令重排。

---

## 2CTA 特有陷阱

### tcgen05.commit 与空 commit group

`tcgen05.commit(mbar, mask, cta_group::2)` 本应在所有待处理的 MMA 完成后给 mbarrier 发信号。但如果 **没有待处理的操作**（空的 commit group），信号只到达本地 CTA 的屏障，不会到达远端 CTA 的。修复：用显式 `mbarrier_arrive(barrier, dst_cta_rank)` 给两个 CTA 都发信号。

> 讲解：`tcgen05` 是 Blackwell 张量核心的异步 MMA 引擎；`tcgen05.commit` 让硬件在流水线排空后自动完成屏障。但在"空提交"时硬件没有工作可等，完成信号只落在本 CTA，跨 CTA 的同步就漏了——这时必须走显式 arrive。

### producer_tail 死锁

默认的 `producer_tail`（继承自 sm90 流水线）通过循环调用 `producer_acquire` 来排空流水线。在 2CTA 模式下这会死锁，因为消费者（MMA warp）可能已经退出而没有释放所有 stage。修复：让 2CTA 的 `producer_tail` 成为 no-op。

### Tile 调度器必须考虑 cluster 形状

cluster 里的两个 CTA 必须得到 **同一个** tile 坐标。裸 `blockIdx.x` 给同一 cluster 内的 CTA 分配连续值。修复：用 `blockIdx.x` 除以 `cluster_shape_m`。

### 跨 CTA 与每 CTA 的流水线

CTA 1 的线程远程 arrive 到 CTA 0 屏障的流水线，需要 cluster 规模的协作组计数。完全限定在单个 CTA 内部的流水线保持每 CTA 计数。

### Softmax 掩码偏移

因果掩码的行位置必须考虑 CTA 在 cluster 内的位置。计算掩码坐标时把 `m_block` 乘以 `cta_group_size`。
