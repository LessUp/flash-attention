# SM90 块大小调优指南

如何为 Hopper（SM90）上的 FlashAttention 选择 tile 尺寸和 MMA 配置。

## 工具

用 `flash_attn/cute/sm90_config_search.py` 枚举可行配置：

```bash
# Both fwd and bwd
python flash_attn/cute/sm90_config_search.py --headdim 128

# Forward only
python flash_attn/cute/sm90_config_search.py --mode fwd --headdim 192-128

# Backward only, custom tile choices
python flash_attn/cute/sm90_config_search.py --mode bwd --headdim 192 --tile-m 64,80 --tile-n 64,96
```

> 讲解：这个工具是一个"配置枚举器"——把硬件约束（smem 容量、寄存器预算、GMMA 原子尺寸）编码成过滤条件，对给定 head 维度暴力列出所有可落地执行的 tile/MMA 配置组合，省去手工试错。

## 硬件约束（H100）

- **SMEM**：总共 228 KB。我们预留约 3 KB 给 LSE、dPsum 和 mbarrier，留给张量缓冲区 **224 KB**。
- **寄存器**：通过 `setmaxnreg` 控制。每个 MMA warp 组的预算：
  - 2 WG：每线程 240 寄存器，减去 24 开销 = **216 可用**
  - 3 WG：每线程 160 寄存器，减去 32 开销 = **128 可用**
- **GMMA 原子**：M 总是 64。swap 之后的有效 M 维必须能被 64 整除。N 维必须能被 `atom_layout_n * 8` 整除。

## 架构：Warp 组

每个 SM90 反向内核有 `num_wg + 1` 个 warp 组（每个 128 线程）：
- **WG0**（生产者）：Q、K、V、dO、LSE、dPsum 的 TMA 加载
- **WG1**（生产者）：dQaccum 存储（TMA reduce-add 到 gmem）
- **WG2..WG(num_wg)**（MMA 消费者）：所有 GEMM

前向内核：`num_wg` 个 MMA WG + 1 个生产者 WG。`tile_m = num_wg * 64`（无 swap）。

## 关键决策

### 1. Warp 组数量（num_wg）

| num_wg | tile_m (fwd) | Threads | Reg budget | Best for |
|--------|-------------|---------|------------|----------|
| 2 | 128 | 384 | 216/thread | hdim <= 128 |
| 3 | 192 | 512 | 128/thread | hdim 129-192 |

WG 越多 = tile_m 越大 = M 方向并行度越好，但寄存器预算更紧、smem 占用更高。

> 讲解：这里的"线程数"是整块 SM 上的线程（3 或 4 个 WG × 128）。WG 数量直接决定 M 维 tile：每个 WG 在 M 方向算 64 行，于是 tile_m = num_wg × 64。M 更大意味着每个输出 tile 覆盖更多查询行、K/V 数据复用率更高，但寄存器与共享内存压力同步上升——这是 FlashAttention 最经典的"访存复用 vs 资源占用"权衡。

### 2. swap_AB

每条 MMA 可以可选地交换它的 A 和 B 操作数。这会转置输出 tile，交换哪个维度映射到 M（必须能被 64 整除）、哪个映射到 N。

**何时 swap：**
- 天然 M 维不能被 64 整除而 N 可以（例如 SdP 的 tile_m=80）
- 想改变哪个操作数在寄存器里、哪个在共享内存里

**前向：** 不需要 swap，因为 tile_m = num_wg * 64 总是能被 64 整除。

**反向**（5 个 MMA）：
- **SdP**（S=Q@K^T, dP=dO@V^T）：输出 (tile_m, tile_n)。若 tile_m % 64 != 0 则 swap。
- **dKV**（dK=dS^T@Q, dV=P^T@dO）：输出 (tile_n, hdim/hdimv)。若 tile_n % 64 != 0 而 hdim % 64 == 0 则 swap。
- **dQ**（dQ=dS@K）：输出 (tile_m, hdim)。若 tile_m % 64 != 0 而 hdim % 64 == 0 则 swap。

### 3. AtomLayout

`atom_layout` 把 WG 分配到 MMA 输出的 M 和 N 维上。`num_wg` 个 MMA WG、`atom_layout_m = A` 时：
- M 方向：A 个 warp 组，每组处理 M/A 行
- N 方向：num_wg/A 个 warp 组，每组处理 N/(num_wg/A) 列

swap 之后 atom layout 也跟着 swap。

**对 smem 流量的影响**：N 方向 WG 更多（`wg_n` 更大）意味着每条指令读更小的 B 切片，但总共有更多指令读重叠的 A 切片。N 方向 WG 更少（`wg_n` 更小）意味着指令更少但每条读更大的 B 切片。通常 **wg_n 更小 = 总 smem 流量更少**。

### 4. mma_dkv_is_rs（dKV 的寄存器源）

当 `AtomLayoutMSdP == 1 && AtomLayoutNdKV == num_wg && SdP_swapAB && !dKV_swapAB` 时，P 和 dS 矩阵可以留在寄存器里，直接作为 dV 和 dK GEMM 的 A 操作数。这：
- **把 sP 从 smem 中消除**（节省 tile_m * tile_n * 2 字节）
- **从 smem 流量中消除 P 的 R2S 存储**
- **消除 dK 和 dV GEMM 的 A 操作数读取**

这是显著的优化——条件满足时总是首选。

> 讲解：R2S 指"寄存器到共享内存（register-to-shared）"的落盘。这条决策本质上是让中间矩阵 P（softmax 后的概率矩阵）和 dS 完全不经过共享内存：谁需要它、谁就在寄存器里直接消费。省下的不仅是 smem 容量，还有一整轮"写 + 读"的带宽。

### 5. 流水线分段（Pipeline Staging）

**前向**：
- Q：1 个 stage（每个 n_block tile 只加载一次）
- K、V：2 个 stage（双缓冲，与 TMA 流水线化）
- O：在 smem 中与 Q 重叠（epilogue 时复用同一缓冲区）

**反向**：
- Q：总是 2 个 stage（双缓冲）
- dO：smem 允许时 2 个 stage（与 Q 流水线对齐），否则 1 个 stage
- PdS：1 个 stage
- K、V：常驻 smem（每个 n_block 只加载一次）

## 寄存器核算

每线程每 WG 的累加器寄存器数 = `M * N / (num_wg * 128)`，其中 M x N 是输出 tile。

**前向峰值寄存器**：
- 有 WG 重叠：`regs_S + regs_P + regs_O`（S、P 为 bf16，O 全部存活）
- 无重叠：`regs_S + regs_O`（S 和 O 交替，P 复用 S 的寄存器）

其中 `regs_P = regs_S / 2`（bf16 对 f32）。

**反向峰值寄存器**：
- `max(2 * regs_SdP, regs_dQ) + regs_dK + regs_dV`
- S 和 dP 累加器同时存活（计算 dP 时 S 仍被 softmax 需要）
- S+dP 被消费后 dQ 复用它们的寄存器空间
- dK 和 dV 跨 m_block 迭代累加

## SMEM 核算

张量缓冲区之和（忽略小的对齐填充）：

**前向**：`max(sQ, sO) + sK*2 + sV*2 + sP`
- sQ = tile_m * hdim * 2
- sK = tile_n * hdim * 2 * 2 stages
- sV = tile_n * hdimv * 2 * 2 stages
- sO = tile_m * hdimv * 2（与 sQ 重叠）
- sP = tile_m * tile_n * 2（RS 时为 0）

**反向**：`sQ*2 + sK + sV + sdO*dO_stage + sP + sdS + sdQaccum`
- sQ = tile_m * hdim * 2 * 2 stages
- sK = tile_n * hdim * 2
- sV = tile_n * hdimv * 2
- sdO = tile_m * hdimv * 2 * dO_stage
- sP = tile_m * tile_n * 2（mma_dkv_is_rs 时为 0）
- sdS = tile_m * tile_n * 2
- sdQaccum = tile_m * hdim * 4（f32）

## SMEM 流量

每轮迭代消耗的 smem 带宽。每条 GMMA 指令读取：
- **A 操作数**：64 * K_red * 2 字节（寄存器源时为 0）
- **B 操作数**：(N_eff / wg_n) * K_red * 2 字节

总指令数 = (M_eff / 64) * wg_n。每条指令独立从 smem 读 A 和 B。

额外流量：P、dS 的 R2S 存储（bf16），dQ 的 smem 存储 + TMA 加载（f32）。

**每块流量**（traffic / (tile_m * tile_n)）归一化后可在不同 tile 尺寸间比较。越低越好。

> 讲解：为什么用"每块流量"而不是绝对流量？tile 越大、K/V 复用率越高，但每块吞吐的绝对数值也随之变大。除以 tile 面积（tile_m × tile_n）之后，不同尺寸才在同一条"每输出元素成本"标尺上可比——这正是选型时判断哪个配置访存效率高的依据。

## 示例配置

### hdim=128（前向）
最佳：tile_m=128，tile_n=192，RS，2 WG。224K smem，9.3 tr/blk。

### hdim=128（反向，非因果）
C++ FA3 配置：tile_m=80，tile_n=128，SdP_swap=T，dKV_swap=F，dQ_swap=T，aSdP=1，adKV=2。mma_dkv_is_rs=True。204K smem，208 regs，39.6 tr/blk。

### hdim=192（反向）
3 WG，tile_m=64，tile_n=96，SdP_swap=F，dKV_swap=T，adKV=1 或 3。216K smem，128 regs。这是 hdim=192 下唯一可行的 tile_n > 64，原因是寄存器压力。

### hdim=192，hdimv=128（DeepSeek 形状）
3 WG 时：需要 AtomLayoutNdKV=3（因为 hdimv=128 不能被 3 整除）。tile_n=96，212K smem。
2 WG 时：tile_n=112 在 210K smem 下可行，或 tile_n=64 在 168K smem 下可行。
