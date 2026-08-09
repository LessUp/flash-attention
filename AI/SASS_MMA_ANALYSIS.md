# 分析 HGMMA 指令的 SASS

## 导出 SASS

```bash
# Compile with cubin output
CUTE_DSL_KEEP_CUBIN=1 python -c "..."

# Find the cubin (saved in cwd with long name)
ls *.cubin

# Disassemble and extract HGMMA instructions
nvdisasm kernel.sm_90a.cubin | grep "HGMMA\."
```

## 阅读 HGMMA 指令

每条 `HGMMA.MxNxK.F32.BF16` 指令是一次 warpgroup 级 MMA：
- **M**：总是 64（一个 warpgroup = 128 线程，硬件固定）
- **N**：每条指令的输出列数（例如 96、128、192）
- **K**：BF16 下总是 16（每条指令一个 K-step）

指令中的关键字段：
```
HGMMA.64x96x16.F32.BF16 R72, gdesc[UR16], RZ, ...
                         ^^^  ^^^^^^^^^^^
                         |    operand A (see RS vs SS below)
                         destination register = accumulator
```

> 讲解：HGMMA 是 Hopper（SM90）上的 warpgroup 级矩阵乘指令（Blackwell 上对应 UMMA）。M=64 固定是因为一个 warpgroup 恰好 128 个线程、每 2 个线程处理一行结果。K=16 表示这条指令只推进 K 维的 16 个元素，完整的 K 归约要靠多条指令循环完成。

## RS 与 SS：解读源操作数

**第 2 个操作数**（operand A）告诉你 MMA 是从共享内存（SS）还是寄存器（RS）读取：

- **`gdesc[UR..]`** —— smem 描述符 → **SS 模式**（A 和 B 都来自共享内存）
- **`R<N>`**（普通寄存器）→ **RS 模式**（A 来自寄存器，B 来自共享内存）

`gdesc[UR..].tnspA.tnspB` 表示带转置操作数的 SS（用于 dV = P.T @ dO 这类输入是 smem 转置视图的 GEMM）。

### RS 何时有用

RS 减少 smem 流量。如果数据已经来自之前的计算并留在寄存器里（例如 softmax 反向的 dS），RS 直接把它喂给下一个 GEMM。

示例 —— `mma_dq_is_rs=True` 时的 dQ = dS @ K：
```
HGMMA.64x192x16.F32.BF16 R24, R232, gdesc[UR4], ...    # RS: A=R232 (dS in regs), B=gdesc (K in smem)
```
对比不带 RS：
```
HGMMA.64x192x16.F32.BF16 R24, gdesc[UR16], gdesc[UR4], ...   # SS: both from smem
```

> 讲解：矩阵乘的 A 操作数走寄存器（RS）还是共享内存（SS），决定了数据在核内要"绕行"多少趟。寄存器是每线程私有的，天然零额外带宽；共享内存则要显式写+读。代价是寄存器占用——所以 RS 适合"中间结果本来就握在寄存器里"的时机，比如 dS 由逐元素运算刚算出来，直接就地消费最省。

## 从 SASS 识别 GEMM

1. **连续多条 HGMMA 使用同一目标寄存器** = 累加到同一个输出 tile（沿 K 维迭代）
2. **同一目标寄存器的指令条数** × 16 = **归约（K）维度**
3. **交错模式中的不同目标寄存器** = 一个大的 MMA 被拆成多个 64 行部分（每个部分 M=64，总 M = 部分数 × 64）

### 示例：`dK = dS.T @ Q`，形状 192×96，K=64

192 行的输出被拆成 3 个 64 行的部分。SASS 显示：
```
HGMMA.64x96x16 dst=R120   # part 0, K-step 0
HGMMA.64x96x16 dst=R72    # part 1, K-step 0
HGMMA.64x96x16 dst=R24    # part 2, K-step 0
HGMMA.64x96x16 dst=R120   # part 0, K-step 1
HGMMA.64x96x16 dst=R72    # part 1, K-step 1
HGMMA.64x96x16 dst=R24    # part 2, K-step 1
...  (4 K-steps total)
```
- 3 个累加器（R120、R72、R24）→ M = 3 × 64 = 192
- 每个累加器 4 条指令 → K = 4 × 16 = 64

> 讲解：反向传播里 dK/dV 的输出形状是 (hdim, tile_n)，而单个 HGMMA 的 M 只有 64，所以 hdim=192 时会被自动拆成 3 个部分、轮流交替推进。用"同一目标寄存器出现次数 ×16"反推 K、用"不同目标寄存器数 ×64"反推 M，是逆向核对 GEMM 形状的快捷方法。

## 案例分析：BWD SM90，hdim=192，hdim_v=128，tile_m=64，tile_n=112

配置：`SdP_WGs=[0], dQ_WGs=[0], dK_WGs=[1], dV_WGs=[0]`，`mma_dq_is_rs=True`
- WG0：S、dP、dV、dQ（256 寄存器）
- WG1：仅 dK（224 寄存器）

### SASS HGMMA 拆分

```
#1-12   64x112x16  dst=R24              src=gdesc[UR..]           SS  ×12  →  S = Q @ K.T
#13-20  64x112x16  dst=R24              src=gdesc[UR..]           SS  ×8   →  dP = dO @ V.T
#21-28  64x112x16  dst=R176/R120 alt    src=gdesc[UR..].tnsp      SS  ×4ea →  dV = P.T @ dO
#29-35  64x192x16  dst=R24              src=R232..R248            RS  ×7   →  dQ = dS @ K
#36-47  64x112x16  dst=R136/R80/R24 cyc src=gdesc[UR8].tnsp       SS  ×4ea →  dK = dS.T @ Q
```

验证：

| GEMM | Atom | # Acc | I/acc | M | N | K | RS/SS | 核对 |
|------|------|-------|-------|---|---|---|-------|------|
| S = Q @ K.T | 64×112×16 | 1 | 12 | 64 | 112 | 12×16=192=hdim | SS | ✓ |
| dP = dO @ V.T | 64×112×16 | 1 | 8 | 64 | 112 | 8×16=128=hdim_v | SS | ✓ |
| dV = P.T @ dO | 64×112×16 | 2 | 4 | 2×64=128=hdim_v | 112 | 4×16=64=tile_m | SS | ✓ |
| dQ = dS @ K | 64×192×16 | 1 | 7 | 64 | 192=hdim | 7×16=112=tile_n | **RS** | ✓ |
| dK = dS.T @ Q | 64×112×16 | 3 | 4 | 3×64=192=hdim | 112 | 4×16=64=tile_m | SS | ✓ |

合计：47 条 HGMMA 指令（40 × 64×112×16 + 7 × 64×192×16）。

dQ 使用 RS 因为 `mma_dq_is_rs=True`：dS 由 SdP 的逐元素运算（P * (dP - dPsum)）在寄存器中计算，直接喂给 dQ GEMM，无需先写入共享内存。
