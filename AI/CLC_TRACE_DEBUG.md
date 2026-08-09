# CLC 追踪调试

当怀疑 CLC 工作调度器做出了令人意外的 tile 分配决定、想从当前内核拿到原始调度器追踪时使用。

> 讲解：CLC（Cluster Launch Control，集群启动控制）是 Blackwell SM100 上由硬件/驱动辅助的动态 tile 调度机制，能把多个逻辑工作块动态分配给空闲的 CTA，用于缓解静态 tile 分配带来的负载不均衡。下面的 `[CLC] query ...` 日志就是内核里调度器 warp 每次"查询下一个该做什么"时留下的痕迹。

## 当前追踪格式

SM100 前向内核在 `FA_LOG_LEVEL=3` 下，每个调度器 warp 查询打印一行：

```text
[CLC] query sm=<smid> cta=<blockIdx.x> (m_blk=<m>,h=<h>,b=<b>,s=<s>) valid=<0|1>
```

当前的输出点：
- `flash_attn/cute/flash_fwd_sm100.py`
- `flash_attn/cute/flash_fwd_mla_sm100.py`

## 如何捕获追踪

注意：
- `FA_LOG_LEVEL=3` 是 `[CLC] query ...` 设备端打印所必需的。
- `FA_CLC=1` 只是请求 CLC；如果形状/特性禁用它，内核仍可能回退。

最小复现模式：

```bash
FA_LOG_LEVEL=3 FA_CLC=1 CUDA_VISIBLE_DEVICES=0 python - <<'PY' \
  > agent_space/clc_trace.log 2>&1
import torch
from flash_attn.cute.interface import flash_attn_func

torch.manual_seed(0)
q = torch.randn(1, 512, 16, 128, device='cuda', dtype=torch.bfloat16)
k = torch.randn(1, 512, 1, 128, device='cuda', dtype=torch.bfloat16)
v = torch.randn(1, 512, 1, 128, device='cuda', dtype=torch.bfloat16)
flash_attn_func(q, k, v, causal=True)
torch.cuda.synchronize()
PY
```

如果希望运行明确显示是否选中了 CLC，也保留主机日志前缀：

```text
[FA] TileScheduler=SingleTileLPTScheduler, scheduling_mode=CLC, USE_2CTA=False
```

## 该看什么

- 主机日志里的 `scheduling_mode=CLC` 确认该形状确实走了 CLC 路径。
- `valid=1` 表示返回的工作 tile 有效。
- `valid=0` 表示对该 CTA/调度器 warp 查询而言调度器已耗尽。
- `m_blk`、`h`、`b`、`s` 是调度器映射之后的逻辑工作坐标。
- `cta` 是物理 `blockIdx.x`；集群（cluster）启动时，多个 CTA 可能参与同一个逻辑 tile。

## 解析追踪

轻量解析器在 `AI/parse_clc_log.py`。

文本摘要：

```bash
python AI/parse_clc_log.py agent_space/clc_trace.log
```

HTML 视图：

```bash
python AI/parse_clc_log.py agent_space/clc_trace.log --html -o agent_space/clc_trace.html
```

## 建议工作流

1. 用 `FA_LOG_LEVEL=3 FA_CLC=1` 复现那个意外的情况。
2. 把 stdout/stderr 存到 `agent_space/clc_trace.log`。
3. 对该日志运行 `AI/parse_clc_log.py`，得到紧凑的按 SM / 按 CTA 摘要。
4. 如果追踪仍可疑，把日志贴到调查线程 / agent 笔记里。
5. 与 `flash_attn/cute/tile_scheduler.py` 中相关的映射逻辑对照。

## 注意事项

- 追踪又吵又贵；先只用一个小的形状。
- 因为打印发生在调度器查询时，工作耗尽后会有大量终态的 `valid=0` 查询行。
- 根据 `flash_attn/cute/interface.py` 当前的启发式，稠密非因果和 varlen MHA 可能有意回退、不走 CLC。
