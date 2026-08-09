# Varlen 预处理 tile 不匹配 bug

## 摘要

`SeqlenInfo.create` 在 `flash_bwd_preprocess.py` 中默认 `tile=128`，但反向内核使用 `tile_m=m_block_size`（例如因果 SM90 为 64）。这导致预处理在 batch 0 之后的所有 batch 上，把 dq_accum 清零并写到错误的 padded offset 上（lse_log2/dpsum 同理）。

## padded_offset 如何工作

对 varlen，dq_accum 这类缓冲区按 tile 对齐的间隔在序列之间留空隙：

```
padded_offset_q = ((offset_q + batch_idx * tile_m) // tile_m) * tile_m
```

空隙大小取决于 `tile_m`。`tile_m=64` 对比 `tile_m=128`，batch 1 在 `offset_q=128` 时：
- tile=64:  padded_offset = ((128 + 64) // 64) * 64  = **192**
- tile=128: padded_offset = ((128 + 128) // 128) * 128 = **256**

预处理在 256 处清零，反向在 192 处写入。

> 讲解：varlen（变长序列）的 batch 里每条序列长度不同，内核按 m-tile 粒度并行处理。为了让每条序列能独立按 tile 边界访问而不互相踩踏，缓冲区在序列之间预留了"对齐空隙"（padded offset）。tile 尺寸不一致时，同一逻辑偏移算出的空隙位置不同，读和写就错位了。

## 症状

- 单独跑测试通过（torch.empty 拿到的是干净内存）
- 连续跑测试失败（CUDA 内存缓存复用了被 NaN 污染的内存）
- 反向内核之后 dq_accum 的合法位置含 NaN
- 用 `torch.zeros` 初始化 dq_accum 会掩盖 bug（到处都是零，包括"正确"的偏移）
- compute-sanitizer 显示 0 错误（地址合法，只是缓冲区内的偏移不对）

> 讲解：这个 bug 最阴险的地方在于"测试环境掩盖缺陷"：torch.empty 往往分到上次没用过的干净内存，单独跑看起来没问题；而 CUDA 显存缓存复用后，被 NaN 污染的内存就暴露了错位。同理，`torch.zeros` 会把错误偏移处的脏值也清零，让症状消失——所以这类 bug 要用"初始化成哨兵值 + 连续跑"的方式去暴露。

## 修复

```python
# flash_bwd_preprocess.py line 216
# Before:
seqlen = SeqlenInfo.create(batch_idx, mO.shape[1], mCuSeqlensQ, mSeqUsedQ)
# After:
seqlen = SeqlenInfo.create(batch_idx, mO.shape[1], mCuSeqlensQ, mSeqUsedQ, tile=self.tile_m)
```

## 教训

任何为 varlen 缓冲区计算 `padded_offset` 的代码，都必须使用与分配并访问这些缓冲区的内核相同的 tile 尺寸。当 `m_block_size != 128` 时，`SeqlenInfo.create` 的默认 `tile=128` 就是个陷阱。
