这个 CUDA 扩展实现了融合的 dropout + 残差 + LayerNorm，基于 Apex 的 [FastLayerNorm](https://github.com/NVIDIA/apex/tree/master/apex/contrib/layer_norm)。
主要改动：
- 增加 dropout 和残差。
- 同时支持 pre-norm 和 post-norm 架构。
- 支持更多隐藏维度（所有能被 8 整除的维度，最大到 8192）。
- 实现 RMSNorm 作为选项。
- 支持带并行残差的 LayerNorm（例如 GPT-J、GPT-NeoX、PaLM）。

如果你需要对大于 8k 的维度使用它，请提交 issue。

这个扩展只在 A100 上测试过。

```sh
cd csrc/layer_norm && pip install .
```

截至 2024-01-05，FlashAttention 仓库不再使用这个扩展。
我们转而使用基于 Triton 的[实现](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/triton/layer_norm.py)。
