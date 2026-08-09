# FlashAttention-4 (CuTeDSL)

FlashAttention-4 是 FlashAttention 的 CuTeDSL 实现，面向 Hopper 和 Blackwell GPU。

## 安装

```sh
pip install flash-attn-4
```

如果使用 CUDA 13，用 `cu13` extra 安装以获得最佳性能：

```sh
pip install "flash-attn-4[cu13]"
```

## 用法

```python
from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

out = flash_attn_func(q, k, v, causal=True)
```

## 开发

```sh
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install -e "flash_attn/cute[dev]"       # CUDA 12.x
pip install -e "flash_attn/cute[dev,cu13]"  # CUDA 13.x（例如 B200）
pytest tests/cute/
```
