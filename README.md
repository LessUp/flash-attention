# FlashAttention

本仓库提供以下论文中 FlashAttention 与 FlashAttention-2 的官方实现。

**FlashAttention：具有 IO 感知的高速、省内存的精确注意力机制（Fast and Memory-Efficient Exact Attention with IO-Awareness）**
Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré
论文：https://arxiv.org/abs/2205.14135
IEEE Spectrum [文章](https://spectrum.ieee.org/mlperf-rankings-2022) 介绍了我们使用 FlashAttention 参加 MLPerf 2.0 基准测试的提交。
![FlashAttention](assets/flashattn_banner.jpg)

**FlashAttention-2：更好的并行性与工作划分，使注意力更快（Faster Attention with Better Parallelism and Work Partitioning）**
Tri Dao

论文：https://tridao.me/publications/flash2/flash2.pdf

![FlashAttention-2](assets/flashattention_logo.png)

## 使用情况

我们很高兴看到 FlashAttention 在发布后短时间内被广泛采用。这个[页面](https://github.com/Dao-AILab/flash-attention/blob/main/usage.md)列出了部分正在使用 FlashAttention 的机构与项目。

FlashAttention 和 FlashAttention-2 可以自由使用和修改（见 LICENSE）。如果使用请引用并致谢 FlashAttention。

## FlashAttention-3 测试版发布

FlashAttention-3 针对 Hopper GPU（如 H100）进行了优化。

博客：https://tridao.me/blog/2024/flash3/

论文：https://tridao.me/publications/flash3/flash3.pdf

![FlashAttention-3 在 H100 80GB SXM5 上 FP16 的加速比](assets/flash3_fp16_fwd.png)

这是一个测试版发布，用于在我们将其集成到仓库其他部分之前进行测试和基准测试。

目前已发布：
- FP16 / BF16 前向与反向，FP8 前向

要求：H100 / H800 GPU，CUDA >= 12.3。

为获得最佳性能，我们强烈建议使用 CUDA 12.8。

安装：
```sh
cd hopper
python setup.py install
```
运行测试：
```sh
export PYTHONPATH=$PWD
pytest -q -s test_flash_attn.py
```
安装完成后，可以这样导入使用：
```python
from flash_attn_3 import flash_attn_interface
flash_attn_interface.flash_attn_func()
```

使用 `uv` 安装，在 `pyproject.toml` 中写入：

```toml
[project]
dependencies = [
    "flash-attn-3"
]

[tool.uv]
no-build-isolation = true

[tool.uv.sources]
flash-attn-3 = { git = "https://github.com/Dao-AILab/flash-attention", subdirectory = "hopper" }
```

## FlashAttention-4 (CuTeDSL)

FlashAttention-4 使用 CuTeDSL 编写，针对 Hopper 和 Blackwell GPU（如 H100、B200）优化。

安装：
```sh
pip install flash-attn-4
```

如果使用 CUDA 13，建议用 `cu13` extra 安装以获得最佳性能：
```sh
pip install "flash-attn-4[cu13]"
```

安装完成后，用法如下：
```python
from flash_attn.cute import flash_attn_func

out = flash_attn_func(q, k, v, causal=True)
```

## 安装与特性
**依赖要求：**
- CUDA toolkit 或 ROCm toolkit
- PyTorch 2.2 及以上。
- `packaging` Python 包（`pip install packaging`）
- `psutil` Python 包（`pip install psutil`）
- `ninja` Python 包（`pip install ninja`）*
- Linux。从 v2.3.2 起可能在 Windows 上工作（我们看到一些[正面反馈](https://github.com/Dao-AILab/flash-attention/issues/595)），但 Windows 编译仍需更多测试。如果你有关于如何为 Windows 设置预编译 CUDA wheel 的想法，请通过 GitHub issue 联系我们。

\* 确保 `ninja` 已安装且工作正常（例如 `ninja
--version` 然后 `echo $?` 应该返回退出码 0）。如果不正常（有时 `ninja
--version` 后 `echo $?` 返回非零退出码），请卸载后重装 `ninja`（`pip uninstall -y ninja && pip install ninja`）。没有 `ninja`，
编译可能需要很长时间（2 小时），因为它不会使用多个 CPU 核心。有了 `ninja`，在使用 CUDA toolkit 的 64 核机器上编译只需 3-5 分钟。

**安装：**
```sh
pip install flash-attn --no-build-isolation
```
也可以从源码编译：
```sh
python setup.py install
```

如果你的机器内存少于 96GB 但 CPU 核心很多，`ninja`
可能启动太多并行编译任务而耗尽内存。要限制并行编译任务数量，可以设置环境变量 `MAX_JOBS`：
```sh
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

**接口：** `src/flash_attention_interface.py`

### NVIDIA CUDA 支持
**依赖要求：**
- CUDA 12.0 及以上。

我们推荐使用 Nvidia 的 [Pytorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
容器，它包含了安装 FlashAttention 所需的全部工具。

FlashAttention-2 的 CUDA 版本目前支持：
1. Ampere、Ada 或 Hopper GPU（如 A100、RTX 3090、RTX 4090、H100）。对于 Turing GPU（T4、RTX 2080），请参见单独的 [flash-attention-turing](https://github.com/ssiu/flash-attention-turing) 仓库，它在 Turing 上支持 FlashAttention 的核心特性子集。
2. fp16 和 bf16 数据类型（bf16 需要 Ampere、Ada 或 Hopper GPU）。
3. 所有不超过 256 的 head 维度。~~head dim > 192 的反向需要 A100/A800 或 H100/H800~~。从 flash-attn 2.5.5 起，head dim 256 的反向在消费级 GPU 上也可用（如果没有 dropout）。

### AMD ROCm 支持
ROCm 版本有两个后端：默认后端是 [composable_kernel](https://github.com/ROCm/composable_kernel)（ck），另有一个 [Triton](https://github.com/triton-lang/triton) 后端。它们都提供了 FlashAttention-2 的实现。

**依赖要求：**
- ROCm 6.0 及以上。

我们推荐使用 ROCm 的 [Pytorch](https://hub.docker.com/r/rocm/pytorch) 容器。

#### Composable Kernel 后端
FlashAttention-2 的 ROCm CK 后端目前支持：
1. MI200x、MI250x、MI300x、MI355x 和 RDNA 3/4 GPU。
2. fp16 和 bf16 数据类型。
3. 前向和反向的 head 维度均支持到 256。

#### Triton 后端
[Flash Attention](https://tridao.me/publications/flash2/flash2.pdf) 的 Triton 实现支持 AMD 的 CDNA（MI200、MI300）和 RDNA GPU，支持 fp16、bf16 和 fp32 数据类型。它提供前向和反向，支持 causal masking、变长序列、任意 Q/KV 序列长度和 head 大小、MQA/GQA、dropout、旋转位置编码（rotary embeddings）、ALiBi、paged attention 和 FP8（通过 Flash Attention v3 接口）。滑窗注意力目前仍在开发中。

Triton 后端的 kernel 由 [aiter](https://github.com/ROCm/aiter) 包提供，作为 git 子模块包含在 `third_party/aiter`，在安装过程中自动安装。

安装时，先从 https://pytorch.org/get-started/locally/ 获取适用于 ROCm 的 PyTorch，然后安装 Flash Attention：
```sh
cd flash-attention
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" pip install --no-build-isolation .
```

要使用特定的 aiter 提交（例如用于测试或开发）：
```sh
cd flash-attention
cd third_party/aiter && git fetch origin && git checkout <commit-sha> && cd ../..
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" pip install --no-build-isolation .
```

运行测试（注意：完整测试套件需要数小时）：
```sh
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" pytest tests/test_flash_attn_triton_amd.py
```

Triton 后端默认使用为确定性和合理性能优化的 kernel 配置。要追求峰值吞吐，可以启用 `FLASH_ATTENTION_TRITON_AMD_AUTOTUNE="TRUE"` 搜索最优设置，这会带来一次性预热成本。

另外，如果*不*使用 autotune，可以用 `FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON` 设置单个 triton 配置，覆盖 `attn_fwd` 的硬编码默认值。例如：
```sh
FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON='{"BLOCK_M":128,"BLOCK_N":64,"waves_per_eu":1,"PRE_LOAD_V":false,"num_stages":1,"num_warps":8}'
```

快速上手 Docker：
```dockerfile
FROM rocm/pytorch:latest

WORKDIR /workspace

# 用 triton 后端构建 flash attention
RUN git clone https://github.com/Dao-AILab/flash-attention &&\ 
    cd flash-attention &&\
    FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" pip install --no-build-isolation .

# 设置工作目录
WORKDIR /workspace/flash-attention

# 设置环境变量以使用 triton 后端
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
```

构建并运行：
```sh
docker build -t flash-attn-triton .
docker run -it --network=host --user root --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ipc=host --shm-size 16G --device=/dev/kfd --device=/dev/dri flash-attn-triton
```

## 如何使用 FlashAttention

主要函数实现了缩放点积注意力（scaled dot product attention，即 softmax(Q @ K^T * softmax_scale) @ V）：
```python
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
```

```python
flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False,
                          window_size=(-1, -1), alibi_slopes=None, deterministic=False):
"""评估（evaluation）时 dropout_p 应设为 0.0。
如果 Q、K、V 已经堆叠成一个张量，这个函数会比分别对 Q、K、V 调用 flash_attn_func 更快，
因为反向传播避免了显式拼接 Q、K、V 的梯度。
如果 window_size != (-1, -1)，则实现滑窗局部注意力（sliding window local attention）。位置 i 的 query
只会关注 [i - window_size[0], i + window_size[1]] 闭区间内的 key。
参数：
    qkv: (batch_size, seqlen, 3, nheads, headdim)
    dropout_p: float。dropout 概率。
    softmax_scale: float。应用 softmax 之前 QK^T 的缩放因子。
        默认为 1 / sqrt(headdim)。
    causal: bool。是否应用因果注意力掩码（例如自回归建模）。
    window_size: (left, right)。如果不是 (-1, -1)，则实现滑窗局部注意力。
    alibi_slopes: (nheads,) 或 (batch_size, nheads)，fp32。对 query i 和 key j 的注意力分数
        加上 (-alibi_slope * |i - j|) 的偏置。
    deterministic: bool。是否使用反向传播的确定性实现，
        这会稍慢且占用更多内存。前向传播始终是确定性的。
返回：
    out: (batch_size, seqlen, nheads, headdim)。
"""
```

```python
flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                window_size=(-1, -1), alibi_slopes=None, deterministic=False):
"""评估时 dropout_p 应设为 0.0。
通过传入 head 数少于 Q 的 KV 来支持多查询和分组查询注意力（MQA/GQA）。
注意 Q 的 head 数必须能被 KV 的 head 数整除。
例如，如果 Q 有 6 个 head，K、V 有 2 个 head，那么 Q 的 head 0、1、2 会关注
K、V 的 head 0，Q 的 head 3、4、5 会关注 K、V 的 head 1。
如果 window_size != (-1, -1)，则实现滑窗局部注意力。位置 i 的 query
只会关注闭区间
[i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]]
内的 key。

参数：
    q: (batch_size, seqlen, nheads, headdim)
    k: (batch_size, seqlen, nheads_k, headdim)
    v: (batch_size, seqlen, nheads_k, headdim)
    dropout_p: float。dropout 概率。
    softmax_scale: float。应用 softmax 之前 QK^T 的缩放因子。
        默认为 1 / sqrt(headdim)。
    causal: bool。是否应用因果注意力掩码（例如自回归建模）。
    window_size: (left, right)。如果不是 (-1, -1)，则实现滑窗局部注意力。
    alibi_slopes: (nheads,) 或 (batch_size, nheads)，fp32。对注意力分数加上
        (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
        偏置（query i 和 key j 之间）。
    deterministic: bool。是否使用反向传播的确定性实现，
        这会稍慢且占用更多内存。前向传播始终是确定性的。
返回：
    out: (batch_size, seqlen, nheads, headdim)。
"""
```

```python
def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 表示无限上下文窗口
    rotary_interleaved=True,
    alibi_slopes=None,
):
    """
    如果 k 和 v 不为 None，k_cache 和 v_cache 会*原地*用 k 和 v 的新值更新。
    这对增量解码（incremental decoding）很有用：你可以传入上一步缓存的 keys/values，
    用当前步的新 keys/values 更新它们，然后对更新后的缓存做注意力，全部在一个 kernel 中完成。

    如果传入 k / v，必须确保缓存足够大以容纳新值。
    例如，KV cache 可以按最大序列长度预先分配，并用 cache_seqlens
    跟踪 batch 中每个序列的当前序列长度。

    如果传入 rotary_cos 和 rotary_sin，还会应用旋转位置编码（rotary embedding）。
    key @k 会在索引 cache_seqlens、cache_seqlens + 1 等处按 rotary_cos 和 rotary_sin 旋转。
    如果是 causal 或局部注意力（即 window_size != (-1, -1)），query @q 会在索引
    cache_seqlens、cache_seqlens + 1 等处旋转。
    如果不是 causal 也不是局部注意力，query @q 只在索引 cache_seqlens 处旋转
    （即认为 @q 中所有 token 都位于位置 cache_seqlens）。

    使用示例见 tests/test_flash_attn.py::test_flash_attn_kvcache。

    通过传入 head 数少于 Q 的 KV 来支持多查询和分组查询注意力（MQA/GQA）。
    注意 Q 的 head 数必须能被 KV 的 head 数整除。
    例如，如果 Q 有 6 个 head，K、V 有 2 个 head，那么 Q 的 head 0、1、2 会关注
    K、V 的 head 0，Q 的 head 3、4、5 会关注 K、V 的 head 1。

    如果 causal=True，因果掩码对齐到注意力矩阵的右下角。
    例如，如果 seqlen_q = 2 且 seqlen_k = 5，因果掩码（1 = 保留，0 = 掩掉）为：
        1 1 1 1 0
        1 1 1 1 1
    如果 seqlen_q = 5 且 seqlen_k = 2，因果掩码为：
        0 0
        0 0
        0 0
        1 0
        1 1
    如果掩码的某一行全为 0，输出将为 0。

    如果 window_size != (-1, -1)，则实现滑窗局部注意力。位置 i 的 query
    只会关注闭区间
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]]
    内的 key。

    注意：不支持反向传播。

    参数：
        q: (batch_size, seqlen, nheads, headdim)
        k_cache: 如果没有 block_table，为 (batch_size_cache, seqlen_cache, nheads_k, headdim)；
            如果有 block_table（即 paged KV cache），为 (num_blocks, page_block_size, nheads_k, headdim)。
            page_block_size 必须是 256 的倍数。
        v_cache: 如果没有 block_table，为 (batch_size_cache, seqlen_cache, nheads_k, headdim)；
            如果有 block_table（即 paged KV cache），为 (num_blocks, page_block_size, nheads_k, headdim)。
        k [可选]: (batch_size, seqlen_new, nheads_k, headdim)。如果不为 None，我们会把 k
            与 k_cache 拼接，从 cache_seqlens 指定的索引开始。
        v [可选]: (batch_size, seqlen_new, nheads_k, headdim)。与 k 类似。
        rotary_cos [可选]: (seqlen_ro, rotary_dim / 2)。如果不为 None，会对 k 和 q 应用旋转位置编码。
            仅当传入 k 和 v 时适用。rotary_dim 必须能被 16 整除。
        rotary_sin [可选]: (seqlen_ro, rotary_dim / 2)。与 rotary_cos 类似。
        cache_seqlens: int 或 (batch_size,)，dtype 为 torch.int32。KV cache 的序列长度。
        block_table [可选]: (batch_size, max_num_blocks_per_seq)，dtype 为 torch.int32。
        cache_batch_idx: (batch_size,)，dtype 为 torch.int32。用于索引 KV cache 的索引。
            如果为 None，我们假设 batch 索引是 [0, 1, 2, ..., batch_size - 1]。
            如果索引不互异，且提供了 k 和 v，缓存中更新的值可能来自任意重复索引。
        softmax_scale: float。应用 softmax 之前 QK^T 的缩放因子。
            默认为 1 / sqrt(headdim)。
        causal: bool。是否应用因果注意力掩码（例如自回归建模）。
        window_size: (left, right)。如果不是 (-1, -1)，则实现滑窗局部注意力。
        rotary_interleaved: bool。仅当传入 rotary_cos 和 rotary_sin 时适用。
            如果为 True，旋转位置编码会组合维度 0 和 1、2 和 3 等。如果为 False，
            旋转位置编码会组合维度 0 和 rotary_dim / 2、1 和 rotary_dim / 2 + 1
            （即 GPT-NeoX 风格）。
        alibi_slopes: (nheads,) 或 (batch_size, nheads)，fp32。对注意力分数加上
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|) 偏置。

    返回：
        out: (batch_size, seqlen, nheads, headdim)。
    """
```

要看这些函数如何用在多头注意力层（包括 QKV 投影、输出投影）中，见 MHA [实现](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/modules/mha.py)。

### 与 🤗 Kernels 一起使用

如果你的硬件环境属于上述任一类型，也可以直接使用 [`kernels` 库](https://github.com/huggingface/kernels) 来使用 Flash Attention 2 和 3。

```py
# pip install kernels

from kernels import get_kernel

# FA2
fa_module = get_kernel("kernels-community/flash-attn2", version=1)
flash_attn_func = fa_module.flash_attn_func

# FA3
fa3_module = get_kernel("kernels-community/flash-attn3", version=1)
flash_attn_func = fa3_module.flash_attn_func
```

## 更新日志（Changelog）

### 2.0：完全重写，快 2 倍
从 FlashAttention (1.x) 升级到 FlashAttention-2

这些函数被重命名：
- `flash_attn_unpadded_func` -> `flash_attn_varlen_func`
- `flash_attn_unpadded_qkvpacked_func` -> `flash_attn_varlen_qkvpacked_func`
- `flash_attn_unpadded_kvpacked_func` -> `flash_attn_varlen_kvpacked_func`

如果同一 batch 中输入的序列长度相同，使用这些函数更简单更快：
```python
flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False)
```
```python
flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)
```
### 2.1：改变 causal 标志的行为

如果 seqlen_q != seqlen_k 且 causal=True，因果掩码对齐到注意力矩阵的右下角，
而不是左上角。

例如，如果 seqlen_q = 2 且 seqlen_k = 5，因果掩码（1 = 保留，0 = 掩掉）为：
v2.0:
    1 0 0 0 0
    1 1 0 0 0
v2.1:
    1 1 1 1 0
    1 1 1 1 1

如果 seqlen_q = 5 且 seqlen_k = 2，因果掩码为：
v2.0:
    1 0
    1 1
    1 1
    1 1
    1 1
v2.1:
    0 0
    0 0
    0 0
    1 0
    1 1
如果掩码的某一行全为 0，输出将为 0。

### 2.2：为推理优化

当 query 序列长度非常小（例如 query 序列长度 = 1）时，为推理（迭代式解码）优化。
这里的瓶颈是如何尽可能快地加载 KV cache，我们把加载工作分散到不同的 thread block 上，
并用一个单独的 kernel 合并结果。

参见 `flash_attn_with_kvcache` 函数，它提供了更多推理特性
（执行旋转位置编码、原地更新 KV cache）。

感谢 xformers 团队，特别是 Daniel Haziza 对这次合作的贡献。

### 2.3：局部（即滑窗）注意力

实现滑窗注意力（即局部注意力）。感谢 [Mistral AI](https://mistral.ai/) 特别是 Timothée Lacroix 的贡献。滑窗注意力被用于 [Mistral 7B](https://mistral.ai/news/announcing-mistral-7b/) 模型。

### 2.4：ALiBi（线性偏置注意力）、确定性反向传播

实现 ALiBi（Press et al., 2021）。感谢 Kakao Brain 的 Sanghun Cho 的贡献。

实现确定性反向传播。感谢 [美团](www.meituan.com) 的工程师的贡献。

### 2.5：Paged KV cache

支持 paged KV cache（即 [PagedAttention](https://arxiv.org/abs/2309.06180)）。
感谢 @beginlner 的贡献。

### 2.6：Softcapping

支持带 softcapping 的注意力，如 Gemma-2 和 Grok 模型中所用。
感谢 @Narsil 和 @lucidrains 的贡献。

### 2.7：兼容 torch compile

感谢 @ani300 的贡献。

## 性能

我们展示了在不同 GPU 上，根据序列长度，使用 FlashAttention 对比 PyTorch 标准注意力的预期加速比（前向 + 反向合计）和内存节省（加速比取决于内存带宽——在更慢的 GPU 内存上我们能看到更多加速）。

我们目前有以下 GPU 的基准测试：
* [A100](#a100)
* [H100](#h100)
<!-- * [RTX 3090](#rtx-3090) -->
<!-- * [T4](#t4) -->

### A100

我们使用以下参数展示 FlashAttention 的加速比：
* Head 维度 64 或 128，隐藏维度 2048（即 32 或 16 个 head）。
* 序列长度 512、1k、2k、4k、8k、16k。
* Batch size 设为 16k / seqlen。

#### 加速比

![FlashAttention 在 A100 80GB SXM5 上 FP16/BF16 的加速比](assets/flash2_a100_fwd_bwd_benchmark.png)

#### 内存

![FlashAttention 内存](assets/flashattn_memory.jpg)

这张图展示了内存节省（注意无论是否使用 dropout 或 masking，内存占用都相同）。
内存节省与序列长度成正比——因为标准注意力的内存随序列长度二次增长，而 FlashAttention 的内存随序列长度线性增长。
我们看到在序列长度 2K 时内存节省 10 倍，4K 时节省 20 倍。
因此，FlashAttention 可以扩展到更长的序列长度。

### H100

![FlashAttention 在 H100 SXM5 上 FP16/BF16 的加速比](assets/flash2_h100_fwd_bwd_benchmark.png)

## 完整模型代码与训练脚本

我们已经发布了完整的 GPT 模型[实现](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/models/gpt.py)。
我们还提供了其他层的优化实现（如 MLP、LayerNorm、交叉熵损失、旋转位置编码）。
总体上，相比 Huggingface 的基线实现，这能提速 3-5 倍，每个 A100 达到 225 TFLOPs/秒，
相当于 72% 的模型 FLOPs 利用率（我们不需要任何激活检查点 activation checkpointing）。

我们还包含一个在 Openwebtext 上训练 GPT2、在 The Pile 上训练 GPT3 的训练[脚本](https://github.com/Dao-AILab/flash-attention/tree/main/training)。

## FlashAttention 的 Triton 实现

Phil Tillet（OpenAI）用 Triton 写了一个 FlashAttention 的实验性实现：
https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py

由于 Triton 是比 CUDA 更高级的语言，理解与实验起来可能更容易。Triton 实现中的记号也更接近我们论文中使用的。

我们还有一个支持注意力偏置（如 ALiBi）的实验性 Triton 实现：
https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py

## 测试
我们测试 FlashAttention 是否与参考实现产生相同的输出和梯度（在一定的数值容差内）。具体来说，我们检查 FlashAttention 的最大数值误差至多是 Pytorch 基线实现数值误差的两倍（针对不同的 head 维度、输入 dtype、序列长度、causal / 非 causal）。

运行测试：
```sh
pytest -q -s tests/test_flash_attn.py
```
## 遇到问题怎么办

这个 FlashAttention-2 的新版本已经在多个 GPT 风格模型上测试过，主要在 A100 GPU 上。

如果遇到 bug，请开一个 GitHub Issue！

## 测试
运行测试：
```sh
pytest tests/test_flash_attn_ck.py
```

## 引用
如果你使用这个代码库，或者觉得我们的工作有价值，请引用：
```
@inproceedings{dao2022flashattention,
  title={Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
  author={Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{'e}, Christopher},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
@inproceedings{dao2023flashattention2,
  title={Flash{A}ttention-2: Faster Attention with Better Parallelism and Work Partitioning},
  author={Dao, Tri},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```
