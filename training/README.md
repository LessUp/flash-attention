# 优化的 Transformer 实现
本仓库包含 FlashAttention 如何集成到模型（例如 GPT、ViT）并端到端训练的示例。我们还提供了其他层的优化实现（例如 MLP、LayerNorm、交叉熵损失、旋转位置编码）。总体上，相比 Huggingface 的基线实现，这能提速 3-5 倍，每个 A100 达到 189 TFLOPs/秒，相当于 60.6% 的模型 FLOPs 利用率（我们不需要任何激活检查点 activation checkpointing）。所有这些都不改变模型架构（即没有近似）。

目标：
- 性能：我们优化模型速度和内存，尤其是在单节点上（例如 8 个 A100）。
- 灵活性：我们提供优化的构建模块（MLP、注意力、LayerNorm），模型代码展示了如何组合这些组件。训练代码也力求与模型和任务无关。

非目标（以及其他资源）：
- 支持尽可能多的模型：Huggingface 的 [transformers](https://github.com/huggingface/transformers) 和 [timm](https://github.com/rwightman/pytorch-image-models/) 在这方面做得很好。
- 大规模分布式训练：我们的代码库已被用于多达 2.7B 参数模型的多 GPU 和多节点训练。不过，如果你在寻找大规模分布式训练技术（例如 pipeline parallelism、tensor parallelism），请看 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/) 和 [DeepSpeed](https://github.com/microsoft/deepspeed)。
- 推理：我们目前专注于训练（未来可能会改变）。如果你想要快速推理，请看 [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)。
- 生产：这个代码库是在几个研究项目中写成的，用于验证加速 ML 模型的想法。

## 模型组件

GPT 模型实现在[这里](https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/models/gpt.py)。下面是一个用旋转位置编码构造 GPT3-1.3B 模型的例子：
```python
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from flash_attn.models.gpt import GPTLMHeadModel

seqlen = 2048
hidden_dim = 2048
nheads = 16
n_layer = 24
rotary_emb_fraction = 0.5
config = GPT2Config(vocab_size=50257, n_positions=seqlen, n_embd=hidden_dim,
                    n_layer=n_layer, n_head=nheads,
                    scale_attn_by_inverse_layer_idx=True,
                    rotary_emb_fraction=rotary_emb_fraction,
                    use_flash_attn=True, fused_mlp=True,
                    fused_bias_fc=True, fused_dropout_add_ln=True,
                    pad_vocab_size_multiple=8)
model = GPTLMHeadModel(config)
```

我们提供以下优化组件：

1. FlashAttention：快速、内存高效的精确注意力。这让注意力更快，并节省大量激活内存。因此我们不需要使用任何激活检查点。
```sh
pip install flash-attn
```

2. 融合的 matmul + bias（前向和反向），以及融合的 matmul + bias + gelu（前向和反向），改编自 Apex 的 [FusedDense](https://github.com/NVIDIA/apex/tree/master/apex/fused_dense)。我们让它支持 bfloat16。为了最佳性能，应使用 CUDA >= 11.8。此前的 CuBLAS 版本对 bfloat16 没有最好的 matmul + bias + gelu 性能。
```sh
cd ../csrc/fused_dense_lib && pip install .
```
3. 优化的交叉熵损失，改编自 Apex 的 [Xentropy](https://github.com/NVIDIA/apex/tree/master/apex/contrib/xentropy)。我们让它支持 bfloat16，并支持原地（in-place）反向以节省内存。
```sh
cd ../csrc/xentropy && pip install .
```
4. 融合的旋转位置编码：
```sh
cd ../csrc/rotary && pip install .
```
5. 融合的 dropout + 残差 + LayerNorm，改编自 Apex 的 [FastLayerNorm](https://github.com/NVIDIA/apex/tree/master/apex/contrib/layer_norm)。我们增加了 dropout 和残差，并让它同时适用于 pre-norm 和 post-norm 架构。支持能被 8 整除的维度，最大到 6144。
```sh
cd ../csrc/layer_norm && pip install .
```

## 训练

我们还提供了在 Openwebtext 上训练 GPT2、在 The Pile 上训练 GPT3 的训练脚本作为示例。你也可以在自己的训练环境中自由使用这个模型。

我们使用 [Hydra](https://hydra.cc/) 做配置，[Pytorch-Lightning](https://github.com/Lightning-AI/lightning) 做训练，[Wandb](https://wandb.ai/) 做日志记录。

我们使用来自 `https://github.com/ashleve/lightning-hydra-template` 的模板。请阅读那里的说明以理解仓库结构。

### 依赖要求

Python 3.8+、Pytorch 1.12+、torchvision、einops、timm、hydra-core、hydra-colorlog、python-dotenv、rich、pytorch-lightning、triton、flash-attn。我们推荐 CUDA 11.8（例如使用 Nvidia 的 Pytorch Docker 镜像 https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch）。

我们提供了一个列出所有所需包的 Dockerfile。

### 数据集准备

运行训练命令会自动下载数据集（Openwebtext、Pile），用 GPT2 tokenizer 分词，拼接所有 token，然后把这个缓存保存到磁盘。你也可以把数据集准备作为一个单独的步骤。

缓存的数据集保存到 `${DATA_DIR}/openwebtext` 和 `${DATA_DIR}/the_pile`。如果没有设置 `${DATA_DIR}`，它们会保存到 `./data/{openwebtext,the_pile}`。

- Openwebtext：
```sh
export PYTHONPATH=$PWD:$PYTHONPATH
pytest -q -s tests/datamodules/test_language_modeling_hf.py -k "openwebtext"
```
在 64 核 CPU 上大约需要 1 小时。处理后的数据集大小为 17GB。

- The Pile：
```sh
export PYTHONPATH=$PWD:$PYTHONPATH
pytest -q -s tests/datamodules/test_language_modeling_hf.py -k "pile"
```
在 64 核 CPU 上大约需要 20 小时。处理后的数据集大小为 699GB。

### 在 Openwebtext 上训练 GPT2
用 8 个 GPU 在 Openwebtext 上训练 GPT2：
```sh
python run.py experiment=owt/gpt2s-flash trainer.devices=8  # 125M
python run.py experiment=owt/gpt2m-flash trainer.devices=8  # 355M
python run.py experiment=owt/gpt2l-flash trainer.devices=8  # 760M
python run.py experiment=owt/gpt2xl-flash trainer.devices=8  # 1.6B
```
默认参数是为 8 x A100 80GB 设置的。

要用 bf16 而不是 fp16 训练，加上 `trainer.precision=bf16`。

### 在 The Pile 上训练 GPT3
用 8 个 GPU 在 The Pile 上训练 GPT3：
```sh
python run.py experiment=pile/gpt3s-flash trainer.devices=8  # 125M
python run.py experiment=pile/gpt3m-flash trainer.devices=8  # 355M
python run.py experiment=pile/gpt3l-flash trainer.devices=8  # 760M
python run.py experiment=pile/gpt3xl-flash trainer.devices=8  # 1.3B
python run.py experiment=pile/gpt3-2.7B-flash-hdim128 trainer.devices=8  # 2.7B
```
默认参数是为 8 x A100 80GB 设置的。我们默认用 bf16 训练。

要使用旋转位置编码训练，运行实验 `pile/gpt3{s,m,l,xl}-flash-rotary`。

### 训练选项

**梯度累积**：要调整设备 batch size 以适配 GPU 内存（全局 batch size 保持不变，梯度累积自动计算），设置 `datamodule.batch_size=blah`。

**多节点**：要在多个节点上训练，加上 `trainer.num_nodes=blah`。

**速度基准测试**：要打印每轮迭代时间，加上 `+callbacks.speed_monitor.verbose=True`。

**可恢复训练**：给这次运行起一个名字，然后恢复时设置 `resume=True`。训练会从完全相同的 batch 重新开始。
```sh
python run.py experiment=pile/gpt3s-flash trainer.devices=8 name=pile-gpt3s-flash resume=True
```

## 训练速度

我们在一个节点上、8 x A100 80GB SXM4（400W）+ NVLink 上测量训练的实际耗时。

FLOPs 使用 [Megatron-LM 论文](https://arxiv.org/abs/2104.04473)（第 5.1 节）的公式计算，但我们乘以 3/4 得到模型 FLOPs（而不是带激活检查点的硬件 FLOPs）。

### GPT2（序列长度 1024）

![GPT2 训练效率](../assets/gpt2_training_efficiency.jpg)

本仓库的实现（FlashAttention）比 Huggingface 的基线实现快 3-4 倍。

### GPT3（序列长度 2048）

![GPT3 训练效率](../assets/gpt3_training_efficiency.jpg)

本仓库的实现（FlashAttention）比 Huggingface 的基线实现快 3-5 倍。

对于 GPT3-2.7B 模型，我们将 head 维度设为 128（而不是 80）以获得更好的效率。

这里我们给出使用 FlashAttention 在 8 x A100 80GB 上训练速度的更多细节。

| 模型        | Batch size (tokens) | 吞吐 (tokens/sec) | 小时 / 10 亿 tokens |
| --------- | ------------------- | ------------------------ | ----------------- |
| GPT3-125M | 0.5M                | 1310k                    |              0.21 |
| GPT3-355M | 0.5M                | 503k                     |              0.55 |
| GPT3-760M | 0.5M                | 245k                     |              1.13 |
| GPT3-1.3B | 1M                  | 169k                     |              1.64 |
| GPT3-2.7B | 1M                  | 85k                      |              3.27 |

举个例子，这意味着可以在 8 x A100 上用大约 43 小时训练一个 26B tokens 的 GPT3-1.3B 模型（根据 Chinchilla 缩放法则，这是计算最优的）。

## 训练质量

这里我们给出在 Openwebtext 上训练 200B tokens 的 GPT2 损失曲线。对于 GPT2，FlashAttention 的运行与 Huggingface 基线实现对于 125M 和 355M 模型产生相同的损失曲线。对于更大的模型，基线实现耗时太长。

![GPT2 训练曲线](../assets/gpt2_training_curve.jpg)

这里我们给出在 The Pile 上训练 400B tokens 的 GPT3 损失曲线。125M、355M、760M 模型的 batch size 是 512k tokens，所以对应 800k 训练步；而 1.3B 和 2.7B 模型的 batch size 是 1M tokens，对应 400k 训练步。

![GPT3 训练曲线](../assets/gpt3_training_curve.jpg)
