# FlashAttention 应用情况

我们很高兴看到 FlashAttention 被许多组织和研究实验室采用，以加速他们的训练 / 推理。
本页面列出了部分正在使用 FlashAttention 的地方。
如果你希望添加你的组织 / 产品 / 代码库的链接，请提交 PR 或给我们发邮件。我们非常期待听到你的消息！

## 已集成到机器学习框架

- Pytorch：[集成](https://github.com/pytorch/pytorch/pull/81434) 到核心 Pytorch 的 nn.Transformer 中。

- Huggingface 的 [transformers](https://github.com/huggingface/transformers) 库。
  [进行中](https://github.com/huggingface/transformers/pull/18439)，博客文章即将发布。

- 微软的 [DeepSpeed](https://github.com/microsoft/DeepSpeed)：
  FlashAttention 被[集成](https://github.com/microsoft/DeepSpeed/blob/ec13da6ba7cabc44bb4745a64a208b8580792954/deepspeed/ops/transformer/inference/triton_ops.py)到 DeepSpeed 的推理引擎中。

- Nvidia 的 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/pull/267)。这个库是训练大规模 Transformer 语言模型的主流框架。

- MosaicML [Composer](https://github.com/mosaicml/composer) [库](https://www.mosaicml.com/blog/gpt-3-quality-for-500k)。Composer 是一个用于高效神经网络训练的库。

- EleutherAI 的 [GPT-NeoX](https://github.com/EleutherAI/gpt-neox/pull/725)。这是一个基于 Nvidia 的 Megatron-LM 和微软的 DeepSpeed、用于大规模训练 Transformer 语言模型的研究库。

- PaddlePaddle：通过 [API](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/nn/functional/flash_attention.py) `paddle.nn.functional.flash_attention` 集成到框架中。

## MLPerf 基准测试

[MLPerf](https://mlcommons.org/en/) 是一个竞争性的机器学习性能基准。FlashAttention
在 MLPerf 训练 2.0（2022 年 6 月）和 MLPerf 训练 2.1（2022 年 11 月）中取得了云端实例上最快的 BERT 训练。

- MLPerf 2.0：[IEEE Spectrum](https://spectrum.ieee.org/mlperf-rankings-2022) 和 [Forbes](https://www.forbes.com/sites/moorinsights/2022/07/12/google-dethrones-nvidia-in-latest-artificial-intelligence-benchmarking-tests/) 关于我们使用 FlashAttention 参加 MLPerf 2.0 基准测试的文章。

- MLPerf 2.1 - [Azure 与 Hazy Research 的合作](https://techcommunity.microsoft.com/t5/azure-high-performance-computing/azure-collaborates-with-hazy-research-and-nvidia-to-achieve/ba-p/3667511)：首次在 16 个节点上不到 2 分钟训练完 MLPerf BERT。

- MLPerf 2.1 - [Nvidia](https://developer.nvidia.com/blog/leading-mlperf-training-2-1-with-full-stack-optimizations-for-ai/)：
  Nvidia 使用 FlashAttention 的技术，让他们（已经高度优化的）BERT 实现更快。

- MLPerf 2.1 - [MosaicML](https://www.mosaicml.com/blog/mlperf-nlp-nov2022)：FlashAttention
  帮助在开放组（open division）中将 BERT 训练加速 2.7 倍。

## 语言模型训练与推理

- [PubMedGPT 2.7B](https://crfm.stanford.edu/2022/12/15/pubmedgpt.html)，斯坦福 CRFM 针对生物医学领域训练的
  领域专用 LLM，在 [MosaicML](https://www.mosaicml.com/blog/introducing-pubmed-gpt) 云上训练。
  仅使用 FlashAttention 就几乎将总训练时间减半。

- Meta 的 [AITemplate](https://ai.facebook.com/blog/gpu-inference-engine-nvidia-amd-open-source/)
  将 FlashAttention 作为其加速 Transformer 推理方案的一部分（在 BERT 上最高 5.3 倍）。

- Nvidia 的 [FasterTransformer](https://github.com/NVIDIA/FasterTransformer) 是最先进的 Transformer
  推理库。从版本 [5.2](https://github.com/NVIDIA/FasterTransformer/commit/b672f49e256ba7a2d4fc9691d270b60b7fc1a2ff)
  开始，FlashAttention 被用作 FasterTransformer 的组件来加速 GPT 推理。

- [Kernl](https://github.com/ELS-RD/kernl) 是一个快速 Transformer 推理库。他们把 FlashAttention 作为
  其[方案](https://twitter.com/pommedeterre33/status/1585284221014245377)的一部分，
  将 Transformer 加速最高 12 倍。

## 扩散模型训练与推理

- Huggingface 的 [diffusers](https://github.com/huggingface/diffusers) 扩散模型库。FlashAttention 被集成到 [diffusers v0.7.0](https://github.com/huggingface/diffusers/releases/tag/v0.7.0) 中。
  推理最高提速 2 倍，且内存占用更低。

- Colossal-AI 的 Stable Diffusion [实现](https://github.com/hpcaitech/ColossalAI/tree/main/examples/images/diffusion)：
  以 FlashAttention 作为组件之一，预训练最高提速 6.5 倍，并将微调的硬件成本降低 7 倍。

- Meta 的 [AITemplate](https://ai.facebook.com/blog/gpu-inference-engine-nvidia-amd-open-source/)，
  以 FlashAttention 作为组件之一，据我们所知是目前[最快](https://twitter.com/bing_xu_/status/1590447334055632897)
  的 Stable Diffusion 推理引擎。

- [Labml.ai](https://twitter.com/labmlai/status/1573634095732490240) 的 Stable Diffusion 推理：加速 50%。

- 我们自己的 Stable Diffusion [fork](https://twitter.com/realDanFu/status/1580641495991754752) 使用 FlashAttention，
  相比原版获得 3-4 倍加速。

## 其他模型

- [Uni-Fold](https://github.com/dptech-corp/Uni-Fold)：Uni-Fold 是一个开源蛋白质模型开发平台，
  超越 AlphaFold。使用 FlashAttention 后，Uni-Fold 比 AlphaFold [快](https://twitter.com/guolin_ke/status/1580532071901995008) 2.6 倍。

- [OpenFold](https://github.com/aqlaboratory/openfold)：一个可训练、省内存且 GPU 友好的 AlphaFold 2 PyTorch 复现。
  以 FlashAttention 作为[组件](https://twitter.com/gahdritz/status/1595420944880779266)之一，
  在短序列推理上比 AlphaFold2 快最多 3 倍，并能预测长 2 倍的结构。

## 不同的实现

- [Triton](https://github.com/openai/triton)：OpenAI 的 Phil Tillet 用 Triton 写的 FlashAttention [实现](https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py)。Triton 是一种基于 Python 的并行编程语言和编译器。

- [xformers](https://github.com/facebookresearch/xformers)：xformers 团队以与 FlashAttention 相似的思路实现了[内存高效注意力](https://twitter.com/fvsmassa/status/1580229170629849089)。
  xformers 会动态分派到可用 / 更快的实现。

- [Jax](https://github.com/google/jax)：[lucidrains](https://github.com/lucidrains/) 用 Jax 写的[实现](https://github.com/lucidrains/flash-attention-jax)。

- [Metal](https://developer.apple.com/metal)：Philip Turner 用 Metal 写的[实现](https://github.com/philipturner/metal-flash-attention)。这个实现把 FlashAttention 移植到 Apple silicon 等移动 GPU 架构。
