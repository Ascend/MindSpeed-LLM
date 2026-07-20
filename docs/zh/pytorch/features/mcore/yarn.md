# YaRN

## 特性介绍

基于 Transformer 的大语言模型几乎已成为众多自然语言处理任务普遍的选择，在这些任务中，诸如上下文学习等长距离处理能力至关重要。在执行自然语言处理任务时，预训练大语言模型的主要限制之一，是由其训练过程所决定的序列最大长度（上下文窗口）。传统 Transformer 模型的计算和内存复杂度是 O(n^2)，其中 n 为序列长度，随着序列长度增加，计算和内存消耗会急剧上升。通过少量微调（或无需微调）动态扩展上下文窗口的相关技术就变得非常重要。

YaRN（Yet another RoPE extensioN）通过 NTK-by-parts 调整位置编码，提升序列扩增后的精度。

![旋转位置编码频率分布示意图](../../figures/yarn/position_embedding.png)

![NTK-by-parts 分段缩放示意图](../../figures/yarn/ntk_by_parts.png)

如图所示，对于一个 token 的 embedding，旋转位置编码 θ 从低维到高维逐渐变大（频率逐渐变低）。较低频的周期数不到 1，较高频的周期数远超过 1。对于高频部分直接外推，对于低频部分线性插值，在两者之间的中间区域进行线性插值过渡。

## 使用方法

使用 RoPE 作为位置编码的模型，在推理时均可使用 YaRN 来扩展上下文长度。以下以 DeepSeek-V2 的配置为例。

在 DeepSeek-V2 系列上，通过 `--rope-scaling-type yarn` 使能，其余配置参数如下：

| 参数 | 说明 |
|---|---|
| `--beta-fast` | 高频旋转周期数，默认值为 32 |
| `--beta-slow` | 低频旋转周期数，默认值为 1 |
| `--rope-scaling-factor` | 上下文扩展倍数，用于高频维度外推。例如，预训练模型的上下文长度为 4K，扩展到 160K 时，该值为 40 |
| `--rope-scaling-mscale` | 注意力缩放系数函数 `yarn_get_mscale` 的入参 |
| `--rope-scaling-mscale-all-dim` | 注意力缩放系数函数 `yarn_get_mscale` 的入参 |
| `--rope-scaling-original-max-position-embeddings` | 预训练模型未扩展时的上下文长度 |

## 使用效果

使用 DeepSeek-V2 系列的 YaRN 默认配置，MMLU 精度测试如下：

| 模型                   | 任务     | MindSpeed-LLM | 社区                                                                    |
|----------------------|--------|-----------|-----------------------------------------------------------------------|
| DeepSeek-V2-Lite-16B | MMLU   | 57.4%     | [58.3%](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite)          |
| DeepSeek-Math-7B     |MMLU-STEM| 56.5%    | [56.5%](https://github.com/deepseek-ai/DeepSeek-Math)          |
| DeepSeek-V2-236B     | MMLU   | 78.1%         | [78.5%](https://huggingface.co/deepseek-ai/DeepSeek-V2)          |
| DeepSeek-V2.5        | MMLU   | 79.3%         | [80.6%](https://huggingface.co/deepseek-ai/DeepSeek-V2.5)          |
