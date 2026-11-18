# NeedleBench 精度测试

本文档介绍NeedleBench精度测试，说明测试集与tiktoken缓存文件的下载方式、参数配置及MindSpeed LLM与社区的精度对比结果，适用于需要评估大语言模型长文本处理能力的用户。

## 概述

NeedleBench 是由上海人工智能实验室和清华大学的研究团队开发的一个用于评估大语言模型（LLM）在处理超长文本（长达百万级上下文窗口）时的检索和推理能力的框架。它专门设计用于压力测试模型在中英双语环境下的长文本处理能力。

下载 NeedleBench 测试集：[opencompass/NeedleBench](https://huggingface.co/datasets/opencompass/NeedleBench)

下载 tiktoken 所需要的缓存文件：[cl100k_base.tiktoken](https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken)

目前 MindSpeed LLM 仓库仅支持 Single-Needle-Retrieval (S-RT) 测试。

## 参数配置

- --max-new-tokens 设置为 64。
- --seq-length、--max-position-embeddings、--max-tokens-to-oom 可设置为 4096、8192、32768、131072、262144、524288、1048576，用于测试对应长度的 NeedleBench 精度。
- 开启 YaRN。YaRN 的配置参照 [YaRN](../../../features/mcore/yarn.md) 的说明文档。

NeedleBench 精度测试如下：

| 模型                                                                 | 任务     | MindSpeed LLM | 社区([OpenCompass](https://doc.opencompass.org.cn/faq/needleinahaystack_eval.html)) |
|--------------------------------------------------------------------|--------|---------------|---------------------|
| [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) | NeedleBench-128K-Single-Needle-Retrieval   | 70.19%        | 70.25%              |
| [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) + YaRN                                       | NeedleBench-128K-Single-Needle-Retrieval | 87.03%        | 88.63%              |
