## NeedleBench精度测试

NeedleBench 是由上海人工智能实验室和清华大学的研究团队开发的一个用于评估大语言模型（LLMs）在处理超长文本（长达百万级上下文窗口）时的检索和推理能力的框架。它专门设计用于压力测试模型在中英双语环境下的长文本处理能力。

下载 NeedleBench 测试集：https://huggingface.co/datasets/opencompass/NeedleBench

下载tiktoken所需要的缓存文件：https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken

目前MindSpeed-LLM仓库仅支持Single-Needle-Retrieval(S-RT)测试，脚本运行参照 examples/mcore/qwen2/evaluate_qwen2_7b_needle_bench_ptd.sh

## 参数配置
- --max-new-tokens 设置为64
- --seq-length, --max-position-embeddings, --max-tokens-to-oom 可设置为4096, 8192, 32768, 131072, 262144, 524288, 1048576，用于测试对应长度的NeedleBench精度
- 开启yarn。yarn的配置参照[yarn](../../../features/yarn.md)的说明文档

NeedleBench精度测试如下：

| 模型                                                                 | 任务     | MindSpeed-LLM | 社区([OpenCompass](https://opencompass.readthedocs.io/en/latest/advanced_guides/needleinahaystack_eval.html)) |
|--------------------------------------------------------------------|--------|---------------|---------------------|
| [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) | NeedleBench-128K-Single-Needle-Retrieval   | 70.19%        | 70.25%              |
| [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) + YARN                                       | NeedleBench-128K-Single-Needle-Retrieval | 87.03%        | 88.63%              |