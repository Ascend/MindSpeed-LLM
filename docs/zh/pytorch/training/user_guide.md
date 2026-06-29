# 模型使用导读

MindSpeed LLM 提供端到端的大语言模型训练方案，涵盖分布式预训练、分布式微调及推理等。

在[快速入门（基于Megatron训练后端）](./quick_start.md)或[快速入门（基于FSDP2训练后端）](./fsdp2_quick_start.md)中，用户可以基于Qwen3-8B模型快速掌握大语言模型的预训练和微调任务，下表详细介绍了MindSpeed LLM的模型使用方法。

**表 1** 模型训练方案及使用说明

| 分类 | 内容 | 使用场景 | 说明 |
| :--- | :--- | :--- | :--- |
| 预训练 | [大模型分布式预训练](./pretrain/mcore/pretrain.md) | 提供数据预处理及预训练脚本，进行模型离线预训练 | <ul><li>如需加载HuggingFace权重，需提前进行权重转换。</li><li>如需将日志存储到脚本文件中，需在运行路径下创建<code>logs</code>文件夹。</li></ul> |
| 预训练 | [数据和权重在线加载训练](./pretrain/mcore/train_from_hf.md) | 将数据预处理、权重转换和训练融为一体，提供从 HuggingFace 开源数据和权重到训练的一键式方案 | <ul><li>当前支持的HuggingFace模型类型包括：<code>qwen3</code>、<code>qwen3-moe</code>、<code>deepseek3</code>、<code>glm45-air</code>、<code>bailing_mini</code>、<code>qwen3-next</code>、<code>seed-oss</code>、<code>deepseek32</code>、<code>magistral</code>、<code>deepseek2-lite</code>。</li><li>当前数据集自动转换功能仅支持以下原始数据格式：<code>parquet, arrow, csv, json, jsonl, txt</code>，暂不支持其他的格式。</li><li>当前权重转换 <code>--enable-mg2hf-convert</code>功能仅支持单机或者共享存储环境，且不支持对Lora微调后的权重做 Megatron→HF 权重转换。</li></ul> |
| 微调 | [单样本微调](./finetune/mcore/single_sample_finetune.md) | 适用于单轮、无历史依赖任务的通用指令微调 | 微调脚本中训练参数的并行配置（如TP/PP/EP/VPP等）需与权重转换时保持一致。 |
| 推理 | [流式推理](./inference/inference.md) | 支持 `greedy_search`、`beam_search` 等多种生成策略的流式输出 | 流式推理当前为固定Instruction输入，用于对比模型推理效果。 |
