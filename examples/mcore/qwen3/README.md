# Qwen3 Mcore 示例链路

本文用于说明 Qwen3 Mcore 示例脚本的常见使用链路，帮助用户按任务阶段查找数据处理、权重转换、训练、推理和评估脚本。具体参数含义和环境准备请以对应任务文档和脚本内容为准。

## 相关文档

- [MindSpeed LLM安装指导](../../../docs/zh/pytorch/training/install_guide.md)
- [mcore 预训练](../../../docs/zh/pytorch/training/pretrain/mcore/pretrain.md)
- [mcore 指令微调](../../../docs/zh/pytorch/training/finetune/mcore/instruction_finetune.md)
- [模型推理](../../../docs/zh/pytorch/training/inference/inference.md)
- [模型评估](../../../docs/zh/pytorch/training/evaluation/evaluation_guide.md)
- [预训练数据处理](../../../docs/zh/pytorch/tools/data_process_pretrain.md)
- 权重转换说明可参考文档导航中的工具链部分。

## 任务链路

| 阶段 | 典型脚本 | 说明 |
| --- | --- | --- |
| 预训练数据处理 | [data_convert_qwen3_pretrain.sh](./data_convert_qwen3_pretrain.sh) | 将预训练数据处理为训练入口可读取的数据格式。 |
| 指令数据处理 | [data_convert_qwen3_instruction.sh](./data_convert_qwen3_instruction.sh) | 处理指令微调场景的数据。 |
| HF 到 mcore 权重转换 | [ckpt_convert_qwen3_hf2mcore.sh](./ckpt_convert_qwen3_hf2mcore.sh) | 将 Hugging Face 权重转换为 mcore 训练、推理或评估可用格式。 |
| mcore 到 HF 权重转换 | [ckpt_convert_qwen3_mcore2hf.sh](./ckpt_convert_qwen3_mcore2hf.sh) | 将 mcore 权重转换回 Hugging Face 格式。 |
| LoRA 权重转换 | [ckpt_convert_qwen3_mcore2hf_lora.sh](./ckpt_convert_qwen3_mcore2hf_lora.sh) | 将 LoRA 相关权重转换为 Hugging Face 格式。 |
| 预训练 | [pretrain_qwen3_8b_4K_ptd.sh](./pretrain_qwen3_8b_4K_ptd.sh) | Qwen3 8B mcore 预训练示例。 |
| 全参微调 | [tune_qwen3_8b_4K_full_ptd.sh](./tune_qwen3_8b_4K_full_ptd.sh) | Qwen3 8B 全参微调示例。 |
| LoRA 微调 | [tune_qwen3_8b_4K_lora_ptd.sh](./tune_qwen3_8b_4K_lora_ptd.sh) | Qwen3 8B LoRA 微调示例。 |
| 推理 | [generate_qwen3_8b_ptd.sh](./generate_qwen3_8b_ptd.sh) | 使用训练或转换后的权重进行文本生成。 |
| 评估 | [evaluate_qwen3_8b_ptd.sh](./evaluate_qwen3_8b_ptd.sh) | 使用评估入口运行 benchmark 任务。 |

## 脚本命名说明

Qwen3 示例脚本通常按任务类型、模型规模和上下文长度命名：

- `pretrain_qwen3_*`：预训练或续训脚本。
- `tune_qwen3_*_full_*`：全参微调脚本。
- `tune_qwen3_*_lora_*`：LoRA 微调脚本。
- `generate_qwen3_*`：推理脚本。
- `evaluate_qwen3_*`：评估脚本。
- `ckpt_convert_qwen3_*`：权重转换脚本。
- `data_convert_qwen3_*`：数据处理脚本。

文件名中的 `0point6b`、`1point7b`、`4b`、`8b`、`14b`、`32b` 表示模型规模，`4K`、`32K`、`256K` 表示常见序列长度配置，`A3` 表示面向特定硬件或集群配置场景的示例脚本，具体含义以脚本内容和相关文档为准。

## 修改前检查

运行或调整脚本前，建议先确认以下内容：

1. 数据、权重、tokenizer、日志和输出目录已替换为实际路径。
2. 模型规模相关参数与目标权重一致。
3. TP、PP、CP 等并行参数与硬件数量和权重切分方式匹配。
4. 数据处理、权重转换、训练、推理和评估阶段使用的路径前后一致。
5. 多机训练时，主节点地址、端口、节点数和节点序号配置正确。
