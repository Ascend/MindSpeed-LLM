# MindSpeed-LLM 任务入口导读

本文帮助用户按任务类型快速定位入口脚本、示例脚本、配置来源和核心实现路径。如需了解安装流程、模型参数或特性说明，请参见对应任务文档。

## 任务入口总览

| 任务类型 | 推荐入口 | 典型示例 | 配置来源 | 入口 / 实现路径 | 适用说明 |
| --- | --- | --- | --- | --- | --- |
| mcore 预训练 | [pretrain_gpt.py](../../pretrain_gpt.py) | [examples/mcore/qwen3/pretrain_qwen3_8b_4K_ptd.sh](../../examples/mcore/qwen3/pretrain_qwen3_8b_4K_ptd.sh) | shell 参数 | [mindspeed_llm/training](../../mindspeed_llm/training) | PyTorch mcore 后端预训练 |
| mcore 后训练 | [posttrain_gpt.py](../../posttrain_gpt.py) | [examples/mcore/qwen3/tune_qwen3_8b_4K_full_ptd.sh](../../examples/mcore/qwen3/tune_qwen3_8b_4K_full_ptd.sh) | shell 参数、`--stage` | [mindspeed_llm/tasks/posttrain](../../mindspeed_llm/tasks/posttrain) | SFT 等后训练任务 |
| DPO 后训练 | [posttrain_gpt.py](../../posttrain_gpt.py) | [examples/mcore/qwen3_moe/dpo_qwen3_30b_a3b_16K_A3_ptd.sh](../../examples/mcore/qwen3_moe/dpo_qwen3_30b_a3b_16K_A3_ptd.sh) | shell 参数、`--stage dpo` | [mindspeed_llm/tasks/posttrain/dpo](../../mindspeed_llm/tasks/posttrain/dpo) | mcore DPO 偏好对齐训练 |
| FSDP2 训练 | [train_fsdp2.py](../../train_fsdp2.py) | [examples/fsdp2/qwen3/pretrain_qwen3_8b_4k_fsdp2_A3.sh](../../examples/fsdp2/qwen3/pretrain_qwen3_8b_4k_fsdp2_A3.sh) | shell + YAML | [mindspeed_llm/fsdp2](../../mindspeed_llm/fsdp2) | PyTorch FSDP2 后端训练 |
| 推理 | [inference.py](../../inference.py)、[inference_fsdp2.py](../../inference_fsdp2.py) | [examples/mcore/qwen3/generate_qwen3_8b_ptd.sh](../../examples/mcore/qwen3/generate_qwen3_8b_ptd.sh) | shell 参数或 YAML | [mindspeed_llm/tasks/inference](../../mindspeed_llm/tasks/inference)、[mindspeed_llm/fsdp2/inference](../../mindspeed_llm/fsdp2/inference) | mcore 与 FSDP2 推理入口不同 |
| 评估 | [evaluation.py](../../evaluation.py) | [examples/mcore/qwen3/evaluate_qwen3_8b_ptd.sh](../../examples/mcore/qwen3/evaluate_qwen3_8b_ptd.sh) | shell 参数、[configs/evaluate](../../configs/evaluate) | [mindspeed_llm/tasks/evaluation](../../mindspeed_llm/tasks/evaluation) | Benchmark 评估任务 |
| 数据预处理 | [preprocess_data.py](../../preprocess_data.py) | [examples/mcore/qwen3/data_convert_qwen3_pretrain.sh](../../examples/mcore/qwen3/data_convert_qwen3_pretrain.sh) | shell 参数 | [mindspeed_llm/tasks/preprocess](../../mindspeed_llm/tasks/preprocess) | 预训练、SFT、DPO 等数据处理 |
| 权重转换 | [convert_ckpt.py](../../convert_ckpt.py)、[convert_ckpt_v2.py](../../convert_ckpt_v2.py) | [examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh](../../examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh) | shell 参数、[configs/checkpoint](../../configs/checkpoint) | [mindspeed_llm/tasks/checkpoint](../../mindspeed_llm/tasks/checkpoint) | HF、Megatron、mcore 等格式转换 |

## mcore 任务链路

典型 mcore 链路包括数据处理、权重转换、训练、推理和评估：

1. 使用 `preprocess_data.py` 处理数据，例如 `examples/mcore/qwen3/data_convert_qwen3_pretrain.sh`。
2. 使用 `convert_ckpt.py` 或 `convert_ckpt_v2.py` 转换权重，例如 `examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh`。
3. 预训练使用 `pretrain_gpt.py`，后训练使用 `posttrain_gpt.py`。
4. 推理使用 `inference.py`，评估使用 `evaluation.py`。

mcore 示例脚本中的参数可按以下层次理解：

| 参数层次 | 常见内容 |
| --- | --- |
| 环境和分布式启动 | `NPUS_PER_NODE`、`NNODES`、`MASTER_ADDR`、`MASTER_PORT` |
| 路径 | 权重路径、数据路径、tokenizer 路径、输出路径 |
| 模型结构 | 层数、hidden size、attention heads、模型 spec |
| 并行策略 | TP、PP、CP、EP 等并行相关参数 |
| 训练超参 | batch size、learning rate、scheduler、训练步数 |
| 性能特性 | flash attention、fused op、overlap、distributed optimizer |
| 输出控制 | log、save、eval interval |

以 Qwen3 mcore 示例为例，常见链路可按阶段查找。更多说明可参考 [Qwen3 mcore 示例链路](../../examples/mcore/qwen3/README.md)。

| 阶段 | 典型脚本 | 说明 |
| --- | --- | --- |
| 预训练数据处理 | [examples/mcore/qwen3/data_convert_qwen3_pretrain.sh](../../examples/mcore/qwen3/data_convert_qwen3_pretrain.sh) | 将预训练数据处理为训练入口可读取的数据格式。 |
| 指令数据处理 | [examples/mcore/qwen3/data_convert_qwen3_instruction.sh](../../examples/mcore/qwen3/data_convert_qwen3_instruction.sh) | 处理指令微调场景的数据。 |
| HF 到 mcore 权重转换 | [examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh](../../examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh) | 将 Hugging Face 权重转换为 mcore 训练可用格式。 |
| mcore 到 HF 权重转换 | [examples/mcore/qwen3/ckpt_convert_qwen3_mcore2hf.sh](../../examples/mcore/qwen3/ckpt_convert_qwen3_mcore2hf.sh) | 将 mcore 权重转换回 Hugging Face 格式。 |
| 预训练 | [examples/mcore/qwen3/pretrain_qwen3_8b_4K_ptd.sh](../../examples/mcore/qwen3/pretrain_qwen3_8b_4K_ptd.sh) | Qwen3 8B mcore 预训练示例。 |
| 全参微调 | [examples/mcore/qwen3/tune_qwen3_8b_4K_full_ptd.sh](../../examples/mcore/qwen3/tune_qwen3_8b_4K_full_ptd.sh) | Qwen3 8B 全参微调示例。 |
| LoRA 微调 | [examples/mcore/qwen3/tune_qwen3_8b_4K_lora_ptd.sh](../../examples/mcore/qwen3/tune_qwen3_8b_4K_lora_ptd.sh) | Qwen3 8B LoRA 微调示例。 |
| 推理 | [examples/mcore/qwen3/generate_qwen3_8b_ptd.sh](../../examples/mcore/qwen3/generate_qwen3_8b_ptd.sh) | 使用训练或转换后的权重进行文本生成。 |
| 评估 | [examples/mcore/qwen3/evaluate_qwen3_8b_ptd.sh](../../examples/mcore/qwen3/evaluate_qwen3_8b_ptd.sh) | 使用评估入口运行 benchmark 任务。 |

## FSDP2 任务链路

FSDP2 任务通常通过 shell 脚本启动分布式训练，并使用 YAML 文件保存模型、数据、并行和训练配置。例如，[examples/fsdp2/qwen3/pretrain_qwen3_8b_4k_fsdp2_A3.sh](../../examples/fsdp2/qwen3/pretrain_qwen3_8b_4k_fsdp2_A3.sh) 会加载 [examples/fsdp2/qwen3/pretrain_qwen3_8b_4k_fsdp2_A3.yaml](../../examples/fsdp2/qwen3/pretrain_qwen3_8b_4k_fsdp2_A3.yaml)，并通过 `train_fsdp2.py` 启动训练。

修改 FSDP2 示例时，优先确认 shell 中的分布式参数和 YAML 中的模型、数据、并行、训练字段是否匹配。

## DPO 任务链路

DPO 属于 mcore DPO 后训练路径，通常通过 `posttrain_gpt.py`、`--stage dpo` 和 shell 参数启动。典型示例为 [examples/mcore/qwen3_moe/dpo_qwen3_30b_a3b_16K_A3_ptd.sh](../../examples/mcore/qwen3_moe/dpo_qwen3_30b_a3b_16K_A3_ptd.sh)，核心实现位于 [mindspeed_llm/tasks/posttrain/dpo](../../mindspeed_llm/tasks/posttrain/dpo)。

修改 DPO 示例时，优先确认模型路径、偏好数据路径、tokenizer、并行参数、训练超参和输出路径。

## 推理、评估、数据预处理和权重转换

| 任务 | 入口 | 常见修改点 |
| --- | --- | --- |
| 推理 | `inference.py`、`inference_fsdp2.py` | 权重路径、tokenizer、并行参数、输出方式 |
| 评估 | `evaluation.py` | 权重路径、评估数据路径、任务列表、batch 参数 |
| 数据预处理 | `preprocess_data.py`、`preprocess_prompt.py` | 原始数据路径、tokenizer、输出前缀、数据格式 |
| 权重转换 | `convert_ckpt.py`、`convert_ckpt_v2.py` | 源权重路径、目标权重路径、模型类型、转换方向、并行切分 |

## 配置来源说明

MindSpeed-LLM 当前包含多种配置形态，使用时无需统一为单一格式。

| 配置形态 | 主要路径 | 常见用途 |
| --- | --- | --- |
| shell 参数 | `examples/mcore` | mcore 训练、后训练、推理、评估和工具链 |
| YAML | `examples/fsdp2`、`configs/fsdp2` | FSDP2 训练配置 |
| JSON | `configs/finetune/templates.json` 等 | prompt template、数据或任务映射 |

跨任务路径迁移时，建议优先对齐模型路径、tokenizer、数据、并行策略、训练超参、checkpoint、日志和输出目录等共同配置。

不同任务路径中的配置概念可以先按下表理解。该表只做概念级对照，具体字段仍以对应示例和专项文档为准。

| 配置概念 | mcore shell | FSDP2 YAML | DPO 后训练 | 说明 |
| --- | --- | --- | --- | --- |
| 模型或权重路径 | 权重加载、保存相关 shell 变量或启动参数 | `model` 相关字段 | policy、reference 等相关模型路径 | 不同任务可能区分初始权重、训练输出和参考模型。 |
| tokenizer | tokenizer 路径或名称参数 | `model` / tokenizer 相关字段 | tokenizer 路径或名称参数 | 需要与模型和数据处理方式保持一致。 |
| 数据输入 | 数据路径、数据前缀或数据集参数 | `data` 相关字段 | 偏好数据路径或数据集参数 | 预训练、指令微调、偏好数据的格式要求不同。 |
| 并行策略 | TP、PP、CP、EP 等 shell 参数 | `parallel` 相关字段 | TP、PP、CP 等 shell 参数 | 并行配置需要与硬件数量、权重切分方式匹配。 |
| 训练超参 | batch size、lr、训练步数等启动参数 | `training` / `optimization` 相关字段 | DPO loss、beta、batch size、lr 等参数 | 同名概念在不同任务中的字段位置可能不同。 |
| checkpoint 与输出 | 加载目录、保存目录、日志目录 | output、checkpoint 相关字段 | 加载目录、保存目录、日志目录 | 建议先确认路径是否存在且可写。 |
| 评估或推理参数 | 推理、评估 shell 参数 | 推理 YAML 或启动参数 | 通常不作为 DPO 主链路配置 | 评估和推理通常需要单独确认数据、任务和输出设置。 |

## 常见修改点

首次修改示例脚本时，建议优先检查以下内容：

- 本机或集群环境参数，例如 NPU 数量、节点数、主节点地址和端口。
- 数据、权重、tokenizer、输出目录等路径。
- 模型规模相关参数是否与目标权重一致。
- TP、PP、CP、EP、FSDP2 等并行配置是否与硬件和权重切分匹配。
- 训练步数、batch size、学习率、保存和评估间隔。
- 日志、checkpoint 和评估输出目录是否可写。

## 谨慎修改的内容

以下内容涉及底层实现或跨模块适配。首次运行示例时，建议先保持默认配置，确认基础流程可用后再按需调整：

- `mindspeed_llm/features_manager` 中的特性 patch 注册逻辑。
- `mindspeed_llm/core` 下的并行、optimizer、transformer 底层实现。
- `mindspeed_llm/training/training.py` 中的主训练循环。
- `mindspeed_llm/fsdp2` 下的 trainer、factory 和 distributed 核心逻辑。

## 运行前检查

首次运行或调整示例脚本前，建议检查以下内容：

| 检查对象 | 建议检查 |
| --- | --- |
| 文档链接 | Markdown 表格和相对链接是否可正常访问。 |
| 示例脚本 | 入口脚本、配套配置文件和源码路径是否存在。 |
| 配置文件 | YAML / JSON 文件格式是否可解析，字段分组是否与对应文档说明一致。 |
| 路径占位 | 数据、权重、tokenizer、输出目录等路径是否已替换为实际路径。 |
| 逻辑调整 | 若调整训练、数据处理或权重转换逻辑，建议先使用小规模输入进行验证。 |

以上检查用于提前发现常见路径和配置问题，不替代完整训练验证。

## 相关文档

- [文档导读](./docs_guide.md)
- [项目导读](./project_guide.md)
- [PyTorch 快速入门](./pytorch/training/quick_start.md)
- [mcore 预训练](./pytorch/training/pretrain/mcore/pretrain.md)
- [mcore 指令微调](./pytorch/training/finetune/mcore/instruction_finetune.md)
- [mcore DPO 后训练](./pytorch/training/finetune/mcore/offline_dpo.md)
- [FSDP2 微调](./pytorch/training/finetune/fsdp2/finetune_fsdp2.md)
- [FSDP2 参数说明](./pytorch/features/fsdp2/arguments.md)
- [模型评估](./pytorch/training/evaluation/evaluation_guide.md)
- [预训练数据处理](./pytorch/tools/data_process_pretrain.md)
- 权重转换说明可参考文档导航中的工具链部分。
