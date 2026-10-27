# MindSpeed LLM Task Entry Guide 

This document helps you quickly locate entry scripts, example scripts, configuration sources, and core implementation paths for each task type. To learn about the installation process, model parameters, or feature descriptions, see the corresponding task documentation.

## Task Entry Overview

| Task Type | Recommended Entry | Typical Example | Configuration Source | Entry/Implementation Path | Applicable Scenario |
| --- | --- | --- | --- | --- | --- |
| MCore pretraining | [pretrain_gpt.py](../../pretrain_gpt.py) | [examples/mcore/qwen3/pretrain_qwen3_8b_4K_ptd.sh](../../examples/mcore/qwen3/pretrain_qwen3_8b_4K_ptd.sh) | Shell parameters | [mindspeed_llm/training](../../mindspeed_llm/training) | Pretraining with the PyTorch MCore backend |
| MCore post-training | [posttrain_gpt.py](../../posttrain_gpt.py) | [examples/mcore/qwen3/tune_qwen3_8b_4K_full_ptd.sh](../../examples/mcore/qwen3/tune_qwen3_8b_4K_full_ptd.sh) | Shell parameters, `--stage` | [mindspeed_llm/tasks/posttrain](../../mindspeed_llm/tasks/posttrain) | Post-training tasks such as SFT |
| FSDP2 training | [train_fsdp2.py](../../train_fsdp2.py) | [examples/fsdp2/qwen3/pretrain_qwen3_8b_4k_fsdp2_A3.sh](../../examples/fsdp2/qwen3/pretrain_qwen3_8b_4k_fsdp2_A3.sh) | Shell + YAML | [mindspeed_llm/fsdp2](../../mindspeed_llm/fsdp2) | Training with the PyTorch FSDP2 backend |
| Inference | [inference.py](../../inference.py), [inference_fsdp2.py](../../inference_fsdp2.py) | [examples/mcore/qwen3/generate_qwen3_8b_ptd.sh](../../examples/mcore/qwen3/generate_qwen3_8b_ptd.sh) | Shell parameters or YAML | [mindspeed_llm/tasks/inference](../../mindspeed_llm/tasks/inference), [mindspeed_llm/fsdp2/inference](../../mindspeed_llm/fsdp2/inference) | The inference entry differs between MCore and FSDP2 |
| Evaluation | [evaluation.py](../../evaluation.py) | [examples/mcore/qwen3/evaluate_qwen3_8b_ptd.sh](../../examples/mcore/qwen3/evaluate_qwen3_8b_ptd.sh) | Shell parameters, [configs/evaluate](../../configs/evaluate) | [mindspeed_llm/tasks/evaluation](../../mindspeed_llm/tasks/evaluation) | Benchmark evaluation tasks |
| Data preprocessing | [preprocess_data.py](../../preprocess_data.py) | [examples/mcore/qwen3/data_convert_qwen3_pretrain.sh](../../examples/mcore/qwen3/data_convert_qwen3_pretrain.sh) | Shell parameters | [mindspeed_llm/tasks/preprocess](../../mindspeed_llm/tasks/preprocess) | Data processing for pretraining, SFT, and similar tasks |
| Weight conversion | [convert_ckpt_v2.py](../../convert_ckpt_v2.py) | [examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh](../../examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh) | Shell parameters, [configs/checkpoint](../../configs/checkpoint) | [mindspeed_llm/tasks/checkpoint](../../mindspeed_llm/tasks/checkpoint) | Conversion among formats such as HF, Megatron, and MCore |

## MCore Task Pipeline

A typical MCore pipeline includes data processing, weight conversion, training, inference, and evaluation:

1. Use `preprocess_data.py` to process data, for example `examples/mcore/qwen3/data_convert_qwen3_pretrain.sh`.
2. Use `convert_ckpt_v2.py` to convert weights, for example `examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh`.
3. Use `pretrain_gpt.py` for pretraining and `posttrain_gpt.py` for post-training.
4. Use `inference.py` for inference and `evaluation.py` for evaluation.

You can understand the parameters in MCore example scripts at the following levels:

| Parameter Level | Common Content |
| --- | --- |
| Environment and distributed launch | `NPUS_PER_NODE`, `NNODES`, `MASTER_ADDR`, `MASTER_PORT` |
| Path | Weight path, data path, tokenizer path, output path |
| Model structure | Number of layers, hidden size, attention heads, model spec |
| Parallelism strategy | Parallelism-related parameters such as TP, PP, CP, and EP |
| Training hyperparameters | Batch size, learning rate, scheduler, number of training steps |
| Performance feature | Flash attention, fused op, overlap, distributed optimizer |
| Output control | Log, save, and eval interval |

Using the Qwen3 MCore example, you can find the common pipeline by stage. For more details, see [Qwen3 MCore example pipeline](../../examples/mcore/qwen3/README_en.md).

| Stage | Typical Script | Description |
| --- | --- | --- |
| Pretraining data processing | [examples/mcore/qwen3/data_convert_qwen3_pretrain.sh](../../examples/mcore/qwen3/data_convert_qwen3_pretrain.sh) | Process pretraining data into the data format that the training entry can read. |
| Instruction data processing | [examples/mcore/qwen3/data_convert_qwen3_instruction.sh](../../examples/mcore/qwen3/data_convert_qwen3_instruction.sh) | Process data for instruction fine-tuning scenarios. |
| HF to MCore weight conversion | [examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh](../../examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh) | Convert Hugging Face weights into the format usable for MCore training. |
| MCore to HF weight conversion | [examples/mcore/qwen3/ckpt_convert_qwen3_mcore2hf.sh](../../examples/mcore/qwen3/ckpt_convert_qwen3_mcore2hf.sh) | Convert MCore weights back to the Hugging Face format. |
| Pretraining | [examples/mcore/qwen3/pretrain_qwen3_8b_4K_ptd.sh](../../examples/mcore/qwen3/pretrain_qwen3_8b_4K_ptd.sh) | Qwen3 8B MCore pretraining example. |
| Full-parameter fine-tuning | [examples/mcore/qwen3/tune_qwen3_8b_4K_full_ptd.sh](../../examples/mcore/qwen3/tune_qwen3_8b_4K_full_ptd.sh) | Qwen3 8B full-parameter fine-tuning example. |
| LoRA fine-tuning | [examples/mcore/qwen3/tune_qwen3_8b_4K_lora_ptd.sh](../../examples/mcore/qwen3/tune_qwen3_8b_4K_lora_ptd.sh) | Qwen3 8B LoRA fine-tuning example. |
| Inference | [examples/mcore/qwen3/generate_qwen3_8b_ptd.sh](../../examples/mcore/qwen3/generate_qwen3_8b_ptd.sh) | Generate text with trained or converted weights. |
| Evaluation | [examples/mcore/qwen3/evaluate_qwen3_8b_ptd.sh](../../examples/mcore/qwen3/evaluate_qwen3_8b_ptd.sh) | Run benchmark tasks with the evaluation entry. |

## FSDP2 Task Pipeline

FSDP2 tasks usually launch distributed training through a shell script and store the model, data, parallelism, and training configuration in a YAML file. For example, [examples/fsdp2/qwen3/pretrain_qwen3_8b_4k_fsdp2_A3.sh](../../examples/fsdp2/qwen3/pretrain_qwen3_8b_4k_fsdp2_A3.sh) loads [examples/fsdp2/qwen3/pretrain_qwen3_8b_4k_fsdp2_A3.yaml](../../examples/fsdp2/qwen3/pretrain_qwen3_8b_4k_fsdp2_A3.yaml) and starts training through `train_fsdp2.py`.

When you modify an FSDP2 example, first check whether the distributed parameters in the shell match the model, data, parallelism, and training fields in the YAML.

## Inference, Evaluation, Data Preprocessing, and Weight Conversion

| Task | Entry | Common Modification Points |
| --- | --- | --- |
| Inference | `inference.py`, `inference_fsdp2.py` | Weight path, tokenizer, parallelism parameters, output method |
| Evaluation | `evaluation.py` | Weight path, evaluation data path, task list, batch parameters |
| Data preprocessing | `preprocess_data.py`, `preprocess_prompt.py` | Raw data path, tokenizer, output prefix, the data format |
| Weight conversion | `convert_ckpt_v2.py` | Source weight path, target weight path, the model type, conversion direction, parallel slicing |

## Configuration Sources

MindSpeed LLM currently provides multiple configuration forms, and you do not need to unify them into a single format.

| Configuration Form | Main Path | Common Use |
| --- | --- | --- |
| Shell parameters | `examples/mcore` | MCore training, post-training, inference, evaluation, and the toolchain |
| YAML | `examples/fsdp2`, `configs/fsdp2` | FSDP2 training configuration |
| JSON | `configs/finetune/templates.json` and so on | Prompt template, data, or task mapping |

When you migrate paths between tasks, you are advised to first align the common configuration such as the model path, tokenizer, data, parallelism strategy, training hyperparameters, checkpoint, logs, and output directory.

You can first understand the configuration concepts in different task paths from the following table. This table only provides a conceptual comparison, and the specific fields are still governed by the corresponding examples and dedicated documents.

| Configuration Concept | MCore Shell | FSDP2 YAML | Description |
| --- | --- | --- | --- |
| Model or weight path | Shell variables or launch parameters for weight loading and saving | `model` fields | Different tasks may distinguish between initial weights, training output, and the reference model. |
| Tokenizer | Tokenizer path or name parameter | `model`/tokenizer fields | Keep it consistent with the model and the data processing method. |
| Data input | Data path, data prefix, or dataset parameter | `data` fields | The format requirements differ for pretraining, instruction fine-tuning, and preference data. |
| Parallelism strategy | Shell parameters such as TP, PP, CP, and EP | `parallel` fields | The parallelism configuration must match the number of hardware devices and the weight slicing method. |
| Training hyperparameters | Launch parameters such as batch size, lr, and number of training steps | `training`/`optimization` fields | The same concept may appear in different field locations across tasks. |
| Checkpoint and output | Load directory, save directory, log directory | Output and checkpoint fields | You are advised to first check whether the paths exist and are writable. |
| Evaluation or inference parameters | Inference and evaluation shell parameters | Inference YAML or launch parameters | Evaluation and inference usually require separate confirmation of the data, task, and output settings. |

## Common Modification Points

When you modify an example script for the first time, you are advised to first check the following:

- Local or cluster environment parameters, such as the number of NPUs, the number of nodes, the master node address, and the port
- Paths for data, weights, tokenizer, and the output directory
- Whether the parameters related to model scale are consistent with the target weights
- Whether the parallelism configuration, such as TP, PP, CP, EP, and FSDP2, matches the hardware and the weight slicing
- The number of training steps, batch size, learning rate, and save and evaluation intervals
- Whether the log, checkpoint, and evaluation output directories are writable

## Content to Modify with Caution

The following content involves low-level implementation or cross-module adaptation. When you run an example for the first time, you are advised to keep the default configuration and adjust it only as needed after you confirm that the basic process works:

- The feature patch registration logic in `mindspeed_llm/features_manager`.
- The low-level implementations of parallelism, optimizer, and transformer in `mindspeed_llm/core`.
- The main training loop in `mindspeed_llm/training/training.py`.
- The core trainer, factory, and distributed logic in `mindspeed_llm/fsdp2`.

## Pre-Run Checks

Before you run or adjust an example script for the first time, you are advised to check the following:

| Check Item | Recommended Check |
| --- | --- |
| Documentation link | Whether the Markdown tables and relative links can be accessed normally. |
| Example script | Whether the entry script, the accompanying configuration files, and the source paths exist. |
| Configuration file | Whether the YAML/JSON file format can be parsed and whether the field groups are consistent with the description in the corresponding document. |
| Path placeholder | Whether paths such as those for data, weights, tokenizer, and the output directory have been replaced with actual paths. |
| Logic adjustment | If you adjust the training, data processing, or weight conversion logic, you are advised to verify it with a small-scale input first. |

These checks help you find common path and configuration problems in advance. They do not replace full training validation.

## Related Documents

- [Documentation Guide](./docs_guide.md)
- [Project Guide](./project_guide.md)
- [PyTorch Quick Start](./pytorch/training/quick_start.md)
- [MCore Pretraining](./pytorch/training/pretrain/mcore/pretrain.md)
- [MCore Instruction Fine-Tuning](./pytorch/training/finetune/mcore/instruction_finetune.md)
- [FSDP2 Fine-Tuning](./pytorch/training/finetune/fsdp2/finetune_fsdp2.md)
- [FSDP2 Parameters](./pytorch/features/fsdp2/arguments.md)
- [Model Evaluation](./pytorch/training/evaluation/evaluation_guide.md)
- [Pretraining Data Processing](./pytorch/tools/data_process_pretrain.md)
- For weight conversion instructions, see the toolchain section in the documentation navigation.
