# Qwen3 MCore Example Pipeline

This document describes the common usage pipeline of the Qwen3 MCore example scripts and helps you locate the scripts for data processing, weight conversion, training, inference, and evaluation by task stage. For the meaning of specific parameters and for environment preparation, refer to the corresponding task documents and script content.

## Related Documents

- [MindSpeed LLM Installation](../../../docs/en/pytorch/training/install_guide.md)
- [MCore pretraining](../../../docs/en/pytorch/training/pretrain/mcore/pretrain.md)
- [MCore instruction fine-tuning](../../../docs/en/pytorch/training/finetune/mcore/instruction_finetune.md)
- [Model inference](../../../docs/en/pytorch/training/inference/inference.md)
- [Model evaluation](../../../docs/en/pytorch/training/evaluation/evaluation_guide.md)
- [Pretraining data processing](../../../docs/en/pytorch/tools/data_process_pretrain.md)
- For weight conversion instructions, see the toolchain section in the documentation navigation.

## Task Pipeline

| Stage | Typical Script | Description |
| --- | --- | --- |
| Pretraining data processing | [data_convert_qwen3_pretrain.sh](./data_convert_qwen3_pretrain.sh) | Processes pretraining data into a data format that the training entry point can read. |
| Instruction data processing | [data_convert_qwen3_instruction.sh](./data_convert_qwen3_instruction.sh) | Processes data for instruction fine-tuning scenarios. |
| HF to MCore weight conversion | [ckpt_convert_qwen3_hf2mcore.sh](./ckpt_convert_qwen3_hf2mcore.sh) | Converts Hugging Face weights into a format available for MCore training, inference, or evaluation. |
| MCore to HF weight conversion | [ckpt_convert_qwen3_mcore2hf.sh](./ckpt_convert_qwen3_mcore2hf.sh) | Converts MCore weights back to the Hugging Face format. |
| LoRA weight conversion | [ckpt_convert_qwen3_mcore2hf_lora.sh](./ckpt_convert_qwen3_mcore2hf_lora.sh) | Converts LoRA-related weights to the Hugging Face format. |
| Pretraining | [pretrain_qwen3_8b_4K_ptd.sh](./pretrain_qwen3_8b_4K_ptd.sh) | Example of Qwen3 8B MCore pretraining. |
| Full-parameter fine-tuning | [tune_qwen3_8b_4K_full_ptd.sh](./tune_qwen3_8b_4K_full_ptd.sh) | Example of Qwen3 8B full-parameter fine-tuning. |
| LoRA fine-tuning | [tune_qwen3_8b_4K_lora_ptd.sh](./tune_qwen3_8b_4K_lora_ptd.sh) | Example of Qwen3 8B LoRA fine-tuning. |
| Inference | [generate_qwen3_8b_ptd.sh](./generate_qwen3_8b_ptd.sh) | Uses trained or converted weights for text generation. |
| Evaluation | [evaluate_qwen3_8b_ptd.sh](./evaluate_qwen3_8b_ptd.sh) | Runs benchmark tasks through the evaluation entry point. |

## Script Naming Conventions

Qwen3 example scripts are usually named by task type, model size, and context length:

- `pretrain_qwen3_*`: pretraining or resume training scripts
- `tune_qwen3_*_full_*`: full-parameter fine-tuning scripts
- `tune_qwen3_*_lora_*`: LoRA fine-tuning scripts
- `generate_qwen3_*`: inference scripts
- `evaluate_qwen3_*`: evaluation scripts
- `ckpt_convert_qwen3_*`: weight conversion scripts
- `data_convert_qwen3_*`: data processing scripts

In file names, `0point6b`, `1point7b`, `4b`, `8b`, `14b`, and `32b` indicate the model size, `4K`, `32K`, and `256K` indicate common sequence length configurations, and `A3` indicates example scripts for specific hardware or cluster configuration scenarios. For the exact meaning, refer to the script content and the related documents.

## Checks Before Modification

Before you run or adjust a script, you are advised to confirm the following:

1. The data, weights, tokenizer, logs, and output directory have been replaced with actual paths.
2. The model size-related parameters match the target weights.
3. The parallel parameters such as TP, PP, and CP match the number of hardware devices and the weight partitioning method.
4. The paths used in the data processing, weight conversion, training, inference, and evaluation stages are consistent.
5. For multi-server training, the master node address, port, node count, and node rank are configured correctly.
