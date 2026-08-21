# FSDP2 Backend Fine-Tuning Guide

## Use Cases

Supervised Fine-Tuning (SFT) continues training a pretrained model with high-quality instruction and response data, enabling the model to learn specific tasks, domain knowledge, or conversational styles. This method applies to scenarios such as question answering, text generation, summarization, translation, code generation, and domain adaptation.

> [!NOTE]
>
> If this is your first time using the FSDP2 backend, we recommend that you first complete an end-to-end walkthrough following the [FSDP2 Quick Start](../../fsdp2_quick_start.md) (Qwen3-8B pretraining and fine-tuning). This document focuses on model, dataset, YAML configuration, and parameter descriptions for full-parameter fine-tuning. It is suitable for scenarios where you change models or use custom datasets.

This document describes how to perform full-parameter fine-tuning based on a HuggingFace-format pretrained model using the FSDP2 backend of MindSpeed LLM. The example uses the Qwen3-8B model and a single `Atlas 900 A2 PoD` (1x8 cluster). The main workflow is as follows:

**Figure 1** FSDP2 backend model fine-tuning workflow

![Fine-tuning workflow](../../../figures/instruction_finetune/process_of_instruction_tuning_fsdp2.png)

## Instructions

### Environment Setup

Before starting fine-tuning, complete the environment installation by referring to the [MindSpeed LLM Installation](../../install_guide.md).

The common environment variables of the FSDP2 backend are located in `examples/fsdp2/env_config.sh`, which is automatically loaded by the example startup scripts. The configuration is as follows:

```bash
export TRAINING_BACKEND=mindspeed_fsdp
export HCCL_CONNECT_TIMEOUT=1800
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export MULTI_STREAM_MEMORY_REUSE=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DEBUG=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Model and Dataset Preparation

**Model Preparation**

For model weight download addresses, see the [Supported Models](../../../models/supported_models.md) list. This example uses the HuggingFace-format weights of [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/tree/main).

A complete model directory should contain the model configuration, Tokenizer configuration, and all weight files. For example:

```text
Qwen3-8B/
├── config.json
├── generation_config.json
├── merges.txt
├── model-00001-of-00005.safetensors
├── model-00002-of-00005.safetensors
├── model-00003-of-00005.safetensors
├── model-00004-of-00005.safetensors
├── model-00005-of-00005.safetensors
├── model.safetensors.index.json
├── tokenizer.json
├── tokenizer_config.json
└── vocab.json
```

The FSDP2 backend can directly load HuggingFace-format weights. No model weight conversion is required before starting training. In version 26.1.0, the model weight directory is configured through `model.model_name_or_path` in the corresponding `examples/fsdp2/**/*.yaml` configuration file by default. For example, when fine-tuning Qwen3-8B, modify the following parameter in `examples/fsdp2/qwen3/tune_qwen3_8b_4k_fsdp2_A2.yaml`:

```yaml
model:
  model_name_or_path: /path/to/Qwen3-8B/
```

**Dataset Preparation**

FSDP2 fine-tuning uses the LLaMA Factory-style data processing workflow and can load data files in formats such as `.parquet`, `.csv`, `.json`, `.jsonl`, `.txt`, and `.arrow`. Data is loaded, format-aligned, and tokenized when the training task starts. No upfront conversion to the Megatron Indexed Dataset is required.

The main data formats currently supported include:

- Alpaca format: Usually contains the `instruction`, `input`, and `output` fields.
- ShareGPT format: Usually stores conversation messages in the `conversations` field.
- OpenAI format: Uses the `messages` field to store conversation messages. Each message usually contains `role` and `content`. The OpenAI format is a data field specification, not a specific dataset.

For detailed format descriptions, see:

- [Alpaca-style datasets](../../../tools/data_process_sft_alpaca_style.md)
- [ShareGPT and OpenAI-style datasets](../../../tools/data_process_sft_sharegpt_style.md)

### Configuring Datasets

In version 26.1.0, the training dataset is configured through `data.dataset` in the corresponding `examples/fsdp2/**/*.yaml` configuration file by default. Dataset parameters support both inline configuration and dataset names registered in `dataset_info.json`.

Take the Qwen3-8B fine-tuning configuration as an example. The `alpaca_full` dataset registered in `dataset_info.json` is used by default:

```yaml
data:
  dataset: alpaca_full
```

**Using Inline Configuration in YAML**

Inline configuration is suitable for quickly validating local datasets. Modify `data.dataset` in the corresponding YAML file:

```yaml
data:
  dataset:
    file_name: ./dataset/train.json
    formatting: alpaca
```

**Using Registered Dataset Names in YAML**

When you need to reuse a dataset, edit `configs/fsdp2/data/dataset_info.json` to add the dataset configuration:

```json
{
  "alpaca_demo": {
    "file_name": "./alpaca_demo.json",
    "formatting": "alpaca"
  },
  "sharegpt_demo": {
    "file_name": "./sharegpt_demo.jsonl",
    "formatting": "sharegpt"
  },
  "openai_demo": {
    "file_name": "./openai_demo.jsonl",
    "formatting": "openai",
    "columns": {
      "messages": "messages"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system",
      "observation_tag": "tool",
      "function_tag": "function_call"
    }
  }
}
```

Then modify `data.dataset` in the corresponding YAML file to specify the dataset by its registered name:

```yaml
data:
  dataset: alpaca_demo
```

To mix multiple datasets, separate multiple registered names with commas:

```yaml
data:
  dataset: alpaca_demo,sharegpt_demo
```

### Configuring Fine-Tuning Parameters

Parameters such as the model weight path, training dataset, parallel scale, batch size, and output directory are all saved in the accompanying YAML file. For detailed configuration, see the [Qwen3-8B fine-tuning configuration file](../../../../../../examples/fsdp2/qwen3/tune_qwen3_8b_4k_fsdp2_A2.yaml). The YAML example content is as follows:

```yaml
model:
  model_name_or_path: /home/data/Qwen3-8B/
  trust_remote_code: False
  train_from_scratch: False

data:
  dataset: alpaca_full
  template: qwen3
  cutoff_len: 4096
  max_samples: 100000
  overwrite_cache: True
  preprocessing_num_workers: 1

parallel:
  fsdp_size: 8
  fsdp_modules:
    - model.layers.{*}
    - model.embed_tokens
    - lm_head
  tp_size: 1
  ep_size: 1
  ep_modules:
    - model.layers.{*}.mlp.experts
  ep_fsdp_size: 1
  ep_fsdp_modules:
    - model.layers.{*}.mlp.experts
  ep_dispatcher: eager
  recompute: True
  recompute_modules:
    - model.layers.{*}

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 1
  dataloader_num_workers: 4
  seed: 42
  dataloader_drop_last: True
  output_dir: ./output
  optimizer: adamw
  lr: 1e-05
  weight_decay: 0.01
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_epsilon: 1e-08
  max_grad_norm: 1.0
  lr_scheduler_type: cosine
  warmup_ratio: 0.0
  min_lr: 1e-06
  max_steps: 2000
  save_steps: 500
  logging_steps: 1
```

The main parameters in the YAML file are described as follows:

| Parameter | Description | Default value |
| --- | --- | --- |
| `model_name_or_path` | Directory of HuggingFace model weights. | `/home/data/Qwen3-8B/` |
| `dataset` | Inline configuration or registered name of the training dataset. | `alpaca_full` |
| `template` | Prompt template selected based on the model. | `qwen3` |
| `cutoff_len` | Maximum length of a training sequence after tokenization. Content exceeding this length is truncated. | `4096` |
| `max_samples` | Maximum number of samples used from each dataset, mainly for debugging. | `100000` |
| `overwrite_cache` | Whether to overwrite the generated data processing cache. | `True` |
| `preprocessing_num_workers` | Number of processes used for data preprocessing. | `1` |
| `fsdp_size` | Fully sharded data parallel size. It should be consistent with `NPUS_PER_NODE * NNODES`. | `8` |
| `fsdp_modules` | Model modules sharded with FSDP. It cannot be empty. | `model.layers.{*}`, `model.embed_tokens`, `lm_head` |
| `tp_size` | Tensor parallel size. | `1` |
| `ep_size` | Expert parallel size. Set it to `1` for dense models. | `1` |
| `ep_fsdp_size` | FSDP size within the expert parallel group. Set it to `1` for dense models. | `1` |
| `recompute` | Whether to enable activation recomputation, trading computation overhead for memory space. | `True` |
| `per_device_train_batch_size` | Training batch size per device. | `1` |
| `gradient_accumulation_steps` | Number of gradient accumulation steps. | `1` |
| `output_dir` | Output directory of training checkpoints. | `./output` |
| `lr` | Initial learning rate. | `1e-05` |
| `max_steps` | Maximum number of training steps. When greater than 0, it overrides `num_train_epochs`. | `2000` |
| `save_steps` | Step interval for saving training checkpoints. | `500` |

For a complete description of the parameters, see [FSDP2 Parameters](../../../features/fsdp2/arguments.md).

### Configuring the Fine-Tuning Script

Open the corresponding `.sh` file to configure distributed training parameters. Model weights, datasets, parallel scale, and the output directory are configured in the accompanying YAML file. For detailed configuration, see the [Qwen3-8B fine-tuning startup script](../../../../../../examples/fsdp2/qwen3/tune_qwen3_8b_4k_fsdp2_A2.sh).

The single-node 8-NPU configuration is as follows:

```bash
NPUS_PER_NODE=8                              # Number of NPUs used on the current node
MASTER_ADDR=localhost                        # IP address of the master node. Set it to localhost for single-node training
MASTER_PORT=6499                             # Communication port of the master node
NNODES=1                                     # Total number of nodes participating in training. Set it to 1 for single-node training
NODE_RANK=0                                  # Index of the current node. Set it to 0 for single-node training
WORLD_SIZE=$((NPUS_PER_NODE * NNODES))       # Total number of NPUs participating in training
```

The multi-node configuration example is as follows:

```bash
NPUS_PER_NODE=8                              # Number of NPUs used on each node
MASTER_ADDR="master node IP"                 # All nodes are configured with the master node IP. localhost is not allowed
MASTER_PORT=6499                             # All nodes use the same master node communication port
NNODES=2                                     # Total number of nodes participating in training
NODE_RANK="current node index"               # Value range: 0 to NNODES-1. It cannot be duplicated across nodes
WORLD_SIZE=$((NPUS_PER_NODE * NNODES))       # Total number of NPUs participating in training
```

`MASTER_ADDR`, `MASTER_PORT`, and `NNODES` must be the same on different nodes. `NODE_RANK` starts from 0 and cannot be duplicated.

The startup script reads the accompanying YAML file with the following command. You do not need to repeatedly configure training parameters in the startup command:

```bash
torchrun $DISTRIBUTED_ARGS train_fsdp2.py \
  examples/fsdp2/qwen3/tune_qwen3_8b_4k_fsdp2_A2.yaml
```

> [!NOTE]
>
> - In this example, `fsdp_size` should be consistent with `NPUS_PER_NODE * NNODES`.
> - During multi-node training, ensure that each node can correctly access the model and dataset paths.

### Starting Fine-Tuning

After configuring the parameters, run the following command in the repository root directory:

```bash
bash examples/fsdp2/qwen3/tune_qwen3_8b_4k_fsdp2_A2.sh
```

For multi-node training, run the startup script on all nodes and set the corresponding `NODE_RANK` on each node. Training logs are saved in the `logs/` directory by default, and training checkpoints are saved in the directory specified by `training.output_dir` in the YAML file.

After the training runs for a while, you can see logs similar to the following in the terminal:

```shell
INFO [2026-06-22 19:25:37] >>  iteration        1/    2000 | consumed samples:          8 | consumed tokens:        564 | elapsed time per iteration (ms): 7827.39 | learning rate: 1.666667E-07 | global batch size:     8 | lm loss: 3.316887E+00 | grad norm: 70.959 | max_memory_allocated(GB): 19.07 | max_memory_reserved(GB): 20.90 |
INFO [2026-06-22 19:25:38] >>  iteration        2/    2000 | consumed samples:         16 | consumed tokens:       1357 | elapsed time per iteration (ms): 1331.74 | learning rate: 3.333333E-07 | global batch size:     8 | lm loss: 2.443476E+00 | grad norm: 41.986 | max_memory_allocated(GB): 19.07 | max_memory_reserved(GB): 22.43 |
INFO [2026-06-22 19:25:38] >>  iteration        3/    2000 | consumed samples:         24 | consumed tokens:       2113 | elapsed time per iteration (ms): 981.08 | learning rate: 5.000000E-07 | global batch size:     8 | lm loss: 2.669216E+00 | grad norm: 45.335 | max_memory_allocated(GB): 19.07 | max_memory_reserved(GB): 22.43 |
```

When the terminal continuously outputs information such as the iteration number, learning rate, loss value, gradient norm, and memory usage, the training task is running normally.

## Usage Constraints

- This guide applies to full-parameter SFT with the FSDP2 backend. It does not cover other post-training methods such as LoRA, DPO, PPO, and reward model training.
- `model_name_or_path`, the dataset, and the output directory should be valid paths accessible in the training environment.
- `template` should match the target model. Otherwise, the training input format may be inconsistent with the model's expectations.
- The model scale, sequence length, batch size, and parallel scale need to be adjusted based on the number of devices and memory capacity.
