# MindSpeed LLM FSDP2 Backend Training Guide

This guide uses gpt-oss 20B as a fine-tuning example to explain how to use the FSDP2 backend of MindSpeed LLM for large language model training. It covers environment setup, configuration details, and the full training startup workflow.

## 1. Environment Setup

See the [MindSpeed LLM Installation Guide](../../install_guide.md) for environment setup.

## 2. Directory Structure

The fine-tuning scripts are located in the following paths.

```bash
MindSpeed-LLM/
├── examples/fsdp2/gpt_oss/
│   ├── tune_gpt_oss_20b_varlen_fsdp2_A3.yaml        # Fine-tuning configuration
│   └── tune_gpt_oss_20b_varlen_fsdp2_A3.sh          # Startup script
├── configs/fsdp2/data/dataset_info.json             # Dataset registry
└── train_fsdp2.py                                   # FSDP2 training entry point
```

## 3. Configuration Changes

### 3.1 Model Path Configuration

```yaml
model:
  model_name_or_path: /path/to/gpt-oss-20b-hf/      # Replace this with your local model path or Hugging Face model ID.
  tokenizer_name_or_path: None                      # Specify this when the model and tokenizer paths differ.
```

### 3.2 Dataset Configuration

#### Option 1: Inline Configuration (Quick Validation with a Single Dataset)

```yaml
  dataset:
    file_name: "./my_data.json"                     # Replace this with the data file path.
    formatting: "alpaca"                            # Choose alpaca, sharegpt, or another format based on the data format.
  cutoff_len: 2048                                  # Truncate input sequences after tokenization when they exceed this length.
```

#### Option 2: Registration through `dataset_info.json`

1. Edit `configs/fsdp2/data/dataset_info.json` and add dataset entries.

    ```json
    {
      "alpaca_full": {
        "file_name": "./train-00000-of-00001.parquet"
      },
      "sharegpt4_zh": {
        "file_name": "./sharegpt_zh.jsonl",
        "formatting": "sharegpt"
      }
    }
    ```

2. Reference them in the YAML configuration.

    ```yaml
    data:
      dataset: alpaca_full, sharegpt4_zh                # Fine-tuning datasets. You can enter comma-separated dataset names, and multiple datasets are supported.
      template: gpt                                     # Template name for building prompts.
      cutoff_len: 2048                                  # Truncate input sequences after tokenization when they exceed this length.
    ```

## 4. Distributed Training Startup Script

`examples/fsdp2/gpt_oss/tune_gpt_oss_20b_varlen_fsdp2_A3.sh`

```bash
source examples/fsdp2/env_config.sh                 # Load the NPU environment variable configuration.

NPUS_PER_NODE=8                                     # Number of NPUs per node.
NNODES=1                                            # Total number of nodes.
MASTER_ADDR=localhost                               # Master node IP address.
MASTER_PORT=6499                                    # Master node communication port.

torchrun \
  --nproc_per_node $NPUS_PER_NODE \
  # Start eight processes on each node.
  --nnodes $NNODES \
  # Use one node in total.
  --node_rank 0 \
  # Current node rank. Adjust this for multi-node training.
  --master_addr $MASTER_ADDR \
  # Master node address.
  --master_port $MASTER_PORT \
  # Master node port.
  train_fsdp2.py examples/fsdp2/gpt_oss/tune_gpt_oss_20b_varlen_fsdp2_A3.yaml
  # Start the training entry point.
```

## 5. Start Training

Run the following command in the repository root.

```bash
bash examples/fsdp2/gpt_oss/tune_gpt_oss_20b_varlen_fsdp2_A3.sh
```

This starts the training job.

## 6. Other Configuration Parameters

### 6.1 Model Configuration

```yaml
model:
  model_id: gpt_oss                                 # Model type identifier. New model types must be registered in the `ModelRegistry` class in `mindspeed_llm/fsdp2/models/model_registry.py`.
  model_name_or_path: /path/to/gpt-oss-20b-hf/      # Local path of the model. This field is required. An exception is raised if it is not specified.
  trust_remote_code: False                          # Whether to allow models from custom modeling files on Hugging Face. Use this to adapt custom model architectures.
  train_from_scratch: False                         # Whether to train the model from scratch with randomly initialized weights without loading model weights.
  tokenizer_name_or_path: None                      # Path or name of the tokenizer. Specify this when it differs from model_name_or_path.
```

### 6.2 Parallelism Strategy

```yaml
parallel:
  fsdp_size: 8                                      # Fully Sharded Data Parallel (FSDP) size. Model parameters are sharded across eight devices.
  fsdp_modules:                                     # List of model layer structures that enable FSDP. This field is required and must not be empty.
    - model.layers.{*}                              # Enable FSDP sharding for all Transformer layers.
    - model.embed_tokens                            # Enable FSDP sharding for the token embedding layer.
    - lm_head                                       # Enable FSDP sharding for the language model output head.
  tp_size: 1                                        # Tensor Parallel size. Model tensors are split across multiple devices by column or row.
  ep_size: 1                                        # Expert Parallel size. This applies to MoE models and splits different experts across multiple devices.
  ep_modules:                                       # Model layer structures that enable expert parallelism. This applies only to MoE models.
    - model.layers.{*}.mlp.experts                  # Enable expert parallelism for the expert modules in all layers.
  ep_fsdp_size: 1                                   # FSDP size within the expert parallel group. This shards the parameters of individual experts on top of expert parallelism.
  ep_fsdp_modules:                                  # Model layer structures that enable FSDP within the expert parallel group.
    - model.layers.{*}.mlp.experts                  # Further shard the parameters inside the expert modules.
  ep_dispatcher: eager                              # MoE expert parallel dispatch strategy: eager, fused, or mc2.
  recompute: True                                   # Whether to enable gradient checkpointing, which saves memory at the cost of extra computation.
  recompute_modules:                                # Model layer structures that enable activation recomputation.
    - model.layers.{*}                              # Enable recomputation for all Transformer layers.
  cp_size: 1                                        # Context Parallel size. Input sequence context is split across multiple devices.
  cp_type: ulysses                                  # Context parallel algorithm type. Currently only the ulysses algorithm is supported.
```

### 6.3 Training Parameters

```yaml
training:
  per_device_train_batch_size: 1                    # Training batch size per device.
  gradient_accumulation_steps: 1                    # Number of gradient accumulation steps. Gradients from multiple batches are accumulated before backpropagation and parameter updates.
  dataloader_num_workers: 1                         # Number of data loading subprocesses. This speeds up data preprocessing.
  disable_shuffling: 1                              # Whether to disable shuffling of the training set.
  seed: 42                                          # Random seed set at the start of training to keep experiments reproducible.
  dataloader_drop_last: True                        # Whether to drop the last incomplete batch when the dataset size is not divisible by the batch size.
  output_dir: ./output                              # Output directory for training results, including model checkpoints, logs, and prediction results. This field is required.
  optimizer: adamw                                  # Optimizer type. Currently only AdamW is supported.
  lr: 1e-05                                         # Initial learning rate for the AdamW optimizer.
  weight_decay: 0.01                                # Weight decay coefficient for the AdamW optimizer.
  adam_beta1: 0.9                                   # beta1 parameter of the AdamW optimizer. This controls the exponential decay rate of the first moment.
  adam_beta2: 0.95                                  # beta2 parameter of the AdamW optimizer. This controls the exponential decay rate of the second moment.
  adam_epsilon: 1e-08                               # epsilon parameter of the AdamW optimizer for numerical stability.
  max_grad_norm: 1.0                                # Maximum norm for gradient clipping to prevent gradient explosion.
  lr_scheduler_type: cosine                         # Learning rate scheduler type: cosine, linear, or constant.
  warmup_ratio: 0.0                                 # Ratio of linear warmup steps to the total training steps.
  min_lr: 1e-06                                     # Minimum learning rate for the cosine scheduler to avoid training stagnation caused by an excessively low learning rate.
  num_train_epochs: 3.0                             # Total number of training epochs. This parameter is overridden when max_steps is greater than 0.
  max_steps: -1                                     # Total number of training steps. This overrides num_train_epochs when greater than 0.
  save_steps: 500                                   # Save a model checkpoint every 500 steps.
  logging_steps: 1                                  # Record training logs every step.
```

### 6.4 Dataset Configuration for Fine-Tuning Scenarios

```yaml
  dataset: alpaca_full                              # Training dataset. You can enter a configuration dictionary or a comma-separated list of dataset names.
  template: gpt                                     # Template name for building prompts.
  cutoff_len: 2048                                  # Truncate input sequences after tokenization when they exceed this length.
  max_samples: 100000                               # For debugging. This truncates the number of samples in each dataset and is mutually exclusive with streaming.
  overwrite_cache: True                             # Whether to overwrite the cached preprocessed dataset.
  preprocessing_num_workers: 1                      # Number of processes for data preprocessing.
```

The `dataset` field supports two configuration methods. You are advised to use the **`dataset_info.json` registration method** to simplify mixed training with multiple datasets.

#### Option 1: Inline Configuration (Suitable for Quick Validation with a Single Dataset)

```yaml
data:
  dataset:
    file_name: "./my_data.json"                     # Data file path.
    formatting: "alpaca"                            # Data format template. Supported formats include alpaca and sharegpt.
```

#### Option 2: Registration through `dataset_info.json`

1. Edit `configs/fsdp2/data/dataset_info.json` and add dataset entries.

    ```json
    {
      "alpaca_full": {
        "file_name": "./train-00000-of-00001.parquet"
      },
      "sharegpt4_zh": {
        "file_name": "./sharegpt_zh.jsonl",
        "formatting": "sharegpt"
      }
    }
    ```

2. Reference them in the YAML configuration.

    ```yaml
    data:
      dataset: alpaca_full, sharegpt4_zh                # Fine-tuning datasets. You can enter comma-separated dataset names configured in dataset_info.json, and multiple datasets are supported.
    ```

### 6.5 Dataset Configuration for Pretraining Scenarios

The dataset configuration method for pretraining scenarios differs from that for fine-tuning scenarios. The following example shows the setup.

```yaml
data:
  dataset: "your original data path, example: /home/train-00000-of-a09b74b3ef9c3b56.parquet" # Enter the original dataset path directly.
  template: gpt                                     # Template name for building prompts.
  cutoff_len: 4096                                  # Truncate input sequences after tokenization when they exceed this length.
  max_samples: 100000                               # For debugging. This truncates the number of samples in each dataset and is mutually exclusive with streaming.
  overwrite_cache: True                             # Whether to overwrite the cached preprocessed dataset.
  preprocessing_num_workers: 1                      # Number of processes for data preprocessing.
  data_manager_type: mg                             # Data manager type. lf indicates fine-tuning data processing, and mg indicates pretraining.
```

For complete parameter descriptions, see [Full Parameter Reference](../../../features/fsdp2/arguments.md).

---
