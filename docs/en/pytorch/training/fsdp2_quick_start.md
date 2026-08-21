# Quick Start: Training the Qwen3-8B Model with the FSDP2 Backend

## Overview

This document provides a simple example to help developers who are new to MindSpeed LLM quickly start model training tasks and complete LLM pretraining and fine-tuning tasks using the FSDP2 backend.
Using Qwen3-8B as an example, this document guides you through LLM pretraining and fine-tuning tasks. The main steps are:

- Prepare the environment: Set up the environment according to the installation guide.
- Prepare weights and datasets: Download the Qwen3-8B open-source model weights from Hugging Face and obtain the Alpaca dataset.
- Start training tasks: Use the FSDP2 backend to run pretraining and fine-tuning on Ascend NPUs.

> [!NOTE]
>
> MindSpeed LLM supports <term>Ascend 950 products</term>, <term>Atlas A3 training products</term>, and <term>Atlas A2 training products</term>. Single-NPU on-chip memory of 64 GB or more is required. For details, see [Supported Models in the PyTorch Framework](../models/supported_models.md).
>
> The current Qwen3-8B example script uses `NPUS_PER_NODE=8`, which requires 8 NPUs. If your actual configuration is lower than this, you may encounter Out of Memory (OOM) issues.

Developer prerequisites:

- Basic experience with PyTorch.
- Basic Python development experience.
- Basic understanding of Fully Sharded Data Parallel (FSDP).

## Environment Preparation

Click [MindSpeed Quick Installation](https://www.hiascend.com/en/developer/software/mindspeed/download) and follow the guidance to set up the environment. For detailed installation instructions, see [MindSpeed LLM Installation](install_guide.md).

## Weight and Dataset Preparation

1. Obtain open-source model weights.

    Create a directory to store the weight files.

    ```shell
    mkdir -p ./model_from_hf/qwen3_hf
    cd ./model_from_hf/qwen3_hf
    ```

    Obtain model weight files from Hugging Face or ModelScope (choose one).

    Method 1: From Hugging Face

    ```shell
    # Use wget to download the weight files
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/config.json
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/generation_config.json
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/merges.txt
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/model-00001-of-00005.safetensors
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/model-00002-of-00005.safetensors
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/model-00003-of-00005.safetensors
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/model-00004-of-00005.safetensors
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/model-00005-of-00005.safetensors
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/model.safetensors.index.json
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/tokenizer.json
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/tokenizer_config.json
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/vocab.json
    ```

    Method 2: From ModelScope (recommended for China)

    ```shell
    # Use wget to download the weight files (from ModelScope)
    wget https://www.modelscope.cn/models/Qwen/Qwen3-8B/resolve/master/config.json
    wget https://www.modelscope.cn/models/Qwen/Qwen3-8B/resolve/master/generation_config.json
    wget https://www.modelscope.cn/models/Qwen/Qwen3-8B/resolve/master/merges.txt
    wget https://www.modelscope.cn/models/Qwen/Qwen3-8B/resolve/master/model-00001-of-00005.safetensors
    wget https://www.modelscope.cn/models/Qwen/Qwen3-8B/resolve/master/model-00002-of-00005.safetensors
    wget https://www.modelscope.cn/models/Qwen/Qwen3-8B/resolve/master/model-00003-of-00005.safetensors
    wget https://www.modelscope.cn/models/Qwen/Qwen3-8B/resolve/master/model-00004-of-00005.safetensors
    wget https://www.modelscope.cn/models/Qwen/Qwen3-8B/resolve/master/model-00005-of-00005.safetensors
    wget https://www.modelscope.cn/models/Qwen/Qwen3-8B/resolve/master/model.safetensors.index.json
    wget https://www.modelscope.cn/models/Qwen/Qwen3-8B/resolve/master/tokenizer.json
    wget https://www.modelscope.cn/models/Qwen/Qwen3-8B/resolve/master/tokenizer_config.json
    wget https://www.modelscope.cn/models/Qwen/Qwen3-8B/resolve/master/vocab.json
    ```

    Verify the correctness and integrity of the weight files by calculating the SHA-256 values with `sha256sum`.

    ```shell
    # Open the file details page to obtain the SHA-256 value at https://huggingface.co/Qwen/Qwen3-8B/blob/main/model-00001-of-00005.safetensors or https://www.modelscope.cn/models/Qwen/Qwen3-8B/file/view/master/model-00001-of-00005.safetensors
    sha256sum ./model-00001-of-00005.safetensors
    sha256sum ./model-00002-of-00005.safetensors
    sha256sum ./model-00003-of-00005.safetensors
    sha256sum ./model-00004-of-00005.safetensors
    sha256sum ./model-00005-of-00005.safetensors
    cd ../..
    ```

2. Obtain the dataset.

    Obtain the Alpaca dataset from Hugging Face.

    ```shell
    mkdir dataset
    cd dataset/
    # Hugging Face dataset link. Choose one
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    # ModelScope dataset link. Choose one
    wget https://www.modelscope.cn/datasets/angelala00/tatsu-lab-alpaca/resolve/master/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    ```

3. Set environment variables.

    ```shell
    source /usr/local/Ascend/cann/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    ```

    The preceding commands use the default installation paths after installation by the root user. Replace them with the actual `set_env.sh` paths in your environment.

## Launching Pretraining

At this stage, we modify the pretraining example script and configuration file, and launch model pretraining. The specific steps are:

1. Edit the pretraining startup script.

    ```shell
    vi examples/fsdp2/qwen3/pretrain_qwen3_8b_4k_fsdp2_A2.sh
    ```

2. Modify and save the distributed parameter configuration.

    The following example shows the configuration:

    ```bash
    source examples/fsdp2/env_config.sh                 # Load the NPU environment variable configuration

    NPUS_PER_NODE=8             # Use 8 NPUs on a single node
    MASTER_ADDR=localhost       # On a single node, use the IP address of this node or localhost. For multi-node training, set all nodes to the master node IP address
    MASTER_PORT=6499            # Port number of this node: 6499
    NNODES=1                    # Configure this according to the number of participating nodes. Use 1 for a single node. For multiple nodes, set the number of nodes
    NODE_RANK=0                 # On a single node, the rank is 0. For multi-node training, use 0 to NNODES-1. Do not reuse the same value on different nodes. The node with NODE_RANK 0 is the master node
    WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))            # World size

    DISTRIBUTED_ARGS="
        --nproc_per_node $NPUS_PER_NODE \
        --nnodes $NNODES \
        --node_rank $NODE_RANK \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT
    "

    torchrun $DISTRIBUTED_ARGS train_fsdp2.py examples/fsdp2/qwen3/pretrain_qwen3_8b_4k_fsdp2_A2.yaml
    ```

3. Edit the training parameter configuration file.

    ```shell
    vi examples/fsdp2/qwen3/pretrain_qwen3_8b_4k_fsdp2_A2.yaml
    ```

4. Modify and save the training parameter configuration.

    The following example shows the configuration:

    ```yaml
    model:
      model_name_or_path: ./model_from_hf/qwen3_hf/     # Replace with the path of the downloaded Hugging Face weights
      trust_remote_code: False
      train_from_scratch: False

    data:
      dataset:
        file_name: ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet   # Replace with the path of the downloaded dataset
      template: qwen3
      cutoff_len: 4096
      max_samples: 100000
      overwrite_cache: True
      preprocessing_num_workers: 1
      data_manager_type: mg

    parallel:
      fsdp_size: 8                                       # Number of FSDP shards. Must be equal to NPUS_PER_NODE * NNODES (that is, the world size)
      fsdp_modules:
        - model.layers.{*}
        - model.embed_tokens
        - lm_head
      tp_size: 1
      recompute: True
      recompute_modules:
        - model.layers.{*}

    training:
      stage: pt                                          # Training stage. pt indicates pretraining
      per_device_train_batch_size: 1
      gradient_accumulation_steps: 1
      dataloader_num_workers: 4
      seed: 42
      dataloader_drop_last: True
      output_dir: ./output                               # Output directory for saving training weights
      optimizer: adamw
      lr: 1e-05
      max_steps: 2000
      save_steps: 500
      logging_steps: 1
    ```

5. Run the pretraining script.

    ```shell
    bash examples/fsdp2/qwen3/pretrain_qwen3_8b_4k_fsdp2_A2.sh
    ```

    After the script runs for a while, you can see the following output in the terminal.

    ```shell
    INFO [2026-06-22 19:28:40] >>  iteration        1/    2000 | consumed samples:          8 | consumed tokens:      32768 | elapsed time per iteration (ms): 10520.60 | learning rate: 9.999994E-06 | global batch size:     8 | lm loss: 1.304194E+00 | grad norm: 8.515 | max_memory_allocated(GB): 19.07 | max_memory_reserved(GB): 20.87 |
    INFO [2026-06-22 19:28:41] >>  iteration        2/    2000 | consumed samples:         16 | consumed tokens:      65536 | elapsed time per iteration (ms): 1879.90 | learning rate: 9.999978E-06 | global batch size:     8 | lm loss: 1.232217E+00 | grad norm: 4.346 | max_memory_allocated(GB): 19.65 | max_memory_reserved(GB): 25.59 |
    INFO [2026-06-22 19:28:43] >>  iteration        3/    2000 | consumed samples:         24 | consumed tokens:      98304 | elapsed time per iteration (ms): 1769.99 | learning rate: 9.999950E-06 | global batch size:     8 | lm loss: 1.134654E+00 | grad norm: 1.550 | max_memory_allocated(GB): 19.65 | max_memory_reserved(GB): 25.59 |
    ```

    When training enters iterations, it indicates that training is proceeding normally.

> [!NOTE]
>
> - For multi-node training, start the pretraining script in multiple terminals at the same time. The pretraining script in each terminal differs only in the `NODE_RANK` parameter. `MASTER_ADDR` is the IP address of the master node for all terminals. All other parameters stay the same.
> - The FSDP2 backend automatically shards the model parameters across NPUs so that each NPU stores only part of the parameters, enabling training of ultra-large-scale models.

## Launching Fine-Tuning

At this stage, we modify the fine-tuning example script and configuration file, and launch model fine-tuning. The specific steps are:

1. Edit the fine-tuning startup script.

    ```shell
    vi examples/fsdp2/qwen3/tune_qwen3_8b_4k_fsdp2_A2.sh
    ```

2. Modify and save the distributed parameter configuration.

    The following example shows the configuration:

    ```bash
    source examples/fsdp2/env_config.sh                 # Load the NPU environment variable configuration

    NPUS_PER_NODE=8             # Use 8 NPUs on a single node
    MASTER_ADDR=localhost       # On a single node, use the IP address of this node or localhost. For multi-node training, set all nodes to the master node IP address
    MASTER_PORT=6499            # Port number of this node: 6499
    NNODES=1                    # Configure this according to the number of participating nodes. Use 1 for a single node. For multiple nodes, set the number of nodes
    NODE_RANK=0                 # On a single node, the rank is 0. For multi-node training, use 0 to NNODES-1. Do not reuse the same value on different nodes. The node with NODE_RANK 0 is the master node
    WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))            # World size

    DISTRIBUTED_ARGS="
        --nproc_per_node $NPUS_PER_NODE \
        --nnodes $NNODES \
        --node_rank $NODE_RANK \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT
    "

    torchrun $DISTRIBUTED_ARGS train_fsdp2.py examples/fsdp2/qwen3/tune_qwen3_8b_4k_fsdp2_A2.yaml
    ```

3. Edit the fine-tuning parameter configuration file.

    ```shell
    vi examples/fsdp2/qwen3/tune_qwen3_8b_4k_fsdp2_A2.yaml
    ```

4. Modify and save the fine-tuning parameter configuration.

    The following example shows the configuration:

    ```yaml
    model:
      model_name_or_path: ./model_from_hf/qwen3_hf/     # Replace with the path of the downloaded Hugging Face weights
      trust_remote_code: False
      train_from_scratch: False

    data:
      dataset:
          file_name: "/home/dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet"   # Replace with the path of the downloaded dataset
          formatting: "alpaca"                                                       # Dataset data format. The default is alpaca
      template: qwen3
      cutoff_len: 4096
      max_samples: 100000
      overwrite_cache: True
      preprocessing_num_workers: 1

    parallel:
      fsdp_size: 8                                       # Number of FSDP shards. Must be equal to NPUS_PER_NODE * NNODES (that is, the world size)
      fsdp_modules:
        - model.layers.{*}
        - model.embed_tokens
        - lm_head
      tp_size: 1
      recompute: True
      recompute_modules:
        - model.layers.{*}

    training:
      per_device_train_batch_size: 1
      gradient_accumulation_steps: 1
      dataloader_num_workers: 4
      disable_shuffling: 1
      seed: 42
      dataloader_drop_last: True
      output_dir: ./output                               # Output directory for fine-tuning results
      optimizer: adamw
      lr: 1e-05
      weight_decay: 0.01
      max_grad_norm: 1.0
      lr_scheduler_type: cosine
      warmup_ratio: 0.0
      min_lr: 1e-06
      max_steps: 2000
      save_steps: 500
      logging_steps: 1
    ```

5. Run the fine-tuning script.

    ```shell
    bash examples/fsdp2/qwen3/tune_qwen3_8b_4k_fsdp2_A2.sh
    ```

    After the script runs for a while, you can see the following output in the terminal.

    ```shell
    INFO [2026-06-22 19:25:37] >>  iteration        1/    2000 | consumed samples:          8 | consumed tokens:        564 | elapsed time per iteration (ms): 7827.39 | learning rate: 1.666667E-07 | global batch size:     8 | lm loss: 3.316887E+00 | grad norm: 70.959 | max_memory_allocated(GB): 19.07 | max_memory_reserved(GB): 20.90 |
    INFO [2026-06-22 19:25:38] >>  iteration        2/    2000 | consumed samples:         16 | consumed tokens:       1357 | elapsed time per iteration (ms): 1331.74 | learning rate: 3.333333E-07 | global batch size:     8 | lm loss: 2.443476E+00 | grad norm: 41.986 | max_memory_allocated(GB): 19.07 | max_memory_reserved(GB): 22.43 |
    INFO [2026-06-22 19:25:38] >>  iteration        3/    2000 | consumed samples:         24 | consumed tokens:       2113 | elapsed time per iteration (ms): 981.08 | learning rate: 5.000000E-07 | global batch size:     8 | lm loss: 2.669216E+00 | grad norm: 45.335 | max_memory_allocated(GB): 19.07 | max_memory_reserved(GB): 22.43 |
    ```

    When training enters iterations, it indicates that training is proceeding normally.

> [!NOTE]
>
> - For multi-node fine-tuning, start the fine-tuning script in multiple terminals at the same time. The fine-tuning script in each terminal differs only in the `NODE_RANK` parameter. `MASTER_ADDR` is the IP address of the master node for all terminals. All other parameters stay the same.
> - Fine-tuning uses the Alpaca dataset format by default. To use other datasets, see the "Data Arguments" section in [Full Parameter Reference (Based on the FSDP2 Training Backend)](../features/fsdp2/arguments.md).

The script includes training parameters. The following table explains some of them.

**Table 1** Training script parameters

|Parameter|Description|Example Configuration|
|----|----|----|
|`fsdp_size`|Size of fully sharded data parallelism. It must be equal to the world size (that is, `NPUS_PER_NODE * NNODES`).|Positive integer, such as 8 or 16|
|`fsdp_modules`|List of model layer structures on which FSDP is enabled.|`["model.layers.{*}", "model.embed_tokens", "lm_head"]`|
|`recompute`|Whether to enable recomputation to save on-chip memory at the cost of some computation.|`True`/`False`|
|`recompute_modules`|Model layer structures on which activation recomputation is enabled.|`["model.layers.{*}"]`|
|`data_manager_type`|Data manager type. `mg` indicates the pretraining scenario and does not need to be configured for fine-tuning.|Pretraining: `mg`. Fine-tuning: not configured|
|`dataset`|Dataset. Use a name from the dataset registry or a local dataset path.|`alpaca_full`, `sharegpt4_zh`|
|`template`|Name of the template used to build prompts during fine-tuning.|`qwen3`, `gpt`|
|`cutoff_len`|Truncation length for input sequences after tokenization. Sequences longer than this value are truncated.|`2048`, `4096`, `16384`|
|`trust_remote_code`|Whether to allow loading models from custom modeling files on Hugging Face.|`True`/`False`|
|`train_from_scratch`|Whether to train the model from scratch with random weights without loading the model weights.|`True`/`False`|

> [!NOTE]
>
> 🔍 For a complete description of the parameters, see [Full Parameter Reference (Based on the FSDP2 Training Backend)](../features/fsdp2/arguments.md).
