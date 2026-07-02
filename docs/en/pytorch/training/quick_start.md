# Quick Start: Qwen3-8B Model Pretraining and Fine-Tuning

## Overview

This document provides a simple example to help developers who are new to MindSpeed LLM quickly start model training tasks and complete instruction fine-tuning with single-sample format data based on a pre-trained language model.

Using Qwen3-8B as an example, this document guides you through the pretraining and fine-tuning tasks for a large language model. The main steps are:

- Prepare the environment: Set up the environment according to the repository guidance.
- Obtain open-source model weights: Download the original Qwen3-8B model from Hugging Face.
- Start training tasks: Run pretraining and fine-tuning on Ascend NPUs.

Developer prerequisites:

- Basic experience with PyTorch.
- Basic Python development experience.
- Basic familiarity with the Megatron-LM repository.

## Environment Preparation

### Environment Setup

For the PyTorch framework, see [MindSpeed LLM Installation Guide](install_guide.md).

### Obtain Open-Source Model Weights

1. Obtain the model weight files from Hugging Face.

    ```shell
    # Create a directory to store the weight files.
    mkdir -p ./model_from_hf/qwen3_hf
    cd ./model_from_hf/qwen3_hf

    # Use wget to download the weight files.
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

2. Verify the integrity of the model weight files with `sha256sum`.

    ```shell
    # Use sha256sum to calculate the SHA-256 value.
    # Open the file details page to obtain the SHA-256 value at https://huggingface.co/Qwen/Qwen3-8B/blob/main/model-00001-of-00005.safetensors.
    sha256sum model-00001-of-00005.safetensors
    sha256sum model-00002-of-00005.safetensors
    sha256sum model-00003-of-00005.safetensors
    sha256sum model-00004-of-00005.safetensors
    sha256sum model-00005-of-00005.safetensors
    ```

### Weight Conversion

Ascend MindSpeed LLM requires model weights in the MCore format. Here, you convert the original Hugging Face weight format to the MCore format. For details, see [Weight Conversion](../tools/checkpoint_convert_hf_mcore.md#21-converting-hugging-face-weights-to-mcore-weights).

Use the official conversion script to obtain the corresponding sharded MCore weights.

1. Edit the weight conversion script.

    ```shell
    cd ../..
    vi examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh
    ```

2. Finish modifying the conversion script and save it.
   The following example shows the adjusted hf2mcore weight conversion script.

    ```bash
    # Change the set_env.sh path according to your actual environment.
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    source /usr/local/Ascend/cann/set_env.sh

    python convert_ckpt_v2.py \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 1 \           # TP partition size.
    --target-pipeline-parallel-size 2 \         # PP partition size.
    --load-dir ./model_from_hf/qwen3_hf/ \      # Hugging Face weight path.
    --save-dir ./model_weights/qwen3_mcore/ \   # MCore weight save path.
    --model-type-hf qwen3
    ```

    **Table 1** Weight conversion parameters

    | Parameter | Description | Required |
    |---|---|---|
    | `--target-tensor-parallel-size` | Tensor parallel size. Recommended value: `1`. | ✅ |
    | `--target-pipeline-parallel-size` | Pipeline parallel size. Recommended value: `2`. | ✅ |
    | `--load-model-type` | Type of the loaded weights. It can be `hf` or `mg`. | ✅ |
    | `--save-model-type` | Type of the saved weights. It can be `hf` or `mg`. | ✅ |
    | `--load-dir` | Weight file load path. | ✅ |
    | `--save-dir` | Weight file save path. | ✅ |
    | `--model-type-hf` | Hugging Face model type. | ✅ |

3. Run the weight conversion script.

    ```shell
    bash examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh
    ```

    After the script runs, you should see log output like the following, which indicates that the weight conversion succeeded:

    ```shell
    INFO:root:Saving to ./model_weights/qwen3_mcore/iter_0000001/mp_rank_00_001/model_optim_rng.pt
    INFO:root:Done!
    ```

> [!NOTE]
>
> For the Qwen3-8B model, the recommended sharding configuration here is `tp1pp2`, which matches the configuration above.

## Launching Pretraining

At this stage, we preprocess the dataset based on the downloaded Hugging Face raw data and then launch pretraining. The specific steps are:

1. Data preprocessing.
2. Launch the pretraining task.

### Data Preprocessing

Preprocess the data in advance to avoid repeatedly loading the raw data. The processed data is stored in two files, `.bin` and `.idx`. For details, see [Pretraining Dataset Processing](../tools/data_process_pretrain.md).

The following example uses the Alpaca dataset for pretraining data processing.

1. Obtain the raw dataset.

    ```shell
    mkdir dataset
    cd dataset/
    # Hugging Face dataset link. Choose one.
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    # ModelScope dataset link. Choose one.
    wget https://www.modelscope.cn/datasets/angelala00/tatsu-lab-alpaca/resolve/master/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    ```

2. Edit the pretraining data processing script.

    ```shell
    vi examples/mcore/qwen3/data_convert_qwen3_pretrain.sh
    ```

3. Finish modifying the data processing script and save it.

    The following example shows the adjusted data processing script.

    ```bash
    # Change the set_env.sh path according to your actual environment.
    source /usr/local/Ascend/cann/set_env.sh

    python ./preprocess_data.py \
      --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
      --tokenizer-name-or-path ./model_from_hf/qwen3_hf/ \         # Ensure that this path matches.
      --tokenizer-type PretrainedFromHF \
      --handler-name GeneralPretrainHandler \
      --output-prefix ./dataset/alpaca \                           # The pretraining dataset generates alpaca_text_document.bin and .idx.
      --json-keys text \
      --workers 4 \
      --log-interval 1000
    ```

    **Table 2** Data preprocessing parameters

    | Parameter | Description | Required |
    |---|---|---|
    | `--input` | Supported input formats include dataset directories or files. If you specify a directory, the script processes all files in it. Supported formats are `.parquet`, `.csv`, `.json`, `.jsonl`, `.txt`, and `.arrow`. All files in the same directory must use the same format. | ✅ |
    | `--tokenizer-type` | Specifies the tokenizer type. When the value is `PretrainedFromHF`, you only need to fill in the model directory for the vocabulary path. | ✅ |
    | `--tokenizer-name-or-path` | Works with `tokenizer-type`. This is the tokenizer source directory of the target model and is used for dataset conversion. | ✅ |
    | `--handler-name` | Specifies the dataset handler class. | ✅ |
    | `--output-prefix` | File name prefix of the converted dataset output. | ✅ |
    | `--workers` | Multi-process dataset processing. | ✅ |
    | `--log-interval` | Number of steps between progress updates. | ✅ |
    | `--json-keys` | List of column names extracted from the file. The default is `text`, and you can use multiple inputs such as `text`, `input`, and `title` according to the actual data. | ✅ |

4. Run the pretraining data processing script.

    ```shell
    bash examples/mcore/qwen3/data_convert_qwen3_pretrain.sh
    ```

    The pretraining dataset processing result looks like this:

    ```shell
    ./dataset/alpaca_text_document.bin
    ./dataset/alpaca_text_document.idx
    ```

### Launching the Pretraining Task

After you finish dataset processing and weight conversion, you can start the pretraining task.

1. Edit the example script.

    ```shell
    vi examples/mcore/qwen3/pretrain_qwen3_8b_4K_ptd.sh
    ```

2. Modify and save the pretraining parameter configuration as shown below:

    ```bash
    NPUS_PER_NODE=8           # Number of NPUs on a single node.
    MASTER_ADDR=localhost     # On a single node, use the IP address of this node. For multi-node training, set all nodes to master_ip.
    MASTER_PORT=6000          # Port number of this node.
    NNODES=1                  # Configure this according to the number of participating nodes. Use 1 for a single node. For multiple nodes, set the number of nodes.
    NODE_RANK=0               # On a single node, the rank is 0. For multi-node training, use 0 to NNODES-1. Do not reuse the same value on different nodes. The master node rank is 0, and its IP address is master_ip.
    WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))

    # Configure the weight save path, weight load path, vocabulary path, and dataset path according to the actual environment. All nodes in a multi-node setup must have the following data.
    CKPT_LOAD_DIR="./model_weights/qwen3_mcore/"   # Weight load path. Use the path saved during weight conversion.
    CKPT_SAVE_DIR="./ckpt/qwen3-8b"                # Weight save path after training completes.
    DATA_PATH="./dataset/alpaca_text_document"     # Dataset path. Use the path saved during data preprocessing. Note that you must add the suffix. If Alpaca preprocessing generates alpaca_text_document.bin and .idx, append alpaca_text_document to the dataset path.
    TOKENIZER_PATH="./model_from_hf/qwen3_hf/"     # Vocabulary path. Use the vocabulary path from the downloaded open-source weights.

    TP=1                # Set this to 1 to match the weight conversion setting `--target-tensor-parallel-size 1`.
    PP=2                # Set this to 2 to match the weight conversion setting `--target-pipeline-parallel-size 2`.
    SEQ_LENGTH=4096     # Set seq_length to 4096.
    MBS=1               # Set micro-batch-size to 1.
    GBS=64              # Set global-batch-size to 64.
    TRAIN_ITERS=2000    # Set the number of training iterations.
    ```

3. Set the environment variables.

    ```shell
    source /usr/local/Ascend/cann/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    ```

    The preceding commands use the default installation paths after installation by the root user. Replace them with the actual `set_env.sh` paths in your environment.

4. Run the pretraining script.

    ```shell
    bash examples/mcore/qwen3/pretrain_qwen3_8b_4K_ptd.sh
    ```

    **Figure 1** Launching pretraining

    ![img_2.png](../figures/quick_start/running_log.png)

    The script includes training parameters and optimization features. The following table explains some of them.

    **Table 3** Training script parameter description

    | Parameter | Description |
    |----|----|
    | `--use-mcore-models` | Use the MCore branch to run the model. |
    | `--disable-bias-linear` | Remove the linear bias term to match the original Qwen model. |
    | `--group-query-attention` | Enable the GQA attention mechanism. |
    | `--num-query-groups 8` | Use with GQA to set the number of groups to 8. |
    | `--position-embedding-type rope` | Use RoPE for positional encoding. |
    | `--untie-embeddings-and-output-weights` | Untie the weights of the output layer and the embedding layer as required by the original model. |
    | `--bf16` | Ascend chips support the `bf16` precision type well, which can significantly improve training speed. |

> [!NOTE]
>
> - For multi-node training, start the pretraining script in multiple terminals at the same time. The pretraining script in each terminal differs only in the `NODE_RANK` parameter. All other parameters stay the same.
> - If you use multi-node training and do not configure shared data, add the `--no-shared-storage` parameter to the training launch script. After you set this parameter, the system determines whether a non-master node needs to load data based on the distributed parameters and checks the corresponding cache and generated data.

## Launching Fine-Tuning

At this stage, we preprocess the dataset based on the downloaded Hugging Face raw data and then launch fine-tuning. The specific steps are:

1. Data preprocessing.
2. Launch the fine-tuning task.

### Data Preprocessing

Preprocess the data in advance to avoid repeatedly loading the raw data. The processed data is stored in two files, `.bin` and `.idx`. For details, see [Alpaca-Style Datasets](../tools/data_process_sft_alpaca_style.md).

The following example uses the Alpaca dataset for data preprocessing.

1. Obtain the raw dataset.

    ```shell
    mkdir dataset
    cd dataset/
    # Hugging Face dataset link.
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    ```

2. Edit the data preprocessing script.

    ```shell
    vi examples/mcore/qwen3/data_convert_qwen3_instruction.sh
    ```

3. Finish modifying the data preprocessing script and save it.

    ```bash
    # Change the set_env.sh path according to your actual environment.
    source /usr/local/Ascend/cann/set_env.sh
    mkdir ./finetune_dataset

    python ./preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./model_from_hf/qwen3_hf/ \
    --output-prefix ./finetune_dataset/alpaca \
    --handler-name AlpacaStyleInstructionHandler \
    --tokenizer-type PretrainedFromHF \
    --workers 4 \
    --log-interval 1000 \
    --enable-thinking true \
    --prompt-type qwen3
    ```

    **Table 4** Data preprocessing parameters

    | Parameter | Description | Required |
    |---|---|---|
    | `--input` | Supported input formats include dataset directories or files. If you specify a directory, the script processes all files in it. Supported formats are `.parquet`, `.csv`, `.json`, `.jsonl`, `.txt`, and `.arrow`. All files in the same directory must use the same format. | ✅ |
    | `--tokenizer-type` | Specifies the tokenizer type. When the value is `PretrainedFromHF`, you only need to fill in the model directory for the vocabulary path. | ✅ |
    | `--tokenizer-name-or-path` | Works with `tokenizer-type`. This is the tokenizer source directory of the target model and is used for dataset conversion. | ✅ |
    | `--output-prefix` | File name prefix of the converted dataset output. | ✅ |
    | `--workers` | Multi-process dataset processing. | ✅ |
    | `--handler-name` | Specifies the dataset handler class. | ✅ |
    | `--log-interval` | Number of steps between progress updates. | ✅ |
    | `--enable-thinking` | Fast-thinking and slow-thinking template switch. | Optional |
    | `--prompt-type` | Used to specify the model template. | ✅ |

4. Run the data processing script.

    ```shell
    bash examples/mcore/qwen3/data_convert_qwen3_instruction.sh
    ```

    The fine-tuning dataset processing results are as follows:

    ```shell
    ./finetune_dataset/alpaca_packed_attention_mask_document.bin
    ./finetune_dataset/alpaca_packed_attention_mask_document.idx
    ./finetune_dataset/alpaca_packed_input_ids_document.bin
    ./finetune_dataset/alpaca_packed_input_ids_document.idx
    ./finetune_dataset/alpaca_packed_labels_document.bin
    ./finetune_dataset/alpaca_packed_labels_document.idx
    ```

### Launching Fine-Tuning Task

After you finish dataset processing and weight conversion, you can start the fine-tuning task.

1. Edit the example script.

    ```shell
    vi examples/mcore/qwen3/tune_qwen3_8b_4K_full_ptd.sh
    ```

2. Modify and save the fine-tuning parameter configuration as shown below:

    ```bash
    NPUS_PER_NODE=8  # Number of NPUs on a single node.
    MASTER_ADDR=localhost
    MASTER_PORT=6000
    NNODES=1
    NODE_RANK=0
    WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
    ```

    Modify the relevant path parameters and model partition configuration in the script:

    ```bash
    CKPT_LOAD_DIR="./model_weights/qwen3_mcore/"  # Use the path saved after weight conversion.
    CKPT_SAVE_DIR="./ckpt/qwen3-8b"               # Use the weight save path after fine-tuning completes.
    DATA_PATH="./finetune_dataset/alpaca"         # Use the processed data path.
    TOKENIZER_PATH="./model_from_hf/qwen3_hf/"    # Use the tokenizer path of the model.
    TP=1                                          # The TP size used during weight conversion. In this example, it is 1.
    PP=2                                          # The PP size used during weight conversion. In this example, it is 2.
    ```

3. Set the environment variables.

    ```shell
    source /usr/local/Ascend/cann/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    ```

    The preceding commands use the default installation paths after installation by the root user. Replace them with the actual `set_env.sh` paths in your environment.

4. Run the fine-tuning script.

    ```shell
    bash examples/mcore/qwen3/tune_qwen3_8b_4K_full_ptd.sh
    ```

    **Figure 2** Launching fine-tuning

    ![tune_log.png](../figures/quick_start/tune_log.png)

    The script includes fine-tuning parameters and optimization features. The following table explains some of them.

    **Table 5** Fine-tuning script parameter description

    | Parameter | Description |
    |----|----|
    | `--finetune` | Start fine-tuning mode. |
    | `--stage` | Training method. |
    | `--is-instruction-dataset` | Specify the instruction fine-tuning dataset to use so that the model is fine-tuned on the specified instruction data. |
    | `--prompt-type` | Specify the model template so that the base model can develop better conversational ability after fine-tuning. You can view the available options in the [templates.json](../../../../configs/finetune/templates.json) file. |
    | `--no-pad-to-seq-lengths` | Support dynamic sequence-length fine-tuning. By default, padding is applied in multiples of 8. |
    | `--sequence-parallel` | Enable sequence parallelism. |
    | `--use-distributed-optimizer` | Enable the distributed optimizer. |
    | `--use-flash-attn` | Enable Flash Attention. |
    | `--bf16` | Ascend chips support the `bf16` precision type well, which can significantly improve training speed. |
