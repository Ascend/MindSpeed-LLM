# Quick Start: Qwen3-0.6B Model Pretraining and Fine-Tuning

## Overview

This document provides a simple example to help developers who are new to MindSpeed LLM quickly start model training tasks and complete instruction fine-tuning with single-sample format data based on a pre-trained language model.

Using the Qwen3-0.6B model as an example, this document guides you through the pretraining and fine-tuning tasks for a large language model. The main steps are:

- Prepare the environment: Set up the environment according to the repository guidance.
- Obtain open-source model weights: Download the original Qwen3-0.6B model from Hugging Face.
- Start training tasks: Run pretraining and fine-tuning on Ascend NPUs.

Developer prerequisites:

- Basic experience with MindSpore.
- Basic Python development experience.
- Basic familiarity with the Megatron-LM repository.

## Environment Preparation

### Environment Setup

For the MindSpore framework, see [MindSpeed LLM Installation Guide](install_guide.md).

### Obtain Open-Source Model Weights

1. Obtain the model weight files from Hugging Face.

    ```shell
    # Create a directory to store the weight files.
    mkdir -p ./model_from_hf/qwen3_hf
    cd ./model_from_hf/qwen3_hf

    # Use wget to download the weight files.
    wget https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/config.json
    wget https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/generation_config.json
    wget https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/merges.txt
    wget https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors
    wget https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json
    wget https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer_config.json
    wget https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/vocab.json
    ```

2. Verify the integrity of the model weight files with `sha256sum`.

    ```shell
    # Use sha256sum to calculate the SHA-256 value.
    # Open the file details to get the SHA-256 value at https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/model.safetensors.
    sha256sum model.safetensors
    ```

### Weight Conversion

Ascend MindSpeed LLM requires model weights in the MCore format. Here, you convert the original Hugging Face weight format to the MCore format. For details, see [Weight Conversion](../pytorch/tools/checkpoint_convert_hf_mcore.md#21-converting-hugging-face-weights-to-mcore-weights).

Use the official conversion script to obtain the corresponding sharded mg weights.

1. Edit the weight conversion script.

    ```shell
    cd MindSpeed-LLM
    vi examples/mindspore/qwen3/ckpt_convert_qwen3_hf2mcore.sh
    ```

2. Finish modifying the conversion script and save it.
   The following example shows the adjusted hf2mcore weight conversion script.

    ```bash
    # Change the set_env.sh path according to your actual environment.
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    source /usr/local/Ascend/cann/set_env.sh

    python ./mindspeed_llm/mindspore/convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --load-dir ./model_from_hf/qwen3_hf/ \
    --save-dir ./model_weights/qwen3_mcore/ \
    --tokenizer-model ./model_from_hf/qwen3_hf/tokenizer.json \
    --params-dtype bf16 \
    --model-type-hf qwen3 \
    --ai-framework mindspore
    ```

    **Table 1** Weight conversion parameters

    | Parameter | Description | Required |
    |---|---|---|
    | `--use-mcore-models` | Convert to the MCore format. | ✅ |
    | `--model-type GPT` | Specify the model type as the GPT series. | ✅ |
    | `--target-tensor-parallel-size` | Tensor parallel size. Recommended value: `1`. | ✅ |
    | `--target-pipeline-parallel-size` | Pipeline parallel size. Recommended value: `1`. | ✅ |
    | `--load-model-type` | The type of the loaded weights. It can be `hf` or `mg`. | ✅ |
    | `--save-model-type` | The type of the saved weights. It can be `hf` or `mg`. | ✅ |
    | `--load-dir` | Weight file load path. | ✅ |
    | `--save-dir` | Weight file save path. | ✅ |
    | `--model-type-hf` | Hugging Face model type. | ✅ |
    | `--params-dtype` | Weight precision after conversion. The default is `fp16`. If the source files use `bf16`, set this option to `bf16`. | ✅ |
    | `--spec` | Transformer layer structure configuration. | ✅ |
    | `--tokenizer-model` | Tokenizer model file path. | ✅ |
    | `--ai-framework` | Training framework. Supported values are `pytorch` and `mindspore`. The default is `pytorch`. Set this option to `mindspore`. | ✅ |

3. Run the weight conversion script.

    ```shell
    bash examples/mindspore/qwen3/ckpt_convert_qwen3_hf2mcore.sh
    ```

    After the script runs, you should see log output like the following, which indicates that the weight conversion succeeded:

    ```shell
    successfully saved checkpoint from iteration 1 to ./model_weights/qwen3_mcore/
    INFO:root:Done!
    ```

> [!NOTE]
>
> - For the Qwen3-0.6B model, the recommended sharding configuration is `tp1pp1`, which matches the configuration above.
> - MindSpore performs weight conversion on the Device side by default. For LLMs, this may cause an OOM risk. Therefore, you are advised to modify `convert_ckpt.py` manually and add the following code when importing the package to run weight conversion on the CPU side:
>
>     ```python
>     import mindspore as ms
>     ms.set_context(device_target="CPU", pynative_synchronize=True)
>     import torch
>     torch.configs.set_pyboost(False)
>     ```
>
> - Model weights converted by MindSpore cannot be used directly for PyTorch training or inference.

## Launching Pretraining

At this stage, you preprocess the dataset based on the downloaded Hugging Face raw data and then launch pretraining. The specific steps are:

1. Data preprocessing.
2. Launch the pretraining task.

### Data Preprocessing

By preprocessing data in various formats in advance, you avoid repeated loading of the raw data. All data is stored in two unified files, `.bin` and `.idx`. For details, see [Pretraining Dataset Processing](../pytorch/tools/data_process_pretrain.md).

The following example uses the Alpaca dataset for pretraining data processing.

1. Obtain the dataset metadata.

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
    vi examples/mindspore/qwen3/data_convert_qwen3_pretrain.sh
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
    bash examples/mindspore/qwen3/data_convert_qwen3_pretrain.sh
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
    cd MindSpeed-LLM
    vi examples/mindspore/qwen3/pretrain_qwen3_0point6b_4K_ms.sh
    ```

2. Modify and save the pretraining parameter configuration as shown below:

    ```bash
    NPUS_PER_NODE=8           # Use 8 NPUs on a single node.
    MASTER_ADDR=localhost     # On a single node, use the local IP address. On multiple nodes, set all nodes to master_ip.
    MASTER_PORT=6011          # Port number of this node.
    NNODES=1                  # Configure this according to the number of participating nodes. Use 1 for a single node. For multiple nodes, set the number of nodes. The master node rank is 0, and its IP address is master_ip.
    NODE_RANK=0               # On a single node, the rank is 0. For multiple nodes, use 0 to NNODES-1, and do not repeat ranks across nodes.
    WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))

    # Configure the weight save path, weight load path, tokenizer path, and dataset path according to the actual environment. All nodes in a multi-node setup must have the following data.
    CKPT_LOAD_DIR="./model_weights/qwen3_mcore/"  # Weight load path. Use the path saved during weight conversion.
    CKPT_SAVE_DIR="./ckpt/qwen3-0point6b"         # Weight save path after training completes.
    DATA_PATH="./dataset/alpaca_text_document"    # Dataset path. Use the path saved during data preprocessing. Note that you must add the suffix. If Alpaca preprocessing generates alpaca_text_document.bin and alpaca_text_document.idx, append alpaca_text_document to the dataset path.
    TOKENIZER_PATH="./model_from_hf/qwen3_hf/"    # Tokenizer path. Use the path of the downloaded open-source weight tokenizer.

    TP=1                # Set this to 1 to match --target-tensor-parallel-size 1 in weight conversion.
    PP=1                # Set this to 1 to match --target-pipeline-parallel-size 1 in weight conversion.
    SEQ_LEN=4096        # Set seq_length to 4096.
    MBS=1               # Set micro-batch-size to 1.
    GBS=8               # Set global-batch-size to 8.
    TRAIN_ITERS=2000    # Set the number of training iterations.
    ```

3. Set the environment variables.

    ```shell
    source /usr/local/Ascend/cann/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    ```

    The preceding commands use the default installation paths after a root user installation. Replace them with the actual `set_env.sh` paths in your environment.

4. Run the pretraining script.

    ```shell
    bash examples/mindspore/qwen3/pretrain_qwen3_0point6b_4K_ms.sh
    ```

    **Figure 1** Launch Pretraining.
    ![img_2.png](../pytorch/figures/quick_start/running_log.png)
