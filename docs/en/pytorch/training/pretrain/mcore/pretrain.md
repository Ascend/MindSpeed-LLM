# Distributed Pretraining for LLMs

## Use Cases

LLM pretraining is a core step in language model development. Its goal is to enable the model to learn language patterns and world knowledge from large-scale unlabeled corpora. The pretraining process focuses more on language modeling itself than on a specific task. Take GPT-like models as an example. They are typical autoregressive language models whose core idea is to predict the next token from historical context. Pretraining repeatedly optimizes this predictive ability. As a result, the model gradually learns to understand context, keep sentences coherent, and master higher-level language structures. Therefore, it provides general-purpose language representation capabilities for many downstream tasks.

Pretraining data is usually plain text and is not task-oriented. For example:

```json
{"text": "Today is a nice day. Let's go hiking together."}
{"text": "Deep learning is changing the world."}
{"text": "The emergence of AI has driven the development of human society."}
```

## Usage Instructions

> [!NOTE]
>
> - If you need to use Pack mode during data preprocessing, refer to [Pack Mode for Distributed LLM Pretraining](./pretrain_eod.md).
> - During pretraining, you can skip loading initial weights. In this case, the model weights are randomly initialized. If you need to load weights, convert them in advance. For details, refer to [Checkpoint Conversion v1](../../../tools/checkpoint_convert_hf_mcore.md) or [Checkpoint Conversion v2](../../../tools/checkpoint_convert_hf_mcore_large_params.md).

The following example uses the Qwen3-8B model to show how to start pretraining. The process for distributed pretraining is as follows:

**Figure 1**  Pretraining process diagram

![Pretraining process diagram](../../../figures/pretrain/process_of_pretraining.png)

1. Environment setup

   Before starting pretraining, refer to [MindSpeed LLM Installation Guide](../../install_guide.md) to complete the environment setup, and ensure that the Ascend NPU toolkit environment variables are configured as follows:

    ```shell
    source /usr/local/Ascend/cann/set_env.sh     # Modify this to the actual installed Toolkit package path.
    source /usr/local/Ascend/nnal/atb/set_env.sh # Modify this to the actual installed nnal package path.
    ```

2. Pretraining data preprocessing

   First, prepare the raw dataset. Common pretraining datasets include:
   - [Alpaca Dataset](https://huggingface.co/datasets/tatsu-lab/alpaca)
   - [Enwiki Dataset](https://huggingface.co/datasets/lsb/enwiki20230101)
   - [C4 Dataset](https://huggingface.co/datasets/allenai/c4)
   - [ChineseWebText](https://huggingface.co/datasets/CASIA-LM/ChineseWebText)

   Then, use the [Enwiki Dataset](https://huggingface.co/datasets/lsb/enwiki20230101) as an example to run data preprocessing. For detailed script configuration, refer to [Qwen3 pretraining data processing script](../../../../../../examples/mcore/qwen3/data_convert_qwen3_pretrain.sh). Modify the following content in the script:

    ```bash
    source /usr/local/Ascend/cann/set_env.sh # Modify this to the actual installed Toolkit package path.

    ......
    --input ./dataset/train-00000-of-00042-d964455e17e96d5a.parquet  # Raw dataset path.
    --tokenizer-name-or-path ./model_from_hf/qwen3_hf                # Hugging Face tokenizer path.
    --output-prefix ./finetune_dataset/alpaca                        # Save path.
    ......
    ```

   Parameters for data preprocessing:

   - `input`: You can point this parameter to a dataset directory or a specific file. If it is a directory, the tool processes all files. It supports the `.parquet`, `.csv`, `.json`, `.jsonl`, `.txt`, and `.arrow` formats. Files in the same folder must use the same format.
   - `handler-name`: The current pretraining pipeline uses `GeneralPretrainHandler` by default. It supports pretraining-style data and extracts the `text` column, as shown here:

        ```shell
        [
            {"text": "document"},
            {"other keys": "optional content"}
        ]
        ```

   - `json-keys`: The list of column names to extract from the file. The default is `text`. You can use multiple inputs such as `text`, `input`, and `title` depending on your needs and the dataset contents, for example:

        ```shell
        --json-keys text input output
        ```

   - `n-subs`: A parallel acceleration parameter for data preprocessing. When the dataset is large, you can speed up preprocessing by setting `--n-subs` to the number of parallel workers. The preprocessing process splits the raw dataset into `n-subs` subsets, processes the subsets in parallel, and then merges them to improve speed. Add this parameter when the dataset is larger than 1 GB.

   Finally, after you set the relevant parameters, run the data preprocessing script:

    ```shell
    bash examples/mcore/qwen3/data_convert_qwen3_pretrain.sh
    ```

3. Configuring single-node or multi-node pretraining scripts

   For detailed parameter configuration, refer to [Qwen3-8B pretraining script](../../../../../../examples/mcore/qwen3/pretrain_qwen3_8b_4K_ptd.sh). The environment variable configuration in the script is described in [Model Script Environment Variables](../../../features/mcore/environment_variable.md).

   After you confirm the environment variables, modify the node-related configuration in the script. The single-node and multi-node configurations are as follows:

   - Single-node configuration

        ```bash
        NPUS_PER_NODE=8  # Number of devices on a single node.
        MASTER_ADDR=localhost
        MASTER_PORT=6000
        NNODES=1
        NODE_RANK=0
        WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
        ```

   - Multi-node configuration

        ```bash
        # Configure distributed parameters according to the actual distributed cluster.
        NPUS_PER_NODE=8                    # Number of devices on each node.
        MASTER_ADDR="your master node IP"  # Change this to the IP address of the master node. It cannot be localhost.
        MASTER_PORT=6000
        NNODES=2                           # Number of nodes in the cluster. Fill in the actual value.
        NODE_RANK="current node id"        # The RANK of the current node. Different nodes cannot reuse the same value. The master node is 0, and other nodes can be 1, 2, and so on.
        WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
        ```

   Then, modify the related path parameters and the model partition configuration in the script:

    ```bash
    CKPT_SAVE_DIR="your model save ckpt path" # Path for saving the weights after training completes.
    DATA_PATH="your data path"                # Dataset path. Fill in the path of the data saved during preprocessing.
    TOKENIZER_PATH="your tokenizer path"      # Vocabulary path. Fill in the path of the downloaded open-source vocabulary.
    CKPT_LOAD_DIR="your model ckpt path"      # Weight loading path. Fill in the path of the weights saved during weight conversion.

    TP=1 # TP size for model weight conversion. In this example, it is 1.
    PP=4 # PP size for model weight conversion. In this example, it is 4.
    ```

   Other parameter descriptions in the script:

   - `DATA_PATH`: The dataset path. Note that the file generated by actual data preprocessing adds `_text_document` to the end. You only need to fill in the dataset file prefix. For example, if the actual relative dataset path is `./finetune_dataset/alpaca/alpaca_text_document.bin`, you only need to fill in `./finetune_dataset/alpaca/alpaca_text_document`.
   - `CKPT_LOAD_DIR`: Weight load path. During pretraining, you can choose to initialize the model weights randomly. In that case, you do not need to configure this parameter, and you must comment out the `--load ${CKPT_LOAD_DIR} \` line in the pretraining script.
   - `tokenizer-type`: When the parameter value is `PretrainedFromHF`, the tokenizer path only needs to point to the model folder and does not need to point to the `tokenizer.model` file. When the parameter value is not `PretrainedFromHF`, for example `Qwen3Tokenizer`, you need to point to the `tokenizer.model` file. The example is as follows:

        ```bash
        # `tokenizer-type` is `PretrainedFromHF`.
        TOKENIZER_PATH="./model_from_hf/Qwen3-8B/"
        --tokenizer-name-or-path ${TOKENIZER_PATH}

        # `tokenizer-type` is not `PretrainedFromHF`.
        TOKENIZER_MODEL="./model_from_hf/Qwen3-8B/tokenizer.model"
        --tokenizer-model ${TOKENIZER_MODEL} \
        ```

   > [!NOTE]
   >
   > - Enclose the provided paths in double quotation marks.
   > - For multi-node training, ensure that the model path and dataset path on each machine are correct. If data sharing is not configured, add the `no-shared-storage` parameter to the training launch script. After you set this parameter, the system determines whether a non-master node needs to load data based on the distributed parameters and checks the corresponding cache and generated data.

4. Starting pretraining

   After you configure the pretraining script, run it to start pretraining. In a multi-node scenario, you need to start the script simultaneously in multiple terminals:

    ```shell
    bash examples/mcore/qwen3/pretrain_qwen3_8b_4K_ptd.sh
    ```

## Usage Constraints

If you need to store logs in the script, create a `logs` folder in the run directory.
