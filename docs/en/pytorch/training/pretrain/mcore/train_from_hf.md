# Training with Online Data and Weight Loading (Train_from_HF)

## Use Cases

In earlier versions, users had to perform weight conversion and data preprocessing offline first. They converted Hugging Face-format weights to the Megatron format, converted the raw dataset to a Megatron-formatted dataset, and then started training. This separate process increased complexity and time overhead.

This feature integrates data preprocessing, weight conversion, and training into a single workflow. You can start training with a single script.

- The integrated weight conversion training feature supports loading from and saving to Hugging Face during training. If the loading directory contains `.safetensors` files or `.bin` files for a Mamba model, and you do not explicitly set conversion flags, the system automatically enables weight conversion. It converts Hugging Face weights to the Megatron format for training, and after each distributed weight save during training, it converts the weights back to the Hugging Face format. You do not need to run a separate weight conversion step. This provides one-click integration from Hugging Face weights into training tasks.
- The data preprocessing feature automatically detects and converts raw data files during model training. You do not need to convert the raw data manually. The system determines whether the input path points to a raw data format, such as `.jsonl` or `.parquet`, and automatically completes the data format conversion during training initialization.

## How to Use

### 1. Weight Conversion

Currently, only standalone storage and shared storage are supported. During training initialization, the system automatically detects whether the training environment uses shared storage.

The system detects the weight files in the loading directory. If the loading directory contains `.safetensors` files or `.bin` files for a Mamba model, and the user does not explicitly set conversion flags, it automatically enables weight conversion. It converts Hugging Face weights to the Megatron format for training, and after each distributed weight save during training, it converts the weights back to the Hugging Face format.

When the `--load` parameter points to a Hugging Face weight path, the path must contain files such as `config.json` for loading parameters. If you do not specify `--model-type-hf`, the system tries to read `{load}/config.json` and infer the supported model type from the configuration file. You must set this parameter manually for Mamba models.

#### Quick Start

When the loading directory contains Hugging Face-format weights, the system automatically enables bidirectional conversion:

```bash
# Load Hugging Face weights, convert automatically, and train.
    --load /path/to/huggingface/model \
    --save /path/to/save/training/results \
    --model-type-hf <model_type>  # Optional. The system infers it automatically.
```

#### Debugging Features

**Scenario 1: Loading from Hugging Face and train**

In the pre-training and fine-tuning scripts `pretrain_xxx.sh` or `tune_xxx.sh`, add the following parameters to enable weight conversion:

```bash
# Load from the Hugging Face format and convert to the Megatron format for training.
--enable-hf2mg-convert \
--model-type-hf <model_type>
```

**Scenario 2: Enabling bidirectional weight conversion**

In the pre-training and fine-tuning scripts `pretrain_xxx.sh` or `tune_xxx.sh`, add the following parameters to enable weight conversion:

```bash
# Save weights in both formats during training. This is equivalent to enabling bidirectional conversion automatically.
    --enable-hf2mg-convert \
    --enable-mg2hf-convert \
    --model-type-hf <model_type>
```

**Scenario 3: Converting Megatron weights saved during training to Hugging Face format**

In the pre-training and fine-tuning scripts `pretrain_xxx.sh` or `tune_xxx.sh`, add the following parameters to enable weight conversion:

```bash
# Convert the Megatron-format weights saved during training to the Hugging Face format each time they are saved.
    --enable-mg2hf-convert \
    --model-type-hf  <model_type>
```

**Scenario 4: Converting only the final saved model to Hugging Face format**

In the pre-training and fine-tuning scripts `pretrain_xxx.sh` or `tune_xxx.sh`, add the following parameters to enable weight conversion:

```bash
# Convert only the Megatron-format weights saved after training ends to the Hugging Face format, and do not convert the Megatron-format weights saved during training.
    --enable-mg2hf-convert \
    --only-convert-last-checkpoint \
    --model-type-hf  <model_type>
```

**Parameters**

| Parameter | Type | Default | Required | Description |
|------|------|--------|------|------|
| `--model-type-hf` | str | None | Optional* | Hugging Face model type. Multiple pretrained model types are supported. |
| `--enable-hf2mg-convert` | flag | False | Optional | Enables Hugging Face-to-Megatron weight conversion only. |
| `--enable-mg2hf-convert` | flag | False | Optional | Enables Megatron-to-Hugging Face weight conversion only. |
| `--only-convert-last-checkpoint` | flag | False | Optional | Converts only the final distributed weights at the end of training. |
| `--mg-save-dir` | str | None | Optional | When converting Hugging Face-to-Megatron weights, specify the Megatron weight save directory. |
| `--hf-save-dir` | str | None | Optional | When converting Megatron-to-Hugging Face weights, specify the Hugging Face weight save directory. |
| `--hf-cfg-dir` | str | None | Optional | Hugging Face configuration directory. Because Megatron-to-Hugging Face conversion generates only the weights and `model.safetensors.index.json`, and does not generate configuration files, the system copies the configuration files from the original Hugging Face model to the Hugging Face weight directory created by the conversion. |

*Note: For special models such as Mamba, you must specify `--model-type-hf` manually.*

#### Notes

1. System resource requirements

    - Drive space: Ensure that you have enough drive space to store the converted weights.
    - Conversion time: After training initialization, weights are converted automatically. Depending on the model size, this process takes about 2 minutes to 2 hours. Please wait patiently.
    - Permission requirements: Ensure that you have read and write permissions for all the following relevant paths:
      - `{load}` - model loading path.
      - `{save}` - training save path.
      - `{mg-save-dir}` - Megatron weight save directory, if specified.
      - `{hf-save-dir}` - Hugging Face weight save directory, if specified.
      - `{hf-cfg-dir}` - Hugging Face configuration directory, if specified.

2. Hugging Face-to-Megatron conversion (`--enable-hf2mg-convert`) constraints

    - Loading path required: When you enable this feature, you must set the `--load` parameter to specify the Hugging Face weight source. Training from random initialization is not supported.
    - Megatron weights not supported: After you enable this parameter, offline-converted Megatron-format weights are not supported.
    - Storage path rules:
      - If you specify `--mg-save-dir`, the converted Megatron weights are saved to that path.
      - If you do not specify it, they are saved by default in the `{load}/megatron_cache_tp{TP}pp{PP}ep{EP}` directory.
      - The training process automatically uses this path as the weight loading path.

3. Megatron-to-Hugging Face conversion (`--enable-mg2hf-convert`) constraints

    - Save path required: When you enable this feature, you must set the `--save` parameter to specify the training output path.
    - Shared storage only: This feature is supported only in a shared-storage environment.
    - LoRA/QLoRA not supported: It does not support Megatron-to-Hugging Face conversion for weights fine-tuned with LoRA or QLoRA.
    - Storage path rules:
      - If you specify `--hf-save-dir`, the converted Hugging Face weights are saved in the `{hf-save-dir}/mg2hf_iteration{iteration}/` directory.
      - If you do not specify it, they are saved by default in the `{save}/mg2hf_iteration{iteration}` directory.
    - Configuration file handling:
      - If you specify `--hf-cfg-dir`, the system copies configuration files from this directory to the converted Hugging Face weight directory.
      - If you do not specify it but bidirectional conversion is enabled, the system copies configuration files from the `{load}` directory.
      - Note: Megatron-to-Hugging Face conversion itself does not generate configuration files. It must copy them from an existing configuration source.

### 2. Data Preprocessing

#### Basic Command

If you want to use the data preprocessing feature, refer to the parameter descriptions and add the relevant parameters for your scenario. Then change the input dataset path specified by `--data-path` to control whether preprocessing runs. The currently supported forms are as follows:

| Input Form | Example | Description |
|-----------|-------|------|
| **Raw file** | `/data/train.jsonl` | Raw dataset. The system automatically identifies it and converts it to `.bin/.idx` format. |
| **Converted prefix** | `/data/train_text_document` | Already converted format. You can use it directly. |

#### Parameters

| Parameter | Type | Required | Description |
|------|------|------|------|
| `--data-path` | `str / list` | Yes | Raw data path or converted prefix. |
| `--handler-name` | `str` | Yes | Name of the data processing handler. |
| `--append-eod` | `bool` | No | Whether to append the `<eod>` token to the end of documents. |
| `--prompt-type` | `str` | Yes (fine-tuning) | Specify the fine-tuning prompt template. |
| `--json-keys` | `list` | No | Fields to extract. The default is `["text"]`. |
| `--workers` | `int` | No | Number of data processing threads. |
| `--n-subs` | `int` | No | Number of data subsets. Multi-process sharding. |
| `--pack` | `bool` | No | Whether to pack samples. Fine-tuning scenario. |
| `--neat-pack` | `bool` | No | Switch that enables the use of a jagged `attention_mask` during computation in pack scenarios. Fine-tuning scenario. |
| `--enable-thinking` | `str` | No | Whether to enable thinking mode. Fine-tuning scenario. |
| `--output-prefix` | `str` | No | Prefix of the output dataset file after conversion. |

Note:

- If you do not specify `--output-prefix`, the processed data file is generated in the same directory as the raw dataset by default.

### 3. Example

Using Qwen3-8B fine-tuning as an example, if you want to enable both data preprocessing and integrated weight-conversion training, add the following parameters to the [Qwen3-8B fine-tuning script](../../../../../../examples/mcore/qwen3/tune_qwen3_8b_4K_full_ptd.sh):

```bash
DATA_PATH="/path/your_dataset/xxx.parquet"
CKPT_LOAD_DIR="/path/to/huggingface_model/Qwen3-8B"
--data-path "$DATA_PATH" \
--load "$CKPT_LOAD_DIR" \
--enable-hf2mg-convert \
--model-type-hf qwen3 \
--handler-name AlpacaStyleInstructionHandler \
--prompt-type qwen3 \
```

## Usage Constraints

- The currently supported Hugging Face model types are `qwen3, qwen3-moe, deepseek3, glm45-air, bailing_mini, qwen3-next, seed-oss, deepseek32, magistral, deepseek2-lite`.

- The current automatic dataset conversion feature supports only the following raw data formats: `parquet, arrow, csv, json, jsonl, txt`.

- The current weight conversion feature `--enable-mg2hf-convert` supports only standalone storage or shared storage environments.

- The current weight conversion feature `--enable-mg2hf-convert` does not support Megatron-to-Hugging Face conversion for weights fine-tuned with LoRA or QLoRA.
