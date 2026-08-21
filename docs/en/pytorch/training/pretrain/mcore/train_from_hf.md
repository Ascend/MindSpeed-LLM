# Training with Online Data and Weight Loading

## Use Cases

Generally, users need to perform weight conversion and data preprocessing offline first, convert Hugging Face-format weights to Megatron format, convert the raw dataset to a Megatron-formatted dataset, and then start the training task. This separate process increases complexity and time cost.

This feature integrates data preprocessing, weight conversion, and training flows. You can start training tasks with a single script.

- **Weight conversion and training integration:** Supports loading, converting, and saving weights from Hugging Face. By automatically detecting the weight file format in the load directory, the system can automatically enable relevant conversion functions to achieve bidirectional automatic conversion from Hugging Face weights to the Megatron format and training integration. You do not need to execute a separate weight conversion step, achieving one-click startup from Hugging Face weights to training tasks.
- **Automatic data preprocessing:** The data preprocessing feature automatically identifies and converts raw data files during model training. You do not need to convert the raw data manually. The system determines whether the input path points to a raw data format (such as `.jsonl`, `.parquet`, and so on) based on the input path, and automatically completes the data format conversion during training initialization.

## How to Use

### Weight Conversion and Training Integration

Currently, only standalone storage and shared storage modes are supported. During training initialization, the system automatically detects whether the current environment uses shared storage.

The system detects the weight files in the loading directory to infer whether automatic conversion is needed. When the loading directory contains `.safetensors` files or `.bin` format files for Mamba models, and the user does not explicitly set the conversion flag, the system automatically enables the weight conversion feature without requiring manual configuration of other parameters. The system converts Hugging Face-format weights to Megatron-format weights for training, and after each distributed weight save, converts them back to Hugging Face-format weights.

When the `--load` parameter is set to a Hugging Face weight path, ensure that the path contains configuration files such as `config.json` for reading parameter configurations. If the `--model-type-hf` parameter is not specified, the system attempts to read the `{load}/config.json` file and automatically infer the supported model type from the configuration file. Note that for Mamba models, you must configure this parameter manually.

#### Quick Start

When the loading directory contains Hugging Face-format weights (that is, `.safetensors` or `.bin` format files exist), the system automatically enables bidirectional conversion.

```bash
# Load Hugging Face weights, convert automatically, and train
    --load /path_to_huggingface_model \           # Set the Hugging Face weight path
    --save /path_to_save_training_results \       # Set the weight save path after training
    --model-type-hf <model_type>                  # Optional. The system infers it automatically
```

#### Usage Notes

In the pretraining and fine-tuning scripts `pretrain_xxx.sh` or `tune_xxx.sh`, add parameters according to the usage scenario to enable weight conversion. For more details, see [Parameters](#parameters).

- Scenario 1: Loading from Hugging Face and training

    ```bash
    # Load from Hugging Face format, automatically convert to Megatron format for training
    --enable-hf2mg-convert \
    --model-type-hf <model_type>
    ```

- Scenario 2: Enabling bidirectional weight conversion

    ```bash
    # Save weights in both formats during training, equivalent to automatically enabling bidirectional conversion
    --enable-hf2mg-convert \
    --enable-mg2hf-convert \
    --model-type-hf <model_type>
    ```

- Scenario 3: Converting Megatron-format weights saved during training to Hugging Face format

    ```bash
    # Convert Megatron-format weights saved each time during training to Hugging Face format
    --enable-mg2hf-convert \
    --model-type-hf  <model_type>
    ```

- Scenario 4: Converting only the final saved model to Hugging Face format

    ```bash
    # Convert only the Megatron-format weights saved after training ends to Hugging Face format, without converting Megatron-format weights saved during intermediate training steps
    --enable-mg2hf-convert \
    --only-convert-last-checkpoint \
    --model-type-hf  <model_type>
    ```

#### Parameters

**Table 1** Parameters

| Parameter | Type | Default | Required | Description |
|------|------|--------|------|------|
| `--load` | string | None | Yes | Directory for loading model weights. In online weight loading training scenarios, points to the Hugging Face weight path. |
| `--save` | string | None | Yes | Directory for saving model weights after training. |
| `--model-type-hf` | string | None | No | Hugging Face model type. Multiple pretrained model types are supported. |
| `--enable-hf2mg-convert` | bool | False | No | Enables Hugging Face-to-Megatron weight conversion only. |
| `--enable-mg2hf-convert` | bool | False | No | Enables Megatron-to-Hugging Face weight conversion only. |
| `--only-convert-last-checkpoint` | bool | False | No | Converts only the final distributed weights at the end of training. |
| `--mg-save-dir` | string | None | No | When converting Hugging Face-to-Megatron weights, specifies the Megatron weight save directory. |
| `--hf-save-dir` | string | None | No | When converting Megatron-to-Hugging Face weights, specifies the Hugging Face weight save directory. |
| `--hf-cfg-dir` | string | None | No | Hugging Face configuration file directory. |

> [!NOTE]
>
> - For special models such as Mamba, you must specify `--model-type-hf` manually.
> - Because Megatron-to-Hugging Face conversion generates only the weights and `model.safetensors.index.json`, and does not generate configuration files, you must use the `--hf-cfg-dir` parameter to copy configuration files from the original Hugging Face model to the Hugging Face weight directory created by the conversion.

#### Resource Requirements

System resource requirements are as follows:

- Drive space: Ensure that you have enough drive space to store the converted weights.
- Conversion time: After training initialization, the system automatically performs weight conversion. Depending on the model size, the expected time ranges from 2 minutes to 2 hours. Please wait patiently.
- Permission requirements: Ensure that you have read and write permissions for all the following relevant paths:
    - `{load}` - model loading path
    - `{save}` - training save path
    - `{mg-save-dir}` - Megatron weight save directory, if specified
    - `{hf-save-dir}` - Hugging Face weight save directory, if specified
    - `{hf-cfg-dir}` - Hugging Face configuration directory, if specified

#### Constraints

- Hugging Face-to-Megatron conversion (`--enable-hf2mg-convert`)
  - Set the loading path. When enabling this feature, you must set the `--load` parameter to specify the Hugging Face weight directory. Training from random initialization is not supported.
  - Megatron-format weights not supported. After you enable this parameter, offline-converted Megatron-format weights are not supported.
  - Storage path rules:
    - If you specify `--mg-save-dir`, the converted Megatron weights are saved to that path.
    - If you do not specify it, they are saved by default in the `{load}/megatron_cache_tp{TP}pp{PP}ep{EP}` directory.
    - The training process automatically uses this path as the weight loading path.

- Megatron-to-Hugging Face conversion (`--enable-mg2hf-convert`)
  - Set the save path. When enabling this feature, you must set the `--save` parameter to specify the training output path.
  - This feature is supported only in standalone storage or shared storage environments.
  - LoRA not supported. Megatron-to-Hugging Face conversion for weights fine-tuned with LoRA is not supported.
  - Storage path rules:
    - If you specify `--hf-save-dir`, the converted Hugging Face weights are saved in the `{hf_save_dir}/mg2hf_iteration{iteration}/` directory.
    - If you do not specify it, they are saved by default in the `{save}/mg2hf_iteration{iteration}` directory.
    - Configuration file handling: If you specify `--hf-cfg-dir`, the system copies configuration files from this directory to the converted Hugging Face weight directory. If you do not specify it but bidirectional conversion is enabled, the system copies configuration files from the `{load}` directory.

> [!NOTE]
>
> Megatron-to-Hugging Face conversion itself does not generate configuration files. You must copy them from an existing configuration source.

### Automatic Data Preprocessing

#### Quick Start

If you want to use the data preprocessing feature, refer to the parameters and add relevant parameters based on your usage scenario. Modify the `--data-path` parameter to specify the input dataset path to determine whether data preprocessing is performed.

The currently supported forms are as follows:

| Input Form | Example | Description |
|-----------|-------|------|
| **Raw file** | `/data/train.jsonl` | Raw dataset. The system automatically identifies it and converts it to `.bin/.idx` format. |
| **Converted prefix** | `/data/train_text_document` | Already converted format. You can use it directly. |

#### Parameters

**Table 2** Parameters

| Parameter | Type | Default | Required | Description |
|------|------|------|------|------|
| `--data-path` | string or list | None | Yes | Raw data path or converted prefix. |
| `--handler-name` | string | "" | Yes | Name of the data processing handler. |
| `--append-eod` | bool | False | No | Whether to append the `<eod>` token to the end of documents. |
| `--prompt-type` | string | None | Yes (fine-tuning) | Specify the fine-tuning prompt template. |
| `--json-keys` | list | `["text"]` | No | Fields to extract. |
| `--workers` | int | 1 | No | Number of data processing threads. |
| `--n-subs` | int | 1 | No | Number of data subsets (multi-process sharding). |
| `--pack` | bool | False | No | Whether to pack samples (fine-tuning scenario). |
| `--neat-pack` | bool | False | No | Switch that enables the use of a jagged `attention_mask` during computation in pack scenarios (fine-tuning scenario). |
| `--enable-thinking` | string | None | No | Whether to enable thinking mode (fine-tuning scenario). |
| `--output-prefix` | string | None | No | Prefix of the output dataset file after conversion. |
| `--seq-length` | int | None | No | In pack mode, specifies the sequence length after data packing. |
| `--reasoning-effort` | string | None | No | Used for DeepSeek-V4 model fine-tuning data processing. Options: max/high. max: inserts the maximum effort instruction prefix into the prompt; high: reserved, currently a no-op. |
| `--drop-thinking` | bool | True | No | In DeepSeek-V4 fine-tuning scenarios, whether to discard historical thinking chains in multi-turn conversations. By default, only the last assistant reasoning is retained as the loss target. Set to False to retain all reasoning turns. |

> [!NOTE]
>
> If you do not specify `--output-prefix`, the processed data file is generated in the same directory as the raw dataset by default.

### Example

Using Qwen3-8B model fine-tuning as an example, to enable both data preprocessing and integrated weight-conversion training, add the following parameters to the [Qwen3-8B fine-tuning script](../../../../../../examples/mcore/qwen3/tune_qwen3_8b_4K_full_ptd.sh):

```bash
DATA_PATH="/path_your_dataset/xxx.parquet"
CKPT_LOAD_DIR="/path_to_huggingface_model/Qwen3-8B"

bash examples/mcore/qwen3/tune_qwen3_8b_4K_full_ptd.sh \
    --data-path "${DATA_PATH}" \
    --load "${CKPT_LOAD_DIR}" \
    --enable-hf2mg-convert \
    --model-type-hf qwen3 \
    --handler-name AlpacaStyleInstructionHandler \
    --prompt-type qwen3
```

## Usage Constraints

- The currently supported Hugging Face model types are: `qwen3`, `qwen3-moe`, `deepseek3`, `glm45-air`, `bailing_mini`, `qwen3-next`, `seed-oss`, `deepseek32`, `magistral`, and `deepseek2-lite`.

- The current automatic dataset conversion feature supports only the following raw data formats: `parquet`, `arrow`, `csv`, `json`, `jsonl`, and `txt`. Other formats are not supported yet.

- The current weight conversion feature `--enable-mg2hf-convert` supports only standalone storage or shared storage environments.

- The current weight conversion feature `--enable-mg2hf-convert` does not support Megatron-to-Hugging Face weight conversion for weights fine-tuned with LoRA.
