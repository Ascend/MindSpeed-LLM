# LoRA Fine-Tuning Porting Guide

MindSpeed LLM supports low-parameter training with LoRA for fine-tuning tasks. You enable it by adding LoRA parameters to the baseline task.

This tutorial provides guidance for porting and developing LoRA fine-tuning workflows for models. The following steps use the Qwen3-8B model and a single Atlas 900 A2 PoD, which is a 1x8 cluster, to explain how to develop a LoRA fine-tuning script step by step.

Before you follow the steps below, complete the environment setup by referring to the [MindSpeed LLM Installation Guide](../../training/install_guide.md), and prepare the model weights and fine-tuning dataset. For model weight downloads, refer to the download links for the corresponding models in the [Supported Models in the PyTorch Framework](../../models/supported_models.md) document. For dataset downloads, refer to the [Alpaca-Style Datasets](../../tools/data_process_sft_alpaca_style.md) and [ShareGPT Datasets](../../tools/data_process_sft_sharegpt_style.md).

## 1. Model Weight Conversion

The LoRA fine-tuning script can use standard base Megatron weights for fine-tuning tasks. Using the Qwen3-8B model with a TP=1 and PP=2 split as an example, see the [Qwen3 Weight Conversion Script](../../../../../examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh) for detailed configuration. You need to modify the related path parameters and model partition settings:

```shell
source /usr/local/Ascend/cann/set_env.sh # Replace this with the actual Toolkit installation path.
......
--target-tensor-parallel-size 1          # TP partition size.
--target-pipeline-parallel-size 2        # PP partition size.
--load-dir ./model_from_hf/qwen3_hf/     # Hugging Face weight path.
--save-dir ./model_weights/qwen3_mcore/   # Megatron weight save path.
```

After you confirm that the paths are correct, run the weight conversion script:

```shell
bash examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh
```

## 2. Data Preprocessing

Use the Alpaca dataset as an example for data preprocessing. For detailed configuration, see the [Qwen3 Data Preprocessing Script](../../../../../examples/mcore/qwen3/data_convert_qwen3_instruction.sh). You need to modify the paths in the script:

```shell
source /usr/local/Ascend/cann/set_env.sh # Replace this with the actual Toolkit installation path.
......
--input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet # Raw dataset path.
--tokenizer-name-or-path ./model_from_hf/qwen3_hf # Hugging Face tokenizer path.
--output-prefix ./finetune_dataset/alpaca  # Save path.
......
```

Parameters for data preprocessing:

- `handler-name`: Specifies the dataset handler class. Common options include `AlpacaStyleInstructionHandler`, `SharegptStyleInstructionHandler`, and `AlpacaStylePairwiseHandler`.
- `tokenizer-type`: Specifies the tokenizer used to process the data. A common value is `PretrainedFromHF`.
- `workers`: The number of parallel workers used to process the dataset.
- `log-interval`: The number of steps between progress updates.
- `enable-thinking`: Enables or disables the fast-thinking and slow-thinking template. You can set it to `[true, false, none]`, and the default value is `none`. When you enable it, the dataset model responses include `<think>` and `</think>`, and these tokens participate in loss calculation. Therefore, all data is treated as slow-thinking data. When you disable it, an empty CoT marker is added to the user input in the dataset, and it does not participate in loss calculation. Therefore, all data is treated as fast-thinking data. Setting it to `none` works well when the original dataset contains a mix of fast-thinking and slow-thinking data. **Currently, this option supports only Qwen3 series models.**
- `prompt-type`: Specifies the model template. It helps the base model gain stronger conversational ability after fine-tuning. You can find the available `prompt-type` options in the [`templates`](../../../../../configs/finetune/templates.json) file.

After you finish configuring the parameters, run the data preprocessing script:

```shell
bash examples/mcore/qwen3/data_convert_qwen3_instruction.sh
```

## 3. LoRA Fine-Tuning Script Development

The repository provides a base fine-tuning script, the [Qwen3-8B Fine-Tuning Script](../../../../../examples/mcore/qwen3/tune_qwen3_8b_4K_full_ptd.sh). LoRA fine-tuning requires you to add LoRA-related parameters on top of this script. Using Qwen3-8B as an example, you can name the corresponding LoRA fine-tuning script `tune_qwen3_8b_4K_lora_ptd.sh`.

Parallel configuration parameters such as TP and PP must match the settings used during weight conversion. You need to modify the related path parameters and model partition settings:

```shell
CKPT_LOAD_DIR="your model ckpt path"      # Weight load path. Enter the weight path saved during conversion.
CKPT_SAVE_DIR="your model save ckpt path" # Weight save path after LoRA fine-tuning finishes.
DATA_PATH="your data path"                # Dataset path. Enter the path saved during data preprocessing, and note that you need to add the suffix.
TOKENIZER_PATH="your tokenizer path"      # Vocabulary path. Enter the vocabulary path from the downloaded open-source weights.
TP=1                                      # The value of target-tensor-parallel-size used during weight conversion.
PP=2                                      # The value of target-pipeline-parallel-size used during weight conversion.
```

To perform LoRA fine-tuning, add the LoRA parameters to `TUNE_ARGS` on top of the full-parameter fine-tuning script:

```shell
--lora-r 16 \
--lora-alpha 32 \
--lora-fusion \
--lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
......
| tee logs/tune_qwen3_8b_4K_lora_ptd.log # Corresponding log file name.
```

Parameters for the fine-tuning script:

- `lora-r`: The LoRA rank, which indicates the dimension of the low-rank matrices. A lower rank uses fewer parameters during training, which reduces compute and memory consumption. However, an excessively low rank may limit model expressiveness.
- `lora-alpha`: Controls the influence ratio of LoRA weights on the original weights. Higher values produce a stronger effect. In general, keep `α/r` at 2.
- `lora-fusion`: Whether to enable the [CCLoRA](../../features/mcore/cc_lora.md) algorithm. This algorithm improves performance by overlapping communication with computation. The current GLM-4.5 model does not support this parameter.
- `lora-target-modules`: Selects the modules that need LoRA. The currently available modules are `linear_qkv`, `linear_proj`, `linear_fc1`, and `linear_fc2`.

After you finish the parameter configuration above, run the LoRA fine-tuning script:

```shell
bash examples/mcore/qwen3/tune_qwen3_8b_4K_lora_ptd.sh
```

## 4. LoRA Inference Script Development

After LoRA fine-tuning completes, you need to further verify whether the model produces the expected outputs. The repository provides a base inference script, the [Qwen3-8B Inference Script](../../../../../examples/mcore/qwen3/generate_qwen3_8b_ptd.sh). LoRA inference requires you to add LoRA-related parameters on top of this script. Using Qwen3-8B as an example, you can name the corresponding LoRA inference script `generate_qwen3_8b_lora_ptd.sh`.

Modify the path parameters and add the LoRA parameters on top of the inference script:

```shell
TOKENIZER_PATH="your tokenizer directory path"   # Vocabulary path. Enter the vocabulary path from the downloaded open-source weights.
CHECKPOINT="your model directory path"           # Weight load path. Enter the weight path saved during conversion.
CHECKPOINT_LORA="your lora model directory path" # Weight save path after LoRA fine-tuning finishes.
......
--lora-load ${CHECKPOINT_LORA}  \
--lora-r 16 \
--lora-alpha 32 \
--lora-fusion \
--lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
--prompt-type qwen3 \
......
| tee logs/generate_qwen3_8b_lora_ptd.log # Corresponding log file name.
```

Parameters:

- `lora-load`: Loads LoRA weight checkpoints to resume training or for inference. For inference, use it together with the `--load` parameter to load LoRA weights from the `CKPT_SAVE_DIR` path.

Parallel configuration parameters such as TP and PP must match the settings used during weight conversion. After you finish configuring the parameters, run the LoRA inference script:

```shell
bash examples/mcore/qwen3/generate_qwen3_8b_lora_ptd.sh
```

The model is expected to correctly answer questions from the dataset without garbled text or repetition.

## 5. LoRA Fine-Tuning Weight Evaluation Script Development

MindSpeed LLM supports accuracy evaluation of large models on public benchmark datasets. For detailed statistics, see [evaluation.md](../../training/evaluation/models_evaluation.md). The repository provides a base weight evaluation script, the [Qwen3-8B Weight Evaluation Script](../../../../../examples/mcore/qwen3/evaluate_qwen3_8b_ptd.sh). LoRA fine-tuning weight evaluation requires you to add LoRA-related parameters on top of this script. Using Qwen3-8B as an example, you can name the corresponding LoRA fine-tuning weight evaluation script `evaluate_qwen3_8b_lora_ptd.sh`.

Modify the path parameters and add the LoRA parameters on top of the weight evaluation script:

```shell
TOKENIZER_PATH="your tokenizer directory path"   # Vocabulary path. Enter the vocabulary path from the downloaded open-source weights.
CHECKPOINT="your model directory path"           # Weight load path. Enter the weight path saved during conversion.
CHECKPOINT_LORA="your lora model directory path" # Weight save path after LoRA fine-tuning finishes.
DATA_PATH="your data path"                       # Downloaded benchmark dataset path, typically MMLU.
......
--lora-load ${CHECKPOINT_LORA} \
--lora-r 16 \
--lora-alpha 32 \
--lora-fusion \
--lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
......
| tee logs/evaluate_qwen3_8b_lora_ptd.log # Corresponding log file name.
```

Parallel configuration parameters such as TP and PP must match the settings used during weight conversion. After you finish configuring the parameters, run the LoRA fine-tuning weight evaluation script:

```shell
bash examples/mcore/qwen3/evaluate_qwen3_8b_lora_ptd.sh
```

Note: Evaluation results are determined by the training dataset and training method. You need to select an appropriate dataset or adjust the LoRA fine-tuning script parameters on your own.

## 6. LoRA Weight and Base Weight Merge and Conversion

The repository supports merging LoRA fine-tuning weights with base model weights and converting them to the Hugging Face format. Simply add the following LoRA parameters to the weight conversion script:

```shell
--lora-load ${CHECKPOINT_LORA}  \
--lora-r 16 \
--lora-alpha 32 \
--lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2
```

Using Qwen3-8B as an example, if you want to merge the LoRA weights and then convert them to the Hugging Face format, you can name the corresponding weight conversion script `ckpt_convert_qwen3_mcore2hf_lora.sh`.

```shell
source /usr/local/Ascend/cann/set_env.sh # Replace this with the actual Toolkit installation path.

python convert_ckpt_v2.py \
    --load-model-type mg \
    --save-model-type hf \
    --load-dir ./model_weights/qwen3_mcore/ \    # Megatron weight save path.
    --lora-load ./ckpt/qwen3_lora \              # Weight save path after LoRA fine-tuning finishes.
    --save-dir ./model_weights/qwen3_mcore2hf/ \ # Converted Hugging Face weight path.
    --hf-cfg-dir ./model_from_hf/qwen3_hf/ \     # Original Hugging Face weight path. Copy the config file into the Hugging Face weight directory generated by the weight conversion process.
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --model-type-hf qwen3
```

After you finish configuring the path parameters, run the weight conversion script:

```shell
bash examples/mcore/qwen3/ckpt_convert_qwen3_mcore2hf_lora.sh
```

Note:

- The LoRA parameter values should match the values used during fine-tuning to ensure that the converted model has the same performance and compatibility.
- Current LoRA fine-tuning does not support the `--mtp-num-layers` parameter.
- After the `peft` library merges the LoRA weights, the weight data type becomes the `float16` type. However, some models, such as the Qwen series, use the `bfloat16` type by default. Converting the merged weights back to the Hugging Face format may cause precision loss. You can temporarily set the data type in the original Hugging Face model's `config.json` file to the `float16` type to work around this issue.
