# Weight Conversion

## Background

As large-scale pretrained models become widely adopted, compatibility issues between different training frameworks and hardware platforms become more visible. Proprietary training frameworks such as MindSpeed LLM usually use custom parallelization strategies, for example tensor parallelism and pipeline parallelism, to address memory and compute bottlenecks during LLM training. As training requirements and hardware change, the partitioning strategy for model parameters must also change accordingly. However, cross-framework weight conversion often faces challenges such as format incompatibility and different partitioning strategies. Weight conversion aims to enable seamless migration and evaluation of large-scale pretrained models across different training frameworks. It also resolves issues such as incompatible weight formats and differences in partitioning strategies between frameworks, and therefore improves model migration flexibility and scalability for broader application scenarios and business needs.

- [Weight Download](#1-weight-download)

  Download open-source model weights from sites such as Hugging Face. You can download them from the CLI or in a browser.
- [Weight Conversion](#2-weight-conversion)
  - [Converting Hugging Face Weights to MCore Weights](#21-converting-hugging-face-weights-to-mcore-weights)

    Convert Hugging Face model weights to the MCore format. It supports multiple parallel partitioning strategies.

  - [Converting MCore Weights to Hugging Face Weights](#22-converting-mcore-weights-to-hugging-face-weights)

    Convert MCore model weights to the Hugging Face format for model migration across different frameworks.

  - [LoRA Weight Conversion](#23-lora-weight-conversion)

    - [Merging MCore Weights](#231-merging-mcore-weights)

      Merge LoRA fine-tuned weights in the MCore format with base model weights and convert them to the MCore or Hugging Face format.

    - [Converting LoRA Weights to Hugging Face Weights](#232-converting-lora-weights-to-hugging-face-weights)

      Convert LoRA fine-tuned weights to the Hugging Face format separately.

- [Weight Conversion Features](#weight-conversion-features)

## How to Use Weight Conversion

Weight conversion addresses weight compatibility issues across different deep learning frameworks and training strategies. It supports efficient weight conversion across multiple models and training configurations. The core features include:

**Bidirectional weight conversion**: Supports weight conversion for more than 100 models. It can convert weight formats between mainstream frameworks such as Hugging Face and Megatron-LM with any parallel partitioning strategy. During conversion, you need to specify `--use-mcore-models` to convert weights to the MCore format.

**Weight conversion for training parallel strategies**: Supports weight conversion across multiple training parallel strategies, including tensor parallelism, pipeline parallelism, expert parallelism, dynamic pipeline partitioning, and virtual pipeline parallelism. Whether you train with different parallel strategies or need to switch between strategies, it enables flexible weight conversion to meet a wide range of training and inference requirements.

**LoRA weight merge and conversion**: Supports merging LoRA weights with base weights, which simplifies the loading steps during model inference. You can use the merged model directly for inference, which significantly improves inference efficiency and reduces unnecessary compute overhead. It also supports converting LoRA fine-tuned weights to the Hugging Face format separately to support downstream customer tasks.

## 1. Weight Download

Download open-source model weights from sites such as Hugging Face.

You can find the training weight links in the `Download Link` column of the [Supported Models in the PyTorch Framework](../models/supported_models.md).

You can download weights directly from the web or from the CLI and save them to the `MindSpeed-LLM/model_from_hf` directory. For example:

```shell
#!/bin/bash
mkdir -p ./model_from_hf/llama-2-7b-hf/
cd ./model_from_hf/llama-2-7b-hf/
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/config.json
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/generation_config.json
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/pytorch_model-00001-of-00002.bin
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/pytorch_model-00002-of-00002.bin
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/pytorch_model.bin.index.json
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/special_tokens_map.json
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/tokenizer.json
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/tokenizer.model
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/tokenizer_config.json
cd ../../
```

## 2. Weight Conversion

### 2.1 Converting Hugging Face Weights to MCore Weights

This conversion turns Hugging Face weights into the MCore format. It supports multiple parallel strategies, such as tensor parallelism and pipeline parallelism, and ensures that the converted weights can continue to be trained and used for inference within the MindSpeed LLM framework.

**Note**:

Before you convert weights, confirm the training parameter configuration first. Then modify the weight conversion script in the repository according to your training setup. These settings change the weight structure, and a mismatch with the training parameters will prevent the training job from loading the weights. The training configurations that you need to confirm are listed in the following table:
<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Description</th>
      <th>Optional/Required</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>--target-tensor-parallel-size</td>
      <td>TP partition count, defaulting to 1.</td>
      <td>Required</td>
    </tr>
    <tr>
      <td>--target-pipeline-parallel-size</td>
      <td>PP partition count, defaulting to 1.</td>
      <td>Required</td>
    </tr>
    <tr>
      <td>--num-layer-list</td>
      <td>Dynamic PP partitioning. Specify the number of layers in each PP stage with a list. Defaults to <code>None</code>.</td>
      <td>Optional</td>
    </tr>
    <tr>
      <td>--num-layers-per-virtual-pipeline-stage</td>
      <td>VPP partitioning. Specify the number of layers in each VPP stage. Defaults to <code>None</code>.</td>
      <td>Optional</td>
    </tr>
    <tr>
      <td>--target-expert-parallel-size</td>
      <td>Expert parallelism. Specify the number of expert-parallel devices. Defaults to 1.</td>
      <td>Optional</td>
    </tr>
    <tr>
      <td>--noop-layers</td>
      <td>Custom no-op layer operation. Specify extra no-op layers at a model layer. After conversion, the number of layers equals the original Hugging Face model layers plus the no-op layers. Defaults to <code>None</code>.</td>
      <td>Optional</td>
    </tr>
    <tr>
      <td>--use-mcore-models</td>
      <td>Convert to MCore weights.</td>
      <td>Required</td>
    </tr>
    <tr>
      <td>--model-type-hf</td>
      <td>Hugging Face model type, defaulting to <code>llama2</code>.</td>
      <td>Optional</td>
    </tr>
    <tr>
      <td>--tokenizer-model</td>
      <td>Specify the exact tokenizer model file, such as <code>tokenizer.model</code>, <code>tokenizer.json</code>, <code>qwen.tiktoken</code>, or <code>None</code>, depending on the format of the vocabulary file in Hugging Face.</td>
      <td>Required</td>
    </tr>
    <tr>
      <td>--params-dtype</td>
      <td>Specify the precision of the converted weights. The default is <code>fp16</code>. If the source file uses <code>bf16</code>, set this to <code>bf16</code> accordingly. This affects inference or evaluation results.</td>
      <td>Required</td>
    </tr>
  </tbody>
</table>

Scenario constraints:

1. The number of model layers must be divisible by the PP partition count. Otherwise, you need to add no-op layers with `--noop-layers` or use dynamic PP.

2. VPP partitioning and dynamic PP partitioning are mutually exclusive. You can use only one of them.

3. Refer to the `"model_mappings"` section in [model_cfg.json](https://gitcode.com/ascend/MindSpeed-LLM/blob/master/configs/checkpoint/model_cfg.json) for the currently supported models.

Here is an example script for converting Llama-2-7B weights from hf to mg for reference only:

```shell
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 4 \
    --num-layer-list 8,8,8,8 \
    --model-type-hf llama2 \
    --use-mcore-models \
    --load-dir ./model_from_hf/llama-2-7b-hf/ \
    --save-dir ./model_weights/llama-2-7b-mcore/ \
    --tokenizer-model ./model_from_hf/llama-2-7b-hf/tokenizer.model
```

**Launch Script**

The MindSpeed LLM naming convention and launch method for the Hugging Face-to-MCore weight conversion script are:

```shell
# Naming and launch:
# bash examples/mcore/model_name/ckpt_convert_xxx_hf2mcore.sh
# Configure the parallel parameters and the paths for loading and saving the weights and vocabulary files.

bash examples/mcore/llama2/ckpt_convert_llama2_hf2mcore.sh
```

### 2.2 Converting MCore Weights to Hugging Face Weights

This conversion turns MCore weights into the Hugging Face format. It supports multiple parallel strategies, such as tensor parallelism and pipeline parallelism. During conversion, the model weights are adapted to the standard Hugging Face format so they can continue training and inference in a Hugging Face environment.

**Note**:

1. When converting to Hugging Face weights, you must set `--target-tensor-parallel-size=1` and `--target-pipeline-parallel-size=1`, because Hugging Face weights do not use parallel partitioning.

2. The output directory after a successful conversion contains only the model weight files. It does not generate `config.json`, `tokenizer.model`, `vocab.json`, or other vocabulary files.

3. You must set `--save-dir` to the original Hugging Face model path, and that path must contain the full Hugging Face model files, including the weights and configuration files.

4. If the MCore weights use no-op layers, you must add the same no-op layer configuration to the CLI when converting MCore weights to the Hugging Face format, and add `--load-checkpoint-loosely`.

Here is a reference conversion script for the Llama-2-7B MCore-to-Hugging Face conversion:

```shell
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --model-type-hf llama2 \
    --use-mcore-models \
    --load-dir ./model_weights/llama-2-7b-mcore/ \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --save-dir ./model_from_hf/llama-2-7b-hf/  # <-- Enter the original Hugging Face model path. The new weights will be saved in ./model_from_hf/llama-2-7b-hf/mg2hf/
```

See Section 2.1 for parameter meanings.

**Launch Script**

The MindSpeed LLM naming convention and launch method for the MCore-to-Hugging Face weight conversion script are:

```shell
# Naming and launch:
# bash examples/mcore/model_name/ckpt_convert_xxx_mcore2hf.sh
# Configure the parallel parameters and the paths for loading and saving the weights and vocabulary files.

bash examples/mcore/llama2/ckpt_convert_llama2_mcore2hf.sh
```

### 2.3 LoRA Weight Conversion

The current repository supports the following two LoRA weight conversion methods:

1. Merge LoRA fine-tuned weights with base model weights and convert them to the Megatron or Hugging Face format.

2. Convert LoRA fine-tuned weights to the Hugging Face format separately. Add the `--lora-ckpt-filter` parameter to the LoRA fine-tuning script to save only the LoRA weights.

#### 2.3.1 Merging MCore Weights

Add the following parameters to the weight conversion command to merge the trained LoRA weights with the converted base weights:

```shell
--lora-load ./ckpt/llama-2-7b-lora  \
--lora-r 16 \
--lora-alpha 32 \
--lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
```

<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Description</th>
      <th>Optional/Required</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>--lora-load</td>
      <td>Loads the weights generated after LoRA fine-tuning.</td>
      <td>Optional</td>
    </tr>
    <tr>
      <td>--lora-r</td>
      <td>LoRA rank. It determines the size of the low-rank matrix.</td>
      <td>Optional</td>
    </tr>
    <tr>
      <td>--lora-alpha</td>
      <td>Defines the scaling factor used by LoRA adaptation. This parameter affects the update speed of the low-rank matrix.</td>
      <td>Optional</td>
    </tr>
    <tr>
      <td>--lora-target-modules</td>
      <td>Defines the LoRA target modules as a space-separated string list with no default value. Each string corresponds to a layer name that needs LoRA fine-tuning, and you can choose only from the four predefined parameter configurations above. You can adjust this parameter as needed.</td>
      <td>Optional</td>
    </tr>
  </tbody>
</table>

**Convert to MCore Weights After Merging**

The following example script merges the LoRA weights of the Llama-2-7B model in the MCore format with the base weights and converts them back to the MCore format for reference only:

```shell
python convert_ckpt.py \
    --model-type GPT \
    --use-mcore-models \
    --load-model-type mg \
    --save-model-type mg \
    --load-dir ./model_weights/llama-2-7b-mcore/ \
    --lora-load ./ckpt/llama-2-7b-lora \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --save-dir ./model_weights/llama-2-7b-lora2mcore
```

<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Description</th>
      <th>Optional/Required</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>--lora-target-modules</td>
      <td>Defines the LoRA target modules as a space-separated string list with no default value. Each string corresponds to a layer name that needs LoRA fine-tuning, and you can choose only from the four predefined parameter configurations above. You can adjust this parameter as needed.</td>
      <td>Optional</td>
    </tr>
  </tbody>
</table>

**Launch Script**

The weight conversion script naming style and launch method are:

```shell
# Use llama2 as the command launch example.
bash examples/mcore/llama2/ckpt_convert_llama2_mg2mg_lora.sh
```

**Convert to Hugging Face Weights After Merging**

The following example script merges the LoRA weights of the Llama-2-7B model in the MCore format with the base weights and converts them to the Hugging Face format for reference only:

```shell
python convert_ckpt.py \
    --model-type GPT \
    --use-mcore-models \
    --load-model-type mg \
    --save-model-type hf \
    --load-dir ./model_weights/llama-2-7b-mcore/ \
    --lora-load ./ckpt/llama-2-7b-lora \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --save-dir ./model_from_hf/llama-2-7b-hf/    # <-- Enter the original Hugging Face model path. The new weights will be saved in ./model_from_hf/llama-2-7b-hf/mg2hg/
```

**Launch Script**

The weight conversion script naming style and launch method are:

```shell
# Use llama2 as the command launch example.
bash examples/mcore/llama2/ckpt_convert_llama2_mcore2hf_lora.sh
```

**Note**:

The LoRA parameter values must match the values used during LoRA fine-tuning, and the LoRA weight partitioning strategy must match the base weight partitioning strategy.

After the `peft` library merges the LoRA weights, the weight data type becomes float16. However, some models, such as the Qwen series, use bfloat16 by default. Converting the merged weights back to the Hugging Face format may therefore cause precision loss. You can temporarily avoid this by changing the data type in `config.json` of the original Hugging Face model to float16.

MoE models do not currently support LoRA weight conversion after `--moe-grouped-gemm` is enabled.

#### 2.3.2 Converting LoRA Weights to Hugging Face Weights

By enabling `--save-lora-to-hf`, you can convert the LoRA weights after fine-tuning to the Hugging Face format. Here is a reference script for converting the Llama-2-7B LoRA weights to the Hugging Face format:

```shell
python convert_ckpt.py \
    --model-type GPT \
    --use-mcore-models \
    --load-model-type mg \
    --save-model-type hf \
    --load-dir ./ckpt/llama2_lora_filter \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --load-checkpoint-loosely \
    --save-lora-to-hf \
    --save-dir ./model_from_hf/llama-2-7b-hf/  # <-- Enter the original Hugging Face model path. The new weights will be saved in ./model_from_hf/llama-2-7b-hf/mg2hf/
```

<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Description</th>
      <th>Optional/Required</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>--save-lora-to-hf</td>
      <td>Set this parameter when you convert LoRA to Hugging Face to convert only the LoRA weights.</td>
      <td>Optional</td>
    </tr>
    <tr>
      <td>--load-checkpoint-loosely</td>
      <td>Allows loose loading. Set this parameter when you convert LoRA weights.</td>
      <td>Optional</td>
    </tr>
  </tbody>
</table>

**Note**:

The source weights contain only LoRA weights and do not include base weights. You need to add `--lora-ckpt-filter` to the LoRA fine-tuning script so that it saves only the LoRA weights.

`--save-lora-to-hf` and `--moe-grouped-gemm` cannot be used together. During LoRA fine-tuning, do not add `--moe-grouped-gemm` to the script.

`--save-lora-to-hf` and `--load-hf-from-config` cannot be used together.

LoRA weight conversion supports only the MCore format. It supports only models whose `fc_type` is `gate_up_down`. Other models are pending adaptation. At present, only Llama-2 and Mixtral are supported.

**Launch Script**

The MindSpeed LLM naming convention and launch method for the LoRA-to-Hugging Face weight conversion script are:

```shell
# Naming and launch:
# bash examples/mcore/model_name/ckpt_convert_xxx_lora2hf.sh
# Configure the parallel parameters and the paths for loading and saving the weights and vocabulary files.

bash examples/mcore/llama2/ckpt_convert_llama2_lora2hf.sh
```

### Weight Conversion Features

MindSpeed LLM supports weight format conversion between Hugging Face and MCore. The supported features are listed below:

<table>
  <thead>
    <tr>
      <th>Source Format</th>
      <th>Target Format</th>
      <th>Supported Feature</th>
      <th>Feature Parameter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="7">Hugging Face </td>
      <td rowspan="7">MCore</td>
      <td>Tensor parallelism</td>
      <td>--target-tensor-parallel-size</td>
    </tr>
    <tr>
      <td>Pipeline parallelism</td>
      <td>--target-pipeline-parallel-size</td>
    </tr>
    <tr>
      <td>Dynamic pipeline partitioning</td>
      <td>--num-layer-list</td>
    </tr>
    <tr>
      <td>Virtual pipeline parallelism</td>
      <td>--num-layers-per-virtual-pipeline-stage</td>
    </tr>
    <tr>
      <td>Expert parallelism</td>
      <td>--target-expert-parallel-size</td>
    </tr>
    <tr>
      <td>Expert tensor parallelism</td>
      <td>--expert-tensor-parallel-size</td>
    </tr>
    <tr>
      <td>Custom no-op layers</td>
      <td>--noop-layers</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="24">MCore </td>
      <td rowspan="8">Hugging Face</td>
      <td>Tensor parallelism</td>
      <td>--target-tensor-parallel-size</td>
    </tr>
    <tr>
      <td>Pipeline parallelism</td>
      <td>--target-pipeline-parallel-size</td>
    </tr>
    <tr>
      <td>Expert parallelism</td>
      <td>--target-expert-parallel-size</td>
    </tr>
    <tr>
      <td>Expert tensor parallelism</td>
      <td>--expert-tensor-parallel-size</td>
    </tr>
    <tr>
      <td>LoRA target modules</td>
      <td>--lora-target-modules</td>
    </tr>
    <tr>
      <td>LoRA weights</td>
      <td>--lora-load</td>
    </tr>
    <tr>
      <td>LoRA rank</td>
      <td>--lora-r</td>
    </tr>
    <tr>
      <td>LoRA alpha</td>
      <td>--lora-alpha</td>
    </tr>
    <tr>
      <td rowspan="10">MCore</td>
      <td>Tensor parallelism</td>
      <td>--target-tensor-parallel-size</td>
    </tr>
    <tr>
      <td>Pipeline parallelism</td>
      <td>--target-pipeline-parallel-size</td>
    </tr>
    <tr>
      <td>Expert parallelism</td>
      <td>--target-expert-parallel-size</td>
    </tr>
    <tr>
      <td>Dynamic pipeline partitioning</td>
      <td>--num-layer-list</td>
    </tr>
    <tr>
      <td>Virtual pipeline parallelism</td>
      <td>--num-layers-per-virtual-pipeline-stage</td>
    </tr>
    <tr>
      <td>LoRA target modules</td>
      <td>--lora-target-modules</td>
    </tr>
    <tr>
      <td>LoRA weights</td>
      <td>--lora-load</td>
    </tr>
    <tr>
      <td>LoRA rank</td>
      <td>--lora-r</td>
    </tr>
    <tr>
      <td>LoRA alpha</td>
      <td>--lora-alpha</td>
    </tr>
    <tr>
      <td>Custom no-op layers</td>
      <td>--noop-layers</td>
    </tr>
  </tbody>
</table>
