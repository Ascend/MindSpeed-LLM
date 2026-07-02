# Weight Conversion

## Weight Conversion Background

As model scale grows from the hundred-million level to the trillion level, terabyte-scale parameter models place extremely high demands on system resources during deployment and migration, and a single device cannot hold the full set of model parameters. MindSpeed LLM uses a memory-efficient weight conversion solution that supports on-demand loading to address the tendency of large-parameter models to crash during conversion. Therefore, it provides a technical foundation for the efficient training and application of ultra-large models.

- [Weight Download](#1-weight-download)

  Download open-source model weights from Hugging Face and other sites. It supports both CLI and web downloads.
- [Weight Conversion](#2-weight-conversion)
  - [Converting Hugging Face Weights to the MCore Format](#21-converting-hugging-face-weights-to-the-mcore-format)

    Convert Hugging Face model weights to the MCore format. It supports multiple parallel sharding schemes.

  - [Converting MCore Weights to the Hugging Face Format](#22-converting-mcore-weights-to-the-hugging-face-format)

    Convert MCore model weights to the Hugging Face format for migration across different frameworks.

  - [Debug Feature: Converting Hugging Face Reduced-Layer Weights to the MCore Format](#23-debug-feature-converting-hugging-face-reduced-layer-weights-to-the-mcore-format)

    Reduced-layer conversion of Hugging Face model weights to the MCore format with multiple parallel sharding schemes is supported.

## How to Use Weight Conversion

Weight conversion addresses compatibility issues for model weights across different deep learning frameworks and training strategies. It supports efficient weight conversion across multiple models and training configurations. The core features include:

**Weight conversion across formats**: It converts weights between the mainstream Hugging Face and Megatron-LM frameworks under any parallel sharding strategy.

**Weight conversion for training parallel strategies**: It supports weight conversion across multiple training parallel strategies, including tensor parallelism (TP), pipeline parallelism (PP), expert parallelism (EP), expert tensor parallelism (ETP), and virtual pipeline parallelism (VPP). Whether you train with different parallel strategies or need to switch between them, it provides flexible weight conversion to meet a wide range of training and inference needs.

## 1. Weight Download

Download open-source model weights from Hugging Face and other sites.

You can find training weight links in the `Download Link` column in the [Supported Models in the PyTorch Framework](../models/supported_models.md).

### Download Methods

#### Method 1. Direct Download on the Web

Open the link in a browser and manually download all weight files.

#### Method 2. CLI Download

Save the weights to the `MindSpeed-LLM/model_from_hf` directory. For example:

```shell
#!/bin/bash
mkdir ./model_from_hf/llama-2-7b-hf/
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

### Common Questions

If you encounter any problems during the download, see:

Hugging Face official documentation: <https://huggingface.co/docs/hub/models-downloading>

ModelScope download guide: <https://modelscope.cn/docs/models/download>

Network troubleshooting: Try a mirror site or a proxy.

### Notes

Ensure that you have enough drive space to store the model weights.

Check file integrity and verify the file size and MD5 value after the download.

Some models may require you to log in or request access before you can download them.

## 2. Weight Conversion

### 2.1 Converting Hugging Face Weights to the MCore Format

Weight conversion converts Hugging Face weights to the MCore format. It supports multiple parallel strategies, such as tensor parallelism and pipeline parallelism, and ensures that you can continue training and inference in the MindSpeed LLM framework after conversion.

**Note**:

Before you convert weights, first confirm the training-time parameter configuration and modify the weight conversion script in the repository according to your training configuration. These configurations change the structure of the weights. If they do not match the training parameters, training cannot load the weights. The full parameter configuration supported by the current weight conversion solution is shown in the following table:
<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Description</th>
      <th>Must Match Training Configuration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>--load-model-type</td>
      <td>Source model type. Options are <code>hf</code> or <code>mg</code>. The default is <code>hf</code>.</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>--save-model-type</td>
      <td>Converted model type. Options are <code>hf</code> or <code>mg</code>. The default is <code>mg</code>.</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>--load-dir</td>
      <td>Source model path.</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>--save-dir</td>
      <td>Path where the converted model weights are stored.</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>--model-type-hf</td>
      <td>Hugging Face model family. The default is <code>qwen3</code>. For already supported models, the script is preconfigured. Therefore, you do not need to change it.</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>--target-tensor-parallel-size</td>
      <td>TP. Specifies the tensor parallel size. The default is 1.</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--target-pipeline-parallel-size</td>
      <td>PP. Specifies the pipeline parallel size. The default is 1.</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--target-expert-parallel-size</td>
      <td>EP. Specifies the expert parallel size. The default is 1.</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--expert-tensor-parallel-size</td>
      <td>ETP. Specifies expert tensor parallelism. It defaults to TP, and after it is enabled, currently only ETP=1 is supported.</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--num-layers-per-virtual-pipeline-stage</td>
      <td>VPP partitioning. Specifies the number of layers in each VPP stage. The default is None.</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--num-layer-list</td>
      <td>Dynamic PP partitioning. It specifies the number of layers in each PP stage through a list. The default is None. When you use it, separate the values with commas. The sum of the list values must equal the total number of model layers, and the length of the list must equal PP. For example, if the model has 14 layers, set <code>--num-layer-list 3,4,4,3</code> and <code>--target-pipeline-parallel-size 4</code>.</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--noop-layers</td>
      <td>Custom noop-layer operation. Specify where to insert noop layers in the model. After conversion, the number of layers equals the original Hugging Face model layer count plus the number of noop layers. The default is None.</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--moe-grouped-gemm</td>
      <td>MoE grouped matrix multiplication optimization.</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--moe-tp-extend-ep</td>
      <td>TP extends EP. When enabled, the TP group in expert layers shards expert parameters.</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--mla-mm-split</td>
      <td>When enabled, it expands the compressed <code>q_compressed</code> and <code>kv_compressed</code> to a higher dimension.</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--mtp-num-layers</td>
      <td>The number of MTP layers. The default is 0.</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--schedules-method</td>
      <td>DualPipeV pipeline scheduling. The default is None, and the available option is <code>dualpipev</code>.</td>
      <td>✅</td>
    </tr>
  </tbody>
</table>

Scenario restrictions:

1. The number of model layers must be divisible by the PP sharding count. Otherwise, add noop layers (`--noop-layers`) or use dynamic PP (`--num-layer-list`).

2. VPP (`--num-layers-per-virtual-pipeline-stage`) and dynamic PP partitioning (`--num-layer-list`) are mutually exclusive.

The following script for the Qwen3-235b model is provided for reference only and shows Hugging Face-to-MCore weight conversion:

```shell
python convert_ckpt_v2.py \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 4 \
    --target-expert-parallel-size 32 \
    --num-layers-per-virtual-pipeline-stage 8 \
    --noop-layers 94,95 \
    --load-dir ./model_from_hf/qwen3_moe_hf/ \
    --save-dir ./model_weights/qwen3_moe_mcore/ \
    --moe-grouped-gemm \
    --model-type-hf qwen3-moe
```

Launch Script

MindSpeed LLM provides prebuilt model weight conversion scripts. The following is the naming style and launch method for MCore to Hugging Face weight conversion scripts. You can search by model category:

```shell
# Naming and launch:
# bash examples/mcore/model_name/ckpt_convert_xxx_mcore2hf.sh
# Configure the parallel parameters and the paths for loading and saving weights and vocabulary files.

bash examples/mcore/qwen3_moe/ckpt_convert_qwen3_moe_235b_mcore2hf.sh
```

### 2.2 Converting MCore Weights to the Hugging Face Format

Weight conversion converts MCore weights to the Hugging Face format. It supports multiple parallel strategies, such as tensor parallelism and pipeline parallelism, and adapts the model weights to the standard Hugging Face format during conversion so that you can continue training and inference in a Hugging Face environment.

**Note**:

1. When you convert to Hugging Face weights, **you do not need to set `--target-tensor-parallel-size`, `--target-pipeline-parallel-size`, `--target-expert-parallel-size`, or `--num-layers-per-virtual-pipeline-stage`**, because Hugging Face weights do not involve parallel sharding.

2. After conversion succeeds, the output directory contains only model weight files. It does not generate model configuration files such as `config.json` or vocabulary files such as `tokenizer.model` and `vocab.json`.

3. If the MCore weights are configured with noop layers, you must use the same noop-layer configuration when you convert the MCore weights to the Hugging Face format.

4. If the expert tensor parallelism (ETP) of the original MCore weights is 1, you must add the **`--expert-tensor-parallel-size 1`** parameter when you run the `mcore2hf` conversion script.

The following `Qwen3-235b` model MCore-to-Hugging Face conversion script is provided for reference only:

```shell
python convert_ckpt_v2.py \
    --load-model-type mg \
    --save-model-type hf \
    --noop-layers 94,95 \
    --load-dir ./model_weights/qwen3_moe_mcore/ \
    --save-dir ./model_from_hf/qwen3_moe_hf/ \
    --moe-grouped-gemm \
    --model-type-hf qwen3-moe
```

Launch Script

MindSpeed LLM provides prebuilt model weight conversion scripts. The following shows the naming convention and launch method for MCore to Hugging Face weight conversion scripts. You can search by model category:

```shell
# Naming and launch:
# bash examples/mcore/model_name/ckpt_convert_xxx_mcore2hf.sh
# Configure the parallel parameters and the paths for loading and saving the weights and vocabulary files.

bash examples/mcore/qwen3_moe/ckpt_convert_qwen3_moe_235b_mcore2hf.sh
```

### 2.3 Debug Feature: Converting Hugging Face Reduced-Layer Weights to the MCore Format

This framework supports **reduced-layer debugging** when converting Hugging Face weights to the MCore format without changing the model configuration file. You can configure the reduction with the following cLI arguments.

`--num-layers`

This specifies the number of layers in the reduced model. It cannot exceed the number of layers in the original model, and this number does **not** include MTP layers. The default value is None. **When you do not use reduced layers, the value comes from the configuration file. Therefore, you do not need to specify this argument.**

If you configure noop layers, the value of `num-layers` should be the real layer count. It should not include MTP layers or the `--noop-layers` count.

If you need to use reduced-layer debugging together with the training script, ensure that this parameter is **consistent with the training script**.

`--first-k-dense-replace`

This specifies the number of dense layers before MoE layers in the reduced model. It cannot exceed the number of dense layers in the original model. The default value is None. **When you do not use reduced layers, the value comes from the configuration file. Therefore, you do not need to specify this argument.**

If you need to use reduced-layer debugging together with the training script, ensure that this parameter is **consistent with the training script**.

`--mtp-num-layers`

This is the number of MTP layers. The default value is 0. It supports configuring MTP layers during reduction, and the value cannot exceed the number of MTP layers in the original model.

If you need to configure MTP layers, you can set them on the CLI, for example `--mtp-num-layers 1`.

If you need to use reduced-layer debugging together with the training script, ensure that this parameter is **consistent with the training script**.

## Usage Constraints

1. Weight Conversion v2 does not currently support converting LoRA/QLoRA weights to Hugging Face. This includes merging LoRA/QLoRA weights with base weights and converting them to the Hugging Face format, as well as converting LoRA/QLoRA weights separately to the Hugging Face format.

2. Weight Conversion v2 does not currently support QLoRA quantized weight conversion from Hugging Face to MCore.

3. Weight Conversion v1 and Weight Conversion v2 are two separate solutions. Do not mix them. For example, do not use v2 for Hugging Face-to-MCore conversion and then use v1 for MCore-to-Hugging Face conversion.

## Community Contributions

Community contributions are welcome. If you have any suggestions for improvement or find any issues during your use of MindSpeed LLM weight conversion, including but not limited to functional and compliance issues, please submit an issue on GitCode. We will review and resolve it promptly.

Thanks to all community members who contribute to the project. 🎉
