# Weight Conversion

## Weight Conversion Background

As model scale grows from the hundred-million level to the trillion level, terabyte-scale parameter models place extremely high demands on system resources during deployment and migration, and a single device cannot hold the full set of model parameters. MindSpeed LLM uses a memory-efficient weight conversion solution that supports on-demand loading to address the tendency of large-parameter models to crash during conversion. Therefore, it provides a technical foundation for the efficient training and application of ultra-large models.

- [Weight Download](#weight-download)

  Download open-source model weights from Hugging Face and other sites. It supports both CLI and web downloads.

- [How to Use Weight Conversion](#how-to-use-weight-conversion)
  - [Converting Hugging Face Weights to the MCore Format](#converting-hugging-face-weights-to-the-mcore-format)

    Convert Hugging Face model weights to the MCore format. It supports multiple parallel sharding schemes.

  - [Converting MCore Weights to the Hugging Face Format](#converting-mcore-weights-to-the-hugging-face-format)

    Convert MCore model weights to the Hugging Face format for migration across different frameworks.

  - [Debug Feature: Converting Hugging Face Reduced-Layer Weights to the MCore Format](#debug-feature-converting-hugging-face-reduced-layer-weights-to-the-mcore-format)

    Reduced-layer conversion of Hugging Face model weights to the MCore format with multiple parallel sharding schemes is supported.

## Weight Conversion Overview

Weight conversion addresses compatibility issues for model weights across different deep learning frameworks and training strategies. It supports efficient weight conversion across multiple models and training configurations. The core features include:

**Weight conversion across formats**: It converts weights between the mainstream Hugging Face and Megatron-LM frameworks under any parallel sharding strategy.

**Weight conversion for training parallel strategies**: It supports weight conversion across multiple training parallel strategies, including tensor parallelism (TP), pipeline parallelism (PP), expert parallelism (EP), expert tensor parallelism (ETP), and virtual pipeline parallelism (VPP). Whether you train with different parallel strategies or need to switch between them, it provides flexible weight conversion to meet a wide range of training and inference needs.

## Weight Download

Download open-source model weights from Hugging Face and other sites. You can find training weight download links in the **Download Link** column in the [Supported Models in the PyTorch Framework](../models/supported_models.md).

### Download Methods

#### Method 1. Direct Download on the Web

Open the link in a browser and manually download all weight files.

#### Method 2. CLI Download

Save the downloaded weights to the `MindSpeed-LLM/model_from_hf` directory. For example:

```shell
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

- Hugging Face official documentation: <https://huggingface.co/docs/hub/models-downloading>

- ModelScope download guide: <https://modelscope.cn/docs/models/download>

> [!NOTE]
>
> If the download is too slow or inaccessible, configure a proxy or a local mirror site and try again.

### Notes

- Ensure that you have enough drive space to store the model weights.

- Check file integrity and verify the file size and MD5 value after the download.

- Some models may require you to log in or request access before you can download them.

## How to Use Weight Conversion

### Converting Hugging Face Weights to the MCore Format

Weight conversion converts Hugging Face weights to the MCore format. It supports multiple parallel strategies, such as tensor parallelism and pipeline parallelism, and ensures that you can continue training and inference in MindSpeed LLM after conversion.

> [!NOTE]
>
> Before you convert weights, first confirm the training-time parameter configuration and modify the weight conversion script in the repository according to your training configuration. These configurations change the structure of the weights. If they do not match the training parameters, training cannot load the weights. For the training configurations that need to be confirmed, see [Table 1](#table1).

**Table 1** Training configuration reference <a id="table1"></a>

| Parameter | Description | Default Value | Must Match Training Configuration |
|-----------|-------------|---------------|-----------------------------------|
| `--load-model-type` | Source model type. Options are `hf` or `mg`. | `hf` | ❌ |
| `--save-model-type` | Converted model type. Options are `hf` or `mg`. | `mg` | ❌ |
| `--load-dir` | Source model path. | None | ❌ |
| `--save-dir` | Path where the converted model weights are stored. | None | ❌ |
| `--hf-cfg-dir` | Path to the source Hugging Face weight configuration directory. This is an optional parameter. During mg2hf conversion, necessary configuration files are copied to the weight save directory to generate ready-to-use Hugging Face format weights. | None | ❌ |
| `--model-type-hf` | Hugging Face model family. The default is `qwen3`. For already supported models, the script is preconfigured. Therefore, you do not need to change it. | `qwen3` | ❌ |
| `--target-tensor-parallel-size` | TP. Specifies the tensor parallel size. | 1 | ✅ |
| `--target-pipeline-parallel-size` | PP. Specifies the pipeline parallel size. | 1 | ✅ |
| `--target-expert-parallel-size` | EP. Specifies the expert parallel size. | 1 | ✅ |
| `--expert-tensor-parallel-size` | ETP. Specifies expert tensor parallelism. Currently only ETP=1 is supported after it is enabled. | None; equals the TP size during actual conversion | ✅ |
| `--num-layers-per-virtual-pipeline-stage` | VPP partitioning. Specifies the number of layers in each VPP stage. | None | ✅ |
| `--num-layer-list` | Dynamic PP partitioning. It specifies the number of layers in each PP stage through a list. When you use it, separate the values with commas. The sum of the list values must equal the total number of model layers, and the length of the list must equal PP. For example, if the model has 14 layers, set `--num-layer-list 3,4,4,3` and `--target-pipeline-parallel-size 4`. | None | ✅ |
| `--noop-layers` | Custom noop-layer operation. Specify where to insert noop layers in the model. After conversion, the number of layers equals the original Hugging Face model layer count plus the number of noop layers. | None | ✅ |
| `--moe-grouped-gemm` | MoE grouped matrix multiplication optimization. | None | ✅ |
| `--moe-tp-extend-ep` | TP extends EP. When enabled, the TP group in expert layers shards expert parameters. | None | ✅ |
| `--mla-mm-split` | When enabled, it expands the compressed `q_compressed` and `kv_compressed` to a higher dimension. | None | ✅ |
| `--mtp-num-layers` | The number of MTP layers. | 0 | ✅ |
| `--schedules-method` | DualPipeV pipeline scheduling. The available option is `dualpipev`. | None | ✅ |

#### Usage Constraints

- The number of model layers must be divisible by the PP sharding count. Otherwise, add noop layers using `--noop-layers` or use dynamic PP with `--num-layer-list`.

- VPP (`--num-layers-per-virtual-pipeline-stage`) and dynamic PP partitioning (`--num-layer-list`) are mutually exclusive.

#### Usage Example

The following Hugging Face-to-MCore weight conversion script for the Qwen3-235b model is provided for reference only:

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

#### Launch Script

MindSpeed LLM provides prebuilt model weight conversion scripts. The following lists the naming style and launch method for Hugging Face to MCore weight conversion scripts. You can search by model category:

```shell
# Script naming:
# bash examples/mcore/model_name/ckpt_convert_xxx_hf2mcore.sh

# Launch method:
bash examples/mcore/qwen3_moe/ckpt_convert_qwen3_moe_235b_hf2mcore.sh
```

> [!NOTE]
>
> Configure the parallel parameters, weight and vocabulary paths, weight loading paths (including vocabulary and other configuration files), and the weight save path in the weight conversion script.

### Converting MCore Weights to the Hugging Face Format

Weight conversion converts MCore weights to the Hugging Face format. It supports multiple parallel strategies, such as tensor parallelism and pipeline parallelism. During conversion, the model weights are adapted to the standard Hugging Face format so that you can continue training and inference in the Hugging Face weight format.

#### Usage Constraints

- Because Hugging Face weights do not involve parallel sharding, **you do not need to set `--target-tensor-parallel-size`, `--target-pipeline-parallel-size`, `--target-expert-parallel-size`, or `--num-layers-per-virtual-pipeline-stage`** when converting to Hugging Face weights.

- After conversion succeeds, the save directory contains only model weight files. It does not generate model configuration files such as `config.json` or vocabulary files such as `tokenizer.model` and `vocab.json`. You can use the `--hf-cfg-dir` parameter to point to the configuration file directory of the original Hugging Face model, and the configuration files will be automatically copied to the weight save directory after mg2hf conversion.

- If the MCore weights are configured with noop layers using `--noop-layers`, you must add the **same noop-layer configuration** when converting MCore weights to the Hugging Face format.

- If the expert tensor parallelism (ETP) of the original MCore weights is 1, you must add the **`--expert-tensor-parallel-size 1`** parameter when you run the mcore2hf conversion script.

#### Usage Example

The following MCore-to-Hugging Face weight conversion script for the Qwen3-235b model is provided for reference only:

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

#### Launch Script

MindSpeed LLM provides prebuilt model weight conversion scripts. The following lists the naming style and launch method for MCore to Hugging Face weight conversion scripts. You can search by model category:

```shell
# Script naming:
# bash examples/mcore/model_name/ckpt_convert_xxx_mcore2hf.sh

# Launch method:
bash examples/mcore/qwen3_moe/ckpt_convert_qwen3_moe_235b_mcore2hf.sh
```

> [!NOTE]
>
> Configure the parallel parameters, weight and vocabulary paths, weight loading paths (including vocabulary and other configuration files), and the weight save path in the weight conversion script.

### Debug Feature: Converting Hugging Face Reduced-Layer Weights to the MCore Format

This framework supports **reduced-layer debugging** when converting Hugging Face weights to the MCore format without changing the model configuration file. You can configure the reduction with the following CLI parameters.

- `--num-layers`

  This specifies the number of layers in the reduced model. It cannot exceed the number of layers in the original model, and this number does **not** include MTP layers. The default value is None. **When you do not use reduced layers, the value comes from the configuration file. Therefore, you do not need to specify this parameter.**

  If you configure noop layers with `--noop-layers`, the value of `num-layers` should be the real layer count. It should not include MTP layers or the `--noop-layers` count.

  If you need to use reduced-layer debugging together with the training script, note the following: when the `--noop-layers` configuration is not used, this parameter must be **consistent with the training script**; when the `--noop-layers` configuration is used, the `--num-layers` parameter for weight conversion is the real layer count (not including noop layers), while the `--num-layers` parameter in the training script is the total layer count (including the `--noop-layers` count).

- `--first-k-dense-replace`

  This specifies the number of dense layers before MoE layers in the reduced model. It cannot exceed the number of dense layers in the original model. The default value is None. **When you do not use reduced layers, the value comes from the configuration file. Therefore, you do not need to specify this parameter.**

  If you need to use reduced-layer debugging together with the training script, ensure that this parameter is **consistent with the training script**.

- `--mtp-num-layers`

  This is the number of MTP layers. The default value is 0. It supports configuring MTP layers during reduction, and the value cannot exceed the number of MTP layers in the original model.

  If you need to configure MTP layers, you can set them on the CLI, for example `--mtp-num-layers 1`.

  If you need to use reduced-layer debugging together with the training script, ensure that this parameter is **consistent with the training script**.
