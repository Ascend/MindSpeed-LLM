# DeepSeek-V3 Model Training

To use DeepSeek-V3 for training, first read the pretraining or fine-tuning usage guide, configure the <a href="../../mcore/deepseek3/">DeepSeek-V3 training scripts</a>, and then run the training process.

## 1. Pretraining Guide

<a href="../../../docs/en/pytorch/training/pretrain/mcore/pretrain.md">Pretraining guide</a>

Use the scripts for the pretraining process in the following order:

```shell
a. data_convert_deepseek3_pretrain.sh

b. pretrain_deepseek3_671b_4k_xxx.sh (Select the A2 or A3 script based on the cluster environment.)
```

## 2. Fine-Tuning Guide

<a href="../../../docs/en/pytorch/training/finetune/mcore/instruction_finetune.md">Fine-tuning guide</a>

Use the scripts for the full-parameter fine-tuning process in the following order:

```shell
a. data_convert_deepseek3_instruction.sh

b. ckpt_convert_deepseek3_hf2mcore.sh

Note: This script is an example. Set the weight conversion parameters based on the tune_deepseek3_671b_4k_full_xxx.sh script in step c. For parameter descriptions, see the following "DeepSeek-V3 Weight Conversion" section.

c. tune_deepseek3_671b_4k_full_xxx.sh (Select the A2 or A3 script based on the cluster environment.)

d. ckpt_convert_deepseek3_mcore2hf.sh (optional)
```

Use the scripts for the LoRA/QLoRA fine-tuning process in the following order:

```shell
a. data_convert_deepseek3_instruction.sh

b. ckpt_convert_deepseek3_hf2mcore.sh

Note: This script is an example. Set the weight conversion parameters based on the tune_deepseek3_671b_4k_xlora_xxx.sh script in step c. For parameter descriptions, see the following "DeepSeek-V3 Weight Conversion" section.

c. tune_deepseek3_671b_4k_xlora_xxx.sh (Select the A2 or A3 script based on the cluster environment.)

d. ckpt_convert_deepseek3_merge_lora2hf.sh (optional)
```

# DeepSeek-V3 Weight Conversion

Important: Before weight conversion, first confirm the parameter configuration used during training. Configure `ckpt_convert_xxx.sh` based on the parameters in the training script `pretrain_xxx.sh` or `tune_xxx.sh`.

## 1. HF2MG/MG2HF Weight Conversion

### 1.1 Conversion Script Configuration and Execution

(1) Hugging Face to Megatron

- The script converts [Hugging Face weights](https://huggingface.co/deepseek-ai/DeepSeek-V3/tree/main) to distributed Megatron MCore weights for tasks such as fine-tuning, inference, and evaluation. The original weights must be dequantized to obtain data in the BF16 format. For the dequantization method, see the [code](https://modelers.cn/models/MindIE/deepseekv3/blob/main/NPU_inference/fp8_cast_bf16.py) officially provided by MindIE.

- Configure the <a href="../../mcore/deepseek3/ckpt_convert_deepseek3_hf2mcore.sh">ckpt_convert_deepseek3_hf2mcore.sh</a> script in the DeepSeek-V3 model directory with the same configuration as the training script, and then run the conversion:

```shell
bash examples/mcore/deepseek3/ckpt_convert_deepseek3_hf2mcore.sh
```

(2) Megatron to Hugging Face

- The script converts trained distributed Megatron MCore weights back to the Hugging Face format.
- Configure the <a href="../../mcore/deepseek3/ckpt_convert_deepseek3_mcore2hf.sh">ckpt_convert_deepseek3_mcore2hf.sh</a> script in the DeepSeek-V3 model directory with the same configuration as the training script, and then run the conversion:

```shell
bash examples/mcore/deepseek3/ckpt_convert_deepseek3_mcore2hf.sh
```

(3) LoRA/QLoRA to Hugging Face

- The script converts trained LoRA/QLoRA weights to the Hugging Face format.

- Configure the <a href="../../mcore/deepseek3/ckpt_convert_deepseek3_merge_lora2hf.sh">ckpt_convert_deepseek3_merge_lora2hf.sh</a> script in the DeepSeek-V3 model directory with the same configuration as the training script, and then run the conversion:

```shell
bash examples/mcore/deepseek3/ckpt_convert_deepseek3_merge_lora2hf.sh
```

### 1.2 Related Parameters

<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Description</th>
      <th>Must match the training configuration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>--target-tensor-parallel-size, --source-tensor-parallel-size</td>
      <td>Tensor parallel size. The default value is 1.</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--target-pipeline-parallel-size, --source-pipeline-parallel-size</td>
      <td>Pipeline parallel size. The default value is 1.</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--target-expert-parallel-size, --source-expert-parallel-size</td>
      <td>Expert parallel size. The default value is 1.</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--num-layers-per-virtual-pipeline-stage</td>
      <td>Virtual pipeline parallelism. The default value is <code>None</code>. Note that <code>--num-layers-per-virtual-pipeline-stage</code> and <code>--num-layer-list</code> cannot be used at the same time.</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--moe-grouped-gemm</td>
      <td>When each expert group contains multiple experts, you can use Grouped GEMM to improve utilization and performance.
Note that this parameter cannot be used with <code>--save-lora-to-hf</code>. That is, after GEMM is enabled, conversion of separate LoRA weights only to the Hugging Face format is not supported.</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--load-dir</td>
      <td>Hugging Face weights that have been dequantized to data in the BF16 format.</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>--save-dir</td>
      <td>Storage path of the converted weights in the Megatron format.</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>--mtp-num-layers</td>
      <td>Number of MTP layers. If MTP layers are not required, set this parameter to 0. The maximum value is 1. The default value is 0.
MTP layer weights are stored in the last PP stage by default.
Note that QLoRA and LoRA weight conversion do not support MTP.</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--num-layers</td>
      <td>Number of model layers. This value does not include MTP layers. The default value is 61.
If noop layers are configured, the value of <code>num-layers</code> must be the total number of layers, excluding MTP layers, plus the number of noop layers specified by <code>--noop-layers</code>.</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--first-k-dense-replace</td>
      <td>Number of dense layers before the MoE layers. The maximum value is 3. The default value is 3.</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--num-layer-list</td>
      <td>Specifies the number of layers for each PP stage. The sum must equal <code>num-layers</code>. Currently, this parameter is supported only when <code>num-layers = 61</code>. This parameter is mutually exclusive with <code>--noop-layers</code>. Use only one of them. The default value is <code>None</code>.</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--noop-layers</td>
      <td>Custom noop layers. This parameter is mutually exclusive with <code>--num-layer-list</code>. Use only one of them. The default value is <code>None</code>.</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--moe-tp-extend-ep</td>
      <td>Extends EP with TP. In the TP group of expert layers, this parameter does not shard expert parameters, but shards the number of experts. The default value is False.</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--mla-mm-split</td>
      <td>In MLA, splits two up-proj matrix multiplication operations into four operations. The default value is False.
Note that QLoRA and LoRA weight conversion does not support this parameter.</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--schedules-method</td>
      <td>Pipeline parallel method. An optional value is <code>dualpipev</code>. The default value is <code>None</code>.</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--qlora-nf4</td>
      <td>Specifies whether to enable the quantized conversion function for QLoRA weights. The default value is False.</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--save-lora-to-hf</td>
      <td>Add this parameter to convert separate LoRA weights that do not contain base weights to the Hugging Face format. This parameter is incompatible with <code>--moe-grouped-gemm</code>.During LoRA fine-tuning, do not add the <code>--moe-grouped-gemm</code> parameter to the script. You can add <code>--lora-ckpt-filter</code> to the fine-tuning script to save only LoRA weights.</td>
      <td>✅</td>
    </tr>
  </tbody>
</table>

## 2. LoRA Weight Conversion

### 2.1 LoRA Weights with Base Weights

If LoRA weights contain base weights and they must be merged before conversion to the Hugging Face format:

Example

```shell
python examples/mcore/deepseek3/convert_ckpt_deepseek3_mcore2hf.py \
    --source-tensor-parallel-size 1 \
    --source-pipeline-parallel-size 4 \
    --source-expert-parallel-size 8 \
    --load-dir ./model_weights/deepseek3-lora \
    --save-dir ./model_from_hf/deepseek3-hf \
    --num-layers 61 \
    --first-k-dense-replace 3 \
    --num-layer-list 16,15,15,15 \
    --lora-r 8 \
    --lora-alpha 16
```

`--load-dir`: Specifies the LoRA weight path. The weights include base weights and LoRA weights.

`--lora-r`: Rank of the LoRA matrix. This value must match the configuration used during LoRA fine-tuning.

`--lora-alpha`: Scaling factor. It scales the contribution of the low-rank matrix and must match the configuration used during LoRA fine-tuning.

[Applicable scenario] During LoRA fine-tuning, if the `--lora-ckpt-filter` parameter is not added, the saved weights include base weights and LoRA weights.

### 2.2 Loading LoRA Weights and Base Weights Separately

If base weights and separate LoRA weights must be merged and converted to the Hugging Face format, specify two paths separately for loading:

Example

```shell
python examples/mcore/deepseek3/convert_ckpt_deepseek3_mcore2hf.py \
    --source-tensor-parallel-size 1 \
    --source-pipeline-parallel-size 4 \
    --source-expert-parallel-size 8 \
    --load-dir ./model_weights/deepseek3-mcore \
    --lora-load ./ckpt/filter_lora \
    --save-dir ./model_from_hf/deepseek3-hf \
    --num-layers 61 \
    --first-k-dense-replace 3 \
    --num-layer-list 16,15,15,15 \
    --lora-r 8 \
    --lora-alpha 16
    # Configure parameters such as --num-layer-list, --noop-layers, and --num-layers-per-virtual-pipeline-stage based on task requirements.
```

`--load-dir`: Specifies the base weight path.

`--lora-load`: Specifies the LoRA weight path. Note that these weights contain only LoRA weights. During LoRA fine-tuning, add `--lora-ckpt-filter` to save only LoRA weights.

`--lora-r` and `--lora-alpha`: These values must match the configuration used during LoRA fine-tuning.

[Applicable scenario] During LoRA fine-tuning, if the `--lora-ckpt-filter` parameter is added, the saved weights contain only LoRA weights, and you must merge LoRA and Hugging Face weights.

### 2.3 Converting Only LoRA Weights to the Hugging Face Format

If separate LoRA weights must be converted to the Hugging Face format:

```shell
python examples/mcore/deepseek3/convert_ckpt_deepseek3_mcore2hf.py \
    --source-tensor-parallel-size 1 \
    --source-pipeline-parallel-size 4 \
    --source-expert-parallel-size 4 \
    --load-dir ./ckpt/lora_v3_filter \
    --save-dir ./model_from_hf/deepseek3-hf \
    --num-layers 61 \
    --first-k-dense-replace 3 \
    --num-layer-list 16,15,15,15 \
    --save-lora-to-hf \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2
```

`--load-dir`: Specifies the LoRA weight path. Note that these weights contain only LoRA weights. During LoRA fine-tuning, add `--lora-ckpt-filter` to save only LoRA weights.

`--lora-target-modules`: Defines the LoRA target modules as a string list separated by spaces. This parameter has no default value. Each string is the name of a layer that requires LoRA fine-tuning.

`--save-lora-to-hf`: Specifies conversion of only LoRA weights to the Hugging Face format. Note that these weights contain only LoRA weights. During LoRA fine-tuning, add `--lora-ckpt-filter` to save only LoRA weights.

[Applicable scenario] During LoRA fine-tuning, if the `--lora-ckpt-filter` parameter is added, the saved weights contain only LoRA weights and only LoRA weights are converted to the Hugging Face format.

## 3. QLoRA Weight Conversion

### 3.1 QLoRA Weights with Base Weights

If QLoRA weights contain base weights and they must be merged before conversion to the Hugging Face format:

Add `--qlora-save-dequantize` to the fine-tuning script to dequantize the weights during saving.

[Applicable scenario] During QLoRA fine-tuning, if the `--lora-ckpt-filter` parameter is not added, the saved weights include base weights and QLoRA weights.

Use the same merge script as `2.1 LoRA Weights with Base Weights`.

### 3.2 Loading QLoRA Weights and Base Weights Separately

If base weights and separate QLoRA weights must be merged and converted to the Hugging Face format, specify two paths separately for loading:

Example

```shell
python examples/mcore/deepseek3/convert_ckpt_deepseek3_mcore2hf.py \
    --source-tensor-parallel-size 1 \
    --source-pipeline-parallel-size 4 \
    --source-expert-parallel-size 8 \
    --load-dir ./model_weights/deepseek3-mcore \
    --lora-load ./ckpt/filter_lora \
    --save-dir ./model_from_hf/deepseek3-hf \
    --num-layers 61 \
    --first-k-dense-replace 3 \
    --num-layer-list 16,15,15,15 \
    --lora-r 8 \
    --lora-alpha 16
    # Configure parameters such as --num-layer-list, --noop-layers, and --num-layers-per-virtual-pipeline-stage based on task requirements.
```

`--load-dir`: Specifies the base weight path. Because QLoRA fine-tuning loads quantized weights, you cannot directly use them as base weights. Export another copy of MCore weights without the `--qlora-nf4` parameter as the base weights for merging.

`--lora-load`: Specifies the QLoRA weight path. Note that these weights contain only QLoRA weights. In the fine-tuning script, add `--qlora-save-dequantize` to dequantize the weights during saving, and add `--lora-ckpt-filter` to save only QLoRA weights.

`--lora-r` and `--lora-alpha`: These values must match the configuration used during LoRA fine-tuning.

[Applicable scenario] During QLoRA fine-tuning, if the `--lora-ckpt-filter` parameter is added, the saved weights contain only QLoRA weights and you must merge QLoRA weights and Hugging Face weights.

### 3.3 Converting Only QLoRA Weights to the Hugging Face Format

If separate QLoRA weights must be converted to the Hugging Face format, add `--qlora-save-dequantize` to the fine-tuning script to dequantize the weights during saving, and add `--lora-ckpt-filter` to save only QLoRA weights.

Use the same conversion script as `2.3 Converting Only LoRA Weights to the Hugging Face Format`.

[Applicable scenario] During QLoRA fine-tuning, if the `--lora-ckpt-filter` parameter is added, the saved weights contain only LoRA weights and only LoRA weights are converted to the Hugging Face format.
