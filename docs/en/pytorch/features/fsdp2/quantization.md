# MindSpeed LLM FSDP2 Back-End Low-Precision Training Guide

## Introduction

This guide helps you implement low-precision training, such as `mxfp8`, in the MindSpeedLLM framework based on the FSDP2 back end.
It improves training efficiency and memory utilization. By configuring a `QuantizationRecipe` and the low-precision all-gather mode, you can significantly reduce communication overhead and memory use while preserving model accuracy, making it suitable for LLM training scenarios.

## Usage

### 1. Parameter Overview

| Parameter | Type | Default | Description |
|------------------------------------------------|------|----------------------|--------------------|
| `--model.quant_recipe_name` | str | `mxfp8` (required) | Name of the quantization recipe. |
| `--model.quant_apply_modules` | str | `'model.layers.{*}'` | Layers or modules to which quantization applies. |
| `--model.quant_ignored_modules` | str | `'*lm_head'`, `'*gate'` | List of submodules to which quantization does not apply. |
| `--model.quant_converters` | str | `'quantize.linear.mx'` | List of quantization converters to use. |
| `--model.enable_fsdp_low_precision_all_gather` | bool | `True` | Whether to enable low-precision communication. |
| `--model.fsdp_low_precision_all_gather_mode` | str | `'on-demand'` | FSDP low-precision all-gather mode. Aggregates the weights needed for the forward or backward pass on demand. |

### 2. Core Parameters

#### ✅ `quant_recipe_name`

The format of `quant_recipe_name` is:

```python
<scaling_strategy>_<scaling_granularity>[-blocksize0-blocksize1-blocksize2]_<inputs_dtype>_<weight_dtype>_<grads_dtype>
```

| Field | Description |
|------|------|
| `scaling_strategy` | Scaling strategy, such as `dynamic` or `delayed`. |
| `scaling_granularity` | Scaling granularity, such as `mx` (the only supported option), `per_tensor`, or `per_channel`. |
| `blocksize0-blocksize1-blocksize2` | Optional block size, used only for block quantization. |
| `inputs_dtype` / `weight_dtype` / `grads_dtype` | Data types for inputs, weights, and gradients, such as `E4M3` and `E5M2`. |

#### Predefined Recipe Example

- `mxfp8`: `dynamic_MX-1-1-32_E4M3_E4M3_E4M3`.
  It supports the MX quantization strategy and suits most scenarios.

> ⚠️ Currently, only the `MX` scaling strategy is supported. More strategies and recipes will be available later.

#### ✅ `quant_apply_modules`

Specify the layers or modules to which quantization applies. Wildcards are supported.
**Example:**

```python
'model.layers.{*}'          # Applies to all Transformer layers.
'model.layers.0.self_attn'  # Applies to the self-attention module in layer 0.
```

#### ✅ `quant_ignored_modules`

Specify the list of submodules to which quantization does not apply. Wildcards are supported.

```python
'*q_proj'        # Does not apply quantization to all `q_proj` submodules.
'*gate'          # Does not apply quantization to the gate part in MLP.
```

#### ✅ `quant_converters`

Specify the quantization converters to use. The following types are currently supported:

- `quantize.linear.mx`: MX-style linear quantization for standard linear layers, such as FFN and Attention.
- `quantize.moe.mx`: MX quantization specifically for expert modules in MoE models.

> 💡 In an MoE model, you can use both `quantize.linear.mx` and `quantize.moe.mx` together.

#### ✅ `enable_fsdp_low_precision_all_gather`

Whether to enable the FSDP low-precision all-gather mode. After you enable it, FSDP performs parameter all-gather operations with low-precision weights, such as `mxfp8`, during the forward and backward passes, significantly reducing communication overhead and memory use.

When you also enable low-precision training, you can turn on this mode to maximize the efficiency gain.

#### ✅ `fsdp_low_precision_all_gather_mode`

Specify the communication mode for low-precision all-gather:

| Mode | Description |
|------|----------------------|
| `on-demand` | Communicates only the weights needed for the forward or backward pass. |
| `all` | Communicates all weights in both the forward and backward passes. |

> ⚠️ If you enable recomputation, the system automatically switches to the `all` mode to ensure computation consistency.

### 3. Example Script

The following example start-up script shows how to configure quantization parameters and low-precision communication:

```bash
QUANT_ARGS="
    --model.quant_recipe_name mxfp8 \
    --model.enable_fsdp_low_precision_all_gather \
    --model.quant_converters quantize.linear.mx quantize.moe.mx \
    --parallel.efsdp_shard_placement_fn shard_by_dim_0
"

bash tests/tools/fsdp2/moe_hf_param_merge_experts.sh
torchrun $DISTRIBUTED_ARGS train_fsdp2.py \
    examples/fsdp2/qwen3_moe/pretrain_qwen3_30b_4k_fsdp2_A3.yaml \
    $QUANT_ARGS \
    | tee logs/pretrain_qwen3_moe_30b_a3b_4K_fsdp2_${TIMESTAMP}.log
```

You only need to add the quantization-related parameters in `QUANT_ARGS` to the existing training script to enable low-precision training and communication.

## Notes

- When you enable efsdp, due to limitations in the underlying framework, set `efsdp_shard_placement_fn` to `shard_by_dim_0` to ensure correct sharding and communication of quantized weights.
