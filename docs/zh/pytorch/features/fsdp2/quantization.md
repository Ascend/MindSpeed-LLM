# MindSpeed LLM FSDP2后端低精度训练指南

## 介绍

本指南旨在帮助用户在 MindSpeedLLM 框架下，基于 FSDP2 后端实现低精度训练（如 mxfp8 等），
提升训练效率与显存利用率。通过配置量化配方（QuantizationRecipe）与低精度all-gather模式，可在保持模型精度的前提下，显著降低通信开销与内存占用，适用于大模型训练场景。

## 使用方法

### 1. 参数概览

| 参数                                             | 类型   | 默认值                  | 说明                 |
|------------------------------------------------|------|----------------------|--------------------|
| `--model.quant_recipe_name`                    | str  | mxfp8（必填）            | 使用的量化配方名           |
| `--model.quant_apply_modules`                  | str  | 'model.layers.{*}'   | 应用量化的层或模块          |
| `--model.quant_ignored_modules`                | str  | '*lm_head'，'*gate'   | 不应用量化的子模块列表        |
| `--model.quant_converters`                     | str  | 'quantize.linear.mx' | 使用的量化转换器列表         |
| `--model.enable_fsdp_low_precision_all_gather` | bool | `True`                | 是否启用低精度通信 |
| `--model.fsdp_low_precision_all_gather_mode`   | str  | 'on-demand'          | FSDP低精度all-gather，按需聚合前向或反向权重|

### 2. 核心参数说明

#### ✅quant_recipe_name

quant_recipe_name的格式为：

```python

<scaling_strategy>_<scaling_granularity>[-blocksize0-blocksize1-blocksize2]_<inputs_dtype>_<weight_dtype>_<grads_dtype>

```

 字段 | 说明 |
|------|------|
| `scaling_strategy` | 缩放策略，如 `dynamic`、`delayed` |
| `scaling_granularity` | 缩放粒度，如 `mx`（仅支持）、`per_tensor`、`per_channel` |
| `blocksize0-blocksize1-blocksize2` | 可选，块大小（仅用于块量化） |
| `inputs_dtype` / `weight_dtype` / `grads_dtype` | 输入、权重、梯度的数据类型，如 `E4M3`、`E5M2` |

#### 预定义配方示例

- `mxfp8`: `dynamic_MX-1-1-32_E4M3_E4M3_E4M3`  
  → 支持 MX 量化策略，适用于大多数场景。

> ⚠️ 当前仅支持 `MX` 缩放策略，后续将支持更多策略与配方。

#### ✅quant_apply_modules

指定需要应用量化的层或模块，支持通配符。
**示例：**

```python

'model.layers.{*}'          # 应用于所有 Transformer 层
'model.layers.0.self_attn' # 应用于第 0 层的自注意力模块

```

#### ✅quant_ignored_modules

指定不应用量化的子模块列表，支持通配符。

```python

'*q_proj'        # 不应用量化到所有的q_proj子模块
'*gate'          # 不应用量化到mlp中的gate部分

```

#### ✅quant_converters

指定使用的量化转换器，目前支持以下类型：

- quantize.linear.mx：适用于普通线性层（如 FFN、Attention）的 MX 策略线性量化。

- quantize.moe.mx：专用于 MoE 模型中专家模块的 MX 量化。

在moe模型中可以同时使用 'quantize.linear.mx' 和 'quantize.moe.mx'，

#### ✅enable_fsdp_low_precision_all_gather

是否启用FSDP的低精度all-gather模式。启用后，在前向/反向传播中，FSDP会以低精度权重（如mxfp8）进行参数的all-gather操作，显著降低通信开销和内存占用。在开启低精度训练的同时，
可以进一步启用该模式以最大化效率提升。

#### ✅fsdp_low_precision_all_gather_mode

指定低精度 all-gather 的通信模式：

 模式 | 说明                   |
|------|----------------------|
| `on-demand` | 仅在前向或反向传播时，通信当前所需的权重 |
| `all` | 前向和反向均通信全部权重         |

> ⚠️ 若启用重计算，系统将自动切换为 'all' 模式，确保计算一致性。

### 3. 示例脚本

以下是一个示例启动脚本，展示了如何配置量化参数与低精度通信：

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
     $QUANT_ARGS\
     | tee logs/pretrain_qwen3_moe_30b_a3b_4K_fsdp2_${TIMESTAMP}.log
     
```

只需要在原有的训练脚本基础上，添加 `QUANT_ARGS` 中的量化相关参数，即可启用低精度训练与通信。

## 注意事项

- 在开启efsdp时，由于底层框架的限制，'efsdp_shard_placement_fn' 需要设置为 'shard_by_dim_0'，以确保量化权重的正确切分与通信。
