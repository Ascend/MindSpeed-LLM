# 模型输入输出采集

本文档适用于快速查看指定训练step的Module前反向输入输出和batch字段等简单定位场景。需要按Module/API分层采集、调用栈或构图信息等复杂精度定位能力时，建议使用[msProbe 精度数据采集](msprobe.md)。

## 概述

MindSpeed LLM提供了独立于msProbe的源码级模型输入输出采集能力，适用于无法安装msProbe、需要查看
batch原始字段，或需要按Module名称直接定位前反向数据的场景。该功能支持：

- 对目标rank、目标训练step注册全模型Module前向Hook和完整反向Hook。
- 默认采集所有叶子Module的输入、输出及梯度统计量。
- 采集`input_ids`、`tokens`、`labels`、`attention_mask`、`position_ids`等batch字段预览。
- 按需将完整Tensor保存为`.pt`文件。

Hook只在配置命中的训练step内存在，step结束后会立即移除。一个step表示一次完整训练迭代，包含该次
迭代的所有micro-batch和optimizer update。

## config.json文件

MindSpeed LLM提供了默认配置文件[`model_io_trace_config.json`](../../../../mindspeed_llm/tools/model_io_trace_config.json)，在未指定自定义配置文件时自动使用该配置。

默认配置显式列出全部支持项，采集rank 0、step 0的叶子Module统计量和常用batch字段，配置示例如下：

```json
{
  "output_format": "text",
  "ranks": [0],
  "steps": [0],
  "module": {
    "leaf_only": true,
    "include": [],
    "exclude": [],
    "forward": true,
    "backward": true
  },
  "tensor": {
    "mode": "statistics",
    "statistics": ["abs_sum", "abs_mean", "max", "min"]
  },
  "batch": {
    "enabled": true,
    "fields": ["input_ids", "tokens", "labels", "attention_mask", "position_ids"],
    "max_rows": 2,
    "max_tokens": 64
  }
}
```

如需修改采集rank、step或Module范围，可复制默认文件并通过启动参数传入自定义配置路径。采集结果根目录不写入`config.json`，统一由FSDP2或Megatron的启动参数传入。

| 配置项 | 是否必选 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `output_format` | 否 | `"text"` | 输出格式：`text` 生成人工可读的文本文件；`jsonl` 生成便于程序解析的结构化记录；`both` 同时生成两种格式。 |
| `ranks` | 否 | `[0]` | 目标 global rank 列表；`[-1]` 表示所有 rank。 |
| `steps` | 否 | `[0]` | 目标 global step；支持整数、闭区间字符串及 `"all"`。 |
| `module.leaf_only` | 否 | `true` | 为 `true` 时只采集无子 Module 的叶子节点。 |
| `module.include` | 否 | `[]` | Module 全名 glob 白名单；空列表表示全部。 |
| `module.exclude` | 否 | `[]` | Module 全名 glob 黑名单。 |
| `module.forward` | 否 | `true` | 是否注册 Module forward Hook。 |
| `module.backward` | 否 | `true` | 是否注册 Module full backward Hook。 |
| `tensor.mode` | 否 | `"statistics"` | `statistics` 仅采集统计量；`tensor` 保存完整 Tensor。 |
| `tensor.statistics` | 否 | `["abs_sum", "abs_mean", "max", "min"]` | 需要计算的统计量，均基于绝对值计算。 |
| `batch.enabled` | 否 | `true` | 是否采集 batch 字段预览。 |
| `batch.fields` | 否 | `["input_ids", "tokens", "labels", "attention_mask", "position_ids"]` | 需要记录的 batch 字段，字段名必须与 `model.forward` 的入参名称一致；位置参数和关键字参数均按该名称匹配。 |
| `batch.max_rows` | 否 | `2` | batch 预览最多记录的行数。 |
| `batch.max_tokens` | 否 | `64` | batch 预览每行最多记录的元素数。 |

- 用于`include`、`exclude`和JSONL记录的Module名称以`model_chunk_0`、`model_chunk_1`开头，便于区分Megatron PP/VPP当前rank上的模型分块。

- 单模型`text`输出会去掉`model_chunk_0`前缀，使Module名称更加简洁；多模型分块场景保留前缀以防止重名。

- `include`和`exclude`使用 glob，例如：

    ```json
    {
      "include": ["model_chunk_0.*.mlp.*"],
      "exclude": ["*.dropout*"]
    }
    ```

## 基于FSDP2后端开启采集

需在YAML配置文件的`training`字段中增加如下示例代码：

```yaml
training:
  model_io_trace: true
  model_io_trace_output_path: /data/model_io_trace/qwen3_8b
  model_io_trace_config_path: /home/custom_config.json
```

也可通过命令行覆盖：

```bash
--training.model_io_trace True \
--training.model_io_trace_output_path /data/model_io_trace/qwen3_8b \
--training.model_io_trace_config_path /home/custom_config.json
```

- `model_io_trace`：可选参数，是否开启模型输入输出采集。MindSpeed LLM默认未开启，启用该功能时必选此参数。

- `model_io_trace_output_path`：必选参数，模型输入输出采集结果保存路径。

- `model_io_trace_config_path`：可选参数，自定义模型输入输出采集配置文件路径。

  > [!NOTE]
  >
  > - `model_io_trace_config_path`参数未配置时，MindSpeed LLM默认使用[`model_io_trace_config.json`](../../../../mindspeed_llm/tools/model_io_trace_config.json)。
  > - 自定义配置文件`custom_config.json`必须是所有训练进程均可访问的文件。

## Qwen3-8B FSDP2使用案例

以[Qwen3-8B FSDP2预训练配置](../../../../examples/fsdp2/qwen3/pretrain_qwen3_8b_4k_fsdp2_A3.yaml)为例，假设首个训练step的loss异常，需要快速查看rank 0、step 0的Module前反向输入输出和batch字段。

1. 启用模型输入输出采集

    默认配置已经采集rank 0、step 0，因此只需在预训练YAML文件的`training`字段中开启功能并指定输出目录。

    ```yaml
    training:
      model_io_trace: true
      model_io_trace_output_path: /data/model_io_trace/qwen3_8b
    ```

2. 编辑训练示例脚本

    根据实际环境修改并保存模型和数据路径，命令如下：

    ```bash
    vi examples/fsdp2/qwen3/pretrain_qwen3_8b_4k_fsdp2_A3.sh
    ```

3. 执行训练脚本

    ```bash
    bash examples/fsdp2/qwen3/pretrain_qwen3_8b_4k_fsdp2_A3.sh
    ```

首个训练step完成后，rank 0的采集结果位于`/data/model_io_trace/qwen3_8b/rank0/step0/`。可先查看`output.txt`和`token_ids.log`，若定位到可疑 Module后，再通过`module.include`缩小采集范围。

## 基于Megatron后端开启采集

基于Megatron后端进行模型训练时，需在训练启动参数中增加如下示例代码：

```bash
--model-io-trace \
--model-io-trace-output-path /data/model_io_trace/qwen3_8b \
--model-io-trace-config-path /home/custom_config.json
```

- `--model-io-trace`：可选参数，是否开启模型输入输出采集。MindSpeed LLM默认未开启，启用该功能时必选此参数。

- `--model-io-trace-output-path`：必选参数，模型输入输出采集结果保存路径。

- `--model-io-trace-config-path`：可选参数，自定义模型输入输出采集配置文件路径。

  > [!NOTE]
  >
  > - `--model-io-trace-config-path`参数未配置时，MindSpeed LLM默认使用[`model_io_trace_config.json`](../../../../mindspeed_llm/tools/model_io_trace_config.json)。
  > - 自定义配置文件`custom_config.json`必须是所有训练进程均可访问的文件。

Megatron的step使用恢复后的`args.iteration`；FSDP2使用恢复后的`global_step`。因此全新训练第一次迭代均对应step 0。

## 输出

典型输出结构如下：

```text
output_path/
└── rank0/
    └── step10/
        ├── output.txt
        ├── token_ids.log
        └── tensors/
```

- `output.txt`：按调用顺序记录Module前反向输入输出、Tensor，以及绝对值`sum/mean/max/min`。
- 单模型场景使用简洁的Module名称；Megatron PP/VPP多模型分块场景增加`model_chunk_N`前缀防止重名。
- `token_ids.log`保存每次模型入口调用的batch字段shape与截断预览，便于NPU与GPU使用同一套文本处理脚本。

配置为`"output_format": "text"`模式时，会将Tensor转为CPU `float32`后写出PyTorch文本表示，即使`tensor.mode`设置为`statistics`也会包含截断后的数值预览。

配置为`"output_format": "jsonl"`时，将生成`module_io.jsonl`和`batch_inputs.jsonl`。前者每行记录一个Module输入、输出或梯度Tensor，包括Module名称、阶段、槽位、shape、dtype、device和统计量；后者保存batch字段shape与截断预览。

配置为`both`时同时生成两套文件。

仅当`tensor.mode`为`tensor`时才创建`tensors/`。

## 使用注意

全模型前反向Hook会显著降低训练性能，完整Tensor还会引入设备到主机同步并快速占满磁盘。首次定位，建议只选一个rank、一个step，保持`leaf_only: true`和`tensor.mode: statistics`；确定可疑Module后，再用`include`缩小范围并将`tensor.mode`改为`tensor`。batch预览和完整Tensor可能包含训练数据，不应写入公共目录。
