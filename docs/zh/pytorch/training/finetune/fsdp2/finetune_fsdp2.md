# FSDP2 后端微调使用指南

## 使用场景

监督微调（Supervised Fine-Tuning，SFT）通过高质量的指令和回复数据继续训练预训练模型，使模型学习特定任务、领域知识或对话方式。该方法适用于问答、文本生成、摘要、翻译、代码生成和领域适配等场景。

> [!NOTE]
>
> 若您是首次使用 FSDP2 后端，建议先按 [FSDP2 快速入门](../../fsdp2_quick_start.md)（Qwen3-8B 预训练与微调）完成一次端到端跟练。本文档侧重全参数微调的模型、数据集、YAML 配置与参数说明，适用于更换模型或自定义数据集场景。

本文介绍如何基于 HuggingFace 格式的预训练模型，使用 MindSpeed LLM FSDP2 后端完成全参数微调。示例采用 Qwen3-8B 模型和单台 `Atlas 900 A2 PoD`（1x8 集群），主要流程如下：

**图 1** FSDP2 后端模型微调流程

![微调流程图](../../../figures/instruction_finetune/process_of_instruction_tuning_fsdp2.png)

## 使用说明

### 环境搭建

启动微调前，请参考 [MindSpeed LLM 安装指导](../../install_guide.md)完成环境安装。

FSDP2 后端的公共环境变量位于 `examples/fsdp2/env_config.sh`，示例启动脚本会自动加载该文件。配置如下：

```bash
export TRAINING_BACKEND=mindspeed_fsdp
export HCCL_CONNECT_TIMEOUT=1800
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export MULTI_STREAM_MEMORY_REUSE=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DEBUG=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### 模型和数据集准备

**模型准备**

模型权重下载地址请参考 [模型支持列表](../../../models/supported_models.md)。本示例使用 [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/tree/main) 的 HuggingFace 格式权重。

完整模型目录应包含模型配置、Tokenizer 配置和全部权重文件，例如：

```text
Qwen3-8B/
├── config.json
├── generation_config.json
├── merges.txt
├── model-00001-of-00005.safetensors
├── model-00002-of-00005.safetensors
├── model-00003-of-00005.safetensors
├── model-00004-of-00005.safetensors
├── model-00005-of-00005.safetensors
├── model.safetensors.index.json
├── tokenizer.json
├── tokenizer_config.json
└── vocab.json
```

FSDP2 后端可直接加载 HuggingFace 格式权重，开始训练前不需要执行模型权重转换。当前仓库默认在对应的 `examples/fsdp2/**/*.sh` 启动脚本中，通过 `--model.model_name_or_path` 配置模型权重目录。例如，Qwen3-8B 微调时需要修改 `examples/fsdp2/qwen3/tune_qwen3_8b_4k_fsdp2_A2.sh` 中的以下参数：

```bash
--model.model_name_or_path /path/to/Qwen3-8B/
```

**数据集准备**

FSDP2 微调使用 LLaMA Factory 风格的数据处理流程，可加载 `.parquet`、`.csv`、`.json`、`.jsonl`、`.txt` 和 `.arrow` 等格式的数据文件。数据在训练任务启动时完成加载、格式对齐和 Tokenize，无需提前转换为 Megatron Indexed Dataset。

当前支持的主要数据格式包括：

- Alpaca 格式：通常包含 `instruction`、`input` 和 `output` 字段。
- ShareGPT 格式：通常在 `conversations` 字段中保存对话消息。
- OpenAI 格式：使用 `messages` 字段保存对话消息，每条消息通常包含 `role` 和 `content`。OpenAI 格式是一种数据字段规范，并非特定数据集。

详细格式说明请参考：

- [Alpaca 风格数据集](../../../tools/data_process_sft_alpaca_style.md)
- [ShareGPT 和 OpenAI 风格数据集](../../../tools/data_process_sft_sharegpt_style.md)

### 配置数据集

当前仓库默认在对应的 `examples/fsdp2/**/*.sh` 启动脚本中，通过 CLI 参数 `--data.dataset` 配置训练数据集，而不是直接修改配套 YAML。数据集参数支持内联配置和 `dataset_info.json` 注册名称两种写法。

以 Qwen3-8B 微调脚本为例，模型权重和数据集的默认配置位置如下：

```bash
torchrun $DISTRIBUTED_ARGS train_fsdp2.py \
  examples/fsdp2/qwen3/tune_qwen3_8b_4k_fsdp2_A2.yaml \
  --model.model_name_or_path /path/to/Qwen3-8B/ \
  --data.dataset alpaca_full \
  --parallel.fsdp_size 8 \
  --parallel.ep_size 1 \
  --parallel.ep_fsdp_size 1 \
  --training.per_device_train_batch_size 1 \
  --training.gradient_accumulation_steps 1 \
  --training.output_dir ./output
```

启动命令中的参数说明如下：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `model_name_or_path` | HuggingFace 模型权重目录。 | `/home/data/Qwen3-8B/` |
| `dataset` | 训练数据集内联配置或注册名称。 | `alpaca_full` |
| `fsdp_size` | 全分片数据并行大小，应与 `NPUS_PER_NODE * NNODES` 保持一致。 | `8` |
| `ep_size` | 专家并行大小；稠密模型配置为 `1`。 | `1` |
| `ep_fsdp_size` | 专家并行组内的 FSDP 大小；稠密模型配置为 `1`。 | `1` |
| `per_device_train_batch_size` | 单卡训练 batch size。 | `1` |
| `gradient_accumulation_steps` | 梯度累积步数。 | `1` |
| `output_dir` | 训练检查点的输出目录。 | `./output` |

**在启动脚本中使用内联配置**

内联配置适合快速验证本地数据集。修改对应 `.sh` 文件中的 `--data.dataset`：

```bash
--data.dataset '{"file_name":"./dataset/train.json","formatting":"alpaca"}'
```

**在启动脚本中使用数据集注册名称**

需要复用数据集时，建议编辑 `configs/fsdp2/data/dataset_info.json`，添加数据集配置：

```json
{
  "alpaca_demo": {
    "file_name": "./alpaca_demo.json",
    "formatting": "alpaca"
  },
  "sharegpt_demo": {
    "file_name": "./sharegpt_demo.jsonl",
    "formatting": "sharegpt"
  },
  "openai_demo": {
    "file_name": "./openai_demo.jsonl",
    "formatting": "openai",
    "columns": {
      "messages": "messages"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system",
      "observation_tag": "tool",
      "function_tag": "function_call"
    }
  }
}
```

然后修改对应 `.sh` 文件中的 `--data.dataset`，通过注册名称指定数据集：

```bash
--data.dataset alpaca_demo
```

如需混合多个数据集，可以使用逗号分隔多个注册名称：

```bash
--data.dataset alpaca_demo,sharegpt_demo
```

### 配置微调参数

模型权重路径、训练数据集、并行规模、batch size 和输出目录等常用参数默认通过 `.sh` 启动脚本传入；其余通用参数保存在配套 YAML 中。详细配置请参考 [Qwen3-8B 微调配置文件](../../../../../../examples/fsdp2/qwen3/tune_qwen3_8b_4k_fsdp2_A2.yaml)。YAML 示例内容如下：

```yaml
model:
  trust_remote_code: False
  train_from_scratch: False

data:
  template: qwen3
  cutoff_len: 4096
  max_samples: 100000
  overwrite_cache: True
  preprocessing_num_workers: 1

parallel:
  fsdp_modules:
    - model.layers.{*}
    - model.embed_tokens
    - lm_head
  ep_modules:
    - model.layers.{*}.mlp.experts
  ep_fsdp_modules:
    - model.layers.{*}.mlp.experts
  ep_dispatcher: eager
  recompute: True
  recompute_modules:
    - model.layers.{*}

training:
  dataloader_num_workers: 4
  seed: 42
  dataloader_drop_last: True
  optimizer: adamw
  lr: 1e-05
  weight_decay: 0.01
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_epsilon: 1e-08
  max_grad_norm: 1.0
  lr_scheduler_type: cosine
  warmup_ratio: 0.0
  min_lr: 1e-06
  max_steps: 2000
  save_steps: 500
  logging_steps: 1
```

YAML 中的主要参数说明如下：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `template` | 根据模型选择的 Prompt 模板。 | `qwen3` |
| `cutoff_len` | 分词后训练序列的最大长度，超过该长度的内容将被截断。 | `4096` |
| `max_samples` | 每个数据集最多使用的样本数量，主要用于调试。 | `100000` |
| `overwrite_cache` | 是否覆盖已经生成的数据处理缓存。 | `True` |
| `preprocessing_num_workers` | 数据预处理进程数。 | `1` |
| `fsdp_modules` | 使用 FSDP 分片的模型模块，不可为空。 | `model.layers.{*}`、`model.embed_tokens`、`lm_head` |
| `recompute` | 是否启用激活重计算，以计算开销换取显存空间。 | `True` |
| `lr` | 初始学习率。 | `1e-05` |
| `max_steps` | 最大训练步数；大于 0 时覆盖 `num_train_epochs`。 | `2000` |
| `save_steps` | 保存训练检查点的步数间隔。 | `500` |

完整参数说明请参考 [FSDP2 参数说明](../../../features/fsdp2/arguments.md)。

### 配置微调脚本

打开对应的 `.sh` 文件，配置模型权重、数据集、输出目录和分布式训练参数。详细配置请参考 [Qwen3-8B 微调启动脚本](../../../../../../examples/fsdp2/qwen3/tune_qwen3_8b_4k_fsdp2_A2.sh)。

单机 8 卡配置如下：

```bash
NPUS_PER_NODE=8                              # 当前节点使用的 NPU 数量
MASTER_ADDR=localhost                        # 主节点 IP 地址，单机可配置为 localhost
MASTER_PORT=6499                             # 主节点通信端口
NNODES=1                                     # 参与训练的节点总数，单机配置为 1
NODE_RANK=0                                  # 当前节点编号，单机配置为 0
WORLD_SIZE=$((NPUS_PER_NODE * NNODES))       # 参与训练的 NPU 总数
```

多机配置示例如下：

```bash
NPUS_PER_NODE=8                              # 每个节点使用的 NPU 数量
MASTER_ADDR="主节点 IP"                      # 所有节点均配置为主节点 IP，不能使用 localhost
MASTER_PORT=6499                             # 所有节点保持相同的主节点通信端口
NNODES=2                                     # 参与训练的节点总数
NODE_RANK="当前节点编号"                      # 取值范围为 0 至 NNODES-1，各节点不能重复
WORLD_SIZE=$((NPUS_PER_NODE * NNODES))       # 参与训练的 NPU 总数
```

不同节点的 `MASTER_ADDR`、`MASTER_PORT` 和 `NNODES` 应保持一致，`NODE_RANK` 从 0 开始且不能重复。

根据实际环境，在 `.sh` 文件的启动命令中修改模型路径、数据集、并行规模和输出路径：

```bash
torchrun $DISTRIBUTED_ARGS train_fsdp2.py \
  examples/fsdp2/qwen3/tune_qwen3_8b_4k_fsdp2_A2.yaml \
  --model.model_name_or_path /path/to/Qwen3-8B/ \
  --data.dataset alpaca_demo \
  --parallel.fsdp_size 8 \
  --parallel.ep_size 1 \
  --parallel.ep_fsdp_size 1 \
  --training.per_device_train_batch_size 1 \
  --training.gradient_accumulation_steps 1 \
  --training.output_dir ./output
```

> [!NOTE]
>
> - 本示例中 `fsdp_size` 应与 `NPUS_PER_NODE * NNODES` 保持一致。
> - 多机训练时，请确保各节点均能正确访问模型和数据集路径。
> - `.sh` 中的 CLI 参数优先级高于 YAML 中的同名参数；若两处同时配置，以 `.sh` 中传入的值为准。

### 启动微调

参数配置完成后，在仓库根目录执行：

```bash
bash examples/fsdp2/qwen3/tune_qwen3_8b_4k_fsdp2_A2.sh
```

多机训练时，需要在所有节点执行启动脚本，并分别设置对应的 `NODE_RANK`。训练日志默认保存在 `logs/` 目录，训练检查点保存在 `--training.output_dir` 指定的目录。

执行一段时间后，可在终端看到如下日志：

```shell
INFO [2026-06-22 19:25:37] >>  iteration        1/    2000 | consumed samples:          8 | consumed tokens:        564 | elapsed time per iteration (ms): 7827.39 | learning rate: 1.666667E-07 | global batch size:     8 | lm loss: 3.316887E+00 | grad norm: 70.959 | max_memory_allocated(GB): 19.07 | max_memory_reserved(GB): 20.90 |
INFO [2026-06-22 19:25:38] >>  iteration        2/    2000 | consumed samples:         16 | consumed tokens:       1357 | elapsed time per iteration (ms): 1331.74 | learning rate: 3.333333E-07 | global batch size:     8 | lm loss: 2.443476E+00 | grad norm: 41.986 | max_memory_allocated(GB): 19.07 | max_memory_reserved(GB): 22.43 |
INFO [2026-06-22 19:25:38] >>  iteration        3/    2000 | consumed samples:         24 | consumed tokens:       2113 | elapsed time per iteration (ms): 981.08 | learning rate: 5.000000E-07 | global batch size:     8 | lm loss: 2.669216E+00 | grad norm: 45.335 | max_memory_allocated(GB): 19.07 | max_memory_reserved(GB): 22.43 |
```

终端开始持续输出迭代次数、学习率、损失值、梯度范数和显存占用等信息，表示训练任务已经正常运行。

## 使用约束

- 本指南适用于 FSDP2 后端全参数 SFT，不涵盖 LoRA、DPO、PPO 和奖励模型训练等其他后训练方法。
- `model_name_or_path`、数据集和输出目录应使用训练环境中可访问的有效路径。
- `template` 应与目标模型匹配，否则可能导致训练输入格式与模型预期不一致。
- 模型规模、序列长度、batch size 和并行规模需要根据设备数量及显存容量进行调整。
