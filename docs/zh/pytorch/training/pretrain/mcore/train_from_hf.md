# 数据和权重在线加载训练

## 使用场景

一般情况下，用户需要先离线执行权重转换和数据预处理，将HuggingFace格式的权重转换为Megatron格式，并将原始数据集转换成Megatron格式的数据集，然后才能启动训练任务。这种分离的操作方式增加了使用复杂度和时间成本。

本功能集成了数据预处理、权重转换和训练流程，通过单脚本即可启动训练任务。

- 权重转换与训练合一：实现了从HuggingFace权重的加载、转换和训练保存。通过自动检测加载目录中的权重文件格式，系统可自动启用相关转换功能，实现HuggingFace权重到Megatron格式的双向自动转换与训练合一。用户无需单独执行权重转换步骤，实现从HuggingFace权重到训练任务的一键式启动。
- 自动数据预处理：数据预处理功能在模型训练时自动识别并转换原始数据文件，用户无需手动执行原始数据转换。系统将根据输入路径自动判断是否为原始数据格式（如 .jsonl、.parquet 等），并在训练初始化阶段自动完成数据格式转换。

## 使用方法

### 权重转换与训练合一

当前仅支持单机或共享存储模式，系统会在训练初始化阶段自动检测当前是否为共享存储环境。

系统通过检测加载目录中的权重文件来推断是否需要自动转换。当加载目录中存在`.safetensors`文件或Mamba模型的`.bin`格式文件，且用户未显式设置转换标志时，系统将自动开启权重转换功能，无需手动配置其他参数。系统会将HuggingFace格式的权重转换为Megatron格式权重用于训练，并在每次保存分布式权重后，将其转回HuggingFace格式权重。

当`--load`参数指定为HuggingFace权重路径时，需确保路径中包含`config.json`等配置文件以读取参数配置。如果未指定`--model-type-hf`参数，系统将尝试读取`{load}/config.json`文件，从配置文件中自动推断匹配支持的模型类型。请注意，对于Mamba模型，需要手动配置此参数。

#### 快速启用

当加载目录中存在HuggingFace格式权重时（即存在`.safetensors`或`.bin`格式文件），系统会自动启用双向转换。

```bash
# 加载HuggingFace权重，自动转换并训练
    --load /path_to_huggingface_model \           # 设置HuggingFace权重路径
    --save /path_to_save_training_results \       # 设置训练后权重保存路径
    --model-type-hf <model_type>                  # 可选，系统会自动推断
```

#### 使用说明

在`pretrain_xxx.sh`或者`tune_xxx.sh`的预训练和微调脚本中，根据使用场景增加参数以开启权重转换，更多详情请参见[参数说明](#参数说明)。

- 场景1：从HuggingFace加载并训练

    ```bash
    # 从HuggingFace格式加载，自动转换为Megatron格式进行训练
    --enable-hf2mg-convert \
    --model-type-hf <model_type>
    ```

- 场景2：开启双向权重转换

    ```bash
    # 训练时同时保存两种格式的权重，作用等同于自动启用双向转换
        --enable-hf2mg-convert \
        --enable-mg2hf-convert \
        --model-type-hf <model_type>
    ```

- 场景3：将训练每次保存的Megatron格式权重转换为HuggingFace格式

    ```bash
    # 将训练过程中每次保存的Megatron格式权重转换为HuggingFace格式
        --enable-mg2hf-convert \
        --model-type-hf  <model_type>
    ```

- 场景4：仅转换最终保存模型为HuggingFace格式

    ```bash
    # 仅将训练结束后保存的Megatron格式权重转换为HuggingFace格式，不转换训练中间过程保存的Megatron格式权重
        --enable-mg2hf-convert \
        --only-convert-last-checkpoint \
        --model-type-hf  <model_type>
    ```

#### 参数说明

**表 1**  参数说明

| 参数 | 类型 | 默认值 | 是否必需 | 描述 |
|------|------|--------|------|------|
| `--load` | string | None | 是 | 加载模型权重的目录，权重在线加载训练场景，指向HuggingFace权重路径 |
| `--save` | string | None | 是 | 训练后保存模型权重的目录 |
| `--model-type-hf` | string | None | 否 | HuggingFace模型类型，支持多种预训练模型 |
| `--enable-hf2mg-convert` | bool | False | 否 | 单独启用HF→Megatron权重转换 |
| `--enable-mg2hf-convert` | bool | False | 否 | 单独启用Megatron→HF权重转换 |
| `--only-convert-last-checkpoint` | bool | False | 否 | 仅在训练结束时转换最后保存的分布式权重 |
| `--mg-save-dir` | string | None | 否 | HF→Megatron权重转换时，指定Megatron权重保存目录 |
| `--hf-save-dir` | string | None | 否 |  Megatron→HF权重转换时，HuggingFace权重保存目录 |
| `--hf-cfg-dir` | string | None | 否 | HuggingFace配置文件目录 |

> [!NOTE]
>
> - 对于Mamba等特殊模型，必须手动指定 `--model-type-hf`。
> - 由于Megatron→HF权重转换仅生成权重以及`model.safetensors.index.json`，不会生成配置文件，因此需要通过指定`--hf-cfg-dir`参数，将原HuggingFace模型的配置文件复制到权重转换所生成的HuggingFace权重目录中。

#### 资源要求

系统资源要求如下：

- 磁盘空间：请确保有足够的磁盘空间存放转换后的权重。
- 转换时间：训练初始化后，系统将自动进行权重转换。根据模型参数规模，预计耗时2分钟至2小时不等，请耐心等待。
- 权限要求：请确保对以下所有相关路径有读写权限。
    - `{load}`：模型加载路径
    - `{save}`：训练保存路径
    - `{mg-save-dir}`：Megatron权重保存目录（如果指定）
    - `{hf-save-dir}`：HuggingFace权重保存目录（如果指定）
    - `{hf-cfg-dir}`：HuggingFace配置文件目录（如果指定）

#### 约束条件

- HF→MG转换 (`--enable-hf2mg-convert`)
  - 设置加载路径。启用此功能时必须设置`--load`参数，用于指定HuggingFace权重目录，不支持从随机初始化开始训练。
  - 不支持Megatron格式权重。配置此参数后，不支持使用离线转换的Megatron格式权重。
  - 存储路径规则：
    - 如果指定`--mg-save-dir`，则转换后的Megatron权重将保存至该指定路径。
    - 如果未指定，则默认保存至`{load}/megatron_cache_tp{TP}pp{PP}ep{EP}`目录。
    - 训练过程将自动使用该路径作为权重加载路径。

- MG→HF转换 (`--enable-mg2hf-convert`)
  - 设置保存路径。启用此功能时必须设置`--save`参数，用于指定训练输出路径。
  - 此功能仅支持在单机或者共享存储环境中使用。
  - 不支持LoRA。不支持对LoRA微调后的权重进行Megatron→HuggingFace转换。
  - 存储路径规则：
    - 如果指定`--hf-save-dir`，则转换后的HuggingFace权重将保存至`{hf_save_dir}/mg2hf_iteration{iteration}/`目录。
    - 如果未指定，则默认保存至`{save}/mg2hf_iteration{iteration}`目录。
    - 配置文件处理：指定`--hf-cfg-dir`时，从该目录复制配置文件至转换后的HuggingFace权重目录；未指定但启用双向转换时，从`{load}`目录复制配置文件。

> [!NOTE]
>
> MG→HF转换本身不会生成配置文件，必须从现有配置文件复制。

### 自动数据预处理

#### 快速启用

如需使用数据预处理功能，请参考参数说明并根据使用场景添加相关参数，通过修改`--data-path`参数指定输入数据集路径，以决定是否进行数据预处理。

目前支持的形式如下：

| 输入形式 | 示例 | 说明 |
|-----------|-------|------|
| **原始文件** | `/data/train.jsonl` | 原始数据集，自动识别并转换为`.bin/.idx`格式 |
| **已转换前缀** | `/data/train_text_document` | 已为转换后的格式，可以直接使用 |

#### 参数说明

**表 2**  参数说明

| 参数 | 类型 | 默认值 |是否必需 | 描述 |
|------|------|------|------|------|
| `--data-path` | string或list | None | 是 |原始数据路径或已转换前缀 |
| `--handler-name` | string | "" | 是 | 数据处理handler名称 |
| `--append-eod` | bool | False | 否 | 是否在文档末尾追加 `<eod>` token |
| `--prompt-type` | string | None | 是（微调）| 指定微调prompt模板 |
| `--json-keys` | list | `["text"]` | 否 | 要提取的字段  |
| `--workers` | int | 1 | 否 | 数据处理线程数 |
| `--n-subs` | int | 1 | 否 | 数据子集数量（多进程切分） |
| `--pack` | bool | False | 否 | 是否对样本进行打包（微调场景）|
| `--neat-pack` | bool | False | 否 | Pack场景下使用锯齿状的`attention_mask`参与计算的开关（微调场景） |
| `--enable-thinking` | string | None | 否 | 是否启用思维模式（微调场景） |
| `--output-prefix` | string | None | 否 | 转换后输出的数据集文件的文件名前缀 |
| `--seq-length` | int | None | 否 | Pack模式下指定数据打包后的序列长度 |
| `--reasoning-effort` | string | None | 否 | 用于DeepSeek-V4模型微调数据处理，可选 max/high。max：在prompt中插入最大努力的指令前缀；high：预留，当前为空操作 |
| `--drop-thinking` | bool | True | 否 | DeepSeek-V4微调场景，是否丢弃多轮对话中的历史思维链。默认仅保留最后一轮assistant的reasoning作为loss目标；设为False则保留所有轮次的reasoning |

> [!NOTE]
>
> 若未指定`--output-prefix`, 处理后的数据文件将默认生成在原始数据集所在的目录下。

### 使用示例

以Qwen3-8B模型微调为例，同时开启数据预处理和权重转换集成训练，则需要在[Qwen3-8B微调脚本](../../../../../../examples/mcore/qwen3/tune_qwen3_8b_4K_full_ptd.sh)基础上增加以下几个参数：

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

## 使用约束

- 当前支持的HuggingFace模型类型：`qwen3`、`qwen3-moe`、`deepseek3`、`glm45-air`、`bailing_mini`、`qwen3-next`、`seed-oss`、`deepseek32`、`magistral`以及`deepseek2-lite`。

- 当前数据集自动转换功能仅支持以下原始数据格式：`parquet`、`arrow`、`csv`、`json`、`jsonl`以及`txt`，暂不支持其他的格式。

- 当前权重转换`--enable-mg2hf-convert`功能仅支持单机或者共享存储环境。

- 当前权重转换`--enable-mg2hf-convert`功能不支持对LoRA微调后的权重进行Megatron→HF权重转换。
