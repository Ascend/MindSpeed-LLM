# 数据/权重在线加载训练 (Train_from_HF)

## 使用场景

在之前的版本中，用户需要先离线执行权重转换和数据预处理，将 HuggingFace 格式的权重转换为 Megatron 格式，并且将原始数据集转换成 Megatron 格式的数据集，然后再启动训练过程。这种分离的操作方式增加了使用复杂度和时间成本。

本功能集成了数据预处理、权重转换和训练为一体，单脚本即可启动训练任务：
- 权重转换合一训练功能实现了从 HuggingFace加载训练和保存功能，通过自动检测加载目录中的权重文件格式，系统可自动启用相关转换功能，实现 HuggingFace 权重到 Megatron 格式的双向自动转换与训练合一，用户无需独立执行权重转换步骤，实现从 HuggingFace 权重到训练任务的一键式集成。
- 数据预处理功能在模型训练时自动识别并转换原始数据文件，无需用户手动执行原始数据转换。系统会根据输入路径自动判断是否为原始数据格式（如 .jsonl、.parquet 等），并在训练初始化阶段自动完成数据格式转换。

## 使用方法
 
### 1. 权重转换功能

当前仅支持单机/共享存储，系统会在训练初始化阶段自动检测是否为共享存储环境。

通过检测加载目录中的权重文件，当加载目录中存在 `.safetensors` 文件或者 mamba 模型的 `.bin` 格式文件，并且用户未显式设置转换标志，系统会自动开启权重转换，无需手动配置其他参数，将 HuggingFace 权重转换为 Megatron 格式权重用于训练，并在训练每次保存分布式权重后，将其转为 HuggingFace 格式权重。

当`--load`参数指定为 HuggingFace 权重路径时，需包含`config.json`等文件用于读取参数配置。如果未指定 `--model-type-hf` 参数，系统会尝试读取 `{load}/config.json` 文件从配置文件自动推断匹配支持的模型类型，请注意，对于mamba模型需要手动配置此参数。


#### 快速开始

当加载目录中存在 HuggingFace 格式权重时，系统会自动启用双向转换：

```bash
# 加载 HuggingFace 权重，自动转换并训练
    --load /path/to/huggingface/model \
    --save /path/to/save/training/results \
    --model-type-hf <model_type>  # 可选，系统会自动推断
```

#### 调试功能

**场景1：从 HuggingFace 加载并训练**

在`pretrain_xxx.sh` 或者`tune_xxx.sh`的预训练和微调脚本中，增加以下参数来开启权重转换：

```bash
# 从 HuggingFace 格式加载，自动转换为 Megatron 格式进行训练
--enable-hf2mg-convert \
--model-type-hf <model_type>
```


**场景2：开启双向权重转换**

在`pretrain_xxx.sh` 或者`tune_xxx.sh`的预训练和微调脚本中，增加以下参数来开启权重转换：

```bash
# 训练时同时保存两种格式的权重，作用等同于自动启用双向转换
    --enable-hf2mg-convert \
    --enable-mg2hf-convert \
    --model-type-hf <model_type>
```


**场景3：将训练保存的 Megatron 格式权重转换为 HuggingFace 格式**

在`pretrain_xxx.sh` 或者`tune_xxx.sh`的预训练和微调脚本中，增加以下参数来开启权重转换：

```bash
# 将训练过程中每次保存的 Megatron 格式权重转换为 HuggingFace 格式
    --enable-mg2hf-convert \
    --model-type-hf  <model_type>
```

**场景4：仅转换最终保存模型为 HuggingFace 格式**

在`pretrain_xxx.sh` 或者`tune_xxx.sh`的预训练和微调脚本中，增加以下参数来开启权重转换：

```bash
# 仅将训练结束后保存的 Megatron 格式权重转换为 HuggingFace 格式，不转换训练中间过程保存的 Megatron 格式权重
    --enable-mg2hf-convert \
    --only-convert-last-checkpoint \
    --model-type-hf  <model_type>
```

**参数说明**

| 参数 | 类型 | 默认值 | 必需 | 描述 |
|------|------|--------|------|------|
| `--model-type-hf` | str | None | 可选* | HuggingFace 模型类型，支持多种预训练模型 |
| `--enable-hf2mg-convert` | flag | False | 可选 | 单独启用 HF→Megatron 权重转换 |
| `--enable-mg2hf-convert` | flag | False | 可选 | 单独启用 Megatron→HF 权重转换 |
| `--only-convert-last-checkpoint` | flag | False | 可选 | 仅在训练结束时转换最后保存的分布式权重 |
| `--mg-save-dir` | str | None | 可选 | HF→Megatron 权重转换时，指定Megatron 权重保存目录 |
| `--hf-save-dir` | str | None | 可选 |  Megatron→HF 权重转换时，HuggingFace 权重保存目录 |
| `--hf-cfg-dir` | str | None | 可选 | HuggingFace 配置文件目录，由于Megatron→HF 权重转换仅生成权重以及`model.safetensors.index.json`，不会生成配置文件，通过指定此参数，将原HuggingFace模型的配置文件复制到权重转换生成的HuggingFace权重目录 |

*注：对于 mamba 等特殊模型，必须手动指定 `--model-type-hf`


#### 注意事项

1. 系统资源要求
- 磁盘空间：请确保有足够的磁盘空间存放转换后的权重
- 转换时间：训练初始化后自动进行权重转换，根据模型参数规模，预计需要 2分钟-2小时 不等，请耐心等待
- 权限要求：请确保对以下所有相关路径有读写权限：
  - `{load}` - 模型加载路径
  - `{save}` - 训练保存路径
  - `{mg-save-dir}` - Megatron权重保存目录（如指定）
  - `{hf-save-dir}` - HuggingFace权重保存目录（如指定）
  - `{hf-cfg-dir}` - HuggingFace配置文件目录（如指定）

2. HF→MG转换 (`--enable-hf2mg-convert`) 约束条件
- 必须设置加载路径：启用此功能时必须设置 `--load` 参数，指定HuggingFace权重来源，不支持从随机初始化开始训练
- 不支持Megatron格式权重：开启此参数后，不支持使用离线转换的Megatron格式权重
- 存储路径规则：
  - 如果指定 `--mg-save-dir`：转换后的Megatron权重保存在该指定路径
  - 如果未指定：默认保存在 `{load}/megatron_cache_tp{TP}pp{PP}ep{EP}` 目录下
  - 训练过程会自动使用该路径作为权重加载路径

3. MG→HF转换 (`--enable-mg2hf-convert`) 约束条件
- 必须设置保存路径：启用此功能时必须设置 `--save` 参数，指定训练输出路径
- 仅支持共享存储：此功能仅支持在共享存储环境中使用
- 不支持LoRA/QLoRA：不支持对LoRA或QLoRA微调后的权重进行Megatron→HuggingFace转换
- 存储路径规则：
  - 如果指定 `--hf-save-dir`：转换后的HuggingFace权重保存在 `{hf_save_dir}/mg2hf_iteration{iteration}/` 目录下
  - 如果未指定：默认保存在 `{save}/mg2hf_iteration{iteration}` 目录下
- 配置文件处理：
  - 如果指定 `--hf-cfg-dir`：将从此目录复制配置文件到转换后的HuggingFace权重目录
  - 如果未指定但启用了双向转换：则从 `{load}` 目录复制配置文件
  - 注意：MG→HF转换本身不会生成配置文件，必须从现有配置文件复制


### 2. 数据预处理功能

#### 基本命令

如果要使用数据预处理功能，请参考参数说明根据使用场景添加相关参数，并修改 `--data-path` 输入数据集路径来决定是否进行数据预处理，目前支持的形式如下：

| 输入形式 | 示例 | 说明 |
|-----------|-------|------|
| **原始文件** | `/data/train.jsonl` | 原始数据集，自动识别并转换为 `.bin/.idx` 格式 |
| **已转换前缀** | `/data/train_text_document` | 已为转换后的格式，可以直接使用 |

#### 参数说明

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `--data-path` | `str / list` | 是 |原始数据路径或已转换前缀 |
| `--handler-name` | `str` | 是 | 数据处理 handler 名称 |
| `--append-eod` | `bool` | 否 | 是否在文档末尾追加 `<eod>` token |
| `--prompt-type` | `str` | 是（微调）| 指定微调 prompt 模板 |
| `--json-keys` | `list` | 否 | 要提取的字段，默认 `["text"]` |
| `--workers` | `int` | 否 | 数据处理线程数 |
| `--n-subs` | `int` | 否 | 数据子集数量（多进程切分） |
| `--pack` | `bool` | 否 | 是否对样本进行打包（微调场景） |
| `--neat-pack` | `bool` | 否 | Pack场景下使用锯齿状的`attention_mask`参与计算的开关（微调场景） |
| `--enable-thinking` | `str` | 否 | 是否启用思维模式（微调场景） |
| `--output-prefix` | `str` | 否 | 转换后输出的数据集文件的文件名前缀 |

注意：
- 若未指定`--output-prefix`, 处理后的数据文件将默认生成在原始数据集所在的目录下。

### 3. 使用示例

以Qwen3-8B模型微调为例, 同时开启数据预处理和权重转换集成训练，则需要在[Qwen3-8B微调脚本](../../../../examples/mcore/qwen3/tune_qwen3_8b_4K_full_ptd.sh)基础上增加以下几个参数：

```bash
DATA_PATH="/path/your_dataset/xxx.parquet"
CKPT_LOAD_DIR="/path/to/huggingface_model/Qwen3-8B"
--data-path DATA_PATH \
--load CKPT_LOAD_DIR \
--enable-hf2mg-convert \
--model-type-hf qwen3 \
--handler-name AlpacaStyleInstructionHandler \
--prompt-type qwen3 \
```

## 使用约束

- 当前支持的 HuggingFace 模型类型：`qwen3, qwen3-moe, deepseek3, glm45-moe, bailing_mini, qwen3-next, seed-oss, deepseek32, magistral, deepseek2-lite`。

- 当前数据集自动转换功能仅支持以下原始数据格式：`parquet, arrow, csv, json, jsonl, txt`, 暂不支持其他的格式。

- 当前权重转换 `--enable-mg2hf-convert`功能仅支持单机或者共享存储环境。

- 当前权重转换 `--enable-mg2hf-convert`功能不支持对Lora/Qlora微调后的权重做 Megatron→HF 权重转换。