# 一键集成训练 (All_in_One_Trainer)

## 使用场景

在之前的版本中，用户需要先离线执行权重转换和数据预处理，将 HuggingFace 格式的权重转换为 Megatron 格式，并且将原始数据集转换成 Megatron 格式的数据集，然后再启动训练过程。这种分离的操作方式增加了使用复杂度和时间成本。

本功能集成了数据预处理、权重转换和训练为一体，单脚本即可启动训练任务：
- 权重转换功能通过 `--enable-hf2mg-convert` 参数实现 HuggingFace 权重到 Megatron 格式的自动转换与训练合一，用户无需独立执行权重转换步骤，真正实现从 HuggingFace 权重到训练任务的一键式集成。
- 数据预处理功能在模型训练时自动识别并转换原始数据文件，无需用户手动执行原始数据转换。系统会根据输入路径自动判断是否为原始数据格式（如 .jsonl、.parquet 等），并在训练初始化阶段自动完成数据格式转换。

## 使用方法
 
### 1. 权重转换功能

当前支持共享存储和非共享存储环境，系统会在训练初始化阶段自动检测存储类型并采用最优的权重转换策略，用户无需手动配置：

- **共享存储环境**：所有计算节点可访问同一存储路径，仅需 rank0 进程执行权重转换，其他进程等待转换完成后开始训练
- **非共享存储环境**：各计算节点使用本地独立存储，每个节点的 local_rank=0 进程分别执行权重转换
- **混合存储环境**：暂不支持部分节点为共享存储、部分节点为非共享存储的异构环境

#### 基本命令

在`pretrain_xxx.sh` 或者`tune_xxx.sh`的预训练和微调脚本中，增加以下参数来开启权重转换：

```bash
--enable-hf2mg-convert \
--model-type-hf <model_type> \
```

#### 参数说明

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `--enable-hf2mg-convert` | flag | 是 | 启用 HF 到 Megatron 权重转换功能 |
| `--model-type-hf` | string | 是 | 指定 HuggingFace 模型类型 |
|`--mg-cache-dir` | string | 否 | 指定转换后mg权重的储存路径, 不指定则默认存到 `{load}/megatron_cache{TP}{PP}{EP}` |
| `--load` | string | 是 | 在增加`--enable-hf2mg-convert`后, `--load`要求权重类型: Huggingface格式权重。Huggingface权重路径下必须包含配置文件 `config.json` 和模型文件（`.bin` 或 `.safetensors` 格式），并且模型文件名中需包含 "model" 关键词 | 

注意：
- 请确保有足够的磁盘空间存放转换后的权重，如果指定`--mg-cache-dir`则权重会储存在该路径, 不指定则默认保存在`{load}/megatron_cache{TP}{PP}{EP}`，训练过程会自动使用该路径作为权重加载路径。
- 训练初始化后自动进行权重转换过程, 根据模型参数时间预计需要2分钟-2小时, 请耐心等待。
- 开启`--enable-hf2mg-convert`参数后, 不支持使用离线转换的mcore格式Megatron权重。
- 请确保对`{load}`路径有读写权限

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