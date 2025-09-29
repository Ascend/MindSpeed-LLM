# 🧠 DeepSeek R1 蒸馏模型微调与推理指南

> **适用场景**：本指南基于华为昇腾 800T A2 服务器，以 Qwen2.5-Math-7B 模型为基础，使用 DeepSeek R1 蒸馏的 OpenR1-Math-220K 数据集为例，演示如何进行模型微调与推理。

---

## 📚 目录导航

- [🧰 环境准备](#-环境准备)
  - [🖥️ 服务器要求](#-服务器要求)
  - [软件安装](#软件安装)
- [📦 模型权重下载](#-模型权重下载)
- [📁 数据集准备](#-数据集准备)
- [🔁 数据格式转换](#-数据格式转换)
- [⚙️ 权重格式转换](#-权重格式转换)
- [🧪 模型推理](#-模型推理)
- [🔧 关键参数说明](#-关键参数说明)
- [🖥️ 分布式配置相关参数](#-分布式配置相关参数)
- [📁 日志与输出控制](#-日志与输出控制)
- [🔄 常见配置更改建议](#-常见配置更改建议)
- [💬 生成模式控制](#-生成模式控制)
- [🎯 全量微调流程](#-全量微调流程)
- [📌 常见问题](#-常见问题)
- [✅ 最佳实践](#-最佳实践)

---

## 🧰 环境准备

### 🖥️ 服务器要求

- **型号**：华为昇腾 800T A2（推荐单机部署）

### 软件安装

[基础软件安装指引](https://gitcode.com/ascend/MindSpeed-LLM/blob/master/docs/mindspore/features/install_guide.md)

```bash
# 安装基础依赖
pip install modelscope pyarrow pandas

# 验证 MindSpore 安装（需与硬件匹配）
python -c "import mindspore as ms; print(ms.__version__)"
```

---

## 📦 模型权重下载

```bash
# 创建工作目录并下载模型
mkdir Qwen25Math7B
cd Qwen25Math7B

modelscope download \
  --model Qwen/Qwen2.5-Math-7B \
  --local_dir ./Qwen25Math7B
```

---

## 📁 数据集准备

```bash
# 创建目录并下载数据
mkdir OpenR1Math220K
cd OpenR1Math220K

# 使用循环批量下载数据分片
for i in {00000..00009}; do
  wget https://hf-mirror.com/datasets/open-r1/OpenR1-Math-220k/resolve/main/data/train-${i}-of-00010.parquet    
done

# 验证数据完整性（检查文件数量）
ls -l | grep parquet | wc -l  # 应输出 10
```

---

## 🔁 数据格式转换

在指令监督微调过程中，`instruction` 列的内容会与 `input` 列拼接作为人类指令（格式为 `instruction\ninput`，其中 `\n` 为换行符），`output` 列则为模型的回答。如果指定了 `history` 字段，则历史对话内容也会被加入；若指定 `system` 字段，则其内容将作为系统提示词。

```shell
# 请根据实际环境 source set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python ./preprocess_data.py \
    --input ./OpenR1Math220K/train-00000-of-00010.parquet \
    --tokenizer-name-or-path ./Qwen25Math7B/ \
    --output-prefix ./OpenR1Math220K_handled/sharegpt \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF \
    --handler-name SharegptStyleInstructionHandler \
    --prompt-type qwen_math_r1 \
    --map-keys '{"messages":"messages", "tags":{"role_tag": "role","content_tag": "content","user_tag": "user","assistant_tag": "assistant","system_tag": "system"}}'
```

### 参数说明：

#### `--input`
支持 `.parquet`、`.csv`、`.json`、`.jsonl`、`.txt`、`.arrow` 格式。输入可以是具体文件或目录，若是目录，则处理该目录下所有文件，且同一目录中数据格式应保持一致。

#### `--map-keys`
用于配置字段映射，以提取对应列。

#### `--prompt-type`
指定模型模板，使 base 模型具备更好的对话能力。

#### `--handler-name`
处理 Alpaca 风格数据集时应设置为 `AlpacaStyleInstructionHandler`，并依据 `--map-keys` 提取相应列。

### 示例：

**示例 1：**

```bash
--map-keys '{"prompt":"notice","query":"question","response":"answer","system":"system_test","history":"histories"}'
```

表示从数据集中提取 `"notice"`、`"question"`、`"answer"`、`"system_test"` 和 `"histories"` 列。

**示例 2：**

```bash
--map-keys '{"history":"histories"}'
```

表示从数据集中提取默认列 `"instruction"`、`"input"`、`"output"` 以及指定列 `"histories"`。

### 启动脚本：

```bash
# 执行转换脚本
bash examples/mindspore/deepseek_r1_distill_qwen/data_convert_distill_qwen_instruction.sh
```

---

## ⚙️ 权重格式转换

### ✅ 基本命令示例：

```bash
python convert_ckpt.py \
       --use-mcore-models \
       --model-type GPT \
       --load-model-type hf \
       --save-model-type mg \
       --target-tensor-parallel-size 2 \
       --target-pipeline-parallel-size 1 \
       --add-qkv-bias \
       --load-dir ./model_from_hf/distill_qwen_mcore/ \
       --save-dir ./model_weights/distill_qwen_mcore/ \
       --tokenizer-model ./model_from_hf/distill_qwen_mcore/tokenizer.json \
       --model-type-hf llama2 \
       --params-dtype bf16
```

### 参数说明：

| 参数名 | 含义 | 示例值 | 修改建议 |
|--------|------|--------|----------|
| `--use-mcore-models` | 使用 MCore 架构模型 | - | 若目标为 MCore 支持的模型，请保留此参数。 |
| `--model-type` | 当前支持的模型类型（如 GPT、BERT 等） | `GPT` | 通常无需更改，除非要转换其他类型模型。 |
| `--load-model-type` | 加载模型的格式 | `hf` (HuggingFace) | 若加载的是 HF 模型，则保留；否则改为对应格式（如 `megatron`）。 |
| `--save-model-type` | 目标保存模型格式 | `mg` (Megatron/MCore) | 若目标为 MCore 格式则保留，否则可改为其他格式。 |
| `--target-tensor-parallel-size` | 张量并行大小（Tensor Parallelism） | `2` | 根据训练/推理设备数量设置，通常是 GPU 数量。 |
| `--target-pipeline-parallel-size` | 流水线并行大小（Pipeline Parallelism） | `1` | 若模型较小，一般保持为 1。 |
| `--add-qkv-bias` | 是否添加 QKV 层偏置项 | - | 如果原始模型包含 QKV 偏置项，请加上此参数。 |
| `--load-dir` | 加载原始模型权重路径 | `./model_from_hf/distill_qwen_mcore/` | 修改为你本地的 HF 模型路径。 |
| `--save-dir` | 保存转换后模型路径 | `./model_weights/distill_qwen_mcore/` | 修改为你希望保存的目录路径。 |
| `--tokenizer-model` | tokenizer.json 文件路径 | `./model_from_hf/distill_qwen_mcore/tokenizer.json` | 若使用自定义分词器，请指定其路径。 |
| `--model-type-hf` | 原始 HF 模型类型 | `llama2` | 若为 LLaMA2 类结构则保留，若为其他结构（如 Qwen、Mistral），请修改为此处支持的名称。 |
| `--params-dtype` | 权重保存的数据类型 | `bf16` | 可选 `fp32`, `fp16`, `bf16`，推荐使用 `bf16` 以节省空间和计算资源。 |
| `--num-layer-list` | 分布式模型各阶段中的层数分配（可选） | `--num-layer-list 11,13,19,21` | 多 stage 流水线并行时需手动分配每 stage 的层数列表。 |

### 💡 使用提示：

- 如果不确定某些参数含义，**不要随意更改默认值**。
- 若目标为多卡分布式训练，请确保 `--target-tensor-parallel-size` 设置正确。
- 若使用非标准模型架构（如 Qwen、ChatGLM、InternLM 等），请确认是否已在代码中注册支持。

### HuggingFace → MCore 格式转换：

```bash
bash examples/mindspore/deepseek_r1_distill_qwen/ckpt_convert_distill_qwen_hf2mcore.sh
```

### MCore → HuggingFace 回转：

```bash
bash examples/mindspore/deepseek_r1_distill_qwen/ckpt_convert_distill_qwen_mcore2hf.sh
```

> **注意事项**：
- 转换后的权重文件大小应与原始权重相近（±5%）

---

## 🧪 模型推理

### 原始权重推理：

```bash
bash examples/mindspore/deepseek_r1_distill_qwen/generate_distill_qwen_7b.sh
```

### 微调后权重推理：

修改推理脚本中的 `CHECKPOINT` 参数：

```bash
# 示例：替换为微调后的权重路径
CHECKPOINT=/path/to/fine_tuned_weights
```

---

## 🔧 关键参数说明

以下是一些在推理过程中需要关注的关键参数及其作用说明：

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `CHECKPOINT` | `"your model directory path"` | 指定模型权重路径，需根据实际路径修改。支持原始权重或微调后的权重。 |
| `TOKENIZER_PATH` | `"your tokenizer directory path"` | 指定分词器路径，通常为 HuggingFace 格式 tokenizer 路径。 |
| `TP` | `1` | Tensor 并行度（Tensor Parallelism），用于控制模型张量切片的数量。多卡推理时可适当增加。 |
| `PP` | `2` | Pipeline 并行度（Pipeline Parallelism），用于控制模型层之间的并行拆分。 |
| `SEQ_LENGTH` | `8192` | 最大上下文长度，设置模型支持的最大输入 token 数量。可根据任务需求调整。 |
| `NPUS_PER_NODE` | `2` | 单个节点使用的 NPU 数量，应根据实际硬件资源进行调整。 |
| `micro-batch-size` | `1` | 推理时每个 micro batch 的大小，默认为 1。批量较大可以提升吞吐但会增加内存占用。 |
| `max-new-tokens` | `256` | 控制生成文本的最大 token 数量，可根据生成内容需求调整。 |
| `prompt-type` | `deepseek3` | 使用的 prompt 类型，决定了输入格式和 prefix 构造方式。 |

---

## 🖥️ 分布式配置相关参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MASTER_ADDR` | `localhost` | 分布式训练/推理主节点地址，多机部署时需改为对应 IP 地址。 |
| `MASTER_PORT` | `6000` | 主节点通信端口，确保未被占用。 |
| `NNODES` | `1` | 总节点数，多机部署时需根据机器数量调整。 |
| `NODE_RANK` | `0` | 当前节点编号，从 0 开始递增。 |
| `WORLD_SIZE` | 自动计算 | 总设备数 = `NPUS_PER_NODE * NNODES`，用于分布式环境初始化。 |

---

## 📁 日志与输出控制

- 所有日志会被重定向到 `logs/generate_mcore_distill_qwen_7b.log`。
- 分布式运行日志保存在 `--log_dir=msrun_log_generate` 指定目录下。
- 可通过 `tee` 实时查看日志输出。

---

## 🔄 常见配置更改建议

### ✅ 修改推理设备数量

```bash
# 如使用单卡推理
NPUS_PER_NODE=1
WORLD_SIZE=1
```

### ✅ 修改模型并行策略

```bash
# 如使用 TP=2, PP=1 的并行方式
TP=2
PP=1
```

### ✅ 更改最大生成长度

```bash
# 修改生成长度为 512
--max-new-tokens 512
```

### ✅ 更改最大上下文长度

```bash
# 改为支持 4096 长度的上下文
SEQ_LENGTH=4096
```

---

## 💬 生成模式控制

你可以通过修改 `--task` 参数来切换不同的推理任务模式，例如：

```bash
# 切换为 chat 模式（适用于对话场景）
--task chat
```

> ⚠️ 注意：不同任务模式可能要求不同的 prompt 构造方式，需配合合适的 `--prompt-type` 使用。

---

## 🎯 全量微调流程

### 配置与执行

1. **参数配置**  
   修改 `tune_distill_qwen_7b_full.sh` 中的关键参数：

   ```bash
   CKPT_LOAD_DIR="your model ckpt path"
   CKPT_SAVE_DIR="your model save ckpt path"
   DATA_PATH="your data path"
   TOKENIZER_PATH="your tokenizer path"
   ```

2. **启动微调**

   ```bash
   bash examples/mindspore/deepseek_r1_distill_qwen/tune_distill_qwen_7b_full.sh
   ```

---

## 📌 常见问题

### 1. **权重转换失败**

- **可能原因**：HuggingFace 权重不完整  
- **解决方案**：重新下载权重并验证文件哈希值

### 2. **数据转换报错**

- **错误示例**：`pyarrow.lib.ArrowException: Unknown error`  
- **处理方法**：检查 Python 环境是否包含 `pyarrow` 依赖

---

## ✅ 最佳实践

**资源监控**：使用以下命令实时监控硬件负载：

```bash
watch -n 1 npu-smi info
```

---