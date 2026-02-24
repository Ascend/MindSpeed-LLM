# MindSpeed LLM SFT模型推理评估

## 使用场景

MindSpeed LLM提供了专门用于SFT（监督微调）模型的推理评估功能，通过`evaluate_for_sft.py`和`inference_fuc.py`两个文件实现。该功能是在`inference.py`基础上增强的，继承了基础推理功能，并添加了数据集相关参数、预处理逻辑和评价规则。

### 适用场景

该评估功能适用于以下场景：

1. **SFT模型性能评估**：对监督微调后的模型进行批量评估，计算准确率等指标
2. **多格式数据集评估**：支持JSON、JSONL、Parquet等多种数据格式的评估
3. **多对话格式支持**：自动识别OpenAI、ShareGPT、Alpaca等常见对话格式
4. **分布式评估**：在多NPU或多节点环境下进行大规模模型评估
5. **错误分析与调试**：通过详细的错误分析定位模型问题，指导模型优化
6. **思维链模式评估**：支持处理包含思考过程的模型输出

### 主要功能特点

- 支持多种数据格式（JSON、JSONL、Parquet）
- 自动识别多种对话数据格式（OpenAI、ShareGPT、Alpaca）
- 灵活的评估配置（批处理大小、采样策略、生成参数等）
- 支持思维链（Think）模式输出处理
- 支持分布式推理评估
- 继承`inference.py`的所有基础推理功能和参数

### 与inference.py的关系

| 特性 | inference.py | evaluate_for_sft.py |
|------|--------------|--------------------|
| 基础推理功能 | ✅ | ✅（继承） |
| 模型加载 | ✅ | ✅（继承） |
| 生成策略 | ✅ | ✅（继承） |
| 数据集评估 | ❌ | ✅ |
| 自动数据格式识别 | ❌ | ✅ |
| 评估指标计算 | ❌ | ✅ |
| 错误分析 | ❌ | ✅ |

`evaluate_for_sft.py`在`inference.py`的基础上添加了：
1. 数据集相关参数（--eval-data-path, --eval-data-size等）
2. 数据预处理逻辑（支持多种数据格式）
3. 评价规则实现（compare_rule函数）
4. 批处理评估功能
5. 详细的评估结果输出

## 使用说明

### 环境准备

在运行推理评估前，请确保已完成以下环境准备：
1. 安装MindSpeed LLM框架及其依赖
2. 准备好SFT微调后的模型权重
3. 准备评估数据集（支持JSON、JSONL、Parquet格式）
4. 准备好对应的分词器

### 基本使用方式

`evaluate_for_sft.py`是在`inference.py`基础上增强的，因此启动时需要`inference.py`的启动参数，同时需要添加评估相关的参数。使用以下命令运行SFT模型推理评估：

```shell
python evaluate_for_sft.py \
    --load <model_weights_path> \
    --tokenizer-name-or-path <tokenizer_path> \
    --eval-data-path <evaluation_data_path> \
    --eval-batch-size <batch_size> \
    [inference.py基础参数] \
```

### 参数详解

#### 评估配置参数（evaluate_for_sft.py新增）

这些参数是`evaluate_for_sft.py`新增的评估相关参数：

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| --eval-data-path | string | 必填 | 评估数据文件路径（支持JSON、JSONL、Parquet格式） |
| --eval-data-size | int | None | 评估数据样本数量（None表示使用所有数据） |
| --eval-shuffle | bool | False | 是否打乱评估数据 |
| --eval-batch-size | int | 10 | 评估批次大小，太大可能引起OOM |
| --rm-think | bool | False | 启用思考模式，移除模型输出中的中间思考过程 |

### 数据格式支持

该推理评估功能支持三种常见的对话数据格式：

#### OpenAI格式

```json
{
  "messages": [
    {
      "role": "system",
      "content": "系统提示（可选）"
    },
    {
      "role": "user",
      "content": "用户指令"
    },
    {
      "role": "assistant",
      "content": "模型响应（作为ground truth）"
    }
  ]
}
```

#### ShareGPT格式

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "用户指令"
    },
    {
      "from": "gpt",
      "value": "模型响应（作为ground truth）"
    }
  ],
  "system": "系统提示（可选）"
}
```

#### Alpaca格式

```json
{
  "instruction": "用户指令（必填）",
  "input": "用户输入（可选）",
  "output": "模型响应（必填，作为ground truth）",
  "system": "系统提示（可选）",
  "history": [
    ["第一轮指令（可选）", "第一轮响应（可选）"],
    ["第二轮指令（可选）", "第二轮响应（可选）"]
  ]
}
```

### 使用示例

#### 自定义比较规则

可以通过修改`compare_rule()`函数来实现自定义的评估比较逻辑，例如支持模糊匹配、部分匹配等：

```python
def compare_rule(trust, prediction):
    # 示例：支持模糊匹配，忽略大小写
    return trust.lower() == prediction.lower()
```

#### 支持新的数据格式

可以扩展`build_prompt_list()`函数来支持新的数据格式，只需添加对应的格式判断和处理逻辑即可。

#### 将现有推理脚本改造成evaluate_for_sft脚本

以下以Qwen2.5模型的推理脚本为例，介绍如何将现有推理脚本改造成`evaluate_for_sft.py`脚本。

##### 示例1：将generate_qwen25_7b_ptd.sh改造成SFT评估脚本

原脚本：`examples/mcore/qwen25/generate_qwen25_7b_ptd.sh`

改造后的脚本：`examples/mcore/qwen25/evaluate_qwen25_7b_sft_ptd.sh`

```bash
#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# please fill these path configurations
CHECKPOINT="your model directory path"
TOKENIZER_PATH="your tokenizer directory path"
EVAL_DATA_PATH="your evaluation data path" # 添加评估数据路径

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TP=4
PP=2
SEQ_LENGTH=32768

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS ./tests/tools/sft_inference/evaluate_for_sft.py \
       --use-mcore-models \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --num-layers 28 \
       --hidden-size 3584  \
       --num-attention-heads 28  \
       --ffn-hidden-size 18944 \
       --max-position-embeddings ${SEQ_LENGTH} \
       --seq-length ${SEQ_LENGTH} \
       --disable-bias-linear \
       --add-qkv-bias \
       --group-query-attention \
       --num-query-groups 4 \
       --swiglu \
       --use-fused-swiglu \
       --normalization RMSNorm \
       --norm-epsilon 1e-6 \
       --use-fused-rmsnorm \
       --position-embedding-type rope \
       --rotary-base 1000000 \
       --use-fused-rotary-pos-emb \
       --make-vocab-size-divisible-by 1 \
       --padded-vocab-size 152064 \
       --micro-batch-size 1 \
       --max-new-tokens 256 \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --tokenizer-not-use-fast \
       --hidden-dropout 0 \
       --attention-dropout 0 \
       --untie-embeddings-and-output-weights \
       --no-gradient-accumulation-fusion \
       --attention-softmax-in-fp32 \
       --seed 42 \
       --load ${CHECKPOINT} \
       --exit-on-missing-checkpoint \
       # 添加SFT评估相关参数 \
       --eval-data-path ${EVAL_DATA_PATH} \
       --eval-batch-size 10 \
       --eval-shuffle \
       --do-sample false \
       | tee logs/evaluate_mcore_qwen25_7b_sft.log
```

##### 示例2：将generate_qwen25_7b_lora_ptd.sh改造成SFT评估脚本

原脚本：`examples/mcore/qwen25/generate_qwen25_7b_lora_ptd.sh`

改造后的脚本：`examples/mcore/qwen25/evaluate_qwen25_7b_lora_sft_ptd.sh`

```bash
#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# please fill these path configurations
TOKENIZER_PATH="your tokenizer directory path"
CHECKPOINT="your model directory path"
CHECKPOINT_LORA="your lora model directory path"
EVAL_DATA_PATH="your evaluation data path" # 添加评估数据路径

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TP=1
PP=8
SEQ_LENGTH=32768

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

LORA_ARGS="
    --lora-load ${CHECKPOINT_LORA} \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-fusion \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2
"

torchrun $DISTRIBUTED_ARGS ./tests/tools/sft_inference/evaluate_for_sft.py \
       --use-mcore-models \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --load ${CHECKPOINT} \
       --num-layer-list 4,4,4,4,3,3,3,3 \
       --num-layers 28 \
       --hidden-size 3584  \
       --num-attention-heads 28  \
       --ffn-hidden-size 18944 \
       --max-position-embeddings ${SEQ_LENGTH} \
       --seq-length ${SEQ_LENGTH} \
       --make-vocab-size-divisible-by 1 \
       --padded-vocab-size 152064 \
       --rotary-base 1000000 \
       --untie-embeddings-and-output-weights \
       --micro-batch-size 1 \
       --swiglu \
       --disable-bias-linear \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --normalization RMSNorm \
       --position-embedding-type rope \
       --norm-epsilon 1e-6 \
       --hidden-dropout 0 \
       --attention-dropout 0 \
       --tokenizer-not-use-fast \
       --add-qkv-bias \
       --max-new-tokens 256 \
       --no-gradient-accumulation-fusion \
       --exit-on-missing-checkpoint \
       --attention-softmax-in-fp32 \
       --seed 42 \
       --group-query-attention \
       --num-query-groups 4 \
       ${LORA_ARGS} \
       # 添加SFT评估相关参数 \
       --eval-data-path ${EVAL_DATA_PATH} \
       --eval-batch-size 8 \
       --eval-data-size 1000 \
       --eval-shuffle \
       --do-sample true \
       --top-k 50 \
       --top-p 0.95 \
       --temperature 0.7 \
       | tee logs/evaluate_mcore_qwen25_7b_lora_sft.log
```

#### 示例输出

```
Processing batch 1/10, data range: 0-9/100
Current batch accuracy: 0.8, index: 10
Average accuracy: 0.8
==================================================
...
===========Prediction Error Detail=============
Prediction Error:
Prompt: <s> [INST] 1+1=? [/INST], Index 3: Ground Truth=2, Prediction=3
Prediction Error:
Prompt: <s> [INST] 2+3=? [/INST], Index 7: Ground Truth=5, Prediction=6
===========Prediction Error Detail End=============
correct = 85
total = 100
```

#### 改造步骤说明

1. **修改脚本名称和输出日志名称**：
   - 将`generate_`前缀改为`evaluate_`
   - 添加`_sft`后缀以标识SFT评估脚本
   - 修改日志文件名以反映SFT评估

2. **替换执行脚本**：
   - 将`inference.py`替换为`evaluate_for_sft.py`

3. **添加评估数据路径**：
   - 新增`EVAL_DATA_PATH`变量，指定评估数据文件路径

4. **添加SFT评估相关参数**：
   - `--eval-data-path`: 指定评估数据路径
   - `--eval-batch-size`: 设置评估批次大小
   - `--eval-shuffle`: 启用数据打乱（可选）
   - `--eval-data-size`: 限制评估数据量（可选）
   - 调整生成策略参数（`--do-sample`, `--top-k`, `--top-p`, `--temperature`）

5. **保留原模型配置参数**：
   - 保留所有原有的模型结构和并行配置参数
   - 确保与训练时的参数保持一致

#### 运行改造后的脚本

1. **准备评估数据**：确保`EVAL_DATA_PATH`指向正确的评估数据文件
2. **设置模型和分词器路径**：正确配置`CHECKPOINT`和`TOKENIZER_PATH`
3. **运行脚本**：
   ```bash
   bash examples/mcore/qwen25/evaluate_qwen25_7b_sft_ptd.sh
   ```

4. **查看评估结果**：评估结果将输出到指定的日志文件中，包含准确率和详细的错误分析

## 使用问题

### Q1: 模型权重加载失败

**A1:** 请检查推理时的参数配置是否与训练时保持一致，特别是并行切分参数（TP/PP/EP等）。

### Q2: 数据解析失败

**A2:** 请确保评估数据格式正确，支持的格式为JSON、JSONL、Parquet，且数据结构符合OpenAI、ShareGPT或Alpaca格式。

### Q3: 生成结果为空或不完整

**A3:** 请检查生成参数配置，特别是`--max-new-tokens`参数，确保其值足够大以生成完整的结果。

### Q4: 评估准确率异常低

**A4:** 请检查：
1. 分词器是否与模型匹配
2. 数据格式是否正确
3. 比较规则是否适合当前任务
4. 生成参数是否合理

### Q5: 如何调整评估批次大小以适应不同的硬件配置？

**A5:** 根据可用NPU内存调整`--eval-batch-size`参数。对于内存较大的NPU，可以适当增加批次大小以提高评估效率；对于内存较小的NPU，应减小批次大小以避免内存溢出。

### Q6: 使用数据并行(DP>1)时，运行脚本后程序卡死怎么办？

**A6:** 当数据并行度(DP)大于1时，需要在命令行中添加`--broadcast`参数，否则可能导致程序卡死。该参数用于确保评估数据在所有进程间正确广播，保证评估过程的正常进行。

示例：
```bash
torchrun $DISTRIBUTED_ARGS evaluate_for_sft.py \
    --broadcast \
    --eval-data-path ${EVAL_DATA_PATH} \
    [其他参数]
```

## 使用总结

MindSpeed LLM的SFT模型推理评估功能提供了一个灵活、高效的评估框架，支持多种数据格式和对话格式，能够帮助用户快速评估SFT模型的性能。通过合理配置评估参数，可以获得准确的评估结果，并通过详细的错误分析定位模型问题，为模型优化提供指导。

将现有推理脚本改造成SFT评估脚本非常简单，只需替换执行脚本并添加评估相关参数即可。这使得用户可以方便地利用现有的模型配置和并行策略进行SFT模型评估，无需重新编写复杂的推理脚本。

### 核心优势

1. **灵活的数据格式支持**：自动识别多种常见对话数据格式，减少数据预处理工作量
2. **完整的评估流程**：从数据加载、模型推理到结果分析，提供端到端的评估解决方案
3. **详细的错误分析**：输出详细的预测错误信息，帮助定位模型问题
4. **分布式评估支持**：支持多NPU和多节点环境下的高效评估
5. **易于扩展**：支持自定义比较规则和数据格式，适应不同的评估需求

### 最佳实践

1. **参数配置**：确保推理时的参数配置与训练时保持一致，特别是并行切分参数
2. **批次大小调整**：根据硬件配置合理设置评估批次大小，平衡评估效率和内存使用
3. **数据准备**：确保评估数据格式正确，数据质量良好
4. **结果分析**：充分利用详细的错误分析信息，指导模型优化
5. **分布式使用**：在多NPU环境下使用时，记得添加`--broadcast`参数

通过遵循这些最佳实践，可以充分发挥MindSpeed LLM SFT评估功能的优势，获得准确可靠的评估结果。
