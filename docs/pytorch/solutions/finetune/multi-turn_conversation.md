# MindSpeed-LLM 多轮对话微调

## 特性介绍

多轮对话的核心思想是让经过训练或微调的模型在与用户交互时，能够记住对话的上下文。这使得模型在不同轮次中保持一致性，从而提供更连贯的回答。通过记忆之前的对话内容，模型能够更好地理解用户的意图，适应用户的需求，并在后续交流中提供更加个性化的响应。

这种特性不仅提升了用户体验，还使得模型在处理复杂问题时更具推理能力。例如，用户可以在多个轮次中逐步提出问题，而模型能够根据之前的对话内容，提供更准确和有针对性的建议或解决方案。此外，多轮对话还可以通过分析用户的反馈和偏好，逐步优化模型的响应策略，使其在长期交互中不断学习和改进。

### 使用说明

可以直接使用[chat](../../../../examples/mcore/gemma2/chat_gemma2_9b_ptd.sh)或者[generate](../../../../examples/mcore/gemma2/generate_gemma2_9b_ptd.sh)脚本，并加载预置权重，与模型进行多轮对话，以达到预期效果。其中预置权重可以是大语言模型的chat权重，base权重或是微调后的权重。

### 效果展示

```text
You >> My sister's birthday is tomorrow and I need a birthday bouquet.
MindSpeed-LLM:
I'd be happy to help you with that! Do you have any specific preferences or theme in mind for the bouquet? Are there any particular flowers or color scheme that you would like to incorporate?

You >> She likes pink roses. The color is pink.
MindSpeed-LLM:
Pink roses it is then! Do you have any other preferences or restrictions, such as budget or flower availability in your area?

You >> Here I go again. Remember why I came to buy flowers yesterday?
MindSpeed-LLM:
Yes, I do remember. You came to buy flowers for your sister's birthday, and you mentioned that she likes pink roses. Is there anything else related to that purchase that I can help you with today?
```
## 开始多轮对话指令微调

有关Alpaca数据集的详细介绍，请参阅[**Alpaca风格数据的说明文档**](datasets/alpaca_dataset.md)。

### 多轮对话数据处理

对使用带有`history`字段的 Alpaca 数据集，并运行如下[脚本](../../../../examples/mcore/gemma2/data_convert_gemma2_instruction.sh)：

```shell
# 请根据 examples/README.md 下 “数据集准备及处理” 章节下载 Alpaca 数据集
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./finetune_dataset

python ./preprocess_data.py \
    --input ./dataset/train-00000-of-00001.parquet \
    --tokenizer-name-or-path ./model_from_hf/gemma2_hf/ \
    --output-prefix ./finetune_dataset/alpaca \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF \
    --handler-name AlpacaStyleInstructionHandler \
    --prompt-type gemma \
    --map-keys '{"history":"histories"}'
    # --map-keys '{"prompt":"instruction","query":"input","response":"output"}' # 默认值，可不传
```

【--map-keys '{"history":"histories"}'】

会提取数据集里的`"instruction"、"input"、"output"、"histories"`列，其中`"instruction"、"input"、"output"`列作为默认值隐式存在。 

### 启动脚本

MindSpeed-LLM微调数据集处理脚本命名风格及启动方法为：

```shell
# Mcore
# 命名及启动：examples/mcore/gemma2/data_convert_gemma2_instruction.sh
bash examples/mcore/gemma2/data_convert_gemma2_instruction.sh
```

指令微调数据集处理结果如下：

```shell
./finetune_dataset/alpaca_packed_attention_mask_document.bin
./finetune_dataset/alpaca_packed_attention_mask_document.idx
./finetune_dataset/alpaca_packed_input_ids_document.bin
./finetune_dataset/alpaca_packed_input_ids_document.idx
./finetune_dataset/alpaca_packed_labels_document.bin
./finetune_dataset/alpaca_packed_labels_document.idx
```

微调时，数据集路径输入 `./finetune_dataset/alpaca` 即可

#### 填写相关路径

`DATA_PATH`：指定数据预处理后的保存路径。

`TOKENIZER_MODEL`：指定模型的分词器路径（例如`tokenizer.model`）。

`CKPT_LOAD_DIR`：指向权重转换后保存的路径。

`CKPT_SAVE_DIR`：指向用户指定的微调后权重保存路径（例如指定保存到`ckpt`文件夹下）。

因此，根据之前的示例，路径应填写如下：

```shell
DATA_PATH="./finetune_dataset/alpaca"
CKPT_SAVE_DIR="./ckpt/"
TOKENIZER_MODEL="./model_from_hf/gemma2_hf/"
CKPT_LOAD_DIR="./model_weights/gemma2-9b-mcore/"
```

**运行多轮对话指令微调脚本** 

```shell
bash examples/mcore/gemma2/chat_gemma2_9b_ptd.sh
```

## 开始多轮对话

使用[对话脚本](../../../../examples/mcore/gemma2/chat_gemma2_9b_ptd.sh)，加载模型分词器和微调过后的权重，进行多轮对话。

填写相关路径

`TOKENIZER_PATH`：指定模型的分词器路径（例如`tokenizer.model`）。

`CHECKPOINT`：指向多轮对话微调后保存权重的路径。

```shell
TOKENIZER_PATH="./model_from_hf/gemma2_hf/"
CHECKPOINT="./ckpt/"
```

**运行多轮对话脚本** 

```shell
bash examples/mcore/gemma2/chat_gemma2_9b_ptd.sh
```

在确认成功加载上微调权重后，在出现`You >> `的提示后，可以进行对话。

## 使用约束

- 当前微调数据预处理使用的默认模板已和LLaMA Factory 0.8.2对齐，如果需要与该版本之后的版本对齐，请在微调数据预处理阶段设置`prompt-type`参数值为`qwen_lf`。