# Alpaca风格数据集

## 常用的Alpaca数据集

常用的对话指令微调数据集有：

- [单轮对话：Alpaca英文数据集](https://huggingface.co/datasets/tatsu-lab/alpaca)
- [单轮对话：Alpaca中文数据集](https://huggingface.co/datasets/llm-wizard/alpaca-gpt4-data-zh/tree/main)
- [多轮对话：AlpacaHistory数据集](https://huggingface.co/datasets/kimnt93/oaast-selected)
- [链式思维 (CoT)：Alpaca数据集](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/Auto-CoT)
- [BELLE：指令微调数据集](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)

## Alpaca风格数据集处理方法

### Alpaca风格数据集下载

Alpaca风格微调数据集下载可以基于网页直接下载，也可以基于命令行下载，比如：

```shell
cd dataset/
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..
```

### Alpaca风格数据集处理

在指令监督微调时，`instruction` 列对应的内容会与 `input` 列对应的内容拼接后作为人类指令，即人类指令为 `instruction\ninput`其中 `\n`为用于连接的换行符。而 `output` 列对应的内容为模型回答。如果指定了history，则会将历史对话内容也加入进来。如果指定system 列，则对应的内容将被作为系统提示词。

```shell
# 请按照您的真实环境 source set_env.sh 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./finetune_dataset

python ./preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./model_from_hf/llama-2-7b-hf/ \
    --output-prefix ./finetune_dataset/alpaca \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF \
    --handler-name AlpacaStyleInstructionHandler \
    --prompt-type llama2  # <-- 需要填入模型模板
    # --map-keys '{"prompt":"instruction","query":"input","response":"output"}' # 默认值，可不传
```

【--input】

可以直接输入到数据集目录或具体文件，如果是目录，则处理全部文件, 支持.parquet/.csv/.json/.jsonl/.txt/.arrow格式， 同一个文件夹下的数据格式需要保持一致 

【--map-keys】

`--map-keys`参数用于配置字段映射来使用数据集。

Alpaca风格示例：

```
[
{
    "instruction": "人类指令（必填）",
    "input": "人类输入（选填）",
    "output": "模型回答（必填）",
    "system": "系统提示词（选填）",
    "history": [
    ["第一轮指令（选填）", "第一轮回答（选填）"],
    ["第二轮指令（选填）", "第二轮回答（选填）"]
    ]
}
]
```

对于上面格式的数据，`--map-keys`参数完整应为

`'{"prompt":"instruction","query":"input","response":"output","system":"system","history":"history"}'`

其中参数的key值`"prompt"、"query"、"response"、"system"、"history"`代表数据集列映射后的属性，在代码中是固定的，不应改变，value值`"instruction"、"input"、"output"、"system"、"history"`对应数据集的列名。

考虑到alpaca数据集大部分都是`["instruction", "input", "output"]`型格式，因此我们为key值`["prompt", "query", "response"]`设置了默认值。因此上面格式`--map-keys`参数可简略为`'{"system": "system","history": "history"}'`

若数据集中无`system`与`history`列，则`--map-keys`可省略。


【--prompt-type】

用于指定模型模板，能够让base模型微调后能具备更好的对话能力。`prompt-type`的可选项可以在[templates](../../../../../mindspeed_llm/tasks/preprocess/templates.py)文件内查看。

【--handler-name】

微调数据预处理Alpaca风格数据集时，应指定为`AlpacaStyleInstructionHandler`，根据`--map-keys`参数提取对应数据的列。

**示例1：**

```
--map-keys '{"prompt":"notice","query":"question","response":"answer","system":"system_test","history":"histories"}'
```

则会提取数据集里的`"notice"、"question"、"answer"、"system_test"、"histories"`列。

**示例2：**

```
--map-keys '{"history":"histories"}'
```

则会提取数据集里的`"instruction"、"input"、"output"、"histories"`列，其中`"instruction"、"input"、"output"`列作为默认值隐式存在。 

### 启动脚本

MindSpeed-LLM微调数据集处理脚本命名风格及启动方法为：

```shell
# 命名及启动：examples/mcore/model_name/data_convert_xxx_instruction.sh
bash examples/mcore/llama2/data_convert_llama2_instruction.sh
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