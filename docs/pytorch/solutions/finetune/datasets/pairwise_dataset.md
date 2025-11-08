# Pairwise 数据集处理

在大模型后训练任务，比如RLHF中，通常需要用到基于人类偏好反馈的数据集，这些语料包含人类对同一个问题的不同回答或不同表述的偏好或评价。在DPO任务中，常用的数据集为pairwise格式数据集，顾名思义，pairwise数据集为配对数据集，即对同一个问题，包含两个答案，一个好的（chosen），一个坏的（rejected）。比如[orca_dpo_pairs](https://huggingface.co/datasets/Intel/orca_dpo_pairs)数据集，包含四个字段：system、question、chosen、rejected。

![](../../../../../sources/images/dpo/orca_rlhf.png)

pairwise配对数据集样本示例：

```
{"system": "You are an AI assistant. You will be given a task. You must generate a detailed and long answer.", 
 "question": "Generate an approximately fifteen-word sentence that describes all this data: Midsummer House eatType restaurant; Midsummer House food Chinese; Midsummer House priceRange moderate; Midsummer House customer rating 3 out of 5; Midsummer House near All Bar One", 
 "chosen": "Midsummer House is a moderately priced Chinese restaurant with a 3/5 customer rating, located near All Bar One.", 
 "rejected": " Sure! Here's a sentence that describes all the data you provided:\n\n\"Midsummer House is a moderately priced Chinese restaurant with a customer rating of 3 out of 5, located near All Bar One, offering a variety of delicious dishes.\""
 }
```

## 常用的Pairwise数据集

常用Pairwise数据集有：

- [orca_dpo_pairs](https://huggingface.co/datasets/Intel/orca_dpo_pairs)
- [orpo-dpo-mix-40k](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k)
- [alpaca_messages_2k_dpo_test](https://huggingface.co/datasets/fozziethebeat/alpaca_messages_2k_dpo_test)

## Pairwise数据集下载

`Pairwise` 数据集下载可以基于网页直接下载，也可以基于命令行下载，比如：

```shell
cd dataset/
wget https://huggingface.co/datasets/Intel/orca_dpo_pairs/resolve/main/orca_rlhf.jsonl
cd ..
```

## Pairwise数据集处理方法

Pairwise格式数据预处理脚本：

```shell
# 请按照您的真实环境 source set_env.sh 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./pairwise_dataset

python ./preprocess_data.py \
    --input ./dataset/orca_rlhf.jsonl \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-not-use-fast \
    --tokenizer-name-or-path ./model_from_hf/Meta-Llama-3-8B-Instruct/ \
    --output-prefix ./pairwise_dataset/orca_rlhf_llama3 \
    --workers 4 \
    --log-interval 1000 \
    --handler-name AlpacaStylePairwiseHandler \
    --prompt-type llama3 \
    --map-keys '{"prompt":"question", "query":"", "system":"system"}'
```

【--prompt-type】

用于指定模型模板，能够让base模型微调后能具备更好的对话能力。`prompt-type`的可选项可以在[templates](../../../../../mindspeed_llm/tasks/preprocess/templates.py)文件内查看。

【--handler-name】

Pairwise数据预处理时，可指定为`AlpacaStylePairwiseHandler`或`SharegptStylePairwiseHandler`，分别处理alpaca风格和sharegpt风格的配对数据集，并根据`--map-keys`参数提取对应数据的列。alpaca风格数据集和sharegpt风格数据集及二者对应map-keys参数说明分别参见[alpaca风格数据集](alpaca_dataset.md)和[sharegpt风格数据集](sharegpt_dataset.md)处理说明。

### 启动脚本

MindSpeed-LLM微调数据集处理脚本命名风格及启动方法为：

```shell
# Mcore
# 命名及启动：examples/mcore/model_name/data_convert_xxx_pairwise.sh
bash examples/mcore/llama3/data_convert_llama3_pairwise.sh
```

指令微调数据集处理结果如下：

```shell
./pairwise_dataset/orca_rlhf_llama3_packed_chosen_input_ids_document.bin
./pairwise_dataset/orca_rlhf_llama3_packed_chosen_input_ids_document.idx
./pairwise_dataset/orca_rlhf_llama3_packed_chosen_labels_document.bin
./pairwise_dataset/orca_rlhf_llama3_packed_chosen_labels_document.idx
./pairwise_dataset/orca_rlhf_llama3_packed_rejected_input_ids_document.bin
./pairwise_dataset/orca_rlhf_llama3_packed_rejected_input_ids_document.idx
./pairwise_dataset/orca_rlhf_llama3_packed_rejected_labels_document.bin
./pairwise_dataset/orca_rlhf_llama3_packed_rejected_labels_document.idx
```

进行DPO训练任务时，数据集路径输入 `./pairwise_dataset/orca_rlhf_llama3` 即可，同时须设置`--is-pairwise-dataset`参数