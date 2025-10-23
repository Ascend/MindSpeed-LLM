# ShareGPT 数据集

## 常用的ShareGPT数据集

常用的对话指令微调数据集有：

- [Roleplay多轮对话：ShareGPT数据集](https://huggingface.co/datasets/shibing624/roleplay-zh-sharegpt-gpt4-data)
- [Chain-of-thought: ShareGPT数据集](https://huggingface.co/datasets/isaiahbjork/chain-of-thought-sharegpt)
- [Capybara: ShareGPT数据集](https://huggingface.co/datasets/Undi95/Capybara-ShareGPT)

## ShareGPT风格数据集下载

`ShareGPT` 风格微调数据集下载可以基于网页直接下载，也可以基于命令行下载，比如：

```shell
cd dataset/
wget https://huggingface.co/datasets/shibing624/roleplay-zh-sharegpt-gpt4-data/resolve/main/sharegpt_formatted_data-evol-gpt4.jsonl
cd ..
```

## ShareGPT风格数据集处理方法

`ShareGPT` 格式支持更多的角色种类，例如 `human、gpt、observation、function_call`等等。它们构成一个对象列表呈现在`conversations`列中。

`ShareGPT`风格示例：

```
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "人类指令"
      },
      {
        "from": "function_call",
        "value": "工具参数"
      },
      {
        "from": "observation",
        "value": "工具结果"
      },
      {
        "from": "gpt",
        "value": "模型回答"
      }
    ],
    "system": "系统提示词（选填）",
    "tools": "工具描述（选填）"
  }
]
```

`ShareGPT`格式数据预处理脚本：

```shell
# 请按照您的真实环境 source set_env.sh 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./finetune_dataset

python ./preprocess_data.py \
    --input ./dataset/sharegpt_formatted_data-evol-gpt4.jsonl \
    --tokenizer-name-or-path ./model_from_hf/llama-2-7b-hf/ \
    --output-prefix ./finetune_dataset/sharegpt \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF \
    --handler-name SharegptStyleInstructionHandler \
    --prompt-type llama2  # <-- 需要填入模型模板
    # --map-keys '{"messages":"conversations", "tags":{"role_tag": "from","content_tag": "value","user_tag": "human","assistant_tag": "gpt","system_tag": "system", "observation_tag":"observation", "function_tag":"function_call"}}' # 默认值，可不传
```

【--prompt-type】

用于指定模型模板，能够让base模型微调后能具备更好的对话能力。`prompt-type`的可选项可以在[templates](../../../../../configs/finetune/templates.json)文件内查看。

【--map-keys】

`--map-keys`参数用于配置字段映射来使用数据集。默认值为

`'{"messages":"conversations", "tags":{"role_tag": "from","content_tag": "value","user_tag": "human","assistant_tag": "gpt","system_tag": "system", "observation_tag":"observation", "function_tag":"function_call"}}'`

其中key值`"messages"、"tags"`代表数据集列映射后的属性，在代码中是固定的，不应改变。value值中`"conversations"`对应数据集的列名、`"from"`对应角色标志、`"human"、"gpt"、"system"、"observation"、"function_call"`对应角色种类、`"value"`对应具体内容标志。


以OpenAI格式为例，OpenAI 格式是 sharegpt 格式的一种特殊情况，其中第一条消息可能是系统提示词。

OpenAI格式示例：

```
[
  {
    "messages": [
      {
        "role": "system",
        "content": "系统提示词（选填）"
      },
      {
        "role": "user",
        "content": "人类指令"
      },
      {
        "role": "assistant",
        "content": "模型回答"
      }
    ]
  }
]
```

OpenAI格式数据预处理脚本：

```shell
# 请按照您的真实环境 source set_env.sh 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./finetune_dataset

python ./preprocess_data.py \
    --input ./dataset/xxx.json \
    --tokenizer-name-or-path ./model_from_hf/llama-2-7b-hf/ \
    --output-prefix ./finetune_dataset/openai \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF \
    --handler-name SharegptStyleInstructionHandler \
    --prompt-type llama2 \
    --map-keys '{"messages":"messages", "tags":{"role_tag": "role","content_tag": "content","user_tag": "user","assistant_tag": "assistant","system_tag": "system"}}'
```

【--handler-name】

微调数据预处理`ShareGPT`风格数据集时，应指定为`SharegptStyleInstructionHandler`，并根据`--map-keys`参数提取对应数据的列。

**示例1：**

```
--map-keys '{"messages":"chat"}'
```

则会提取数据集里的`"chat"`列，其中`"tags"`属性包含角色格式和内容格式，做为默认值隐式存在，角色格式可以为：`"from": "human"、"from": "gpt"、"from": "observation"、"from": "function_call"`，内容格式为`"value": "具体内容"`

**示例2：**

```
--map-keys '{"messages":"messages", "tags":{"role_tag": "role","content_tag": "content","user_tag": "user","assistant_tag": "assistant"}}'
```

则会提取数据集里的`"messages"`列，其中角色格式可以为：`"role": "user"、"role": "assistant"`，内容格式为`"content": "具体内容"`

### 启动脚本

MindSpeed-LLM微调数据集处理脚本命名风格及启动方法为：

```shell
# mcore
# 命名及启动：examples/mcore/model_name/data_convert_xxx_instruction.sh
bash examples/mcore/llama2/data_convert_llama2_instruction.sh
```

指令微调数据集处理结果如下：

```shell
./finetune_dataset/openai_packed_attention_mask_document.bin
./finetune_dataset/openai_packed_attention_mask_document.idx
./finetune_dataset/openai_packed_input_ids_document.bin
./finetune_dataset/openai_packed_input_ids_document.idx
./finetune_dataset/openai_packed_labels_document.bin
./finetune_dataset/openai_packed_labels_document.idx
```

微调时，数据集路径输入 `./finetune_dataset/openai` 即可