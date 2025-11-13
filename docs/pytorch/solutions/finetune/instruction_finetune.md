# 大模型指令微调

## 使用场景

大模型虽然在预训练后拥有强大的语言能力，但它们往往缺乏任务意识或交互能力。通过在多任务、多样化的数据集上进行微调，指令微调（Instruction Fine-Tuning）使得模型在应对各种任务时更加灵活并更具泛化能力。  
指令微调首先收集多个不同任务的数据集，并将每个任务转换为指令形式的输入，帮助大模型在多样化任务上提升泛化能力。具体来说，就是通过“指令-输出”的配对样本，使用有监督学习的方式，让模型学会执行具体任务。指令微调的基本原理如下：

![指令微调原理](../../../../sources/images/instruction_finetune/General_pipline_of_instruction_tuning.png)  
*图源：[Instruction Tuning for Large Language Models: A Survey.](https://arxiv.org/pdf/2308.10792v5)*

根据指令微调的数据格式，可分为以下三种常见的使用场景：

### 单样本微调

每条数据为一个独立的任务样本，包括指令和目标回复。适合用于问答、翻译、单轮任务等。

- 数据格式示例：

    ```json
    {
        "instruction": "请将下面的句子翻译成英文：我爱自然语言处理。",
        "response": "I love natural language processing."
    }
    ```

### [多样本pack微调](./multi_sample_pack_finetune.md)

为了提高训练效率，将多个样本拼接打包成一个长序列，减少填充，提高显存利用率。

- 原始样本：

    ```json
    [
        {
            "instruction": "请将以下句子翻译成英文：我们正在开发一款新的人工智能助手。",
            "response": "We are developing a new AI assistant."
        },
        {
            "instruction": "请列出三个可再生能源的例子。",
            "response": "太阳能、风能和水能是三种常见的可再生能源。"
        }
    ]
    ```

- 拼接后输入示例：

    ```json
    <bos> Instruction: 请将以下句子翻译成英文：我们正在开发一款新的人工智能助手。
    Response: We are developing a new AI assistant. <eos>
    Instruction: 请列出三个可再生能源的例子。
    Response: 太阳能、风能和水能是三种常见的可再生能源。 <eos>
    ```

### [多轮对话微调](./multi-turn_conversation.md)

用于训练模型进行连续对话，保留上下文信息。每条样本包含一轮以上的用户和模型对话。

- 数据示例：

    ```json
    {
        "conversations": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，有什么我可以帮你的吗？"},
            {"role": "user", "content": "什么是强化学习？"},
            {"role": "assistant", "content": "强化学习是一种让智能体通过试错方式学习策略的方法。"}
        ]
    }
    ```

- 构造的输入示例：

    ```json
    <user>: 你好
    <assistant>: 你好，有什么我可以帮你的吗？
    <user>: 什么是强化学习？
    <assistant>: 强化学习是一种让智能体通过试错方式学习策略的方法。
    ```
## 使用方法

本章节介绍如何基于预训练语言模型，使用单样本格式数据完成指令微调任务，其他数据格式请参考[多样本pack微调](./multi_sample_pack_finetune.md)和[多轮对话微调](./multi-turn_conversation.md)。该使用方法是基于Qwen3-8B模型和单台`Atlas 900 A2 POD`（1x8集群）进行全参数微调。大模型微调主要包含以下流程：  

![微调流程图](../../../../sources/images/instruction_finetune/process_of_instruction_tuning.png)

第一步，请参考[安装指导](../../install_guide.md)，完成环境安装。请注意由于Qwen3要求使用`transformers>=4.51.0`，因此Python需使用3.9及以上版本。请在训练开始前配置好昇腾NPU套件相关的环境变量，如下所示：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh # 以具体的ascend-toolkit路径为主
source /usr/local/Ascend/nnal/atb/set_env.sh # 以具体的nnal路径为主
```
第二步，准备好模型权重和微调数据集。模型权重下载请参考[Dense模型](../../models/dense_model.md)、[MOE模型](../../models/moe_model.md)和[SSM模型](../../models/ssm_model.md)文档中对应模型的下载链接。以[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/tree/main)模型为例，完整的模型文件夹应该包括以下内容：

```shell
.
├── README.md                      # 模型说明文档
├── config.json                   # 模型结构配置文件
├── generation_config.json       # 文本生成时的配置
├── merges.txt                   # tokenizer的合并规则文件
├── model-00001-of-00005.safetensors  # 模型权重文件第1部分（共5部分）
├── model-00002-of-00005.safetensors  # 模型权重文件第2部分
├── model-00003-of-00005.safetensors  # 模型权重文件第3部分
├── model-00004-of-00005.safetensors  # 模型权重文件第4部分
├── model-00005-of-00005.safetensors  # 模型权重文件第5部分
├── model.safetensors.index.json      # 权重分片索引文件，指示各个权重参数对应的文件
├── tokenizer.json               # Hugging Face格式的tokenizer
├── tokenizer_config.json       # tokenizer相关配置
└── vocab.json                  # 模型词表文件
```

数据集准备请参考[Alpaca风格数据集](datasets/alpaca_dataset.md)、[ShareGPT风格数据集](datasets/sharegpt_dataset.md)和[Pairwise风格数据集](datasets/pairwise_dataset.md)的相关内容，目前已支持`.parquet`, `.csv`, `.json`, `.jsonl`, `.txt`, `.arrow`的格式的数据文件。

第三步，进行[权重转换](../checkpoint/checkpoint_convert.md)，即将模型原始的HF权重转换成Megatron权重，以Qwen3-8B模型在TP1PP4切分为例，详细配置请参考[Qwen3-8B权重转换脚本](../../../../examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh)。需要修改脚本中的以下参数配置：

```shell
--load-dir ./model_from_hf/qwen3_hf/ # HF权重路径
--save-dir ./model_weights/qwen3_mcore/ # Megatron权重保存路径
--tokenizer-model ./model_from_hf/qwen3_hf/tokenizer.json # HF的tokenizer路径
--target-tensor-parallel-size 1 # TP切分大小
--target-pipeline-parallel-size 4 # PP切分大小
```

确认路径无误后运行权重转换脚本：

```shell
bash examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh
```

第四步，进行数据预处理。因为不同数据集使用的处理方法不同，请先确认好预处理的数据格式，详细使用说明跳转到以下文档：

- [Alpaca微调数据使用文档](datasets/alpaca_dataset.md)
- [ShareGPT微调数据使用文档](datasets/sharegpt_dataset.md) 
- [Pairwise微调数据使用文档](datasets/pairwise_dataset.md)

接下来将以Alpaca数据集为例执行数据预处理，详细配置请参考[Qwen3数据预处理脚本](../../../../examples/mcore/qwen3/data_convert_qwen3_instruction.sh)。需要修改脚本内的路径：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh # 修改为真实的ascend-toolkit路径
......
--input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet # 原始数据集路径 
--tokenizer-name-or-path ./model_from_hf/qwen3_hf # HF的tokenizer路径
--output-prefix ./finetune_dataset/alpaca  # 保存路径
......
```
数据预处理相关参数说明：

- `handler-name`：指定数据集的处理类，常用的有`AlpacaStyleInstructionHandler`，`SharegptStyleInstructionHandler`，`AlpacaStylePairwiseHandler`等。
- `tokenizer-type`：指定处理数据的tokenizer，常用`PretrainedFromHF`。
- `workers`：处理数据集的并行数。
- `log-interval`：处理进度更新的间隔步数。
- `enable-thinking`：快慢思考模板开关，可设定为`[true,false,none]`，默认值是`none`。开启后，会在数据集的模型回复中添加`<think>`和`</think>`，并参与到loss计算，所有数据被当成慢思考数据；当关闭后，空的CoT标志将被添加到数据集的用户输入中，不参与loss计算，所有数据被当成快思考数据；设置为`none`时适合原始数据集是混合快慢思考数据的场景。**目前只支持Qwen3系列模型**。
- `prompt-type`：用于指定模型模板，能够让base模型微调后能具备更好的对话能力。`prompt-type`的可选项可以在[`templates`](../../../../configs/finetune/templates.json)文件内查看。

相关参数设置完毕后，运行数据预处理脚本：

```shell
bash examples/mcore/qwen3/data_convert_qwen3_instruction.sh
```

第五步，配置模型微调脚本，详细的参数配置请参考[Qwen3-8b微调脚本](../../../../examples/mcore/qwen3/tune_qwen3_8b_4K_full_ptd.sh)。脚本中的环境变量配置见[环境变量说明](../../features/environment_variable.md)。模型微调可在单机或者多机上运行，以下是单机运行的相关参数配置说明：

```shell
# 单机配置
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1  
NODE_RANK=0  
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
```

环境变量确认无误后，需要修改相关路径参数和模型切分配置：

```shell
CKPT_LOAD_DIR="your model ckpt path"  # 指向权重转换后保存的路径
CKPT_SAVE_DIR="your model save ckpt path" # 指向用户指定的微调后权重保存路径
DATA_PATH="your data path" # 指定处理后的数据路径
TOKENIZER_PATH="your tokenizer path" # 指定模型的tokenizer路径
TP=1 # 模型权重转换的tp大小，在本例中是1
PP=4 # 模型权重转换的pp大小，在本例中是4
```

微调脚本相关参数说明
- `DATA_PATH`：数据集路径。请注意实际数据预处理生成文件末尾会增加`_input_ids_document`等后缀，该参数填写到数据集的前缀即可。例如实际的数据集相对路径是`./finetune_dataset/alpaca/alpaca_packed_input_ids_document.bin`等，那么只需要填`./finetune_dataset/alpaca/alpaca`即可。
- `is-instruction-dataset`：用于指定微调过程中采用指令微调数据集，以确保模型依据特定指令数据进行微调。
- `no-pad-to-seq-lengths`：在不同的mini-batch间支持以动态的序列长度进行微调，默认padding到`8`的整数倍，可以通过`pad-to-multiple-of`参数来修改padding的倍数。假设微调时指定`--seq-length`序列长度为1024，开启`--no-pad-to-seq-lengths`后，序列长度会padding到真实数据长度的8整数倍。如下图所示：  
![variable-seq-lengths图示](../../../../sources/images/instruction_finetune/variable_seq_lengths.png)

第六步，启动微调脚本。参数配置完毕后，如果是单机运行场景，只需要在一台机器上启动微调脚本：

```shell
bash examples/mcore/qwen3/tune_qwen3_8b_4K_full_ptd.sh
```

如果是多机运行，则需要在单机的脚本上修改以下参数：

```shell
# 多机配置 
# 根据分布式集群实际情况配置分布式参数
NPUS_PER_NODE=8  # 每个节点的卡数
MASTER_ADDR="your master node IP"  # 都需要修改为主节点的IP地址（不能为localhost）
MASTER_PORT=6000
NNODES=2  # 集群里的节点数，以实际情况填写,
NODE_RANK="current node id"  # 当前节点的RANK，多个节点不能重复，主节点为0, 其他节点可以是1,2..
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
```

最后确保每台机器上的模型路径和数据集路径等无误后，在多个终端上同时启动微调脚本即可开始训练。

第七步，进行模型验证。完成微调后，需要进一步验证模型是否具备了预期的输出能力。我们提供了简单的模型生成脚本，只需要加载微调后的模型权重，便可观察模型在不同生成参数配置下的回复，详细配置请参考[Qwen3-8B推理脚本](../../../../examples/mcore/qwen3/generate_qwen3_8b_ptd.sh)。需要在脚本中修改以下参数：

```shell
CKPT_DIR="your model save ckpt path" # 指向微调后权重的保存路径
TOKENIZER_PATH="your tokenizer path" # 指向模型tokenizer的路径
```

然后运行推理脚本：

```shell
bash examples/mcore/qwen3/generate_qwen3_8b_ptd.sh
```

此外，若想要验证模型在不同任务下的表现，请参考[模型评估](../evaluation/evaluation_guide.md)来更全面评估微调效果。

## 使用约束

- 序列类型的不同，其对应的微调脚本和数据预处理方式也不同，这里以Qwen3的指令微调举例：

    | 序列长度   | 特点                             | 训练脚本 | 数据预处理方式   |
    |--------|--------------------------------|----|---------------------------------------------------------|
    | 固定长度序列 | 性能低，不推荐使用 |  训练时不使用`--no-pad-to-seq-lengths`参数    | 使用默认预处理脚本，如`data_convert_qwen3_instruction.sh`         |
    | 动态长度序列 | sample吞吐高  |   训练脚本需要使用`--no-pad-to-seq-lengths`参数   | 使用默认预处理脚本，如`data_convert_qwen3_instruction.sh`         |
    | 样本拼接序列 | token吞吐高，支持长序列并行 |   训练脚本需要使用`--reset-attention-mask`参数，不启用`--no-pad-to-seq-lengths`   | 使用pack配置的预处理脚本，详见[多样本pack微调](./multi_sample_pack_finetune.md) |

    请根据自己的使用场景，灵活选择对应类型的指令微调训练脚本和数据预处理脚本。

- 当前微调数据预处理使用的默认模板已和LLaMA Factory 0.8.2对齐，如果需要与该版本之后的版本对齐，请修改[templates](../../../../configs/finetune/templates.json)文件中相应模板的`template_class`的键值为`LFDefaultTemplate`。例如使用`qwen`模板时，在[templates](../../../../configs/finetune/templates.json)中找到`"name": "qwen"`的字典，然后加上新的键值对`"template_class": "LFDefaultTemplate"`即可。