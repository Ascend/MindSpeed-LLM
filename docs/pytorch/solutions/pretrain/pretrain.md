# 大模型分布式预训练

## 使用场景

大模型预训练（Pretraining）是语言模型发展的核心步骤，目标是让模型通过大规模无标签语料学习语言规律与世界知识。预训练过程更关注语言建模本身，而非具体任务执行。以GPT类模型为例，它是一种典型的自回归语言模型，其核心思想是基于历史上下文预测下一个标记。预训练的过程就是通过反复优化这种预测能力，模型逐渐学会如何理解语境、保持句子连贯性，并掌握更高层次的语言结构，为多种下游任务提供通用的语言表示能力。  
预训练数据通常为纯文本格式，无任务导向，例如：

```json
{"text": "今天是个好天气，我们一起去爬山。"}
{"text": "深度学习正在改变世界。"}
{"text": "AI的出现推动了人类社会的发展。"}
```

## 使用方法

本章节以Qwen3-8B模型为例，介绍了预训练启动方法。大模型分布式预训练主要包含以下流程:

![预训练流程图](../../../../sources/images/pretrain/process_of_pretraining.png)

注意：
- 数据预处理时如果需要使用数据集pack模式，请参考[大模型预训练pack模式](./pretrain_eod.md)。
- 预训练时可以不加载初始权重，此时模型权重采用随机初始化。如果需要加载权重，则需提前进行权重转换，具体请参考[权重转换](../checkpoint_convert.md)章节

第一步，请在训练开始前参考[安装指导](../../install_guide.md)，完成环境安装。请注意由于Qwen3要求使用`transformers>=4.51.0`，因此Python需使用3.9及以上版本。环境搭建完成后，确保在预训练开始前已经配置好昇腾NPU套件相关的环境变量，如下所示：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh # 以具体的ascend-toolkit路径为主
source /usr/local/Ascend/nnal/atb/set_env.sh # 以具体的nnal路径为主
```

第二步，数据预处理。首先准备好原始数据集，常见的预训练数据集有：
- [Alpaca数据集](https://huggingface.co/datasets/tatsu-lab/alpaca)
- [Enwiki数据集](https://huggingface.co/datasets/lsb/enwiki20230101)
- [C4数据集](https://huggingface.co/datasets/allenai/c4)
- [ChineseWebText](https://huggingface.co/datasets/CASIA-LM/ChineseWebText)

接下来以[Enwiki数据集](https://huggingface.co/datasets/lsb/enwiki20230101)为例执行数据预处理，详细的脚本配置可参考[Qwen3预训练数据处理脚本](../../../../examples/mcore/qwen3/data_convert_qwen3_pretrain.sh)，需要修改脚本中的以下路径：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh # 修改为真实的ascend-toolkit路径
......
--input ./dataset/train-00000-of-00042-d964455e17e96d5a.parquet # 原始数据集路径 
--tokenizer-name-or-path ./model_from_hf/qwen3_hf # HF的tokenizer路径
--output-prefix ./finetune_dataset/alpaca  # 保存路径
......
```

数据预处理相关参数说明:

- `input`：可以直接输入到数据集目录或具体文件，如果是目录，则处理全部文件, 支持`.parquet`，`.csv`，`.json`，`.jsonl`，`.txt`，`.arrow`格式， 同一个文件夹下的数据格式需要保持一致。
- `handler-name`：当前预训练默认使用 `GeneralPretrainHandler`，支持的是预训练数据风格，提取数据的`text`列，格式如下：

    ```shell
    [
        {"text": "document"},
        {"other keys": "optional content"}
    ]
    ```
- `json-keys`：从文件中提取的列名列表，默认为 `text`，可以为 `text`, `input`, `title` 等多个输入，结合具体需求及数据集内容使用，如：

    ```shell
    --json-keys text input output
    ```
- `n-subs`：数据预处理并行加速参数。当需要预处理的数据集比较大时，可以通过并行处理进行加速，方法为设置参数`--n-subs`，通过该参数设置并行处理数量。在数据预处理过程会将原始数据集切分为`n-subs`个子集，对子集进行并行处理，然后合并，从而实现加速。建议预处理数据集超过GB级别时加上该参数。

相关参数设置完毕后，运行数据预处理脚本：

```shell
bash examples/mcore/qwen3/data_convert_qwen3_pretrain.sh
```

第三步，配置模型预训练脚本，详细的参数配置请参考[Qwen3-8B预训练脚本](../../../../examples/mcore/qwen3/pretrain_qwen3_8b_4K_ptd.sh)。脚本中的环境变量配置见[环境变量说明](../../features/environment_variable.md)。单机运行的配置说明如下：

```shell
NPUS_PER_NODE=8 # 节点的卡数
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1  
NODE_RANK=0  
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
```

环境变量确认无误后，需要在脚本中修改相关路径参数和模型切分配置：

```shell
# 注意：提供的路径需要加双引号
CKPT_SAVE_DIR="your model save ckpt path" # 训练完成后的权重保存路径
DATA_PATH="your data path" # 数据集路径，填入数据预处理时保存的数据路径
TOKENIZER_PATH="your tokenizer path" # 词表路径，填入下载的开源权重词表路径
CKPT_LOAD_DIR="your model ckpt path" # 权重加载路径，填入权重转换时保存的权重路径

TP=1 # 模型权重转换的tp大小，在本例中是1
PP=4 # 模型权重转换的pp大小，在本例中是4
```

脚本内的相关参数说明:

- `DATA_PATH`：数据集路径。请注意实际数据预处理生成文件末尾会增加`_text_document`，该参数填写到数据集的文件前缀即可。例如实际的数据集相对路径是`./finetune_dataset/alpaca/alpaca_text_document.bin`等，那么只需要填`./finetune_dataset/alpaca/alpaca_text_document`即可。
- `CKPT_LOAD_DIR`: 权重加载路径。预训练时可以选择随机初始化模型权重，此时该参数不用配置，同时需要注释掉预训练脚本中的`--load ${CKPT_LOAD_DIR} \`代码行。
- `tokenizer-type`：参数值为PretrainedFromHF时， 词表路径仅需要填到模型文件夹即可，不需要到tokenizer.model文件；参数值不为PretrainedFromHF时，例如Qwen3Tokenizer，需要指定到tokenizer.model文件。示例如下

    ```shell 
    # tokenizer-type为PretrainedFromHF
    TOKENIZER_PATH="./model_from_hf/Qwen3-8B/"
    --tokenizer-name-or-path ${TOKENIZER_PATH}

    
    # tokenizer-type不为PretrainedFromHF
    TOKENIZER_MODEL="./model_from_hf/Qwen3-8B/tokenizer.model"
    --tokenizer-model ${TOKENIZER_MODEL} \
    ```

第四步，预训练脚本配置完毕后，运行脚本启动预训练。

```shell
bash examples/mcore/qwen3/pretrain_qwen3_8b_4K_ptd.sh
```

如果是多机运行，那么需要在[Qwen3-8B预训练脚本](../../../../examples/mcore/qwen3/pretrain_qwen3_8b_4K_ptd.sh)中修改以下参数：

```shell
# 根据分布式集群实际情况配置分布式参数
NPUS_PER_NODE=8  # 每个节点的卡数
MASTER_ADDR="your master node IP"  # 都需要修改为主节点的IP地址（不能为localhost）
MASTER_PORT=6000
NNODES=2  # 集群里的节点数，以实际情况填写,
NODE_RANK="current node id"  # 当前节点的RANK，多个节点不能重复，主节点为0, 其他节点可以是1,2..
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
```

最后确保每台机器上的模型路径和数据集路径等无误后，在多个终端上同时启动预训练脚本即可开始训练。如果使用多机训练，且没有设置数据共享，需要在训练启动脚本中增加`no-shared-storage`参数。设置此参数之后将会根据布式参数判断非主节点是否需要load数据，并检查相应缓存和生成数据。

## 使用约束

- 如需存储日志到脚本文件中，请在运行路径目录下创建`logs`文件夹