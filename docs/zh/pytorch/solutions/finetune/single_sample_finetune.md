# 单样本微调

## 使用场景

单样本微调是最基础、最通用的指令微调形式。每条样本由独立的“指令”和“目标回复”构成，不依赖任何上下文信息。该模式适用于**单轮、无历史依赖**的任务，例如：

- 问答（知识问答、常识推理）
- 文本翻译
- 文本摘要与改写
- 情感分析、意图分类
- 代码生成、数学计算

其特点是数据构造简单、任务边界清晰，便于大规模扩充指令数据，是训练模型掌握基本任务能力的主要方式。

## 使用说明

本章节介绍如何基于预训练语言模型，使用单样本格式数据完成指令微调任务，其他数据格式请参考[多样本pack微调](./multi_sample_pack_finetune.md)和[多轮对话微调](./multi-turn_conversation.md)。该使用方法是基于Qwen3-8B模型和单台`Atlas 900 A2 POD`（1x8集群）进行全参数微调。大模型微调主要包含以下流程：  

**图 1**  单样本微调流程图  
![微调流程图](../../figures/instruction_finetune/process_of_instruction_tuning.png)

1. 环境搭建  
    启动微调前请参考[安装指导](../../install_guide.md)完成环境安装，并确保已完成昇腾NPU套件相关的环境变量配置，如下所示：

    ```shell
    source /usr/local/Ascend/cann/set_env.sh # 修改为实际安装的Toolkit包路径
    source /usr/local/Ascend/nnal/atb/set_env.sh # 修改为实际安装的nnal包路径
    ```

2. 模型和数据集准备  
    - 模型准备  
        模型权重下载请参考[Dense模型](../../models/dense_model.md)、[MOE模型](../../models/moe_model.md)和[SSM模型](../../models/ssm_model.md)文档中对应模型的下载链接。以[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/tree/main)模型为例，完整的模型文件夹应该包括以下内容：

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

    - 数据集准备  
        数据集准备请参考[Alpaca风格数据集](datasets/alpaca_dataset.md)、[ShareGPT风格数据集](datasets/sharegpt_dataset.md)和[Pairwise风格数据集](datasets/pairwise_dataset.md)的相关内容，目前已支持`.parquet`, `.csv`, `.json`, `.jsonl`, `.txt`, `.arrow`的格式的数据文件。

3. 模型权重转换  
    请参考[权重转换v1](../checkpoint/checkpoint_convert.md)和[权重转换v2](../checkpoint/checkpoint_convert_v2.md)，即将模型原始的HF权重转换成Megatron权重，以Qwen3-8B模型在TP1PP4切分为例，详细配置请参考[Qwen3-8B权重转换脚本](../../../../../examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh)。

    首先需要修改脚本中的以下参数配置：

    ```bash
    --load-dir ./model_from_hf/qwen3_hf/ # HF权重路径
    --save-dir ./model_weights/qwen3_mcore/ # Megatron权重保存路径
    --tokenizer-model ./model_from_hf/qwen3_hf/tokenizer.json # HF的tokenizer路径
    --target-tensor-parallel-size 1 # TP切分大小
    --target-pipeline-parallel-size 4 # PP切分大小
    ```

    然后，确认路径无误后运行权重转换脚本：

    ```shell
    bash examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh
    ```

4. 数据预处理  
    因为不同数据集使用的处理方法不同，请先确认好预处理的数据格式，详细使用说明跳转到以下文档：

    - [Alpaca微调数据使用文档](datasets/alpaca_dataset.md)
    - [ShareGPT微调数据使用文档](datasets/sharegpt_dataset.md) 
    - [Pairwise微调数据使用文档](datasets/pairwise_dataset.md)

    接下来将以Alpaca数据集为例执行数据预处理，详细配置请参考[Qwen3数据预处理脚本](../../../../../examples/mcore/qwen3/data_convert_qwen3_instruction.sh)。需要修改脚本内的路径：

    ```shell
    source /usr/local/Ascend/cann/set_env.sh # 修改为实际安装的Toolkit包路径
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
    - `prompt-type`：用于指定模型模板，能够让base模型微调后能具备更好的对话能力。`prompt-type`的可选项可以在[`templates`](../../../../../configs/finetune/templates.json)文件内查看。

    相关参数设置完毕后，运行数据预处理脚本：

    ```shell
    bash examples/mcore/qwen3/data_convert_qwen3_instruction.sh
    ```

5. 配置单机或多机微调脚本  
    详细的参数配置请参考[Qwen3-8b微调脚本](../../../../../examples/mcore/qwen3/tune_qwen3_8b_4K_full_ptd.sh)。脚本中的环境变量配置见[环境变量说明](../../features/environment_variable.md)。

    环境变量确认无误后，需要在脚本中修改节点相关配置，单机和多机配置如下：

    - 单机配置

        ```bash
        NPUS_PER_NODE=8 # 单节点的卡数
        MASTER_ADDR=localhost
        MASTER_PORT=6000
        NNODES=1  
        NODE_RANK=0  
        WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
        ```

    - 多机配置

        ```bash
        # 根据分布式集群实际情况配置分布式参数
        NPUS_PER_NODE=8  # 每个节点的卡数
        MASTER_ADDR="your master node IP"  # 都需要修改为主节点的IP地址（不能为localhost）
        MASTER_PORT=6000
        NNODES=2  # 集群里的节点数，以实际情况填写,
        NODE_RANK="current node id"  # 当前节点的RANK，多个节点不能重复，主节点为0, 其他节点可以是1,2..
        WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
        ```

    然后需要在脚本中修改相关路径参数和模型切分配置：

    ```bash
    CKPT_LOAD_DIR="your model ckpt path"  # 指向权重转换后保存的路径
    CKPT_SAVE_DIR="your model save ckpt path" # 指向用户指定的微调后权重保存路径
    DATA_PATH="your data path" # 指定处理后的数据路径
    TOKENIZER_PATH="your tokenizer path" # 指定模型的tokenizer路径
    TP=1 # 模型权重转换的tp大小，在本例中是1
    PP=4 # 模型权重转换的pp大小，在本例中是4
    ```

    微调脚本相关参数说明：

    - `DATA_PATH`：数据集路径。请注意实际数据预处理生成文件末尾会增加`_input_ids_document`等后缀，该参数填写到数据集的前缀即可。例如实际的数据集相对路径是`./finetune_dataset/alpaca/alpaca_packed_input_ids_document.bin`等，那么只需要填`./finetune_dataset/alpaca/alpaca`即可。
    - `is-instruction-dataset`：用于指定微调过程中采用指令微调数据集，以确保模型依据特定指令数据进行微调。
    - `no-pad-to-seq-lengths`：在不同的mini-batch间支持以动态的序列长度进行微调，默认padding到8的整数倍，可以通过`pad-to-multiple-of`参数来指定修改padding到几的倍数。假设微调时指定`--seq-length`序列长度为1024，开启`--no-pad-to-seq-lengths`后，序列长度会padding到大于等于真实数据长度且为8的整数倍的值。

        **图 2**  variable-seq-lengths图示  
        ![variable-seq-lengths图示](../../figures/instruction_finetune/variable_seq_lengths.png)

    > [!NOTE]
    > - 提供的路径需要加双引号。
    > - 多机训练中请确保每台机器上的模型路径和数据集路径等无误。
    > - 训练参数的并行配置，如TP/PP/EP/VPP等（具体列表查看[权重转换指南](../../../pytorch/solutions/checkpoint/checkpoint_convert_v2.md#21-huggingface权重转换到megatron-mcore格式)）需要与第3步保持一致。

6. 启动微调  
    参数配置完毕后，可运行脚本启动微调（多机场景中需要在多个终端上同时启动脚本）：

    ```shell
    bash examples/mcore/qwen3/tune_qwen3_8b_4K_full_ptd.sh
    ```

7. 推理验证  
    完成微调后，需要进一步验证模型是否具备了预期的输出能力。我们提供了简单的模型生成脚本，只需要加载微调后的模型权重，便可观察模型在不同生成参数配置下的回复，详细配置请参考[Qwen3-8B推理脚本](../../../../../examples/mcore/qwen3/generate_qwen3_8b_ptd.sh)。

    首先在脚本中修改以下参数：

    ```shell
    CKPT_DIR="your model save ckpt path" # 指向微调后权重的保存路径
    TOKENIZER_PATH="your tokenizer path" # 指向模型tokenizer的路径
    ```

    然后运行推理脚本：

    ```shell
    bash examples/mcore/qwen3/generate_qwen3_8b_ptd.sh
    ```

    此外，若想要验证模型在不同任务下的表现，请参考[模型评估](../evaluation/evaluation_guide.md)来更全面评估微调效果。
