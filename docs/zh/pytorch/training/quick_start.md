# 快速入门：Qwen3-8B 模型预训练及微调

## 概述

本文档提供了一个简易示例，帮助初次接触MindSpeed LLM的开发者快速启动模型训练任务，并基于预训练语言模型，使用单样本格式数据完成指令单机微调任务。
以下将以Qwen3-8B模型为例，指导开发者完成大语言模型的预训练和微调任务，主要步骤包括：

- 环境准备：根据仓库指导文件搭建环境
- 准备开源模型权重：从HuggingFace下载Qwen3-8B原始模型
- 启动训练任务：在昇腾NPU上进行模型预训练和微调

开发者入门基础：

- 具备基础的PyTorch使用经验
- 具备初级的Python开发经验
- 对Megatron-LM仓库有基本的了解

## 环境准备

### 环境搭建

基于PyTorch框架，环境搭建请参考[MindSpeed LLM安装指导](install_guide.md)。

### 获取开源模型权重

1. 通过HuggingFace获取模型权重文件。

    ```shell
    # 创建一个目录存储权重文件
    mkdir -p ./model_from_hf/qwen3_hf
    cd ./model_from_hf/qwen3_hf

    # wget获取权重文件
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/config.json
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/generation_config.json
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/merges.txt
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/model-00001-of-00005.safetensors
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/model-00002-of-00005.safetensors
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/model-00003-of-00005.safetensors
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/model-00004-of-00005.safetensors
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/model-00005-of-00005.safetensors
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/model.safetensors.index.json
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/tokenizer.json
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/tokenizer_config.json
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/vocab.json
    ```

2. 通过sha256sum验证模型权重文件完整性。

    ```shell
    # 利用sha256sum计算sha256数值
    # 打开文件明细可获取sha256值，https://huggingface.co/Qwen/Qwen3-8B/blob/main/model-00001-of-00005.safetensors
    sha256sum model-00001-of-00005.safetensors
    sha256sum model-00002-of-00005.safetensors
    sha256sum model-00003-of-00005.safetensors
    sha256sum model-00004-of-00005.safetensors
    sha256sum model-00005-of-00005.safetensors
    ```

### 权重转换

昇腾MindSpeed LLM要求模型权重采用Megatron-Mcore格式，在这里我们将原始HuggingFace权重格式转换为Megatron-Mcore格式，详见[hf2mg权重转换](../../pytorch/tools/checkpoint_convert_hf_mcore.md#21-huggingface权重转换到megatron-mcore格式)。

使用官方提供的转换脚本，获取对应切分的mg权重。

1. 进入MindSpeed LLM目录编辑权重转换脚本。

    ```shell
    cd ../..
    vi examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh
    ```

2. 完成转换脚本的修改配置并保存。
    如下为调整后的hf2mcore权重转换示例脚本。

    ```bash
    # 请按照实际环境修改set_env.sh路径
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    source /usr/local/Ascend/cann/set_env.sh

    python convert_ckpt_v2.py \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 1 \           # TP切分大小
    --target-pipeline-parallel-size 2 \         # PP切分大小
    --load-dir ./model_from_hf/qwen3_hf/ \      # HF权重路径
    --save-dir ./model_weights/qwen3_mcore/ \   # Megatron权重保存路径
    --model-type-hf qwen3
    ```

    **表 1**  权重转换参数解析

    |参数|说明|必填|
    |---|---|---|
    |`--target-tensor-parallel-size`|张量并行度设置（建议配置1）| ✅ |
    |`--target-pipeline-parallel-size`|流水线并行度设置（建议保持2）| ✅ |
    |`--load-model-type`|加载权重的类别（可以是hf、mg）| ✅ |
    |`--save-model-type`|存储权重的类别（可以是hf、mg）| ✅ |
    |`--load-dir`|权重文件加载路径| ✅ |
    |`--save-dir`|权重文件保存路径| ✅ |
    |`--model-type-hf`|HuggingFace模型类别| ✅ |

3. 执行权重转换脚本。

    ```shell
    bash examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh
    ```

    运行脚本后，预期会看到类似以下的日志输出，表示权重转换成功：

    ```shell
    INFO:root:Saving to ./model_weights/qwen3_mcore/iter_0000001/mp_rank_00_001/model_optim_rng.pt
    INFO:root:Done!
    ```

> [!NOTE]
>
> 对于Qwen3-8B模型，此处推荐的切分配置是tp1pp2，对应上述配置。

## 启动预训练

在这一阶段，我们将基于下载的HuggingFace原数据，完成数据集预处理，并启动模型预训练，具体步骤如下：

1. 数据预处理
2. 启动预训练任务

### 数据预处理

通过对各种格式的数据做提前预处理，避免原始数据的反复处理加载，将所有的数据都统一存储到.bin和.idx两个文件中，详见[预训练数据处理](../../pytorch/tools/data_process_pretrain.md)。

如下以Alpaca数据集为例，进行预训练数据集处理示例。

1. 获取数据集原数据。

    ```shell
    mkdir dataset
    cd dataset/
    # HuggingFace 数据集链接（择一获取）
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    # ModelScope 数据集链接（择一获取）
    wget https://www.modelscope.cn/datasets/angelala00/tatsu-lab-alpaca/resolve/master/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    ```

2. 编辑预训练数据处理脚本。

    ```shell
    vi examples/mcore/qwen3/data_convert_qwen3_pretrain.sh
    ```

3. 完成数据处理脚本的修改配置并保存。

    如下为调整后的数据处理示例脚本。

    ```bash
    # 请按照实际环境修改set_env.sh路径
    source /usr/local/Ascend/cann/set_env.sh

    python ./preprocess_data.py \
      --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
      --tokenizer-name-or-path ./model_from_hf/qwen3_hf/ \         # 注意此处路径是否一致
      --tokenizer-type PretrainedFromHF \
      --handler-name GeneralPretrainHandler \
      --output-prefix ./dataset/alpaca \                           # 预训练数据集会生成alpaca_text_document.bin和.idx
      --json-keys text \
      --workers 4 \
      --log-interval 1000
    ```

    **表 2**  数据预处理参数解析

    |参数|说明|必填|
    |---|---|---|
    |`--input`|支持输入数据集目录或文件，目录则处理全部文件, 支持.parquet、.csv、.json、.jsonl、.txt、.arrow格式，同一目录要求数据格式保持一致| ✅ |
    |`--tokenizer-type`|说明使用tokenizer类别，参数值为PretrainedFromHF时，词表路径填写模型目录即可| ✅ |
    |`--tokenizer-name-or-path`|配合tokenizer-type，目标模型的tokenizer原数据文件夹，用于数据集的转换| ✅ |
    |`--handler-name`|指定数据集的处理类| ✅ |
    |`--output-prefix`|转换后输出的数据集文件的文件名前缀 | ✅ |
    |`--workers`|多进程数据集处理| ✅ |
    |`--log-interval`|处理进度更新的间隔步数| ✅ |
    |`--json-keys`|从文件中提取的列名列表，默认为`text`，可以为`text`、`input`及`title`等多个输入，结合实际情况及数据集内容使用| ✅ |

4. 执行预训练数据处理脚本。

    ```shell
    bash examples/mcore/qwen3/data_convert_qwen3_pretrain.sh
    ```

    预训练数据集处理结果如下：

    ```shell
    ./dataset/alpaca_text_document.bin
    ./dataset/alpaca_text_document.idx
    ```

### 启动预训练任务

完成了数据集处理和权重转换之后，可以开始启动预训练任务。

1. 编辑示例脚本。

    ```shell
    vi examples/mcore/qwen3/pretrain_qwen3_8b_4K_ptd.sh
    ```

2. 修改并保存预训练参数配置，配置示例如下：

    ```bash
    NPUS_PER_NODE=8           # 使用单节点的8卡NPU
    MASTER_ADDR=localhost     # 单机使用本节点ip，多机所有节点都配置为master_ip
    MASTER_PORT=6000          # 本节点端口号为6000
    NNODES=1                  # 根据参与节点数量配置，单机为1，多机即多节点
    NODE_RANK=0               # 单机RANK为0，多机为(0,NNODES-1)，不同节点不可重复，master_node rank为0，其ip为master_ip
    WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))

    # 根据实际情况配置权重保存、权重加载、词表、数据集路径，多机中所有节点都要有如下数据
    CKPT_LOAD_DIR="./model_weights/qwen3_mcore/"   # 权重加载路径，填入权重转换时保存的权重路径
    CKPT_SAVE_DIR="./ckpt/qwen3-8b"                # 训练完成后的权重保存路径
    DATA_PATH="./dataset/alpaca_text_document"     # 数据集路径，填入数据预处理时保存的数据路径，注意需要添加后缀，如使用Alpaca数据集预处理会生成alpaca_text_document.bin和.idx，则在数据集路径后再加上alpaca_text_document
    TOKENIZER_PATH="./model_from_hf/qwen3_hf/"     # 词表路径，填入下载的开源权重词表路径

    TP=1                # 权重转换设置--target-tensor-parallel-size 1，修改为1
    PP=2                # 权重转换设置--target-pipeline-parallel-size 2，修改为2，与权重转换时一致
    SEQ_LENGTH=4096     # 设置seq_length为4096
    MBS=1               # 设置micro-batch-size为1
    GBS=64              # 设置global-batch-size为64
    TRAIN_ITERS=2000    # 设置训练迭代步数
    ```

3. 设置环境变量。

    ```shell
    source /usr/local/Ascend/cann/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    ```

    以上命令以root用户安装后的默认路径为例，请用户根据set_env.sh的实际路径进行替换。

4. 执行预训练脚本。

    ```shell
    bash examples/mcore/qwen3/pretrain_qwen3_8b_4K_ptd.sh
    ```

    **图 1**  启动预训练

    ![img_2.png](../../pytorch/figures/quick_start/running_log.png)

    脚本中包含训练参数或优化特性，下表为部分参数解释。

    **表 3**  训练脚本参数说明

    |参数名|说明|
    |----|----|
    |`--use-mcore-models`|使用Mcore分支运行模型|
    |`--disable-bias-linear`|去掉linear的偏移值，与Qwen原模型一致|
    |`--group-query-attention`|开启GQA注意力处理机制|
    |`--num-query-groups 8`|配合GQA使用，设置groups为8|
    |`--position-embedding-type rope`|位置编码采用RoPE方案|
    |`--untie-embeddings-and-output-weights`|根据原模型要求将output层和embedding层的权重解耦|
    |`--bf16`|昇腾芯片对bf16精度支持良好，可显著提升训练速度|

> [!NOTE]
>
> - 多机训练需在多个终端同时启动预训练脚本（每个终端的预训练脚本只有NODE_RANK参数不同，其他参数均相同）。
> - 如果使用多机训练，且没有设置数据共享，需要在训练启动脚本中增加`--no-shared-storage`参数，设置此参数之后将会根据分布式参数判断非主节点是否需要load数据，并检查相应缓存和生成数据。

## 启动微调

在这一阶段，我们将基于下载的HuggingFace原数据，完成数据集预处理，并启动模型微调，具体步骤如下：

1. 数据预处理
2. 启动微调任务

### 数据预处理

通过对各种格式的数据做提前预处理，避免原始数据的反复处理加载，将所有的数据都统一存储到.bin和.idx两个文件中，详见[Alpaca微调数据使用文档](../../pytorch/tools/data_process_sft_alpaca_style.md)。

如下以Alpaca数据集为例，进行数据预处理示例。

1. 获取数据集原数据。

    ```shell
    mkdir dataset
    cd dataset/
    # HuggingFace 数据集链接
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    ```

2. 编辑数据预处理脚本。

    ```shell
    vi examples/mcore/qwen3/data_convert_qwen3_instruction.sh
    ```

3. 完成数据预处理脚本的修改配置并保存。

    ```bash
    # 请根据实际环境修改set_env.sh路径
    source /usr/local/Ascend/cann/set_env.sh
    mkdir ./finetune_dataset

    python ./preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./model_from_hf/qwen3_hf/ \
    --output-prefix ./finetune_dataset/alpaca \
    --handler-name AlpacaStyleInstructionHandler \
    --tokenizer-type PretrainedFromHF \
    --workers 4 \
    --log-interval 1000 \
    --enable-thinking true \
    --prompt-type qwen3
    ```

    **表 4**  数据预处理参数解析

    |参数|说明|必填|
    |---|---|---|
    |`--input`|支持输入数据集目录或文件，目录则处理全部文件, 支持.parquet、.csv、.json、.jsonl、.txt、.arrow格式，同一目录要求数据格式保持一致| ✅ |
    |`--tokenizer-type`|说明使用tokenizer类别，参数值为PretrainedFromHF时，词表路径填写模型目录即可| ✅ |
    |`--tokenizer-name-or-path`|配合tokenizer-type，目标模型的tokenizer原数据文件夹，用于数据集的转换| ✅ |
    |`--output-prefix`|转换后输出的数据集文件的文件名前缀 | ✅ |
    |`--workers`|多进程数据集处理| ✅ |
    |`--handler-name`|指定数据集的处理类| ✅ |
    |`--log-interval`|处理进度更新的间隔步数| ✅ |
    |`--enable-thinking`|快慢思考模板开关|  |
    |`--prompt-type`|用于指定模型模板| ✅ |

4. 执行数据处理脚本。

    ```shell
    bash examples/mcore/qwen3/data_convert_qwen3_instruction.sh
    ```

    微调数据集处理结果如下：

    ```shell
    ./finetune_dataset/alpaca_packed_attention_mask_document.bin
    ./finetune_dataset/alpaca_packed_attention_mask_document.idx
    ./finetune_dataset/alpaca_packed_input_ids_document.bin
    ./finetune_dataset/alpaca_packed_input_ids_document.idx
    ./finetune_dataset/alpaca_packed_labels_document.bin
    ./finetune_dataset/alpaca_packed_labels_document.idx
    ```

### 启动微调任务

完成了数据集处理和权重转换之后，可以开始启动微调任务。

1. 编辑示例脚本。

    ```shell
    vi examples/mcore/qwen3/tune_qwen3_8b_4K_full_ptd.sh
    ```

2. 修改并保存微调参数配置，配置示例如下：

    ```bash
    NPUS_PER_NODE=8  # 单节点的卡数
    MASTER_ADDR=localhost
    MASTER_PORT=6000
    NNODES=1
    NODE_RANK=0
    WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
    ```

    在脚本中修改相关路径参数和模型切分配置：

    ```bash
    CKPT_LOAD_DIR="./model_weights/qwen3_mcore/"  # 指向权重转换后保存的路径
    CKPT_SAVE_DIR="./ckpt/qwen3-8b"               # 指向用户指定的微调后权重保存路径
    DATA_PATH="./finetune_dataset/alpaca"         # 指定处理后的数据路径
    TOKENIZER_PATH="./model_from_hf/qwen3_hf/"    # 指定模型的tokenizer路径
    TP=1                                          # 模型权重转换的tp大小，在本例中是1
    PP=2                                          # 模型权重转换的pp大小，在本例中是2
    ```

3. 设置环境变量。

    ```shell
    source /usr/local/Ascend/cann/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    ```

    以上命令以root用户安装后的默认路径为例，请用户根据set_env.sh的实际路径进行替换。

4. 执行微调脚本。

    ```shell
    bash examples/mcore/qwen3/tune_qwen3_8b_4K_full_ptd.sh
    ```

    脚本中包含微调参数或优化特性，下表为部分参数解释。

    **表 5**  微调脚本参数说明

    |参数名|说明|
    |----|----|
    |`--finetune`|启动模型的微调模式。|
    |`--stage`|训练方法。|
    |`--is-instruction-dataset`|用于指定微调过程中采用指令微调数据集，以确保模型依据特定指令数据进行微调。|
    |`--prompt-type`|用于指定模型模板，能够让base模型微调后能具备更好的对话能力。可在[templates.json](../../../../configs/finetune/templates.json)文件内查看`prompt-type`的可选项。|
    |`--no-pad-to-seq-lengths`|支持动态序列长度微调，默认按照8的倍数进行padding。|
    |`--sequence-parallel`|开启序列并行。|
    |`--use-distributed-optimizer`|启用分布式优化器。|
    |`--use-flash-attn`|启用Flash Attention。|
    |`--bf16`|昇腾芯片对bf16精度支持良好，可显著提升训练速度。|
