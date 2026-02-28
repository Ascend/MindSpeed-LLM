# 快速入门：Qwen2.5 模型预训练

## 概述

本文档提供了一个简易示例，帮助初次接触MindSpeed LLM的开发者快速启动模型训练任务。以下将以Qwen2.5-7B模型为例，指导开发者完成Qwen2.5-7B大语言模型的预训练任务，主要步骤包括：

- 环境准备：根据仓库指导文件搭建环境
- 获取开源模型权重：从HuggingFace下载Qwen2.5-7B原始模型
- 启动预训练：在昇腾NPU上进行模型预训练

开发者入门基础：

- 具备基础的PyTorch使用经验
- 具备初级的Python开发经验
- 对Megatron-LM仓库有基本的了解

## 环境准备

### 环境搭建

基于不同的框架，环境搭建请参考[MindSpeed LLM安装指导-PyTorch框架](./pytorch/install_guide.md)和[MindSpeed LLM安装指导-MindSpore框架](./mindspore/install_guide.md)。

### 获取开源模型权重

1. 获取模型权重文件。
    - 通过HuggingFace

        ```shell
        # 创建一个目录存储权重文件
        mkdir -p ./model_from_hf/qwen2.5-7b-hf
        cd ./model_from_hf/qwen2.5-7b-hf

        # wget获取权重文件
        wget https://huggingface.co/Qwen/Qwen2.5-7B/resolve/main/config.json
        wget https://huggingface.co/Qwen/Qwen2.5-7B/resolve/main/generation_config.json
        wget https://huggingface.co/Qwen/Qwen2.5-7B/resolve/main/merges.txt
        wget https://huggingface.co/Qwen/Qwen2.5-7B/resolve/main/model-00001-of-00004.safetensors
        wget https://huggingface.co/Qwen/Qwen2.5-7B/resolve/main/model-00002-of-00004.safetensors
        wget https://huggingface.co/Qwen/Qwen2.5-7B/resolve/main/model-00003-of-00004.safetensors
        wget https://huggingface.co/Qwen/Qwen2.5-7B/resolve/main/model-00004-of-00004.safetensors
        wget https://huggingface.co/Qwen/Qwen2.5-7B/resolve/main/model.safetensors.index.json
        wget https://huggingface.co/Qwen/Qwen2.5-7B/resolve/main/tokenizer.json
        wget https://huggingface.co/Qwen/Qwen2.5-7B/resolve/main/tokenizer_config.json
        wget https://huggingface.co/Qwen/Qwen2.5-7B/resolve/main/vocab.json
        ```

    - 通过ModelScope 

        ```shell
        # 创建一个目录存储权重文件
        mkdir -p ./model_from_hf/qwen2.5-7b-hf
        cd ./model_from_hf/qwen2.5-7b-hf

        # wget获取权重文件
        wget https://www.modelscope.cn/models/Qwen/Qwen2.5-7B/resolve/master/config.json
        wget https://www.modelscope.cn/models/Qwen/Qwen2.5-7B/resolve/master/generation_config.json
        wget https://www.modelscope.cn/models/Qwen/Qwen2.5-7B/resolve/master/merges.txt
        wget https://www.modelscope.cn/models/Qwen/Qwen2.5-7B/resolve/master/model-00001-of-00004.safetensors
        wget https://www.modelscope.cn/models/Qwen/Qwen2.5-7B/resolve/master/model-00002-of-00004.safetensors
        wget https://www.modelscope.cn/models/Qwen/Qwen2.5-7B/resolve/master/model-00003-of-00004.safetensors
        wget https://www.modelscope.cn/models/Qwen/Qwen2.5-7B/resolve/master/model-00004-of-00004.safetensors
        wget https://www.modelscope.cn/models/Qwen/Qwen2.5-7B/resolve/master/model.safetensors.index.json
        wget https://www.modelscope.cn/models/Qwen/Qwen2.5-7B/resolve/master/tokenizer.json
        wget https://www.modelscope.cn/models/Qwen/Qwen2.5-7B/resolve/master/tokenizer_config.json
        wget https://www.modelscope.cn/models/Qwen/Qwen2.5-7B/resolve/master/vocab.json
        ```

2. 通过sha256sum验证模型权重文件完整性。  

    ```shell
    # 利用sha256sum计算sha256数值
    # 打开文件明细可获取sha256值，https://huggingface.co/Qwen/Qwen2.5-7B/blob/main/model-00001-of-00004.safetensors
    # 如果从ModelScope下载，则打开 https://www.modelscope.cn/models/Qwen/Qwen2.5-7B/file/view/master/model-00001-of-00004.safetensors
    sha256sum model-00001-of-00004.safetensors
    sha256sum model-00002-of-00004.safetensors
    sha256sum model-00003-of-00004.safetensors
    sha256sum model-00004-of-00004.safetensors
    ```

    **图 1**  计算本地权重文件sha256数值  
    ![img.png](./pytorch/figures/quick_start/sha256.png)

    **图 2**  从对应官网获取sha256数值  
    ![img_1.png](./pytorch/figures/quick_start/sha256_hf.png)

## 启动预训练

在这一阶段，我们将基于下载的HuggingFace原数据，完成权重转换和数据集预处理，并启动模型预训练，具体步骤如下：

1. HuggingFace权重转换为Megatron权重
2. 处理预训练数据集
3. 启动预训练任务

### 权重转换

昇腾MindSpeed LLM要求模型权重采用Megatron-Mcore格式，在这里我们将原始HuggingFace权重格式转换为Megatron-Mcore格式。
详见[hf2mg权重转换](./pytorch/solutions/checkpoint/checkpoint_convert.md#21-huggingface权重转换到megatron-mcore格式)

使用官方提供的转换脚本，获取对应切分的mg权重。

1. 编辑权重转换脚本。

    ```shell
    cd MindSpeed-LLM
    vi examples/mcore/qwen25/ckpt_convert_qwen25_hf2mcore.sh
    ```

2. 完成转换脚本的修改配置并保存。

    如下为调整后的hf2mcore权重转换示例脚本。

    ```bash
    # 请按照实际环境修改 set_env.sh 路径
    source /usr/local/Ascend/cann/set_env.sh

    python convert_ckpt.py \
           --use-mcore-models \
           --model-type GPT \
           --load-model-type hf \
           --save-model-type mg \
           --target-tensor-parallel-size 1 \    # 将切分调整为tp1
           --target-pipeline-parallel-size 4 \  # 将切分调整为pp4
           --add-qkv-bias \
           --load-dir ./model_from_hf/qwen2.5-7b-hf/ \
           --save-dir ./model_weights/qwen2.5_mcore/ \
           --tokenizer-model ./model_from_hf/qwen2.5-7b-hf/tokenizer.json \
           --model-type-hf llama2 \
           --params-dtype bf16
    ```

      **表 1**  权重转换参数解析

    |参数|说明|必填|
    |---|---|---|
    |`--model-type GPT`|指定模型类型为GPT系列| ✅ |
    |`--use-mcore-models`|转换为Megatron-Mcore格式| ✅ |
    |`--target-tensor-parallel-size`|张量并行度设置（建议配置1）| ✅ |
    |`--target-pipeline-parallel-size`|流水线并行度设置（建议保持4）| ✅ |
    |`--tokenizer-model`|指定分词器路径| ✅ |
    |`--load-model-type`|加载权重的类别（可以是hf、mg）| ✅ |
    |`--save-model-type`|存储权重的类别（可以是hf、mg）| ✅ |
    |`--load-dir`|权重文件加载路径| ✅ |
    |`--save-dir`|权重文件保存路径| ✅ |
    |`--model-type-hf`|HuggingFace模型类别，默认为llama2|   |
    |`--params-dtype`|指定权重转换后的权重精度模式，默认为fp16，如果源文件格式为bf16，则需要设置为bf16 | ✅ |

3. 执行权重转换脚本。

    ```shell
    bash examples/mcore/qwen25/ckpt_convert_qwen25_hf2mcore.sh
    ```

> [!NOTE]  
>
> - 对于Qwen2.5-7B模型，此处推荐的切分配置是tp1pp4，对应上述配置。
> - MindSpore框架尚不支持QLoRA权重量化转换，【--qlora-nf4】参数仅可置为False。
> - MindSpore框架默认在Device侧进行权重转换，在模型较大时存在OOM风险，因此建议用户手动修改`convert_ckpt.py`，在包导入时加> 入如下代码设置CPU侧执行权重转换：
>
>     ```python
>     import mindspore as ms
>     ms.set_context(device_target="CPU", pynative_synchronize=True)
>     import torch
>     torch.configs.set_pyboost(False)
>     ```
>
> - MindSpore框架转换出的模型权重无法直接用于PyTorch框架训练或推理。

### 预训练数据集处理

通过对各种格式的数据做提前预处理，避免原始数据的反复处理加载，将所有的数据都统一存储到.bin和.idx两个文件中，详见[预训练数据处理](./pytorch/solutions/pretrain/pretrain_dataset.md)

常用的预训练数据集包括alpaca、enwiki、c4等，[预训练数据处理](./pytorch/solutions/pretrain/pretrain_dataset.md)中提供了数据集下载地址。

如下以Alpaca数据集为例，进行预训练数据集示例。

1. 获取数据集元数据。

    ```shell
    mkdir dataset
    cd dataset/
    # HuggingFace 数据集链接（择一获取）
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    # ModelScope 数据集链接（择一获取）
    wget https://www.modelscope.cn/datasets/angelala00/tatsu-lab-alpaca/resolve/master/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    ```

2. 编辑预训数据处理脚本。

    ```shell
    vi examples/mcore/qwen25/data_convert_qwen25_pretrain.sh
    ```

3. 完成数据处理脚本的修改配置并保存。

    如下为调整后的数据处理示例脚本。

    ```bash
    # 请按照实际环境修改 set_env.sh 路径
    source /usr/local/Ascend/cann/set_env.sh

    python ./preprocess_data.py \
      --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
      --tokenizer-name-or-path ./model_from_hf/qwen2.5-7b-hf/ \         # 注意此处路径是否一致
      --output-prefix ./dataset/alpaca \                                # 预训练数据集会生成alpaca_text_document.bin和.idx
      --tokenizer-type PretrainedFromHF \
      --workers 4 \
      --log-interval 1000
    ```

    **表 2**  数据预处理参数解析

    |参数|说明|必填|
    |---|---|---|
    |`--input`|支持输入数据集目录或文件，目录则处理全部文件, 支持.parquet、.csv、.json、.jsonl、.txt、.arrow格式，同一目录要求数据格式保持一致| ✅ |
    |`--tokenizer-type`|说明使用tokenizer类别，参数值为PretrainedFromHF时，词表路径填写模型目录即可| ✅ |
    |`--tokenizer-name-or-path`|配合tokenizer-type，目标模型的tokenizer原数据文件夹，用于数据集的转换|  |
    |`--tokenizer-model`|配合指定分词器模型的路径，路径具体到tokenizer.model文件|  |
    |`--output-prefix`|转换后输出的数据集文件的文件名前缀 | ✅ |
    |`--workers`|多进程数据集处理| ✅ |

4. 执行预训数据处理脚本。

    ```shell
    bash examples/mcore/qwen25/data_convert_qwen25_pretrain.sh
    ```

### 启动预训练任务

完成了数据集处理和权重转换之后，可以开始拉起预训练任务。

1. 编辑示例脚本。

    ```shell
    vi examples/mcore/qwen25/pretrain_qwen25_7b_32k_ptd.sh
    ```

2. 修改并报错预训练参数配置，配置示例如下：

    ```bash
    NPUS_PER_NODE=8           # 使用单节点的8卡NPU
    MASTER_ADDR=localhost     # 单机使用本节点ip，多机所有节点都配置为master_ip
    MASTER_PORT=6000          # 本节点端口号为6000
    NNODES=1                  # 根据参与节点数量配置，单机为1，多机即多节点
    NODE_RANK=0               # 单机RANK为0，多机为(0,NNODES-1)，不同节点不可重复，master_node rank为0，其ip为master_ip
    WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))

    # 根据实际情况配置权重保存、权重加载、词表、数据集路径，多机中所有节点都要有如下数据
    CKPT_LOAD_DIR="./model_weights/qwen2.5_mcore/"  # 权重加载路径，填入权重转换时保存的权重路径
    CKPT_SAVE_DIR="./ckpt/qwen25-7b"                # 训练完成后的权重保存路径
    DATA_PATH="./dataset/alpaca_text_document"      # 数据集路径，填入数据预处理时保存的数据路径，注意需要添加后缀,如使用alpaca数据集预处理会生成alpaca_text_document.bin和.idx，则在数据集路径后再加上alpaca_text_document
    TOKENIZER_PATH="./model_from_hf/qwen2.5-7b-hf/" # 词表路径，填入下载的开源权重词表路径

    TP=1                # 权重转换设置--target-tensor-parallel-size 1，修改为1
    PP=4                # 权重转换设置--target-pipeline-parallel-size 4，修改为4，与权重转换时一致
    SEQ_LEN=4096        # 修改seq_length为4096 
    MBS=1               # 设置micro-batch-size为1
    GBS=64              # 设置global-batch-size为64
    ```

3. 设置环境变量。

    ```shell
    source /usr/local/Ascend/cann/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    ```

    以上命令以root用户安装后的默认路径为例，请用户根据set_env.sh的实际路径进行替换。

4. 执行预训练脚本。

    ```shell
    bash examples/mcore/qwen25/pretrain_qwen25_7b_32k_ptd.sh
    ```

    **图 1**  启动预训练  
    ![img_2.png](./pytorch/figures/quick_start/running_log.png)

    脚本中特性包含训练参数或优化特性，下表为部分参数解释。

    **表 3**  训练脚本参数说明  

    |参数名|说明|
    |----|----| 
    |`--use-mcore-models`|使用Mcore分支运行模型|
    |`--disable-bias-linear`|去掉linear的偏移值，与Qwen原模型一致|
    |`--add-qkv-bias`|增加Q、K、V的偏移值，是权重的组成部分|
    |`--group-query-attention`|开启GQA注意力处理机制|
    |`--num-query-groups 4`|配合GQA使用，设置groups为4|
    |`--position-embedding-type rope`|位置编码采用RoPE方案|
    |`--untie-embeddings-and-output-weights`|根据原模型要求将output层和embedding层的权重解耦|
    |`--bf16`|昇腾芯片对BF16精度支持良好，可显著提升训练速度|

> [!NOTE]
>
> - 多机训练需在多个终端同时启动预训练脚本（每个终端的预训练脚本只有NODE_RANK参数不同，其他参数均相同）。
> - 如果使用多机训练，且没有设置数据共享，需要在训练启动脚本中增加`--no-shared-storage`参数，设置此参数之后将会根据分布式参数判断非主节点是否需要load数据，并检查相应缓存和生成数据。
