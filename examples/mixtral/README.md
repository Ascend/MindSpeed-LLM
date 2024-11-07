# Mixtral

<p align="left">
        <b>简体中文</b> |
        <b><a href="README_en.md">English</a> </b> 
</p>

# 目录

- [Mixtral](#mixtral)
- [目录](#目录)
- [8x7B](#8x7b)
  - [硬件要求](#硬件要求)
  - [准备工作](#准备工作)
  - [模型训练](#模型训练)
  - [模型推理](#模型推理)
  - [模型评估](#模型评估)

# 8x7B

## 硬件要求

训练的最低硬件配置:

| 硬件 |        配置        |
| :--: |:----------------:|
| NPU | 32 x Ascend NPUs |

推理的推荐硬件配置:

| 硬件 |      配置      |
| :--: | :-------------: |
| NPU | 8 x Ascend NPUs |

## 准备工作

1. 克隆仓库到本地服务器

    ```shell
    git clone https://gitee.com/ascend/ModelLink.git
    cd ModelLink
    git checkout 1.0.RC2
    cd ..
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    git checkout core_r0.6.0
    cp -r megatron ../ModelLink/
    cd ..
    cd ModelLink
    mkdir logs
    mkdir model_from_hf
    mkdir dataset
    mkdir ckpt
    ```

2. 搭建环境

    ```bash
    # python3.8
    conda create -n test python=3.8
    conda activate test

    # 安装 torch 和 torch_npu
    pip install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl
    pip install torch_npu-2.1.0*-cp38-cp38m-linux_aarch64.whl
    pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

    # 修改 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # 安装加速库
    git clone https://gitee.com/ascend/MindSpeed.git
    cd MindSpeed
    git checkout 2b0edd2
    pip install -r requirements.txt 
    pip3 install -e .
    cd ..

    # 安装其余依赖库
    pip install -r requirements.txt 
    ```

3. 下载 Mixtral-8x7B 的 [预训练权重和词表](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/tree/main)（*建议仅下载使用safetensors格式的权重*）

    ```shell
    #!/bin/bash
    cd ./model_from_hf/
    git lfs install
    git clone https://huggingface.co/mistralai/Mixtral-8x7B-v0.1
    mv Mixtral-8x7B-v0.1 Mixtral-8x7B
    cd ..
    ```

4. 权重转换
    HuggingFace权重 --> 任意并行切分策略的Megatron权重
    ***（该场景一般用于使能开源的HuggingFace模型在Megatron上进行训练）***

    ```bash
    # 修改 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    
    # HF 转 tp8-pp4-ep1
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader mixtral_hf \
        --saver mixtral \
        --load-dir ./model_from_hf/Mixtral-8x7B/ \
        --save-dir ./model_weights/Mixtral-8x7B-v0.1-tp8-pp4-ep1/ \
        --tokenizer-model ./model_from_hf/Mixtral-8x7B/tokenizer.model \
        --target-tensor-parallel-size 8 \
        --target-pipeline-parallel-size 4 \
        --target-expert-parallel-size 1 
    ```

    任意并行切分策略的Megatron权重 --> 任意并行切分策略的Megatron权重
    ***（该场景一般用于重新配置切分后模型的权重，比如在四机32卡 TP8-PP4策略下训练完了，想在单机8卡 TP8上进行推理）***

    ```bash
    # 修改 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    
    # tp1-pp8-ep2 转 tp1-pp8-ep1
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader mixtral_mg \
        --saver mixtral \
        --load-dir ./model_weights/Mixtral-8x7B-v0.1-tp8-pp4-ep1/ \
        --save-dir ./model_weights/Mixtral-8x7B-v0.1-tp8-pp1-ep1/ \
        --target-tensor-parallel-size 8 \
        --target-pipeline-parallel-size 1 \
        --target-expert-parallel-size 1 
    ```

    任意并行切分策略的Megatron权重 --> HuggingFace权重
    ***（该场景一般用于将训练好的megatron模型重新转回HuggingFace格式）***

    ```bash
    # 修改 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    
    # tp8-pp4-ep1 转 HF
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader mixtral_mg \
        --saver mixtral \
        --save-model-type huggingface \
        --load-dir ./model_weights/Mixtral-8x7B-v0.1-tp8-pp4-ep1/ \
        --save-dir ./model_from_hf/Mixtral-8x7B/    # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/Mixtral-8x7B/mg2hg/
    ```

## 模型训练

1. 准备数据集

    下载 Mixtral-8x7B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

    ```shell
    # 下载数据
    cd ./dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    # 处理数据   
    mkdir ./dataset/Mixtral-8x7B/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/Mixtral-8x7B/ \
        --output-prefix ./dataset/Mixtral-8x7B/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF
    ```

    配置 Mixtral-8x7B 预训练脚本：***examples/mixtral/pretrain_mixtral_8x7b_ptd.sh***

    ```shell
    # 设置 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # 根据实际情况配置词表、数据集、模型参数保存路径
    DATA_PATH="./dataset/Mixtral-8x7B/alpaca_text_document"
    TOKENIZER_MODEL="./model_from_hf/Mixtral-8x7B/"
    CKPT_SAVE_DIR="./ckpt/Mixtral-8x7B/"

    # 根据分布式集群实际情况配置分布式参数
    GPUS_PER_NODE=8
    MASTER_ADDR="your master node IP"
    MASTER_PORT=6000
    NNODES=4
    NODE_RANK="current node id"
    WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

    # 训练并行策略
    TP=8
    PP=4
    EP=1
    ```

    启动 Mixtral-8x7B 预训练脚本: ***examples/pretrain_mixtral_8x7b_ptd.sh***

    ```shell
    bash examples/mixtral/pretrain_mixtral_8x7b_ptd.sh
    ```

    **注意**：如果使用多机训练，且没有设置数据共享，需要在训练启动脚本中增加`--no-shared-storage`参数，设置此参数之后将会根据分布式参数判断非主节点是否需要load数据，并检查相应缓存和生成数据。

2. 微调

    下载微调数据集 [这里](https://huggingface.co/datasets/silk-road/alpaca-data-gpt4-chinese/blob/main/Alpaca_data_gpt4_zh.jsonl)

    ```shell
    # 下载数据集
    mkdir finetune_dataset
    cd ./finetune_dataset
    wget https://huggingface.co/datasets/silk-road/alpaca-data-gpt4-chinese/blob/main/Alpaca_data_gpt4_zh.jsonl
    cd ..
    
    # 处理微调数据集  
    mkdir ./finetune_dataset/Mixtral-8x7B/
    python ./tools/preprocess_data.py \
        --input ./finetune_dataset/Alpaca_data_gpt4_zh.jsonl \
        --output-prefix ./finetune_dataset/Mixtral-8x7B/alpaca \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path ./model_from_hf/Mixtral-8x7B/ \
        --append-eod \
        --tokenizer-not-use-fast \
        --handler-name GeneralInstructionHandler \
        --workers 4
    ```


3. 指令微调

    微调的配置脚本基本和预训练脚本一致. *区别是数据集，以及增加训练参数--is-instruction-dataset

    增加微调参数--finetune，增加预训练权重加载参数--load，使微调从第一步开始：

    ```bash
    DATA_PATH="./finetune_dataset/Mixtral-8x7B/alpaca"
    CKPT_PATH="./ckpt/Mixtral-8x7B/"
        --load ${CKPT_PATH} \
        --finetune \
        --is-instruction-dataset
    ```


## 模型推理

首先需要配置推理脚本: ***examples/mixtral/generate_mixtral_8x7b_ptd.sh***

```bash
# 根据您自己的 ascend-toolkit 路径，执行set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# 修改模型权重路径和词表路径
CHECKPOINT="./model_weights/Mixtral-8x7B-v0.1-tp8-pp1-ep1/"
TOKENIZER_MODEL="./model_from_hf/Mixtral-8x7B/"

# 根据实际加载的模型权重修改并行配置
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=8
TP=8
PP=1

# 注意
该文档中使用的Mixtral-8x7B-v0.1模型为L0模型，只有续写能力，推理不带任何模板并且容易出现重复或回答不停止的情况。

如若想要有较好的人机对话能力，请使用Mixtral-8x7B-Instruct-v0.1模型，该模型为指令遵从度训练后需要配合模板使用，基本操作同上，仅inference-prompt-type有变化：
--inference-prompt-type mixtral
```

然后可直接启动

```bash
bash examples/mixtral/generate_mixtral_8x7b_ptd.sh
```

推理的示例如下:
![Inference](https://gitee.com/ascend/ModelLink/raw/master/sources/images/mixtral/generate_demo.png)

## 模型评估

使用 MMLU数据集评估模型. 数据集下载路径 [这里](https://huggingface.co/datasets/cais/mmlu).
配置评估脚本: examples/mixtral/evaluate_mixtral_8x7b_ptd.sh

```bash
# ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# 修改模型参数路径和词表路径
TOKENIZER_PATH="./model_from_hf/Mixtral-8x7B/"                    #词表路径
CHECKPOINT="./model_weights/Mixtral-8x7B-v0.1-tp8-pp1-ep1"        #模型路径
# 配置任务和数据集路径
DATA_PATH="./mmlu/test/"
TASK="mmlu"
```

启动评估

```bash
bash examples/mixtral/evaluate_mixtral_8x7b_ptd.sh
```

评估结果如下

| 数据集 | 总问题数 | 参考准确率 | NPU准确率 |
| :----: | :------: | :--------: | :-------: |
|  MMLU  |  14042  |   0.658   |   0.660   |
