# LLaMA3  $\color{black}{\bf\tiny{【昇腾贡献模型】}}$

<p align="left">
        <b>简体中文</b> |
        <b><a href="README_en.md">English</a> </b> 
</p>

# 目录

- [LLaMA3](#llama3)
- [目录](#目录)
- [LLAMA3-8B](#llama3-8b)
  - [训练](#训练)
    - [脚本](#脚本)
  - [推理-8B](#推理-8b)
  - [评估-8B](#评估-8b)
- [LLAMA3-70B](#llama3-70b)
  - [训练](#训练)
    - [脚本](#脚本)
  - [推理-70B](#推理-70b)
  - [评估-70B](#评估-70b)

# LLAMA3-8B

## 训练

LLAMA3-8B 训练的硬件配置:

| 硬件 |      配置      |
| :--: | :-------------: |
| NPU | 8 x Ascend NPUs |

### 脚本

1. 克隆仓库到本地服务器

    ```shell
    git clone https://gitee.com/ascend/MindSpeed-LLM.git 
    cd MindSpeed-LLM
    git checkout 1.0.RC2
    cd ..
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    git checkout core_r0.6.0
    cp -r megatron ../MindSpeed-LLM/
    cd ..
    cd MindSpeed-LLM
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
3. 下载 LLAMA3-8B 的 [预训练权重和词表](https://huggingface.co/unsloth/llama-3-8B/tree/main)

    ```shell
    #!/bin/bash
    mkdir ./model_from_hf/llama-3-8b-hf/
    cd ./model_from_hf/llama-3-8b-hf/
    wget https://huggingface.co/unsloth/llama-3-8B/raw/main/config.json
    wget https://huggingface.co/unsloth/llama-3-8B/raw/main/generation_config.json
    wget https://huggingface.co/unsloth/llama-3-8B/raw/main/model-00001-of-00004.safetensors
    wget https://huggingface.co/unsloth/llama-3-8B/raw/main/model-00002-of-00004.safetensors
    wget https://huggingface.co/unsloth/llama-3-8B/raw/main/model-00003-of-00004.safetensors
    wget https://huggingface.co/unsloth/llama-3-8B/raw/main/model-00004-of-00004.safetensors
    wget https://huggingface.co/unsloth/llama-3-8B/raw/main/model.safetensors.index.json
    wget https://huggingface.co/unsloth/llama-3-8B/raw/main/special_tokens_map.json
    wget https://huggingface.co/unsloth/llama-3-8B/raw/main/tokenizer.json
    wget https://huggingface.co/unsloth/llama-3-8B/raw/main/tokenizer_config.json
    cd ../../
    ```
4. 权重转换

    4.1 将权重从 huggingface 格式转化为 megatron 格式
    ***（该场景一般用于使能开源的HuggingFace模型在Megatron上进行训练）***

    ```bash
    # 修改 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # 权重格式转换
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader llama2_hf \
        --saver megatron \
        --target-tensor-parallel-size 8 \
        --target-pipeline-parallel-size 1 \
        --load-dir ./model_from_hf/llama-3-8b-hf/ \
        --save-dir ./model_weights/llama-3-8b-hf-v0.1-tp8-pp1/ \
        --tokenizer-model ./model_from_hf/llama-3-8b-hf/tokenizer.json
    ```

    4.2 任意并行切分策略的 Megatron 权重 格式转化为 HuggingFace权重
    ***（该场景一般用于将训练好的megatron模型重新转回HuggingFace格式）***

    ```shell
    # 请按照您的真实环境修改 set_env.sh 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader megatron \
        --saver megatron \
        --save-model-type save_huggingface_llama \
        --load-dir ./model_weights/llama-3-8b-hf-v0.1-tp8-pp1/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --save-dir ./model_from_hf/llama-3-8b-hf/     # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/llama-3-8b-hf/mg2hg/
    ```

    权重转换适用于预训练、微调、推理和评估，根据任务不同调整参数 `target-tensor-parallel-size`和 `target-pipeline-parallel-size`。

5. 预训练

    5.1 准备数据集

    下载 LLaMA3-8B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/blob/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

    ```shell
    # 下载数据
    cd ./dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    # 处理数据   
    mkdir ./dataset/llama-3-8b-hf/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/llama-3-8b-hf/ \
        --output-prefix ./dataset/llama-3-8b-hf/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF
    ```

    5.2 预训练
    配置llama3-8B 预训练脚本: examples/llama3/pretrain_llama3_8b_ptd.sh

    ```shell
    # 设置 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    
    # 根据实际情况配置词表、数据集、模型参数保存路径
    CKPT_SAVE_DIR="./ckpt/llama-3-8b-hf/"
    TOKENIZER_MODEL="./model_from_hf/llama-3-8b-hf/"  #词表路径
    DATA_PATH="./dataset/llama-3-8b-hf/alpaca_text_document"  #数据集路径
    CKPT_LOAD_DIR="./model_weights/llama-3-8b-hf-v0.1-tp8-pp1/" #权重路径
    ```

    多机运行增加参数--overlap-grad-reduce

    启动 LLaMA3-8B 预训练脚本: examples/llama3/pretrain_llama3_8b_ptd.sh

    ```shell
    bash examples/llama3/pretrain_llama3_8b_ptd.sh
    ```

    **注意**：如果使用多机训练，且没有设置数据共享，需要在训练启动脚本中增加`--no-shared-storage`参数，设置此参数之后将会根据分布式参数判断非主节点是否需要load数据，并检查相应缓存和生成数据。
6. 微调

    6.1 准备微调数据集
    
    下载微调数据集 [这里](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

    ```shell
    # 下载数据集
    mkdir finetune_dataset
    cd ./finetune_dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    
    # 处理微调数据集  
    mkdir ./finetune_dataset/llama-3-8b-hf/
    python ./tools/preprocess_data.py \
        --input ./finetune_dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/llama-3-8b-hf/ \
        --output-prefix ./finetune_dataset/llama-3-8b-hf/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF \
        --handler-name GeneralInstructionHandler \
        --append-eod
    ```

    6.2 全参微调
    
    全参微调的配置脚本基本和预训练脚本一致. *区别是数据集，以及增加训练参数--is-instruction-dataset*
    增加微调参数--finetune，增加预训练权重加载参数--load，使微调从第一步开始。更改为以下参数：

    ```bash
    DATA_PATH="./finetune_dataset/llama-3-8b-hf/alpaca"
    TOKENIZER_PATH="./model_from_hf/llama-3-8b-hf/"
    CKPT_PATH="./ckpt/llama-3-8b-hf/"
        --load ${CKPT_PATH} \
        --finetune \
        --is-instruction-dataset \
        --tokenizer-not-use-fast \
    ```


## 推理-8B

配置llama3-8B 推理脚本: examples/llama3/generate_llama3_8b_ptd.sh

```bash
# 根据您自己的 ascend-toolkit 路径，执行set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
# 修改模型权重路径和词表路径
CHECKPOINT="./model_weights/llama-3-8b-hf-v0.1-tp8-pp1"
TOKENIZER_PATH="./model_from_hf/llama-3-8b-hf/"
```

启动llama3-8B 推理脚本

```bash
bash examples/llama3/generate_llama3_8b_ptd.sh
```

**如果想要使用大模型聊天功能，请下载对应的Instruct/Chat权重，并转换为Megatron格式，配置相关路径，
启动聊天脚本: examples/llama3/generate_llama3_8b_chat_ptd.sh**

LLAMA3-8B-Instruct 权重[下载](https://huggingface.co/unsloth/llama-3-8b-Instruct)

聊天示例如下：
```
You >> 你好
MindSpeed-LLM:
你好！我是你的中文语言模型，很高兴和你交流！有什么想聊的题目或问题，请随时问我！

You >> 介绍一下llama3

MindSpeed-LLM:
LLaMA 3 is a large-scale language model developed by Meta AI, a leading artificial intelligence research organization. It's a significant upgrade to the previous LLaMA models, offering improved performance and capabilities.

Here are some key features and facts about LLaMA 3:

Scale: LLaMA 3 is a massive model, consisting of 13.5 billion parameters, making it one of the largest language models in the world.
Architecture: It's based on the Transformer architecture, which is a type of neural network designed specifically for natural language processing tasks.
Training data: LLaMA 3 was trained on a massive dataset of text from the internet, including books, articles, and websites. This training data allows the model to learn about various topics, styles, and language patterns.
Language understanding: LLaMA 3 is capable of understanding natural language, including nuances, idioms, and figurative language. It can also recognize and generate text in multiple languages.
Conversational capabilities: The model can engage in conversations, answering questions, and responding to prompts. It can also generate text based on a given topic or theme.
Creative writing: LLaMA
You >> 能用中文回答吗

MindSpeed-LLM:

LLaMA 3 是 Meta AI 开发的一种大规模语言模型，是之前 LLama 模型的升级版本，提供了改进的性能和能力。

以下是 LLama 3 的一些关键特点和事实：

规模：LLaMA 3 是一个庞大的模型，具有 13.5 亿参数，世界上最大的语言模型之一。
架构：它基于 Transformer 架构，这是一种专门为自然语言处理任务设计的神经网络。
训练数据：LLaMA 3 在互联网上的大量文本数据上进行了训练，包括书籍、文章和网站。这使得模型能够学习各种主题、风格和语言模式。
语言理解：LLaMA 3 能够理解自然语言，包括 nuance、idiom 和 figurative language。它也能识别和生成多种语言的文本。
对话能力：模型能够参与对话，回答问题和响应提示。它也能基于给定的主题或主题生成文本。
创作写作：LLa
```

## 评估-8B

使用 MMLU数据集评估模型. 数据集下载路径 [这里](https://huggingface.co/datasets/cais/mmlu).
配置llama3-8B 评估脚本: examples/llama3/evaluate_llama3_8b_ptd.sh

```bash
# ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# 修改模型参数路径和词表路径
TOKENIZER_PATH="./model_from_hf/llama-3-8b-hf/"  #词表路径
CHECKPOINT="./model_weights/llama-3-8b-hf-v0.1-tp8-pp1"  #模型路径
# 配置任务和数据集路径
DATA_PATH="./mmlu/data/test/"
TASK="mmlu"
```

启动评估

```bash
bash examples/llama3/evaluate_llama3_8b_ptd.sh
```

评估结果如下

| 数据集 | 总学科数 | 总问题数 | 参考准确率 | NPU准确率 |
| :----: | :------: | :------: | :--------: | :-------: |
|  MMLU  |    57    |  14042  |   0.666   |  0.653  |

# LLAMA3-70B

## 训练

LLAMA3-70B 训练的硬件配置:

| 硬件 |      配置      |
| :--: | :-------------: |
| NPU | 64 x Ascend NPUs |

### 脚本

1. 克隆仓库到本地服务器

    ```shell
    git clone https://gitee.com/ascend/MindSpeed-LLM.git 
    cd MindSpeed-LLM
    git checkout 1.1
    cd ..
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    git checkout core_r0.6.0
    cp -r megatron ../MindSpeed-LLM/
    cd ..
    cd MindSpeed-LLM
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
3. 下载 LLAMA3-70B 的 [预训练权重和词表](https://huggingface.co/v2ray/Llama-3-70B/tree/main)

    ```shell
    #!/bin/bash
    mkdir ./model_from_hf/llama-3-70b-hf/
    cd ./model_from_hf/llama-3-70b-hf/
    wget https://huggingface.co/v2ray/Llama-3-70B/raw/main/config.json
    wget https://huggingface.co/v2ray/Llama-3-70B/raw/main/generation_config.json
    wget https://huggingface.co/v2ray/Llama-3-70B/raw/main/model-00001-of-00030.safetensors
    wget https://huggingface.co/v2ray/Llama-3-70B/raw/main/model-00002-of-00030.safetensors
    wget https://huggingface.co/v2ray/Llama-3-70B/raw/main/model-00003-of-00030.safetensors
    wget https://huggingface.co/v2ray/Llama-3-70B/raw/main/model-00004-of-00030.safetensors
    ...
    wget https://huggingface.co/v2ray/Llama-3-70B/raw/main/model-00030-of-00030.safetensors
    wget https://huggingface.co/v2ray/Llama-3-70B/raw/main/model.safetensors.index.json
    wget https://huggingface.co/v2ray/Llama-3-70B/raw/main/special_tokens_map.json
    wget https://huggingface.co/v2ray/Llama-3-70B/raw/main/tokenizer.json
    wget https://huggingface.co/v2ray/Llama-3-70B/raw/main/tokenizer_config.json
    cd ../../
    ```
4. 权重转换

    4.1 将权重从 huggingface 格式转化为 megatron 格式
    ***（该场景一般用于使能开源的HuggingFace模型在Megatron上进行训练）***

    ```bash
    # 修改 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # 权重格式转换
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader llama2_hf \
        --saver megatron \
        --target-tensor-parallel-size 8 \
        --target-pipeline-parallel-size 8 \
        --load-dir ./model_from_hf/llama-3-70b-hf/ \
        --save-dir ./model_weights/llama-3-70b-hf-v0.1-tp8-pp8/ \
        --tokenizer-model ./model_from_hf/llama-3-70b-hf/tokenizer.json
    ```

    4.2 任意并行切分策略的 Megatron 权重 格式转化为 HuggingFace权重
    ***（该场景一般用于将训练好的megatron模型重新转回HuggingFace格式）***

    ```shell
    # 请按照您的真实环境修改 set_env.sh 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader megatron \
        --saver megatron \
        --save-model-type save_huggingface_llama \
        --load-dir ./model_weights/llama-3-70b-hf-v0.1-tp8-pp8/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --save-dir ./model_from_hf/llama-3-70b-hf/     # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/llama-3-70b-hf/mg2hg/
    ```

    权重转换适用于预训练、微调、推理和评估，根据任务不同调整参数 `target-tensor-parallel-size`和 `target-pipeline-parallel-size`。

5. 预训练

    5.1 准备数据集

    下载 LLaMA3-70B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

    ```shell
    # 下载数据
    cd ./dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    # 处理数据   
    mkdir ./dataset/llama-3-70b-hf/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/llama-3-70b-hf/ \
        --output-prefix ./dataset/llama-3-70b-hf/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF
    ```

    5.2 预训练
    配置llama3-70B 预训练脚本: examples/llama3/pretrain_llama3_70b_ptd.sh
    ```shell
    # 设置 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # 根据实际情况配置词表、数据集、模型参数保存路径
    CKPT_SAVE_DIR="./ckpt/llama-3-70b-hf/"
    TOKENIZER_MODEL="./model_from_hf/llama-3-70b-hf/"  #词表路径
    DATA_PATH="./dataset/llama-3-70b-hf/alpaca_text_document"  #数据集路径
    CKPT_LOAD_DIR="./model_weights/llama-3-70b-hf-v0.1-tp8-pp8/" #权重路径
    ```

    多机运行增加参数--overlap-grad-reduce

    启动 LLaMA3-70B 预训练脚本: examples/llama3/pretrain_llama3_70b_ptd.sh

    ```shell
    bash examples/llama3/pretrain_llama3_70b_ptd.sh
    ```

    **注意**：如果使用多机训练，且没有设置数据共享，需要在训练启动脚本中增加`--no-shared-storage`参数，设置此参数之后将会根据分布式参数判断非主节点是否需要load数据，并检查相应缓存和生成数据。

6. 微调

    6.1 准备微调数据集
    
    下载微调数据集 [这里](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

    ```shell
    # 下载数据集
    mkdir finetune_dataset
    cd ./finetune_dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    
    # 处理微调数据集  
    mkdir ./finetune_dataset/llama-3-70b-hf/
    python ./tools/preprocess_data.py \
        --input ./finetune_dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/llama-3-70b-hf/ \
        --output-prefix ./finetune_dataset/llama-3-70b-hf/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF \
        --handler-name GeneralInstructionHandler \
        --append-eod
    ```

    6.2 全参微调
    
    全参微调的配置脚本基本和预训练脚本一致. *区别是数据集，以及增加训练参数--is-instruction-dataset*
    增加微调参数--finetune，增加预训练权重加载参数--load，使微调从第一步开始。更改为以下参数：

    ```bash
    DATA_PATH="./finetune_dataset/llama-3-70b-hf/alpaca"
    TOKENIZER_PATH="./model_from_hf/llama-3-70b-hf/"
    CKPT_PATH="./ckpt/llama-3-70b-hf/"
        --load ${CKPT_PATH} \
        --finetune \
        --is-instruction-dataset \
        --tokenizer-not-use-fast \
    ```


## 推理-70B

配置llama3-70B 推理脚本: examples/llama3/generate_llama3_70b_ptd.sh

```bash
# 根据您自己的 ascend-toolkit 路径，执行set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
# 修改模型权重路径和词表路径
CHECKPOINT="./model_weights/llama-3-70b-hf-v0.1-tp8-pp1"
TOKENIZER_PATH="./model_from_hf/llama-3-70b-hf/"
```

启动llama3-70B 推理脚本

```bash
bash examples/llama3/generate_llama3_70b_ptd.sh
```

## 评估-70B

使用 MMLU数据集评估模型. 数据集下载路径 [这里](https://huggingface.co/datasets/cais/mmlu).
配置llama3-70B 评估脚本: examples/llama3/evaluate_llama3_70b_ptd.sh

```bash
# ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# 修改模型参数路径和词表路径
TOKENIZER_PATH="./model_from_hf/llama-3-70b-hf/"  #词表路径
CHECKPOINT="./model_weights/llama-3-70b-hf-v0.1-tp8-pp1"  #模型路径
# 配置任务和数据集路径
DATA_PATH="./mmlu/data/test/"
TASK="mmlu"
```

启动评估

```bash
bash examples/llama3/evaluate_llama3_70b_ptd.sh
```

评估结果如下

| 数据集 | 总学科数 | 总问题数 | 参考准确率 | NPU准确率 |
| :----: | :------: | :------: | :--------: | :-------: |
|  MMLU  |    57    |  14042  |   0.795   |  0.783  |