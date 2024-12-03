# Qwen1.5 $\color{black}{\bf\tiny{【社区贡献模型】}}$

<p align="left">
        <b>简体中文</b> |
        <b><a href="README_en.md">English</a> </b> 
</p>

# 目录
- [Qwen1.5-0.5B](#Qwen1.5-0.5b)
  - [训练-0.5B](#训练-0.5b)
    - [脚本-0.5B](#脚本-0.5b)
  - [推理-0.5B](#推理-0.5b)
  - [评估-0.5B](#评估-0.5b)
- [Qwen1.5-1.8B](#Qwen1.5-1.8b)
  - [训练-1.8B](#训练-1.8b)
    - [脚本-1.8B](#脚本-1.8b)
  - [推理-1.8B](#推理-1.8b)
  - [评估-1.8B](#评估-1.8b)
- [Qwen1.5-4B](#Qwen1.5-4b)
  - [训练-4B](#训练-4b)
    - [脚本-4B](#脚本-4b)
  - [推理-4B](#推理-4b)
  - [评估-4B](#评估-4b)
- [Qwen1.5-7B](#qwen15-7b)
  - [训练-7B](#训练-7b)
	- [脚本-7B](#脚本-7b)
  - [推理-7B](#推理-7b)
  - [评估-7B](#评估-7b)
- [Qwen1.5-14B](#qwen15-14b)
  - [训练-14B](#训练-14b)
	- [脚本-14B](#脚本-14b)
  - [推理-14B](#推理-14b)
  - [评估-14B](#评估-14b)
- [Qwen1.5-32B](#qwen15-32b)
  - [训练-32B](#训练-32b)
    - [脚本-32B](#脚本-32b)
  - [推理-32B](#推理-32b)
  - [评估-32B](#评估-32b)
- [Qwen1.5-72B](#qwen15-72b)
  - [训练-72B](#训练-72b)
    - [脚本-72B](#脚本-72b)
  - [推理-72B](#推理-72b)
  - [评估-72B](#评估-72b)
# Qwen1.5-0.5B

## 训练-0.5B
Qwen1.5-0.5B 训练的硬件配置:

| 硬件  |       配置        |
|:---:|:---------------:|
| NPU | 8 x Ascend NPUs |
### 脚本-0.5B

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
   
    # 安装依赖库
    pip install -r requirements.txt 
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

    ```
3. 下载 Qwen1.5-0.5B 的 [预训练权重和词表](https://huggingface.co/Qwen/Qwen1.5-0.5B/tree/main)

    ```shell
    #!/bin/bash
    mkdir ./model_from_hf/qwen15-0.5b-hf/
    cd ./model_from_hf/qwen15-0.5b-hf/
    wget https://huggingface.co/Qwen/Qwen1.5-0.5B/resolve/main/config.json
    wget https://huggingface.co/Qwen/Qwen1.5-0.5B/resolve/main/generation_config.json
    wget https://huggingface.co/Qwen/Qwen1.5-0.5B/resolve/main/merges.txt
    wget https://huggingface.co/Qwen/Qwen1.5-0.5B/resolve/main/model.safetensors
    wget https://huggingface.co/Qwen/Qwen1.5-0.5B/resolve/main/tokenizer.json
    wget https://huggingface.co/Qwen/Qwen1.5-0.5B/resolve/main/tokenizer_config.json
    wget https://huggingface.co/Qwen/Qwen1.5-0.5B/resolve/main/vocab.json
    cd ../../
    ```
4. 权重转换

    4.1 将权重从 huggingface 格式转化为 magatron 格式
    ***（该场景一般用于使能开源的HuggingFace模型在Megatron上进行训练）***

    ```bash
    # 修改 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # 权重格式转换
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader llama2_hf \
        --saver megatron \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --add-qkv-bias \
        --load-dir ./model_from_hf/qwen15-0.5b-hf/ \
        --save-dir ./model_weights/qwen15-0.5b-hf-v0.1-tp1-pp1/ \
        --tokenizer-model ./model_from_hf/qwen15-0.5b-hf/tokenizer.json
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
        --load-dir ./ckpt/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --add-qkv-bias \
        --save-dir ./model_from_hf/qwen15-0.5b-hf/     # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/qwen15-0.5b-hf/mg2hg/
    ```

    权重转换适用于预训练、微调、推理和评估，根据任务不同调整参数 `target-tensor-parallel-size`和 `target-pipeline-parallel-size`。
5. 预训练

    5.1 准备数据集

    下载 Qwen1.5-0.5B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

    ```shell
    # 下载数据
    cd ./dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    # 处理数据   
    mkdir ./dataset/qwen15-0.5b-hf/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/qwen15-0.5b-hf/ \
        --output-prefix ./dataset/qwen15-0.5b-hf/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF
    ```

    5.2 预训练

    ```shell
    # 设置 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # 根据实际情况配置词表、数据集、模型参数保存路径
    CKPT_SAVE_DIR="./ckpt/qwen15-0.5b-hf/"
    TOKENIZER_PATH="./model_from_hf/qwen15-0.5b-hf"  #词表路径
    DATA_PATH="./dataset/qwen15-0.5b-hf/alpaca_text_document"  #数据集路径
    CKPT_LOAD_DIR="./model_weights/qwen15-0.5b-hf-v0.1-tp1-pp1"
    ```
    启动 Qwen1.5-0.5B 预训练脚本: examples/qwen15/pretrain_qwen15_0point5b_ptd.sh

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
    mkdir ./finetune_dataset/qwen15-0.5b-hf/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/qwen15-0.5b-hf/ \
        --output-prefix ./finetune_dataset/qwen15-0.5b-hf/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF \
        --handler-name GeneralInstructionHandler \
        --append-eod
    ```

    6.2 全参微调
    全参微调的配置脚本基本和预训练脚本一致. *区别是数据集，以及增加训练参数--is-instruction-dataset*

    增加微调参数--finetune，参数如下：

    ```bash
    DATA_PATH="./finetune_dataset/qwen15-0.5b-hf/alpaca"
    TOKENIZER_PATH="./model_from_hf/qwen15-0.5b-hf/"
    CKPT_PATH="./model_weights/qwen15-0.5b-hf-v0.1-tp1-pp1/"
        --load ${CKPT_PATH} \
        --finetune \
        --is-instruction-dataset \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path ${TOKENIZER_PATH} \
        --tokenizer-not-use-fast \
    ```


## 推理-0.5B

配置Qwen1.5-0.5B 推理脚本: examples/qwen15/generate_qwen15_0point5b_ptd.sh

```bash
# 根据您自己的 ascend-toolkit 路径，执行set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
# 修改模型权重路径和词表路径
CHECKPOINT="./model_weights/qwen15-0.5b-hf-v0.1-tp1-pp1"
TOKENIZER_PATH="./model_from_hf/qwen15-0.5b-hf/"
```

启动Qwen1.5-0.5B 推理脚本

```bash
bash examples/qwen15/generate_qwen15_0point5b_ptd.sh
```

推理示例如下：

![Inference](../../sources/images/qwen15/qwen15_0point5b_inference.png)

## 评估-0.5B

使用 MMLU数据集评估模型. 数据集下载路径 [这里](https://huggingface.co/datasets/cais/mmlu).
配置Qwen1.5-0.5B 评估脚本: examples/qwen15/evaluate_qwen1.5_0point5b_ptd.sh

```bash
# ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# 修改模型参数路径和词表路径
TOKENIZER_PATH="./model_from_hf/qwen15-0.5b-hf/"  #词表路径
CHECKPOINT="./model_weights/qwen15-0.5b-hf-v0.1-tp1-pp1"  #模型路径
# 配置任务和数据集路径
DATA_PATH="./mmlu/data/test/"
TASK="mmlu"
```

启动评估

```bash
bash examples/qwen15/evaluate_qwen15_0point5b_ptd.sh
```

评估结果如下

| 数据集 | 总学科数 | 总问题数 | 参考准确率 | NPU准确率 |
| :----: | :------: | :------: |:-----:|:------:|
|  MMLU  |    57    |  14042  | 0.318 | 0.318  |
# Qwen1.5-1.8B

## 训练-1.8B
Qwen1.5-1.8B 训练的硬件配置:

| 硬件  |       配置        |
|:---:|:---------------:|
| NPU | 8 x Ascend NPUs |
### 脚本-1.8B

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
    # 安装依赖库
    pip install -r requirements.txt 
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

    ```
3. 下载 Qwen1.5-1.8B 的 [预训练权重和词表](https://huggingface.co/Qwen/Qwen1.5-1.8B/tree/main)

    ```shell
    #!/bin/bash
    mkdir ./model_from_hf/qwen15-1.8b-hf/
    cd ./model_from_hf/qwen15-1.8b-hf/
    wget https://huggingface.co/Qwen/Qwen1.5-1.8B/resolve/main/config.json
    wget https://huggingface.co/Qwen/Qwen1.5-1.8B/resolve/main/generation_config.json
    wget https://huggingface.co/Qwen/Qwen1.5-1.8B/resolve/main/merges.txt
    wget https://huggingface.co/Qwen/Qwen1.5-1.8B/resolve/main/model.safetensors
    wget https://huggingface.co/Qwen/Qwen1.5-1.8B/resolve/main/tokenizer.json
    wget https://huggingface.co/Qwen/Qwen1.5-1.8B/resolve/main/tokenizer_config.json
    wget https://huggingface.co/Qwen/Qwen1.5-1.8B/resolve/main/vocab.json
    cd ../../
    ```
4. 权重转换

    4.1 将权重从 huggingface 格式转化为 magatron 格式
    ***（该场景一般用于使能开源的HuggingFace模型在Megatron上进行训练）***

    ```bash
    # 修改 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # 权重格式转换
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader llama2_hf \
        --saver megatron \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --params-dtype bf16 \
        --add-qkv-bias \
        --load-dir ./model_from_hf/qwen15-1.8b-hf/ \
        --save-dir ./model_weights/qwen15-1.8b-hf-v0.1-tp1-pp1/ \
        --tokenizer-model ./model_from_hf/qwen15-1.8b-hf/tokenizer.json
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
        --load-dir ./ckpt/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --add-qkv-bias \
        --save-dir ./model_from_hf/qwen15-1.8b-hf/     # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/qwen15-1.8b-hf/mg2hg/
    ```

    权重转换适用于预训练、微调、推理和评估，根据任务不同调整参数 `target-tensor-parallel-size`和 `target-pipeline-parallel-size`。
5. 预训练

    5.1 准备数据集

    下载 Qwen1.5-1.8B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

    ```shell
    # 下载数据
    cd ./dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    # 处理数据   
    mkdir ./dataset/qwen15-1.8b-hf/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/qwen15-1.8b-hf/ \
        --output-prefix ./dataset/qwen15-1.8b-hf/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF
    ```

    5.2 预训练

    ```shell
    # 设置 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # 根据实际情况配置词表、数据集、模型参数保存路径
    CKPT_SAVE_DIR="./ckpt/qwen15-1.8b-hf/"
    TOKENIZER_PATH="./model_from_hf/qwen15-1.8b-hf"  #词表路径
    DATA_PATH="./dataset/qwen15-1.8b-hf/alpaca_text_document"  #数据集路径
    CKPT_LOAD_DIR="./model_weights/qwen15-1.8b-hf-v0.1-tp1-pp1"
    ```
    启动 Qwen1.5-1.8B 预训练脚本: examples/qwen15/pretrain_qwen15_1point8b_ptd.sh

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
    mkdir ./finetune_dataset/qwen15-1.8b-hf/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/qwen15-1.8b-hf/ \
        --output-prefix ./finetune_dataset/qwen15-1.8b-hf/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF \
        --handler-name GeneralInstructionHandler \
        --append-eod
    ```

    6.2 全参微调
    全参微调的配置脚本基本和预训练脚本一致. *区别是数据集，以及增加训练参数--is-instruction-dataset*

    增加微调参数--finetune，参数如下：

    ```bash
    DATA_PATH="./finetune_dataset/qwen15-1.8b-hf/alpaca"
    TOKENIZER_PATH="./model_from_hf/qwen15-1.8b-hf/"
    CKPT_PATH="./model_weights/qwen15-1.8b-hf-v0.1-tp1-pp1/"
        --load ${CKPT_PATH} \
        --finetune \
        --is-instruction-dataset \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path ${TOKENIZER_PATH} \
        --tokenizer-not-use-fast \
    ```


## 推理-1.8B

配置Qwen1.5-1.8B 推理脚本: examples/qwen15/generate_qwen15_1point8b_ptd.sh

```bash
# 根据您自己的 ascend-toolkit 路径，执行set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
# 修改模型权重路径和词表路径
CHECKPOINT="./model_weights/qwen15-1.8b-hf-v0.1-tp1-pp1"
TOKENIZER_PATH="./model_from_hf/qwen15-1.8b-hf/"
```

启动Qwen1.5-1.8B 推理脚本

```bash
bash examples/qwen15/generate_qwen15_1point8b_ptd.sh
```

推理示例如下：

![Inference](../../sources/images/qwen15/qwen15_1point8b_inference.png)

## 评估-1.8B

使用 MMLU数据集评估模型. 数据集下载路径 [这里](https://huggingface.co/datasets/cais/mmlu).
配置Qwen1.5-1.8B 评估脚本: examples/qwen15/evaluate_qwen15_1point8b_ptd.sh

```bash
# ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# 修改模型参数路径和词表路径
TOKENIZER_PATH="./model_from_hf/qwen15-1.8b-hf/"  #词表路径
CHECKPOINT="./model_weights/qwen15-1.8b-hf-v0.1-tp1-pp1"  #模型路径
# 配置任务和数据集路径
DATA_PATH="./mmlu/data/test/"
TASK="mmlu"
```

启动评估

```bash
bash examples/qwen15/evaluate_qwen15_1point8b_ptd.sh
```

评估结果如下

| 数据集 | 总学科数 | 总问题数 |                      参考准确率                       | NPU准确率 |
| :----: | :------: | :------: |:------------------------------------------------:|:------:|
|  MMLU  |    57    |  14042  | [0.468](https://qwenlm.github.io/zh/blog/qwen1.5/) | 0.462  |

# Qwen1.5-4B

## 训练-4B
Qwen1.5-4B 训练的硬件配置:

| 硬件  |       配置        |
|:---:|:---------------:|
| NPU | 8 x Ascend NPUs |
### 脚本-4B

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
    # 安装依赖库
    pip install -r requirements.txt
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
    ```
3. 下载 Qwen1.5-4B 的 [预训练权重和词表](https://huggingface.co/Qwen/Qwen1.5-4B/tree/main)

    ```shell
    #!/bin/bash
    mkdir ./model_from_hf/qwen15-4b-hf/
    cd ./model_from_hf/qwen15-4b-hf/
    wget https://huggingface.co/Qwen/Qwen1.5-4B/resolve/main/config.json
    wget https://huggingface.co/Qwen/Qwen1.5-4B/resolve/main/generation_config.json
    wget https://huggingface.co/Qwen/Qwen1.5-4B/resolve/main/merges.txt
    wget https://huggingface.co/Qwen/Qwen1.5-4B/resolve/main/model-00001-of-00002.safetensors
    wget https://huggingface.co/Qwen/Qwen1.5-4B/resolve/main/model-00002-of-00002.safetensors
    wget https://huggingface.co/Qwen/Qwen1.5-4B/resolve/main/model.safetensors.index.json
    wget https://huggingface.co/Qwen/Qwen1.5-4B/resolve/main/tokenizer.json
    wget https://huggingface.co/Qwen/Qwen1.5-4B/resolve/main/tokenizer_config.json
    wget https://huggingface.co/Qwen/Qwen1.5-4B/resolve/main/vocab.json
    cd ../../
    ```
4. 权重转换

    4.1 将权重从 huggingface 格式转化为 magatron 格式
    ***（该场景一般用于使能开源的HuggingFace模型在Megatron上进行训练）***

    ```bash
    # 修改 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # 权重格式转换
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader llama2_hf \
        --saver megatron \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 2 \
        --params-dtype bf16 \
        --add-qkv-bias \
        --load-dir ./model_from_hf/qwen15-4b-hf/ \
        --save-dir ./model_weights/qwen15-4b-hf-v0.1-tp1-pp2/ \
        --tokenizer-model ./model_from_hf/qwen15-4b-hf/tokenizer.json
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
        --load-dir ./ckpt/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --add-qkv-bias \
        --save-dir ./model_from_hf/qwen15-4b-hf/     # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/qwen15-4b-hf/mg2hg/
    ```

    权重转换适用于预训练、微调、推理和评估，根据任务不同调整参数 `target-tensor-parallel-size`和 `target-pipeline-parallel-size`。
5. 预训练

    5.1 准备数据集

    下载 Qwen1.5-4B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

    ```shell
    # 下载数据
    cd ./dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    # 处理数据   
    mkdir ./dataset/qwen15-4b-hf/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/qwen15-4b-hf/ \
        --output-prefix ./dataset/qwen15-4b-hf/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF
    ```

    5.2 预训练

    ```shell
    # 设置 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # 根据实际情况配置词表、数据集、模型参数保存路径
    CKPT_SAVE_DIR="./ckpt/qwen15-4b-hf/"
    TOKENIZER_PATH="./model_from_hf/qwen15-4b-hf"  #词表路径
    DATA_PATH="./dataset/qwen15-4b-hf/alpaca_text_document"  #数据集路径
    CKPT_LOAD_DIR="./model_weights/qwen15-4b-hf-v0.1-tp1-pp2"
    ```
    启动 Qwen1.5-4B 预训练脚本: examples/qwen15/pretrain_qwen15_4b_ptd.sh

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
    mkdir ./finetune_dataset/qwen15-4b-hf/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/qwen15-4b-hf/ \
        --output-prefix ./finetune_dataset/qwen15-4b-hf/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF \
        --handler-name GeneralInstructionHandler \
        --append-eod
    ```

    6.2 全参微调
    全参微调的配置脚本基本和预训练脚本一致. *区别是数据集，以及增加训练参数--is-instruction-dataset*

    增加微调参数--finetune，参数如下：

    ```bash
    DATA_PATH="./finetune_dataset/qwen15-4b-hf/alpaca"
    TOKENIZER_PATH="./model_from_hf/qwen15-4b-hf/"
    CKPT_PATH="./model_weights/qwen15-4b-hf-v0.1-tp1-pp2/"
        --load ${CKPT_PATH} \
        --finetune \
        --is-instruction-dataset \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path ${TOKENIZER_PATH} \
        --tokenizer-not-use-fast \
    ```


## 推理-4B

配置Qwen1.5-4B 推理脚本: examples/qwen15/generate_qwen15_4b_ptd.sh

```bash
# 根据您自己的 ascend-toolkit 路径，执行set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
# 修改模型权重路径和词表路径
CHECKPOINT="./model_weights/qwen15-4b-hf-v0.1-tp1-pp2"
TOKENIZER_PATH="./model_from_hf/qwen15-4b-hf/"
```

启动Qwen1.5-4B 推理脚本

```bash
bash examples/qwen15/generate_qwen15_4b_ptd.sh
```

推理示例如下：

![Inference](../../sources/images/qwen15/qwen1.5_4b_inference.png)

## 评估-4B

使用 MMLU数据集评估模型. 数据集下载路径 [这里](https://huggingface.co/datasets/cais/mmlu).
配置Qwen1.5-4B 评估脚本: examples/qwen15/evaluate_qwen15_4b_ptd.sh

```bash
# ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# 修改模型参数路径和词表路径
TOKENIZER_PATH="./model_from_hf/qwen15-4b-hf/"  #词表路径
CHECKPOINT="./model_weights/qwen15-4b-hf-v0.1-tp1-pp2"  #模型路径
# 配置任务和数据集路径
DATA_PATH="./mmlu/data/test/"
TASK="mmlu"
```

启动评估

```bash
bash examples/qwen15/evaluate_qwen15_4b_ptd.sh
```

评估结果如下

| 数据集 | 总学科数 | 总问题数 |                       参考准确率                       | NPU准确率 |
| :----: | :------: | :------: |:-------------------------------------------------:| :-------: |
|  MMLU  |    57    |  14042  | [0.561](https://qwenlm.github.io/zh/blog/qwen1.5) |  0.550  |


# Qwen1.5-7B

## 训练-7B

Qwen1.5-7B 训练的硬件配置:

| 硬件  |       配置        |
|:---:|:---------------:|
| NPU | 8 x Ascend NPUs |

### 脚本-7B

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
   
   # 安装其余依赖库
   pip install -r requirements.txt
   
   # 安装 torch 和 torch_npu
   pip install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl
   pip install torch_npu-2.1.0*-cp38-cp38m-linux_aarch64.whl
   pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

   # 安装加速库
   git clone https://gitee.com/ascend/MindSpeed.git
   cd MindSpeed
   git checkout 2b0edd2
   pip install -r requirements.txt
   pip install -e .
   cd ..
   ```
3. 下载 Qwen1.5-7B 的 [预训练权重和词表](https://huggingface.co/Qwen/Qwen1.5-7B/tree/main)

   ```bash
   mkdir ./model_from_hf/Qwen1.5-7B/
   cd ./model_from_hf/Qwen1.5-7B/
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/config.json
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/generation_config.json
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/merges.txt
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/model-00001-of-00004.safetensors
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/model-00002-of-00004.safetensors
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/model-00003-of-00004.safetensors
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/model-00004-of-00004.safetensors
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/model.safetensors.index.json
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/special_tokens_map.json
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/tokenizer.json
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/tokenizer_config.json
   wget https://huggingface.co/Qwen/Qwen1.5-7B/resolve/main/vocab.json
   cd ../../
   ```
4. 权重转换

   将权重从 huggingface 格式转化为 megatron 格式
   ***（该场景一般用于使能开源的HuggingFace模型在Megatron上进行训练）***

   ```shell
   # 修改 ascend-toolkit 路径
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   
   python tools/checkpoint/convert_ckpt.py \
       --model-type GPT \
       --loader llama2_hf \
       --saver megatron \
       --target-tensor-parallel-size 8 \
       --target-pipeline-parallel-size 1 \
       --make-vocab-size-divisible-by 16 \
       --load-dir ./model_from_hf/Qwen1.5-7B/ \
       --save-dir ./model_weights/Qwen1.5-7B-v0.1-tp8-pp1/ \
       --tokenizer-model ./model_from_hf/Qwen1.5-7B/tokenizer.json \
       --add-qkv-bias \
       --params-dtype bf16 
   ```

   任意并行切分策略的Megatron权重 格式转化为 HuggingFace权重
   ***（该场景一般用于将训练好的megatron模型重新转回HuggingFace格式）***

   ```bash
   # 请按照您的真实环境修改 set_env.sh 路径
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   python tools/checkpoint/convert_ckpt.py \
       --model-type GPT \
       --loader megatron \
       --saver megatron \
       --save-model-type save_huggingface_llama \
       --target-tensor-parallel-size 1 \
       --target-pipeline-parallel-size 1 \
       --add-qkv-bias \
       --load-dir ./model_weights/Qwen1.5-7B-v0.1-tp8-pp1 \
       --save-dir ./model_from_hf/Qwen1.5-7B 		# 需要填入原始HF模型路径，新权重会存于./model_from_hf/Qwen1.5-7B/mg2hg/
   ```
   
5. 预训练

   5.1 准备数据集

   下载Qwen1.5-7B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

   ```shell
   # 下载数据
   cd ./dataset
   wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
   cd ..
   
   # 处理数据   
   mkdir ./dataset/Qwen1.5-7B/
   python ./tools/preprocess_data.py \
       --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
       --tokenizer-name-or-path ./model_from_hf/Qwen1.5-7B \
       --output-prefix ./dataset/Qwen1.5-7B/alpaca \
       --tokenizer-type PretrainedFromHF \
       --seq-length 8192 \
       --workers 4 \
       --log-interval 1000
   ```
   
   5.2 预训练

   配置Qwen1.5-7B 预训练脚本: examples/qwen15/pretrain_qwen15_7b_ptd.sh

   ```shell
   # 设置 ascend-toolkit 路径
   source /usr/local/Ascend/ascend-toolkit/set_env.sh 

   # 根据实际情况配置词表、数据集、模型参数保存路径
   CKPT_SAVE_DIR="./ckpt/Qwen1.5-7B"
   TOKENIZER_PATH="./model_from_hf/Qwen1.5-7B"  #词表路径
   DATA_PATH="./dataset/Qwen1.5-7B/alpaca_text_document"  #数据集路径
   CKPT_LOAD_DIR="./model_weights/Qwen1.5-7B-v0.1-tp8-pp1"
   ```
   多机运行增加参数 `--overlap-grad-reduce`。

   启动 Qwen1.5-7B 预训练脚本: examples/qwen15/pretrain_qwen15_7b_ptd.sh

   ```shell
    bash examples/qwen15/pretrain_qwen15_7b_ptd.sh
   ```
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
   mkdir ./finetune_dataset/Qwen1.5-7B/
   python ./tools/preprocess_data.py \
       --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
       --tokenizer-name-or-path ./model_from_hf/Qwen1.5-7B/ \
       --output-prefix ./finetune_dataset/Qwen1.5-7B/alpaca \
       --workers 4 \
       --log-interval 1000 \
       --tokenizer-type PretrainedFromHF \
       --handler-name GeneralInstructionHandler \
       --append-eod
   ```
   6.2 全参微调 全参微调的配置脚本基本和预训练脚本一致。

   *区别是数据集，以及增加训练参数`--is-instruction-dataset`，增加微调参数`--finetune`，增加预训练权重加载参数`--load`
   ，使微调从第一步开始。*

   修改如下：

   ```bash
   CKPT_LOAD_DIR="./model_weights/Qwen1.5-7B-v0.1-tp8-pp1/"
   CKPT_SAVE_DIR="./ckpt/Qwen1.5-7B/"
   DATA_PATH="./finetune_dataset/Qwen1.5-7B/alpaca"
   TOKENIZER_PATH="./model_from_hf/Qwen1.5-7B/"

   --load ${CKPT_LOAD_DIR} \
   --finetune \
   --is-instruction-dataset \
   --tokenizer-not-use-fast \
   ```


## 推理-7B

配置 Qwen1.5-7B 推理脚本：examples/qwen15/generate_qwen15_7b_ptd.sh

```bash
# ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 修改模型权重路径和词表路径
CHECKPOINT="./model_weights/Qwen1.5-7B-v0.1-tp8-pp1"
TOKENIZER_PATH="./model_from_hf/Qwen1.5-7B"
```

启动Qwen1.5-7B推理脚本

```bash
bash examples/qwen15/generate_qwen15_7b_ptd.sh
```

推理示例如下：

![Inference](../../sources/images/qwen15/qwen1.5_7b_inference.png)

## 评估-7B

使用[CEval数据集](https://huggingface.co/datasets/ceval/ceval-exam)
和[MMLU数据集](https://huggingface.co/datasets/cais/mmlu)评估模型.

配置Qwen1.5-7B评估脚本: examples/qwen15/evaluate_qwen15_7b_ptd.sh

```bash
# ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# 修改模型参数路径和词表路径
TOKENIZER_PATH="./model_from_hf/Qwen1.5-7B/"  #词表路径
CHECKPOINT="./model_weights/Qwen1.5-7B-v0.1-tp8-pp1/"  #模型路径

# 配置任务和数据集路径
DATA_PATH="./mmlu/data/test/"  # ceval任务配置为 "./ceval/val/"
TASK="mmlu"  # ceval任务配置为 "ceval"
```

启动评估

```bash
bash examples/qwen15/evaluate_qwen15_7b_ptd.sh
```

|  数据集  | 总学科数 | 总问题数  |                        参考准确率                         | NPU准确率 |
|:-----:|:----:|:-----:|:----------------------------------------------------:|:------:|
| MMLU  |  57  | 14042 |   [61.0](https://qwenlm.github.io/zh/blog/qwen1.5)   | 0.603  |


# Qwen1.5-14B

## 训练-14B

Qwen1.5-14B 训练的硬件配置:

| 硬件  |       配置        |
|:---:|:---------------:|
| NPU | 8 x Ascend NPUs |

### 脚本-14B

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

   # 安装依赖库
   pip install -r requirements.txt
   
   # 安装 torch 和 torch_npu
   pip install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl
   pip install torch_npu-2.1.0*-cp38-cp38m-linux_aarch64.whl
   pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

   # 安装加速库
   git clone https://gitee.com/ascend/MindSpeed.git
   cd MindSpeed
   git checkout 2b0edd2
   pip install -r requirements.txt
   pip install -e .
   cd ..
   ```
3. 下载 Qwen1.5-14B 的 [预训练权重和词表](https://huggingface.co/Qwen/Qwen1.5-14B/tree/main)

   ```bash
   mkdir ./model_from_hf/Qwen1.5-14B/
   cd ./model_from_hf/Qwen1.5-14B/
   wget https://huggingface.co/Qwen/Qwen1.5-14B/blob/main/config.json
   wget https://huggingface.co/Qwen/Qwen1.5-14B/blob/main/generation_config.json
   wget https://huggingface.co/Qwen/Qwen1.5-14B/blob/main/merges.txt
   wget https://huggingface.co/Qwen/Qwen1.5-14B/blob/main/model.safetensors.index.json
   wget https://huggingface.co/Qwen/Qwen1.5-14B/blob/main/special_tokens_map.json
   wget https://huggingface.co/Qwen/Qwen1.5-14B/blob/main/tokenizer.json
   wget https://huggingface.co/Qwen/Qwen1.5-14B/blob/main/tokenizer_config.json
   wget https://huggingface.co/Qwen/Qwen1.5-14B/blob/main/vocab.json
   wget https://huggingface.co/Qwen/Qwen1.5-14B/blob/main/model-00001-of-00008.safetensors
   ...
   cd ../../
   ```
4. 权重转换

   将权重从 huggingface 格式转化为 megatron 格式
   ***（该场景一般用于使能开源的HuggingFace模型在Megatron上进行训练）***

   ```shell
   # 修改 ascend-toolkit 路径
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   
   python tools/checkpoint/convert_ckpt.py \
       --model-type GPT \
       --loader llama2_hf \
       --saver megatron \
       --target-tensor-parallel-size 4 \
       --target-pipeline-parallel-size 2 \
       --make-vocab-size-divisible-by 16 \
       --load-dir ./model_from_hf/Qwen1.5-14B/ \
       --save-dir ./model_weights/Qwen1.5-14B-v0.1-tp4-pp2/ \
       --tokenizer-model ./model_from_hf/Qwen1.5-14B/tokenizer.json \
       --add-qkv-bias \
       --params-dtype bf16 
   ```

   任意并行切分策略的Megatron权重 格式转化为 HuggingFace权重
   ***（该场景一般用于将训练好的megatron模型重新转回HuggingFace格式）***

   ```bash
   # 请按照您的真实环境修改 set_env.sh 路径
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   python tools/checkpoint/convert_ckpt.py \
       --model-type GPT \
       --loader megatron \
       --saver megatron \
       --save-model-type save_huggingface_llama \
       --target-tensor-parallel-size 1 \
       --target-pipeline-parallel-size 1 \
       --add-qkv-bias \
       --load-dir ./model_weights/Qwen1.5-14B-v0.1-tp4-pp2 \
       --save-dir ./model_from_hf/Qwen1.5-14B 		# 需要填入原始HF模型路径，新权重会存于./model_from_hf/Qwen1.5-14B/mg2hg/
   ```
   
5. 预训练

   5.1 准备数据集

   下载Qwen1.5-14B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

   ```shell
   # 下载数据
   cd ./dataset
   wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
   cd ..
   
   # 处理数据   
   mkdir ./dataset/Qwen1.5-14B/
   python ./tools/preprocess_data.py \
       --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
       --tokenizer-name-or-path ./model_from_hf/Qwen1.5-14B \
       --output-prefix ./dataset/Qwen1.5-14B/alpaca \
       --tokenizer-type PretrainedFromHF \
       --seq-length 8192 \
       --workers 4 \
       --log-interval 1000
   ```
   
   5.2 预训练

   配置Qwen1.5-14B 预训练脚本: examples/qwen15/pretrain_qwen15_14b_ptd.sh

   ```shell
   # 设置 ascend-toolkit 路径
   source /usr/local/Ascend/ascend-toolkit/set_env.sh 

   # 根据实际情况配置词表、数据集、模型参数保存路径
   CKPT_SAVE_DIR="./ckpt/Qwen1.5-14B"
   TOKENIZER_PATH="./model_from_hf/Qwen1.5-14B"  #词表路径
   DATA_PATH="./dataset/Qwen1.5-14B/alpaca_text_document"  #数据集路径
   CKPT_LOAD_DIR="./model_weights/Qwen1.5-14B-v0.1-tp4-pp2"
   ```
   多机运行增加参数 `--overlap-grad-reduce`。

   启动 Qwen1.5-14B 预训练脚本: examples/qwen15/pretrain_qwen15_14b_ptd.sh

   ```shell
    bash examples/qwen15/pretrain_qwen15_14b_ptd.sh
   ```
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
   mkdir ./finetune_dataset/Qwen1.5-14B/
   python ./tools/preprocess_data.py \
       --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
       --tokenizer-name-or-path ./model_from_hf/Qwen1.5-14B/ \
       --output-prefix ./finetune_dataset/Qwen1.5-14B/alpaca \
       --workers 4 \
       --log-interval 1000 \
       --tokenizer-type PretrainedFromHF \
       --handler-name GeneralInstructionHandler \
       --append-eod
   ```
   6.2 全参微调 全参微调的配置脚本基本和预训练脚本一致。

   *区别是数据集，以及增加训练参数`--is-instruction-dataset`，增加微调参数`--finetune`，增加预训练权重加载参数`--load`
   ，使微调从第一步开始}`。*

   修改如下：

   ```bash
   CKPT_LOAD_DIR="./model_weights/Qwen1.5-14B-v0.1-tp4-pp2/"
   CKPT_SAVE_DIR="./ckpt/Qwen1.5-14B/"
   DATA_PATH="./finetune_dataset/Qwen1.5-14B/alpaca"
   TOKENIZER_PATH="./model_from_hf/Qwen1.5-14B/"

   --load ${CKPT_LOAD_DIR} \
   --finetune \
   --is-instruction-dataset \
   --tokenizer-not-use-fast \
   ```


## 推理-14B

配置 Qwen1.5-14B 推理脚本：examples/qwen15/generate_qwen15_14b_ptd.sh

```bash
# ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 修改模型权重路径和词表路径
CHECKPOINT="./model_weights/Qwen1.5-14B-v0.1-tp4-pp2"
TOKENIZER_PATH="./model_from_hf/Qwen1.5-14B"
```

启动Qwen1.5-14B推理脚本

```bash
bash examples/qwen15/generate_qwen15_14b_ptd.sh
```

推理示例如下：

![Inference](../../sources/images/qwen15/qwen1.5_14b_inference.png)

## 评估-14B

使用[CEval数据集](https://huggingface.co/datasets/ceval/ceval-exam)
和[MMLU数据集](https://huggingface.co/datasets/cais/mmlu)评估模型.

配置Qwen1.5-14B评估脚本: examples/qwen15/evaluate_qwen15_14b_ptd.sh

```bash
# ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# 修改模型参数路径和词表路径
TOKENIZER_PATH="./model_from_hf/Qwen1.5-14B/"  #词表路径
CHECKPOINT="./model_weights/Qwen1.5-14B-v0.1-tp4-pp2/"  #模型路径

# 配置任务和数据集路径
DATA_PATH="./mmlu/data/test/"  # ceval任务配置为 "./ceval/val/"
TASK="mmlu"  # ceval任务配置为 "ceval"
```

启动评估

```bash
bash examples/qwen15/evaluate_qwen15_14b_ptd.sh
```

|  数据集  | 总学科数 | 总问题数  |                      参考准确率                       | NPU准确率 |
|:-----:|:----:|:-----:|:------------------------------------------------:|:------:|
| MMLU  |  57  | 14042 | [67.6](https://qwenlm.github.io/zh/blog/qwen1.5) |  67.3  |

# Qwen1.5-32B

## 训练-32B
| 硬件  | 序列长度 |        配置        |
|:---:|:----:|:----------------:|
| NPU |  8k  | 32 x Ascend NPUs |

### 脚本-32B
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

    # 安装依赖库
   pip install -r requirements.txt
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

   ```
   
3. 下载 Qwen1.5-32B 的 [预训练权重和词表](https://huggingface.co/Qwen/Qwen1.5-32B/tree/main)
   ```bash
   mkdir ./model_from_hf/Qwen1.5-32B/
   cd ./model_from_hf/Qwen1.5-32B/
   wget https://huggingface.co/Qwen/Qwen1.5-32B/blob/main/config.json
   wget https://huggingface.co/Qwen/Qwen1.5-32B/blob/main/generation_config.json
   wget https://huggingface.co/Qwen/Qwen1.5-32B/blob/main/merges.txt
   wget https://huggingface.co/Qwen/Qwen1.5-32B/blob/main/model-00001-of-00017.safetensors
   wget https://huggingface.co/Qwen/Qwen1.5-32B/blob/main/model-00002-of-00017.safetensors
   wget https://huggingface.co/Qwen/Qwen1.5-32B/blob/main/model-00003-of-00017.safetensors
   wget https://huggingface.co/Qwen/Qwen1.5-32B/blob/main/model-00004-of-00017.safetensors
   wget https://huggingface.co/Qwen/Qwen1.5-32B/blob/main/model-00005-of-00017.safetensors
   wget https://huggingface.co/Qwen/Qwen1.5-32B/blob/main/model-00006-of-00017.safetensors
   wget https://huggingface.co/Qwen/Qwen1.5-32B/blob/main/model-00007-of-00017.safetensors
   wget https://huggingface.co/Qwen/Qwen1.5-32B/blob/main/model-00008-of-00017.safetensors
   wget https://huggingface.co/Qwen/Qwen1.5-32B/blob/main/model-00009-of-00017.safetensors
   wget https://huggingface.co/Qwen/Qwen1.5-32B/blob/main/model-00010-of-00017.safetensors
   wget https://huggingface.co/Qwen/Qwen1.5-32B/blob/main/model-00011-of-00017.safetensors
   wget https://huggingface.co/Qwen/Qwen1.5-32B/blob/main/model-00012-of-00017.safetensors
   wget https://huggingface.co/Qwen/Qwen1.5-32B/blob/main/model-00013-of-00017.safetensors
   wget https://huggingface.co/Qwen/Qwen1.5-32B/blob/main/model-00014-of-00017.safetensors
   wget https://huggingface.co/Qwen/Qwen1.5-32B/blob/main/model-00015-of-00017.safetensors
   wget https://huggingface.co/Qwen/Qwen1.5-32B/blob/main/model-00016-of-00017.safetensors
   wget https://huggingface.co/Qwen/Qwen1.5-32B/blob/main/model-00017-of-00017.safetensors
   wget https://huggingface.co/Qwen/Qwen1.5-32B/blob/main/model.safetensors.index.json
   wget https://huggingface.co/Qwen/Qwen1.5-32B/blob/main/tokenizer.json
   wget https://huggingface.co/Qwen/Qwen1.5-32B/blob/main/tokenizer_config.json
   wget https://huggingface.co/Qwen/Qwen1.5-32B/blob/main/vocab.json
   cd ../../
   ```
4. 权重转换

   4.1 将权重从 huggingface 格式转化为 megatron 格式
   ***（该场景一般用于使能开源的HuggingFace模型在Megatron上进行训练）***
   ```bash
   # 修改 ascend-toolkit 路径
   source /usr/local/Ascend/ascend-toolkit/set_env.sh

   python tools/checkpoint/convert_ckpt.py \
       --model-type GPT \
       --loader llama2_hf \
       --saver megatron \
       --target-tensor-parallel-size 8 \
       --target-pipeline-parallel-size 4 \
       --num-layers-per-virtual-pipeline-stage 2 \
       --params-dtype bf16 \
       --load-dir ./model_from_hf/Qwen1.5-32B/ \
       --save-dir ./model_weights/Qwen1.5-32B-v0.1-tp8-pp4-vpp2/ \
       --tokenizer-model ./model_from_hf/Qwen1.5-32B/tokenizer.json \
       --add-qkv-bias \
       --max-queue-size 70
   ```
   **注意**: --num-layers-per-virtual-pipeline-stage参数只有在--target-pipeline-parallel-size > 2时才会生效，如果target-pipeline-parallel-size不满足>2的条件，请在脚本中删除--num-layers-per-virtual-pipeline-stage参数

   4.2 任意并行切分策略的Megatron权重 格式转化为 HuggingFace权重
   ***（该场景一般用于将训练好的megatron模型重新转回HuggingFace格式）***

   ```shell
   # 请按照您的真实环境修改 set_env.sh 路径
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   
   python tools/checkpoint/convert_ckpt.py \
       --model-type GPT \
       --loader megatron \
       --saver megatron \
       --save-model-type save_huggingface_llama \
       --load-dir ./model_weights/Qwen1.5-32B-v0.1-tp8-pp4-vpp2/ \
       --target-tensor-parallel-size 1 \
       --target-pipeline-parallel-size 1 \
       --num-layers-per-virtual-pipeline-stage 2 \
       --add-qkv-bias \
       --save-dir ./model_from_hf/Qwen1.5-32B/     # 需要填入原始HF模型路径，新权重会存于./model_from_hf/Qwen1.5-32B/mg2hg/
   ```
   权重转换适用于预训练、微调、推理和评估，根据任务不同调整参数 `target-tensor-parallel-size`和 `target-pipeline-parallel-size`。
5. 预训练

    5.1 准备数据集
    下载 Qwen1.5-32B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)
     ```shell
     # 下载数据
     cd ./dataset
     wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
     cd ..
     # 处理数据   
     mkdir ./dataset/qwen1.5-32B-hf/
     python ./tools/preprocess_data.py \
         --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
         --tokenizer-name-or-path ./model_from_hf/Qwen1.5-32B/ \
         --output-prefix ./dataset/qwen1.5-32B-hf/alpaca \
         --workers 4 \
         --log-interval 1000 \
         --tokenizer-type PretrainedFromHF
     ```
    5.2 预训练
     ```shell
     # 设置 ascend-toolkit 路径
     source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
     # 根据实际情况配置词表、数据集、模型参数保存路径
     CKPT_SAVE_DIR="./ckpt/Qwen1.5-32B/"
     TOKENIZER_PATH="./model_from_hf/Qwen1.5-32B/"  #词表路径
     DATA_PATH="./dataset/qwen1.5-32B-hf/alpaca_text_document"  #数据集路径
     CKPT_LOAD_DIR="./model_weights/qwen1.5-32B-v0.1-tp8-pp4-vpp2/"
    ```
   
    启动 Qwen1.5-32B 预训练脚本: examples/qwen15/pretrain_qwen15_32b_ptd.sh
    
    ```shell
     bash examples/qwen15/pretrain_qwen15_32b_ptd.sh
    ```
    **注意**：
    + 如果使用多机训练，且没有设置数据共享，需要在训练启动脚本中增加`--no-shared-storage`参数，设置此参数之后将会根据分布式参数判断非主节点是否需要load数据，并检查相应缓存和生成数据。
    + --num-layers-per-virtual-pipeline-stage参数只有在--target-pipeline-parallel-size > 2时才会生效，如果target-pipeline-parallel-size不满足>2的条件，请在脚本中删除--num-layers-per-virtual-pipeline-stage参数
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
    mkdir ./finetune_dataset/qwen-1.5-32b-hf/
    python ./tools/preprocess_data.py \
        --input ./finetune_dataset/ train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/Qwen1.5-32B/ \
        --output-prefix ./finetune_dataset/qwen-1.5-32b-hf/alpaca \
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
    DATA_PATH="./finetune_dataset/qwen-1.5-32b-hf/alpaca"
    TOKENIZER_PATH="./model_from_hf/Qwen1.5-32B/"
    CKPT_PATH="./ckpt/Qwen1.5-32B/"
        --load ${CKPT_PATH} \
        --finetune \
        --is-instruction-dataset \
        --tokenizer-not-use-fast \
    ```
   
    6.3 Lora微调

    Lora微调的脚本配置是在全参微调脚本基础上加上lora参数，如下所示:
    ```bash
   CKPT_SAVE_DIR="./ckpt/Qwen1.5-32B-lora-tp8-pp1/"
        --lora-target-modules query_key_value dense dense_h_to_4h dense_4h_to_h \
        --lora-r 16 \
        --lora-alpha 32 \
    ```
    如果模型的词表变化了，可以加上以下参数（词表不变不建议添加）
    ```bash
        --lora-modules-to-save word_embeddings output_layer \
    ```
    启动qwen1.5-32B Lora微调脚本: examples/qwen15/tune_qwen15_32b_ptd.sh
    ```shell
    bash examples/qwen15/tune_qwen15_32b_ptd.sh
    ```



## 推理-32B

配置 qwen1.5-32b 推理脚本：examples/qwen15/generate_qwen15_32b_ptd.sh

```bash
# ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 修改模型权重路径和词表路径
CHECKPOINT="./model_weights/Qwen1.5-32B-v0.1-tp8-pp1/"
TOKENIZER_PATH="/model_from_hf/Qwen1.5-32B/"
```

启动qwen1.5-32b推理脚本

```bash
bash examples/qwen15/generate_qwen15_32b_ptd.sh
```

推理示例如下：

![Inference](../../sources/images/qwen15/qwen15_32b_inference.png)

配置 Qwen1.5-32B lora推理脚本： examples/qwen15/generate_qwen15_32b_lora_chat_ptd.sh

```bash
# 修改lora权重路径
CHECKPOINT_LORA="./ckpt/Qwen1.5-32B-lora-tp8-pp1/"
```
Qwen1.5-32B启动lora推理:

```bash
bash ./examples/qwen15/generate_qwen15_32b_lora_chat_ptd.sh
```

lora微调后的推理效果如下：
![Inference](../../sources/images/qwen15/qwen15_32b_lora_inference.png)

## 评估-32B

使用[MMLU数据集](https://huggingface.co/datasets/cais/mmlu)评估模型.

配置qwen1.5-32b评估脚本: examples/qwen15/evaluate_qwen15_32b_ptd.sh

```bash
# ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# 修改模型参数路径和词表路径
TOKENIZER_PATH="./model_from_hf/Qwen1.5-32B/"  # 词表路径
CHECKPOINT="./model_weights/Qwen1.5-32B-v0.1-tp8-pp1/"  # 模型路径

# 配置任务和数据集路径
DATA_PATH="./mmlu/data/test/"  # ceval任务配置为 "./ceval/val/"
TASK="mmlu"  # ceval任务配置为 "ceval"
```

启动评估

```bash
bash examples/qwen15/evaluate_qwen15_32b_ptd.sh
```

| 数据集 | 总学科数 | 总问题数 |                       参考准确率                       | NPU准确率 |
|:---:|:---:|:---:|:-------------------------------------------------:|:------:|
| MMLU | 57 | 14042 | [73.4](https://qwenlm.github.io/zh/blog/qwen1.5/) |  72.6  |

# Qwen1.5-72B

## 训练-72B
| 硬件  | 序列长度 |        配置        |
|:---:|:----:|:----------------:|
| NPU |  8k  | 64 x Ascend NPUs |

### 脚本-72B
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
   
   # 安装依赖库
   pip install -r requirements.txt
   
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
   ```
3. 下载 Qwen1.5-72B 的 [预训练权重和词表](https://huggingface.co/Qwen/Qwen1.5-72B/tree/main)
   ```bash
   mkdir ./model_from_hf/Qwen1.5-72B/
   cd ./model_from_hf/Qwen1.5-72B/
   wget https://huggingface.co/Qwen/Qwen1.5-72B/blob/main/config.json
   wget https://huggingface.co/Qwen/Qwen1.5-72B/blob/main/generation_config.json
   wget https://huggingface.co/Qwen/Qwen1.5-72B/blob/main/merges.txt
   wget https://huggingface.co/Qwen/Qwen1.5-72B/blob/main/model-00001-of-00038.safetensors
   wget https://huggingface.co/Qwen/Qwen1.5-72B/blob/main/model-00002-of-00038.safetensors
   wget https://huggingface.co/Qwen/Qwen1.5-72B/blob/main/model-00003-of-00038.safetensors
   wget https://huggingface.co/Qwen/Qwen1.5-72B/blob/main/model-00004-of-00038.safetensors
   wget https://huggingface.co/Qwen/Qwen1.5-72B/blob/main/model-00005-of-00038.safetensors
   ...
   wget https://huggingface.co/Qwen/Qwen1.5-72B/blob/main/model.safetensors.index.json
   wget https://huggingface.co/Qwen/Qwen1.5-72B/blob/main/tokenizer.json
   wget https://huggingface.co/Qwen/Qwen1.5-72B/blob/main/tokenizer_config.json
   wget https://huggingface.co/Qwen/Qwen1.5-72B/blob/main/vocab.json
   cd ../../
   ```
4. 权重转换

   4.1 将权重从 huggingface 格式转化为 megatron 格式
   ***（该场景一般用于使能开源的HuggingFace模型在Megatron上进行训练）***
   ```bash
   # 修改 ascend-toolkit 路径
   source /usr/local/Ascend/ascend-toolkit/set_env.sh

   python tools/checkpoint/convert_ckpt.py \
       --model-type GPT \
       --loader llama2_hf \
       --saver megatron \
       --target-tensor-parallel-size 8 \
       --target-pipeline-parallel-size 8 \
       --num-layers-per-virtual-pipeline-stage 2 \
       --params-dtype bf16 \
       --load-dir ./model_from_hf/Qwen1.5-72B/ \
       --save-dir ./model_weights/Qwen1.5-72B-v0.1-tp8-pp8-vpp2/ \
       --tokenizer-model ./model_from_hf/Qwen1.5-72B/tokenizer.json \
       --add-qkv-bias \
       --max-queue-size 90
   ```
    **注意**： --num-layers-per-virtual-pipeline-stage参数只有在--target-pipeline-parallel-size > 2时才会生效，如果target-pipeline-parallel-size不满足>2的条件，请在脚本中删除--num-layers-per-virtual-pipeline-stage参数

   4.2 任意并行切分策略的Megatron权重 格式转化为 HuggingFace权重
   ***（该场景一般用于将训练好的megatron模型重新转回HuggingFace格式）***

   ```shell
   # 请按照您的真实环境修改 set_env.sh 路径
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   
   python tools/checkpoint/convert_ckpt.py \
       --model-type GPT \
       --loader megatron \
       --saver megatron \
       --save-model-type save_huggingface_llama \
       --load-dir ./model_weights/Qwen1.5-72B-v0.1-tp8-pp8-vpp2/ \
       --target-tensor-parallel-size 1 \
       --target-pipeline-parallel-size 1 \
       --num-layers-per-virtual-pipeline-stage 2 \
       --add-qkv-bias \
       --save-dir ./model_from_hf/Qwen1.5-72B/     # 需要填入原始HF模型路径，新权重会存于./model_from_hf/Qwen1.5-72B/mg2hg/
   ```
   权重转换适用于预训练、微调、推理和评估，根据任务不同调整参数 `target-tensor-parallel-size`和 `target-pipeline-parallel-size`。

5. 预训练

    5.1 准备数据集
    下载 Qwen1.5-72B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)
     ```shell
     # 下载数据
     cd ./dataset
     wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
     cd ..
     # 处理数据   
     mkdir ./dataset/Qwen1.5-72B-hf/
     python ./tools/preprocess_data.py \
         --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
         --tokenizer-name-or-path ./model_from_hf/Qwen1.5-72B/ \
         --output-prefix ./dataset/Qwen1.5-72B-hf/alpaca \
         --workers 4 \
         --log-interval 1000 \
         --tokenizer-type PretrainedFromHF
     ```
    5.2 预训练
     ```shell
     # 设置 ascend-toolkit 路径
     source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
     # 根据实际情况配置词表、数据集、模型参数保存路径
     CKPT_SAVE_DIR="./ckpt/Qwen1.5-72B/"
     TOKENIZER_PATH="./model_from_hf/Qwen1.5-72B/"  #词表路径
     DATA_PATH="./dataset/Qwen1.5-72B-hf/alpaca_text_document"  #数据集路径
     CKPT_LOAD_DIR="./model_weights/Qwen1.5-72B-v0.1-tp8-pp8-vpp2/"
    ```
   
    启动 Qwen1.5-72B 预训练脚本: examples/qwen15/pretrain_qwen15_72b_ptd.sh
    
    ```shell
     bash examples/qwen15/pretrain_qwen15_72b_ptd.sh
    ```
    **注意**：
    + 如果使用多机训练，且没有设置数据共享，需要在训练启动脚本中增加`--no-shared-storage`参数，设置此参数之后将会根据分布式参数判断非主节点是否需要load数据，并检查相应缓存和生成数据。
    + --num-layers-per-virtual-pipeline-stage参数只有在--target-pipeline-parallel-size > 2时才会生效，如果target-pipeline-parallel-size不满足>2的条件，请在脚本中删除--num-layers-per-virtual-pipeline-stage参数
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
    mkdir ./finetune_dataset/qwen-1.5-72b-hf/
    python ./tools/preprocess_data.py \
        --input ./finetune_dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/Qwen1.5-72B/ \
        --output-prefix ./finetune_dataset/qwen-1.5-72b-hf/alpaca \
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
    DATA_PATH="./finetune_dataset/qwen-1.5-72b-hf/alpaca"
    TOKENIZER_PATH="./model_from_hf/Qwen1.5-72B/"
    CKPT_PATH="./ckpt/Qwen1.5-72B/"
        --load ${CKPT_PATH} \
        --finetune \
        --is-instruction-dataset \
        --tokenizer-not-use-fast \
    ```
   
    6.3 Lora微调

    Lora微调的脚本配置是在全参微调脚本基础上加上lora参数，如下所示:
    ```bash
   CKPT_SAVE_DIR="./ckpt/Qwen1.5-72B-lora-tp8-pp2/"
        --lora-target-modules query_key_value dense dense_h_to_4h dense_4h_to_h \
        --lora-r 16 \
        --lora-alpha 32 \
    ```
    如果模型的词表变化了，可以加上以下参数（词表不变不建议添加）
    ```bash
        --lora-modules-to-save word_embeddings output_layer \
    ```
    启动qwen1.5-72B Lora微调脚本: examples/qwen15/tune_qwen15_72b_ptd.sh
    ```shell
    bash examples/qwen15/tune_qwen15_72b_ptd.sh
    ```


## 推理-72B

配置 qwen1.5-72b 推理脚本：examples/qwen15/generate_qwen15_72b_ptd.sh

```bash
# ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 修改模型权重路径和词表路径
CHECKPOINT="./model_weights/Qwen1.5-72B-v0.1-tp8-pp1/"
TOKENIZER_PATH="./model_from_hf/Qwen1.5-72B/"
```

启动qwen1.5-72b推理脚本

```bash
bash examples/qwen15/generate_qwen15_72b_ptd.sh
```

推理示例如下：

![Inference](../../sources/images/qwen15/qwen15_72b_inference.png)

配置 Qwen1.5-72B lora推理脚本： examples/qwen15/generate_qwen15_72b_lora_chat_ptd.sh

```bash
# 修改lora权重路径
CHECKPOINT_LORA="./ckpt/Qwen1.5-72B-lora-tp8-pp2/"
```
Qwen1.5-72B启动lora推理:

```bash
bash ./examples/qwen15/generate_qwen15_72b_lora_chat_ptd.sh
```

lora微调后的推理效果如下：
![Inference](../../sources/images/qwen15/qwen15_72b_lora_inference.png)

## 评估-72B

使用[MMLU数据集](https://huggingface.co/datasets/cais/mmlu)评估模型.

配置qwen1.5-72b评估脚本: examples/qwen15/evaluate_qwen15_72b_ptd.sh

```bash
# ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# 修改模型参数路径和词表路径
TOKENIZER_PATH="./model_from_hf/Qwen1.5-72B/"  # 词表路径
CHECKPOINT="./model_weights/Qwen1.5-72B-v0.1-tp8-pp1/"  # 模型路径

# 配置任务和数据集路径
DATA_PATH="./mmlu/data/test/"  # ceval任务配置为 "./ceval/val/"
TASK="mmlu"  # ceval任务配置为 "ceval"
```

启动评估

```bash
bash examples/qwen15/evaluate_qwen15_72b_ptd.sh
```

| 数据集 | 总学科数 | 总问题数 |                       参考准确率                       | NPU准确率 |
|:---:|:---:|:---:|:-------------------------------------------------:|:------:|
| MMLU | 57 | 14042 | [77.5](https://qwenlm.github.io/zh/blog/qwen1.5/) |  77.5  |
