# ChatGLM3

<p align="left">
        <b>简体中文</b> |
        <b><a href="README_en.md">English</a> </b> 
</p>

# 目录

- [ChatGLM3](#ChatGLM3)
- [目录](#目录)
- [ChatGLM3-6B](#ChatGLM3-6B)
  - [训练-6B](#训练)
    - [脚本](#脚本)
  - [推理-6B](#推理-6B)
  - [评估-6B](#评估-6B)

# ChatGLM3-6B

## 训练

ChatGLM3-6B 训练的硬件配置:

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
    pip install -e .
    cd ..
    
    # 安装其余依赖库
    pip install -r requirements.txt 
    ```
3. 下载 ChatGLM3-6B 的 [预训练权重和词表](https://huggingface.co/THUDM/chatglm3-6b/tree/main)

    ```shell
    #!/bin/bash
    mkdir ./model_from_hf/chatglm3_6b_hf/
    cd ./model_from_hf/chatglm3_6b_hf/
    wget https://huggingface.co/THUDM/chatglm3-6b-base/resolve/main/config.json
    wget https://huggingface.co/THUDM/chatglm3-6b-base/resolve/main/configuration_chatglm.py
    wget https://huggingface.co/THUDM/chatglm3-6b-base/resolve/main/modeling_chatglm.py
    wget https://huggingface.co/THUDM/chatglm3-6b-base/resolve/main/pytorch_model-00001-of-00007.bin
    wget https://huggingface.co/THUDM/chatglm3-6b-base/resolve/main/pytorch_model-00002-of-00007.bin
    wget https://huggingface.co/THUDM/chatglm3-6b-base/resolve/main/pytorch_model-00003-of-00007.bin
    wget https://huggingface.co/THUDM/chatglm3-6b-base/resolve/main/pytorch_model-00004-of-00007.bin
    wget https://huggingface.co/THUDM/chatglm3-6b-base/resolve/main/pytorch_model-00005-of-00007.bin
    wget https://huggingface.co/THUDM/chatglm3-6b-base/resolve/main/pytorch_model-00006-of-00007.bin
    wget https://huggingface.co/THUDM/chatglm3-6b-base/resolve/main/pytorch_model-00007-of-00007.bin
    wget https://huggingface.co/THUDM/chatglm3-6b-base/resolve/main/pytorch_model.bin.index.json
    wget https://huggingface.co/THUDM/chatglm3-6b-base/resolve/main/quantization.py
    wget https://huggingface.co/THUDM/chatglm3-6b-base/resolve/main/tokenization_chatglm.py
    wget https://huggingface.co/THUDM/chatglm3-6b-base/resolve/main/tokenizer.model
    wget https://huggingface.co/THUDM/chatglm3-6b-base/resolve/main/tokenizer_config.json
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
        --loader chatglm3_hf \
        --saver megatron \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 2 \
        --load-dir ./model_from_hf/chatglm3_6b_hf/ \
        --save-dir ./model_weights/chatglm3_6b_tp1pp2/ \
        --tokenizer-model ./model_from_hf/chatglm3_6b_hf/tokenizer.model \
        --add-qkv-bias
    ```

    注意：chatglm3的--target-tensor-parallel-size跟config.json中的multi_query_attention配置有关，这里multi_query_attention设置的是2。

    4.2 任意并行切分策略的 Megatron 权重 格式转化为 HuggingFace权重
    ***（该场景一般用于将训练好的megatron模型重新转回HuggingFace格式）***

    ```shell
    # 请按照您的真实环境修改 set_env.sh 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader megatron \
        --saver megatron \
        --save-model-type save_huggingface_chatglm3 \
        --load-dir ./model_weights/chatglm3_6b_tp1pp2/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --add-qkv-bias \
        --save-dir ./model_from_hf/chatglm3_6b_hf/     # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/chatglm3_6b_hf/mg2hg/
    ```

5. 预训练

    5.1 准备数据集

    下载 ChatGLM3-6B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

    ```shell
    # 下载数据
    cd ./dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    
    # 处理数据    
    mkdir ./dataset/chatglm3_6b_hf/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/chatglm3_6b_hf/ \
        --output-prefix ./dataset/chatglm3_6b_hf/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF
    ```

    5.2 用ptd模式预训练
    配置ChatGLM3-6B PTD 预训练脚本: examples/chatglm3/pretrain_chatglm3_6B_8K.sh

    ```shell
    # 设置 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    
    # 根据实际情况配置词表、数据集、模型参数加载和保存路径
    LOAD_CHECKPOINT_PATH="./model_weights/chatglm3_6b_tp1pp2/"
    SAVE_CHECKPOINT_PATH="./ckpt/chatglm3_6b_hf/"
    TOKENIZER_PATH="./model_from_hf/chatglm3_6b_hf/"  #词表路径
    DATA_PATH="./dataset/chatglm3_6b_hf/alpaca_text_document"  #数据集路径
    ```

    多机运行增加参数--overlap-grad-reduce

    启动 ChatGLM3-6B PTD预训练脚本: examples/chatglm3/pretrain_chatglm3_6B_8K.sh

    ```shell
    bash examples/chatglm3/pretrain_chatglm3_6B_8K.sh
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
    mkdir ./finetune_dataset/chatglm3-6b-hf/
    python ./tools/preprocess_data.py \
        --input ./finetune_dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/chatglm3_6b_hf/ \
        --output-prefix ./finetune_dataset/chatglm3-6b-hf/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF \
        --handler-name GeneralInstructionHandler \
        --append-eod
    ```

    6.2 全参微调
    全参微调的配置脚本基本和预训练脚本一致. *区别是数据集，以及增加训练参数--is-instruction-dataset*

    增加微调参数--finetune，增加权重加载参数--load，使微调从第一步开始。使用--tokenizer-padding-side left。修改tokenizer参数，更改为以下参数：

    ```bash
    DATA_PATH="./finetune_dataset/chatglm3-6b-hf/alpaca"
    TOKENIZER_PATH="./model_from_hf/chatglm3_6b_hf/"
    CKPT_LOAD_DIR="./model_weights/chatglm3_6b_tp1pp2/"
        --load ${CKPT_LOAD_DIR} \
        --finetune \
        --is-instruction-dataset \
        --tokenizer-padding-side left \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-not-use-fast \
    ```
    启动 ChatGLM3-6B 全参微调脚本: examples/chatglm3/tune_chatglm3_6B_8K.sh

    ```shell
    bash examples/chatglm3/tune_chatglm3_6B_8K.sh
    ```


## 推理

我们在ChatGLM3_6B中支持推理来生成文本。
推理不同于预训练，比如我们需要加载预训练检查点和输出样本的长度:

配置 ChatGLM3-6B 推理脚本: examples/chatglm3/generate_chatglm3_6B.sh

```shell
# 修改模型权重路径以及词表路径
CHECKPOINT="./model_weights/chatglm3_6b_tp1pp2/"
TOKENIZER_PATH="./model_from_hf/chatglm3_6b_hf/"
```

启动推理脚本

```shell
bash ./examples/chatglm3/generate_chatglm3_6B.sh
```

推理结果示例如下:
![ChatGLM3-6B-generate.png](https://gitee.com/ascend/MindSpeed-LLM/raw/master/sources/images/chatglm3/ChatGLM3-6B-generate.png)


## 评估

使用mmlu基准来评估模型。mmlu基准[下载](https://github.com/FranxYao/chain-of-thought-hub/tree/main/MMLU/data/test).



因评估代码限制，参考 4.1权重转换 设置--target-tensor-parallel-size 2 --target-pipeline-parallel-size 4做权重转换，保存新权重到chatglm3_6b_tp2pp4目录。

配置chatglm3-6b评估脚本: examples/chatglm3/evaluate_chatglm3_6B.sh

```bash
# ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# 修改模型参数路径和词表路径
TOKENIZER_PATH="./model_from_hf/chatglm3_6b_hf/"  #词表路径
CHECKPOINT="./model_weights/chatglm3_6b_tp2pp4/"  #模型路径
# 配置任务和数据集路径
DATA_PATH="./mmlu/data/test/"
TASK="mmlu"
```

启动评估

```bash
bash examples/chatglm3/evaluate_chatglm3_6B.sh
```

| 数据集 | 总学科数 | 总问题数 |                   参考准确率                   | NPU准确率 |
|:---:|:---:|:---:|:-----------------------------------------:|:------:|
| MMLU | 57 | 14042 | [61.4](https://github.com/THUDM/ChatGLM3) |  61.5  |
