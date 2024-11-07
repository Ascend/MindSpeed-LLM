# LLaMA

<p align="left">
        <b>简体中文</b> |
        <b><a href="README_en.md">English</a></b>
</p>

- [LLaMA-7B/13B](#llama-7b13b)
  - [训练](#训练)
    - [脚本](#脚本)
    - [推理](#推理)
      - [脚本](#脚本)
    - [使用基线数据集进行评估](#使用基线数据集进行评估)
- [LLaMA-33B/65B](#llama-33b65b)
  - [训练](#训练)
    - [数据集](#数据集)
    - [脚本](#脚本)
    - [推理](#推理)
      - [脚本](#脚本)
    - [使用基线数据集进行评估](#使用基线数据集进行评估)
- [引用](#引用)

# LLaMA-7B/13B

## 训练

LLaMA-7B/13B 训练的硬件配置如下:

| 硬件 |      配置      |
| :--: | :-------------: |
| NPU | 8 x Ascend NPUs |

### 脚本

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
    pip install torch-2.1.0-cp38-cp38m-linux_aarch64.whl
    pip install torch_npu-2.1.0.post5_XXXXXX-cp38-cp38m-linux_aarch64.whl
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

3. 下载 LLaMA-7B [权重和词表](https://huggingface.co/ruibin-wang/llama-7b-hf/tree/main) 或 LLaMA-13B [权重和词表](https://huggingface.co/ruibin-wang/llama-13b-hf/tree/main)

    ```shell
    cd ./model_from_hf
    # 需要安装 git-lfs: git lfs install
    git clone https://huggingface.co/ruibin-wang/llama-7b-hf
    cd ..
    ```

    or

    ```shell
    cd ./model_from_hf
    # 需要安装 git-lfs: git lfs install
    git clone https://huggingface.co/ruibin-wang/llama-13b-hf
    cd ..
    ```

4. 权重转换

    4.1 将模型权重文件从 huggingface 格式转化为 megatron 格式
    ***（该场景一般用于使能开源的HuggingFace模型在Megatron上进行训练）***

    LLaMA-7B

    ```shell
    # 修改 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # 权重格式转换
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader llama2_hf \
        --saver megatron \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 8 \
        --load-dir ./model_from_hf/llama-7b-hf/ \
        --save-dir ./model_weights/llama-7b-hf-v0.1-tp1-pp8/ \
        --tokenizer-model ./model_from_hf/llama-7b-hf/tokenizer.model
    ```

    LLaMA-13B

    ```shell
    # 修改 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # 权重格式转换
    # 单机8卡
    mkdir model_weights
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader llama2_hf \
        --saver megatron \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 8 \
        --load-dir ./model_from_hf/llama-13b-hf/ \
        --save-dir ./model_weights/llama-13b-hf-v0.1-tp1-pp8/ \
        --tokenizer-model ./model_from_hf/llama-13b-hf/tokenizer.model
    ```

    4.2 将模型权重文件从 megatron 格式转化为 huggingface 格式
    ***（该场景一般用于将训练好的megatron模型重新转回HuggingFace格式）***

    LLaMA-7B

    ```shell
    # 请按照您的真实环境修改 set_env.sh 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader megatron \
        --saver megatron \
        --save-model-type save_huggingface_llama \
        --load-dir ./model_weights/llama-7b-hf-v0.1-tp1-pp8/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --save-dir ./model_from_hf/llama-7b-hf/  # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/llama-7b-hf/mg2hg/
    ```

    LLaMA-13B

    ```shell
    # 请按照您的真实环境修改 set_env.sh 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader megatron \
        --saver megatron \
        --save-model-type save_huggingface_llama \
        --load-dir ./model_weights/llama-13b-hf-v0.1-tp1-pp8/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --save-dir ./model_from_hf/llama-13b-hf/  # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/llama-13b-hf/mg2hg/
    ```

    权重转换适用于预训练、微调、推理和评估，根据任务不同调整参数`target-tensor-parallel-size`和`target-pipeline-parallel-size`。

5. 预训练

    5.1 准备预训练数据集

    下载 LLaMA-7B/13B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet) 

    ```shell
    cd dataset/
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    ```

    处理数据集

    LLaMA-7B

    ```shell
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    mkdir ./dataset/llama-7b-hf/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/llama-7b-hf/ \
        --output-prefix ./dataset/llama-7b-hf/alpaca \
        --workers 4 \
        --log-interval 1000  \
        --tokenizer-type PretrainedFromHF  
    ```

    LLaMA-13B

    ```shell
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    mkdir ./dataset/llama-7b-hf/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/llama-13b-hf/ \
        --output-prefix ./dataset/llama-7b-hf/alpaca \
        --workers 4 \
        --log-interval 1000  \
        --tokenizer-type PretrainedFromHF  
    ```

    5.2 配置 LLaMA-7B/13B 预训练脚本

    LLaMA-7B

    ```shell
    # 设置 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    # 根据实际情况配置词表、数据集、模型参数保存路径
    TOKENIZER_MODEL="./model_from_hf/llama-7b-hf/tokenizer.model"
    DATA_PATH="./dataset/llama/alpaca_text_document"  #数据集 路径
    LOAD_CHECKPOINT_PATH="./model_weights/llama-7b-hf-v0.1-tp1-pp8"
    SAVE_CHECKPOINT_PATH="./ckpt/llama-7b-hf/"
    ```

    LLaMA-13B

    ```shell
    # 设置 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    # 根据实际情况配置词表、数据集、模型参数保存路径
    TOKENIZER_MODEL="./model_from_hf/llama-13b-hf/tokenizer.model" 
    DATA_PATH="./dataset/llama-13b-hf/alpaca_text_document"  #数据集 路径
    LOAD_CHECKPOINT_PATH="./model_weights/llama-13b-hf-v0.1-tp1-pp8"
    SAVE_CHECKPOINT_PATH="./ckpt/llama-13b-hf/"
    ```

    5.3 启动 LLaMA-7B/13B 预训练脚本

    **注意**：如果使用多机训练，且没有设置数据共享，需要在训练启动脚本中增加`--no-shared-storage`参数，设置此参数之后将会根据分布式参数判断非主节点是否需要load数据，并检查相应缓存和生成数据。

    LLaMA-7B

    ```shell
    bash examples/llama/pretrain_llama_7b_ptd.sh
    ```

    LLaMA-13B

    ```shell
    # 单机8卡
    bash examples/llama/pretrain_llama_13b_ptd.sh 
    ```

6. 微调

    6.1 准备微调数据集

    下载 LLaMA-7B/13B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

    ```shell
    mkdir finetune_dataset
    cd ./finetune_dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    ```

    处理指令数据集

    LLaMA-7B

    ```shell
    mkdir ./finetune_dataset/llama-7b-hf/
    python ./tools/preprocess_data.py \
        --input ./finetune_dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/llama-7b-hf/ \
        --output-prefix ./finetune_dataset/llama-7b-hf/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF \
        --handler-name GeneralInstructionHandler \
        --append-eod
    ```

    LLaMA-13B

    ```shell
    mkdir ./finetune_dataset/llama-13b-hf/
    python ./tools/preprocess_data.py \
        --input ./finetune_dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/llama-13b-hf/ \ 
        --output-prefix ./finetune_dataset/llama-13b-hf/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF \
        --handler-name GeneralInstructionHandler \
        --append-eod
    ```

    6.2 配置 LLaMA-7B/13B 微调脚本

    LLaMA-7B

    ```shell
    # 设置 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    # 根据实际情况配置词表、数据集、模型参数保存路径
    TOKENIZER_PATH="./model_from_hf/llama-7b-hf/"  #tokenizer 路径
    DATA_PATH="./finetune_dataset/llama-7b-hf/alpaca"  #数据集 路径
    LORA_CHECKPOINT="your lora weight"
    LOAD_CHECKPOINT_PATH="./model_weights/llama-13b-hf-v0.1-tp1-pp8"
    SAVE_CHECKPOINT_PATH="./ckpt/llama-7b-hf/"
    ```

    LLaMA-13B

    ```shell
    # 设置 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    # 根据实际情况配置词表、数据集、模型参数保存路径
    TOKENIZER_PATH="./model_from_hf/llama-13b-hf/"  #tokenizer 路径
    DATA_PATH="./finetune_dataset/llama-13b-hf/alpaca"  #数据集 路径
    LORA_CHECKPOINT="your lora weight"
    LOAD_CHECKPOINT_PATH="your init model load path"
    SAVE_CHECKPOINT_PATH="your model ckpt save path"
    ```

    增加微调参数--finetune，使微调从第一步开始。

    6.3 启动 LLaMA-7B/13B 微调脚本

    LLaMA-7B

    ```shell
    bash examples/llama/tune_llama_7b_ptd.sh
    ```

    LLaMA-13B

    ```shell
    # 单机8卡
    bash examples/llama/tune_llama_13b_ptd.sh 
    ```




## 推理

我们支持使用 LLaMA-7B 和 LLaMA-13B 进行文本生成的推理。
推理与预训练不同，比如我们需要加载预训练权重和输出样本的长度：

配置LLaMA-7B推理脚本 `examples/llama/generate_llama_7b_ptd.sh`和LLaMA-13B推理脚本 `examples/llama/generate_llama_13b_ptd.sh`。

```shell
# 修改模型权重路径和分词器路径
CHECKPOINT=<checkpoint-path>
TOKENIZER_PATH=<tokenizer-path>
```

LLaMA-7B:

```shell
bash ./examples/llama/generate_llama_7b_ptd.sh
```

LLaMA-13B:

```shell
bash ./examples/llama/generate_llama_13b_ptd.sh
```

配置LLaMA-7B lora推理脚本 `examples/llama/generate_llama_7b_lora_ptd.sh`和LLaMA-13B lora推理脚本 `examples/llama/generate_llama_13b_lora_ptd.sh`。

```bash
# 修改lora权重路径
CHECKPOINT_LORA="your lora model directory path"
```

LLaMA-7B:

```shell
bash ./examples/llama/generate_llama_7b_lora_ptd.sh
```

LLaMA-13B:

```shell
bash ./examples/llama/generate_llama_13b_lora_ptd.sh
```

部分推理样本如下：

LLaMA-7B:

![llama-7B_generate.png](https://gitee.com/ascend/ModelLink/raw/master/sources/images/llama/llama-7B_generate.png)

LLaMA-13B:

![llama-13B_generate.png](https://gitee.com/ascend/ModelLink/raw/master/sources/images/llama/llama-13B_generate.png)

## 使用基线数据集进行评估

我们使用 Boolq benchmark 来评估我们的模型。Benchmark下载[此处](https://huggingface.co/datasets/boolq)。

配置LLaMA-7B评估脚本 `examples/llama/evaluate_llama_7B_ptd.sh` 和 LLaMA-13B评估脚本 `examples/llama/evaluate_llama_13B_ptd.sh`：

修改权重路径, 词表路径和数据集任务路径：

```shell
CHECKPOINT=<checkpoint-path>
TOKENIZER_PATH=<tokenizer-path>
DATA_PATH="./boolq/data/test/"
TASK="boolq"
```

修改最大生成词参数：

```shell
--max-new-tokens 1 
```

开始评估：

```shell
bash examples/llama/evaluate_llama_7B_ptd.sh
bash examples/llama/evaluate_llama_13B_ptd.sh
```

LLaMA-7B/13B在**Ascend NPU**中的评测表现：

| 任务                                                  | 模型        | 昇腾值  | 社区值  |
|-----------------------------------------------------|-----------|------|------|
| [Boolq](https://huggingface.co/datasets/boolq) | LLaMA-7B  | 74.6 | [75.4](https://hub.opencompass.org.cn/dataset-detail/BoolQ) | 
| [Boolq](https://huggingface.co/datasets/boolq) | LLaMA-13B | 79.6 | [78.7](https://hub.opencompass.org.cn/dataset-detail/BoolQ) |

# LLaMA-33B/65B

LLaMA 模型源于: [LLaMA: OPen and Efficient Foundation Language Models](https://arxiv.org/pdf/2302.13971v1.pdf)

>Touvron, Hugo, et al. "LLaMA: OPen and Efficient Foundation Language Models." arXiv preprint arXiv:2302.13971 (2023).

## 训练

LLaMA-33B/65B 训练的硬件配置:

|  硬件 |        配置        |
|:---:|:----------------:|
| NPU | 32 x Ascend NPUs |

### 数据集

模型使用 alpaca 数据集训练

### 脚本

1. 克隆仓库到本地服务器

    ```shell
    git clone https://gitee.com/ascend/ModelLink.git 
    cd ModelLink
    git checkout 1.1
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

    ```shell
    # python3.8
    conda create -n test python=3.8
    conda activate test

    # 安装 torch 和 torch_npu
    # ARM
    wget https://download.pytorch.org/whl/torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl
    pip install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl
    pip install torch_npu-2.1.0.post4_XXXXXX-cp38-cp38m-manylinux2014_aarch64.whl
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

3. 下载权重和词表

    llama-33B 权重

    ```shell
    cd ./model_from_hf/

    # 需要安装 git-lfs: git lfs install
    git clone https://huggingface.co/pinkmanlove/llama-33b-hf
    cd ..
    # 将 tokenizer_config.json 中的 "LLaMATokenizer" 修改为 "LLaMTokenizer" （这是hf的一个bug）
    ```

    llama-65B 权重

    ```shell
    cd ./model_from_hf/

    # 需要安装 git-lfs: git lfs install
    git clone https://huggingface.co/pinkmanlove/llama-65b-hf
    cd ..
    # 将 tokenizer_config.json 中的 "LLaMATokenizer" 修改为 "LLaMTokenizer" （这是hf的一个bug）
    ```

4. 权重转换

    4.1 预训练权重从 huggingface 格式转换为 megatron 格式
    ***（该场景一般用于使能开源的HuggingFace模型在Megatron上进行训练）***

    llama-33B

    ```shell
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader llama2_hf \
        --saver megatron \
        --target-tensor-parallel-size 4 \
        --target-pipeline-parallel-size 4 \
        --load-dir ./model_from_hf/llama-33b-hf/ \
        --save-dir ./model_weights/llama-33b-hf-v0.1-tp4-pp4/ \
        --tokenizer-model ./model_from_hf/llama-33b-hf/tokenizer.model
    ```

    llama-65B

    ```shell
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader llama2_hf \
        --saver megatron \
        --target-tensor-parallel-size 8 \
        --target-pipeline-parallel-size 4 \
        --load-dir ./model_from_hf/llama-65b-hf/ \
        --save-dir ./model_weights/llama-65b-hf-v0.1-tp8-pp4/ \
        --tokenizer-model ./model_from_hf/llama-65b-hf/tokenizer.model
    ```

    4.2 预训练权重从 megatron 格式转换为 huggingface 格式
    ***（该场景一般用于将训练好的megatron模型重新转回HuggingFace格式）***

    llama-33B

    ```shell
    # 请按照您的真实环境修改 set_env.sh 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader megatron \
        --saver megatron \
        --save-model-type save_huggingface_llama \
        --load-dir /model_weights/llama-33b-hf-v0.1-tp4-pp4/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --save-dir  ./model_from_hf/llama-33b-hf/    # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/llama-33b-hf/mg2hg/
    ```

    llama-65B

    ```shell
    # 请按照您的真实环境修改 set_env.sh 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader megatron \
        --saver megatron \
        --save-model-type save_huggingface_llama \
        --load-dir /model_weights/llama-65b-hf-v0.1-tp8-pp4/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --save-dir ./model_from_hf/llama-65b-hf/    # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/llama-65b-hf/mg2hg/
    ```
    权重转换适用于预训练、微调、推理和评估，根据任务不同调整参数`target-tensor-parallel-size`和`target-pipeline-parallel-size`。

5. 预训练

    5.1 准备预训练数据集

    ```shell
    cd dataset/
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    ```

    处理数据集

    LLaMA-33B

    ```shell
    mkdir ./dataset/llama-33b-hf/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/llama-33b-hf \
        --output-prefix ./dataset/llama-33b-hf/alpaca \
        --workers 4 \
        --log-interval 1000  \
        --tokenizer-type PretrainedFromHF 
    ```

    LLaMA-65B

    ```shell
    mkdir ./dataset/llama-65b-hf/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/llama-65b-hf \
        --output-prefix ./dataset/llama-65b-hf/alpaca \
        --workers 4 \
        --log-interval 1000  \
        --tokenizer-type PretrainedFromHF 
    ```

    5.2 配置 LLaMA-33B/65B 预训练脚本

    LLaMA-33B

    ```bash
    # 修改 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # 配置词表和数据路径等
    TOKENIZER_MODEL="./model_from_hf/llama-33b-hf/tokenizer.model"
    DATA_PATH="./dataset/llama-33b-hf/alpaca_text_document"
    LOAD_CHECKPOINT_PATH="./model_weights/llama-33b-hf-v0.1-tp4-pp4/"
    SAVE_CHECKPOINT_PATH="./ckpt/llama-33b-hf/"
    ```

    LLaMA-65B

    ```bash
    # 修改 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # 配置词表和数据路径等
    TOKENIZER_MODEL=./model_from_hf/llama-65b-hf/tokenizer.model
    DATA_PATH="./dataset/llama-65b-hf/alpaca_text_document"
    LOAD_CHECKPOINT_PATH="./model_weights/llama-65b-hf-v0.1-tp8-pp4/"
    SAVE_CHECKPOINT_PATH="./ckpt/llama-65b-hf/"
    ```

    5.3 启动预训练脚本:

    **注意**：如果使用多机训练，且没有设置数据共享，需要在训练启动脚本中增加`--no-shared-storage`参数，设置此参数之后将会根据分布式参数判断非主节点是否需要load数据，并检查相应缓存和生成数据。

    启动 llama-33B 预训练脚本 : ./examples/llama/pretrain_llama_33B_ptd_32p.sh

    ```bash
    bash examples/llama/pretrain_llama_33B_ptd_32p.sh
    ```

    启动 llama-65B 预训练脚本 : ./examples/llama/pretrain_llama_65b_ptd.sh

    ```bash
    # 四机32卡
    bash examples/llama/pretrain_llama_65b_ptd.sh
    ```

    为多节点配置 llama-33B/65B 预训练脚本 (在集群的每个节点上启动脚本):

    ```shell
    MASTER_ADDR=localhost
    MASTER_PORT=6001
    NNODES=4
    NODE_RANK=0
    ```


6. 微调

    6.1 准备微调数据集

    下载数据集

    ```shell
    mkdir finetune_dataset
    cd ./finetune_dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    ```

    处理指令数据集

    LLaMA-33B

    ```shell
    mkdir ./finetune_dataset/llama-33b-hf/
    python ./tools/preprocess_data.py \
        --input ./dataset_llama2/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/llama-33b-hf/ \ 
        --output-prefix ./finetune_dataset/llama-33b-hf/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF \
        --handler-name GeneralInstructionHandler \
        --append-eod
    ```

    LLaMA-65B

    ```shell
    mkdir ./finetune_dataset/llama-65b-hf/
    python ./tools/preprocess_data.py \
        --input ./dataset_llama2/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/llama-65b-hf/  \
        --output-prefix ./finetune_dataset/llama-65b-hf/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF \
        --handler-name GeneralInstructionHandler \
        --append-eod
    ```

    6.2 配置 LLaMA-33B/65B 微调脚本

    LLaMA-33B

    ```shell
    # 设置 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    # 根据实际情况配置词表、数据集、模型参数保存路径
    TOKENIZER_PATH="./model_from_hf/llama-33b-hf/"  #tokenizer 路径
    DATA_PATH="./finetune_dataset/llama-33b-hf/alpaca"  #数据集 路径
    LORA_CHECKPOINT="your lora weight"
    LOAD_CHECKPOINT_PATH="./model_weights/llama-33b-hf-v0.1-tp4-pp4/"
    SAVE_CHECKPOINT_PATH="./ckpt/llama-33b-hf/"
    ```

    LLaMA-65B

    ```shell
    # 设置 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    # 根据实际情况配置词表、数据集、模型参数保存路径
    TOKENIZER_PATH="./model_from_hf/llama-65b-hf/"  #tokenizer 路径
    DATA_PATH="./finetune_dataset/llama-65b-hf/alpaca"  #数据集 路径
    LORA_CHECKPOINT="your lora weight"
    LOAD_CHECKPOINT_PATH="./model_weights/llama-65b-hf-v0.1-tp8-pp4/"
    SAVE_CHECKPOINT_PATH="./ckpt/llama-65b-hf/"
    ```

    增加微调参数--finetune，使微调从第一步开始。

    6.3 启动 LLaMA-33B/65B 微调脚本

    LLaMA-33B

    ```shell
    bash examples/llama/tune_llama_33b_ptd.sh
    ```

    LLaMA-65B

    ```shell
    # 双机16卡
    bash examples/llama/tune_llama_65b_ptd.sh 
    ```



## 推理

我们支持使用 LLaMA-33B 和 LLaMA-65B 进行文本生成的推理。
推理与预训练不同，比如我们需要加载预训练权重和输出样本的长度：

配置LLaMA-33B推理脚本 `examples/llama/generate_llama_33b_ptd.sh`。

配置LLaMA-65B推理脚本 `examples/llama/generate_llama_65b_ptd.sh`。

```shell
# 修改模型权重路径和分词器路径
CHECKPOINT=<checkpoint-path>
TOKENIZER_PATH=<tokenizer-path>
```

LLaMA-33B:

```shell
bash ./examples/llama/generate_llama_33b_ptd.sh
```

LLaMA-65B:

```shell
bash ./examples/llama/generate_llama_65b_ptd.sh
```

配置LLaMA-33B lora推理脚本 `examples/llama/generate_llama_33B_lora_ptd.sh`。

配置LLaMA-65B lora推理脚本 `examples/llama/generate_llama_65B_lora_ptd.sh`。

```bash
# 修改lora权重路径
CHECKPOINT_LORA="your lora model directory path"
```

LLaMA-33B:

```shell
bash ./examples/llama/generate_llama_33b_lora_ptd.sh
```

LLaMA-65B:

```shell
bash ./examples/llama/generate_llama_65b_lora_ptd.sh
```

部分推理样本如下：

LLaMA-33B:

![llama-13B_generate.png](https://gitee.com/ascend/ModelLink/raw/master/sources/images/llama/llama33B_generate.png)

LLaMA-65B:

![llama-65B_generate.png](https://gitee.com/ascend/ModelLink/raw/master/sources/images/llama/llama-65B_generate.png)

## 使用基线数据集进行评估

我们使用 Boolq benchmark 来评估我们的模型。Benchmark下载[此处](https://huggingface.co/datasets/boolq)。

配置LLaMA-33B评估脚本：examples/llama/evaluate_llama_33B_ptd.sh

配置LLaMA-65B评估脚本：examples/llama/evaluate_llama_65B_ptd.sh

修改权重路径, 词表路径和数据集任务路径：

```shell
CHECKPOINT=<checkpoint-path>
TOKENIZER_PATH=<tokenizer-path>
DATA_PATH="./boolq/data/test/"
TASK="boolq"
```

修改最大生成词参数：

```shell
--max-new-tokens 1 
```

开始评估

```shell
# llama-33B评估
bash examples/llama/evaluate_llama_33B_ptd.sh
# llama-65B评估
bash examples/llama/evaluate_llama_65B_ptd.sh
```

LLaMA-33B和LLaMA-65B在**Ascend NPU**中的评测表现：

| 任务                                             | 模型        | 昇腾值  | 社区值                                                                 |
|------------------------------------------------|-----------|------|---------------------------------------------------------------------|
| [Boolq](https://huggingface.co/datasets/boolq) | LLaMA-33B | 83.2 | [83.1](https://paperswithcode.com/sota/question-answering-on-boolq) |
| [Boolq](https://huggingface.co/datasets/boolq) | LLaMA-65B | 85.7 | [86.6](https://paperswithcode.com/sota/question-answering-on-boolq) |

## 引用

```shell
@article{Touvron2023llama,
  title={LLaMA: OPen and Efficient Foundation Language Models},
  author={Hugo Touvron*, Thibaut Lavril*, Gautier Izacard*, Xavier Martinet, Marie-Anne Lachaux, Timothee Lacroix, Baptiste Roziere, Naman Goyal,
  Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave*, Guillaume Lample*},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}}
```
