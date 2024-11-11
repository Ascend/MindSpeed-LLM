# Bloom  $\color{black}{\bf\tiny{【昇腾贡献模型】}}$

<p align="left">
        <b>简体中文</b> |
        <b><a href="README_en.md">English</a> </b> 
    </p>
</p>

[toc]

# Bloom-7B

## 训练

Bloom-7B 训练的硬件配置如下：

| 硬件 |      配置      |
| :--: | :-------------: |
| NPU | 8 x Ascend NPUs |

### 脚本

1. 克隆仓库到本地服务器

```shell
git clone https://gitee.com/ascend/ModelLink.git
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout -f bcce6f
cp -r megatron ../ModelLink/
cd ..
cd ModelLink
git checkout 1.0
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
pip install torch-2.1.0-cp37-cp37m-linux_aarch64.whl
pip install torch_npu-2.1.0.XXX-cp37-cp37m-linux_aarch64.whl
pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 安装加速库
git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
git checkout 224ae35e8fc96778f957029d1371ddb623452a50
pip install -r requirements.txt 
pip3 install -e .
cd ..

# 安装其余依赖库
pip install -r requirements.txt 
```

3. 准备预训练权重

首先下载 Bloom-7B 的 [权重](https://huggingface.co/bigscience/bloom-7b1/tree/main)

```shell
mkdir ./model_from_hf/Bloom-7B/
cd ./model_from_hf/Bloom-7B/
cd tokenizer
wget https://huggingface.co/bigscience/bloom/resolve/main/special_tokens_map.json
wget https://huggingface.co/bigscience/bloom/resolve/main/tokenizer.json
wget https://huggingface.co/bigscience/bloom/resolve/main/tokenizer_config.json
...
cd ../../
```

4. 权重转换

将模型权重文件从 HuggingFace权重 格式转化为 Megatron 权重
***（该场景一般用于使能开源的HuggingFace模型在Megatron上进行训练）***

```shell
python tools/checkpoint/util.py \
    --model-type GPT \
    --loader loader_bloom_hf \
    --saver saver_megatron \
    --target-tensor-parallel-size 8 \
    --target-pipeline-parallel-size 1 \
    --load-dir ./model_from_hf/Bloom-7B/ \
    --save-dir ./model_weights/Bloom-7B-v0.1-tp8-pp1/ \
    --tokenizer-model None 
```

任意并行切分策略的Megatron权重 格式转化为 HuggingFace权重
***（该场景一般用于将训练好的megatron模型重新转回HuggingFace格式）***

```shell
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python tools/checkpoint/util.py \
    --model-type GPT \
    --loader megatron \
    --saver megatron \
    --save-model-type save_huggingface_llama \
    --load-dir ./model_weights/Bloom-7B-v0.1-tp8-pp1/ \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --embed-layernorm \
    --save-dir ./model_from_hf/Bloom-7B/     # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/Bloom-7B/mg2hg/
```

5. 准备数据集

下载 Bloom 7B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

```shell
# 下载数据
cd dataset/
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..

# 处理数据         
mkdir ./dataset/Bloom-7B/
python ./tools/preprocess_data.py \
  --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
  --tokenizer-name-or-path ./model_from_hf/Bloom-7B/ \
  --output-prefix ./dataset/Bloom-7B/alpaca \
  --workers 4 \
  --log-interval 1000 \
  --tokenizer-type PretrainedFromHF
```

6. 配置 Bloom-7B 预训练脚本(Bloom-7B暂不支持Flash Attention): examples/bloom/pretrain_bloom_ptd_7B.sh

```shell
# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

CKPT_SAVE_DIR="./ckpt/Bloom-7B/"
DATA_PATH="./dataset/Bloom-7B/alpaca_text_document"
TOKENIZER_PATH="./model_from_hf/Bloom-7B/"
CKPT_LOAD_DIR="./model_weights/Bloom-7B-v0.1-tp8-pp1/"
```

7. 启动 Bloom-7B 预训练脚本: examples/bloom/pretrain_bloom_ptd_7B.sh

```shell
bash examples/bloom/pretrain_bloom_ptd_7B.sh 
```

**注意**：如果使用多机训练，且没有设置数据共享，需要在训练启动脚本中增加`--no-shared-storage`参数，设置此参数之后将会根据分布式参数判断非主节点是否需要load数据，并检查相应缓存和生成数据。

### 性能

#### 吞吐

Bloom-7B

|  设备  |    模型     | 迭代数  | 样本吞吐 (samples/s) | tokens吞吐 (tokens/s/p) | 单步迭代时间 (s/step) | 
|:----:|:---------:|:----:|:---------------------:|:---------------:|:----------------:|
| NPUs | Bloom 7b | 1000 | 7.95 | 2034 | 64.55
|  参考  | Bloom 7B | 1000 | 9.894 |  2525 | 19.40 



## Bloom-7B推理
首先配置Bloom-7B 推理脚本: tasks/inference/generate_bloom_ptd_7B.sh 
```bash
# 根据您自己的 ascend-toolkit 路径，执行set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
# 请按实际情况修改模型权重路径和分词器路径
CHECKPOINT="./model_weights/Bloom-7B-Base-v0.1-tp8-pp1/"
TOKENIZER_PATH="./model_from_hf/Bloom-7B-Base/"
```

然后可直接启动generate_bloom_7b_ptd.sh

```bash
bash tasks/inference/generate_bloom_7b_ptd.sh
```

推理示例如下：

![Inference](../../sources/images/bloom/bloom7b-generate.png)

## Bloom-7B评测

配置Bloom-7B 评估脚本: tasks/evaluation/evaluate_bloom_7b_ptd.sh

```bash
# ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# 请按实际情况修改模型权重路径和分词器路径
CHECKPOINT="./model_weights/Bloom-7B-Base-v0.1-tp8-pp1/"
TOKENIZER_PATH="./model_from_hf/Bloom-7B-Base/"
# 配置任务和数据集路径
DATA_PATH="your dataset path"
TASK="your task"
```

启动评估

```bash
bash tasks/evaluation/evaluate_bloom_7B_ptd.sh
```

MMLU评测得分

|  数据集 |验证集  |参考准确率|NPU准确率|
|:---:|:---:|:---:|:---:|
| mmlu | test |0.254|0.251|

# Bloom-176B

## 训练

BLOOM 模型源于: [A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100).

> Scao, Teven Le, et al. "Bloom: A 176b-parameter open-access multilingual language model." arXiv preprint arXiv:2211.05100 (2022).

Bloom-176B 训练的硬件配置:

| **硬件** | **配置**        |
| -------- | --------------- |
| NPU      | 12x8 Atlas 900 A2 PODc |

## 脚本

1. 克隆仓库到本地服务器

```shell
git clone https://gitee.com/ascend/ModelLink.git
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout -f bcce6f
cp -r megatron ../ModelLink/
cd ..
cd ModelLink
git checkout 1.0
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
pip install torch-2.1.0-cp37-cp37m-linux_aarch64.whl
pip install torch_npu-2.1.0.XXX-cp37-cp37m-linux_aarch64.whl
pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 安装加速库
git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
git checkout 224ae35e8fc96778f957029d1371ddb623452a50
pip install -r requirements.txt 
pip3 install -e .
cd ..

# 安装其余依赖库
pip install -r requirements.txt 
```

3. 准备预训练权重

下载 Bloom-176B [权重](https://huggingface.co/bigscience/bloom/tree/main)

```shell
mkdir ./model_from_hf/Bloom-176B/
cd ./model_from_hf/Bloom-176B/
wget https://huggingface.co/bigscience/bloom/resolve/main/special_tokens_map.json
wget https://huggingface.co/bigscience/bloom/resolve/main/tokenizer.json
wget https://huggingface.co/bigscience/bloom/resolve/main/tokenizer_config.json
...
cd ../../
```

4. 权重转换
   将模型权重文件从 HuggingFace权重 格式转化为 Megatron 权重
   ***（该场景一般用于使能开源的HuggingFace模型在Megatron上进行训练）***

```shell
python tools/checkpoint/util.py \
    --model-type GPT \
    --loader loader_bloom_hf \
    --saver saver_megatron \
    --target-tensor-parallel-size 8 \
    --target-pipeline-parallel-size 5 \
    --load-dir ./model_from_hf/Bloom-176B/ \
    --save-dir ./model_weights/Bloom-176B-v0.1-pt8-pp5/ \
    --tokenizer-model None \
    --params-dtype bf16  
    # config.json中同字段对应的key值与其他模型不一致，将文件中的n_embed改为hidden_size， 将num_attention_heads修改为n_head。
```

任意并行切分策略的Megatron权重 格式转化为 HuggingFace权重
***（该场景一般用于将训练好的megatron模型重新转回HuggingFace格式）***

```shell
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python tools/checkpoint/util.py --model-type GPT \
    --loader megatron \
    --saver megatron \
    --save-model-type save_huggingface_llama \
    --load-dir ./model_weights/Bloom-176B-v0.1-pt8-pp5/ \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --embed-layernorm \
    --params-dtype bf16 \
    --save-dir ./model_from_hf/Bloom-176B/     # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/Bloom-176B/mg2hg/
```

5. 准备数据集

下载 Bloom 176B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

```shell
# 下载数据
cd dataset/
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..

# 处理数据        
mkdir ./dataset/Bloom-176B/
python ./tools/preprocess_data.py \
  --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
  --tokenizer-name-or-path ./model_from_hf/Bloom-176B/ \
  --output-prefix ./dataset/Bloom-176B/alpaca \
  --workers 4 \
  --log-interval 1000 \
  --tokenizer-type PretrainedFromHF
```

6. 配置 Bloom-176B 预训练脚本(Bloom-176B暂不支持Flash Attention): examples/bloom/pretrain_bloom_176b.sh

```shell
# 修改 MASTER_ADDR 为主节点
MASTER_ADDR=localhost

# 修改每个节点的节点序号，主节点序号为 0, 其余节点的序号依次增长到集群节点数量-1
NODE_RANK=0

# 修改数据集路径和词表路径
TOKENIZER_NAME_OR_PATH=./model_from_hf/Bloom-176B/
DATA_PATH=./dataset/Bloom-176B/alpaca_text_document
```

7. 启动 Bloom-176B 预训练脚本: examples/bloom/pretrain_bloom_176b.sh

在集群中的每个节点上启动 examples/bloom/pretrain_bloom_176b.sh 脚本

```shell
bash examples/bloom/pretrain_bloom_176b.sh
```

**注意**：如果使用多机训练，且没有设置数据共享，需要在训练启动脚本中增加`--no-shared-storage`参数，设置此参数之后将会根据分布式参数判断非主节点是否需要load数据，并检查相应缓存和生成数据。

## 性能

### 吞吐

Bloom-176B 在 **昇腾芯片** 和 **参考芯片** 上的性能对比:

| 设备 | 模型       | 总迭代数 | tokens吞吐 (tokens/p/s) |
| ---- | ---------- | -------- | ----------------------- |
| NPUs | Bloom-176B | 1000     | 100                    |
| 参考 | Bloom-176B | NA       | 107                     |


## 推理
首先配置Bloom-176B 推理脚本: tasks/inference/generate_bloom_ptd_176B.sh 
bloom 176b的推理需要5机，因此要用上面的  权重转换脚本重新切分，tp=8，pp=5

```bash
# 根据您自己的 ascend-toolkit 路径，执行set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
# 请按实际情况修改模型权重路径和分词器路径
CHECKPOINT="./model_weights/Bloom-176B-v0.1-tp8-pp5/"
TOKENIZER_PATH="./model_from_hf/Bloom-176B/"
```

然后可直接启动generate_bloom_176b_ptd.sh

```bash
bash tasks/inference/generate_bloom_176b_ptd.sh
```

推理示例如下：

![Inference](../../sources/images/bloom/bloom176b-generate.png)

## 评估

配置Bloom-176B 评估脚本: tasks/evaluation/evaluate_bloom_176b_ptd.sh

```bash
# ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# 请按实际情况修改模型权重路径和分词器路径
CHECKPOINT="./model_weights/Bloom-176B-v0.1-tp8-pp5/"
TOKENIZER_PATH="./model_from_hf/Bloom-176B/"
# 配置任务和数据集路径
DATA_PATH="your dataset path"
TASK="your task"
```

启动评估

```bash
bash tasks/evaluation/evaluate_bloom_176B_ptd.sh
```

评测得分

|  数据集 |验证集  |参考准确率|NPU准确率|
|:---:|:---:|:---:|:---:|
| boolq | dev |/|0.645|

