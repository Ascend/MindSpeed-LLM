## 模型脚本声明

目前tests/minspore/0day/qwen3下的模型仅支持0day首发下基本功能跑通，处于内部测试阶段，未完成充分的性能测试和验收。在实际使用中可能存在未被发现的问题，待后续充分验证后会发布正式版本。

## 权重转换
### 使用说明
支持将HuggingFace权重转换为mcore权重，用于与训练、微调等任务。
### 启动脚本
使用Qwen3模型目录下的[HuggingFace转Megatron脚本](https://gitcode.com/ascend/MindSpeed-LLM/blob/master/convert_ckpt.py)
```commandline
#稠密模型
bash tests/mindspore/0day/qwen3/ckpt_convert_qwen3_dense_hf2mcore.sh

#稀疏模型
bash tests/mindspore/0day/qwen3/ckpt_convert_qwen3_moe_hf2mcore.sh
```
### 相关参数说明
【--target-tensor-parallel-size】

张量并行度，默认值为1。

【--target-pipeline-parallel-size】

流水线并行度，默认值为1。

【--target-expert-parallel-size】

专家并行度，默认值为1。

【--load-dir】

已经反量化为bf16数据格式的huggingface权重。

【--save-dir】

转换后的megatron格式权重的存储路径。

【--num-layers】

模型层数,如配置空操作层，num-layers的值应为总层数加上空操作层层数。

【--spec】

qwen3模型此处传入 mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec

【--model-type-hf】

qwen3稀疏模型传入qwen3，稠密模型传入qwen3-moe

【--moe-grouped-gemm】

模型是否开启通信掩盖

## 数据预处理
### 常用与训练数据集

[Alpaca数据集](https://huggingface.co/datasets/tatsu-lab/alpaca)

[Enwiki数据集](https://huggingface.co/datasets/lsb/enwiki20230101)

[C4数据集](https://huggingface.co/datasets/allenai/c4)

[ChineseWebText](https://huggingface.co/datasets/CASIA-LM/ChineseWebText)

### 数据集下载
数据集下载可以基于网页直接下载，也可以基于命令行下载，比如：
```commandline
mkdir dataset
cd dataset/
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..
```

### 数据集处理
#### 数据集处理方法
执行以下脚本进行数据处理：
```commandline
bash ./tests/mindspore/0day/qwen3/data_convert_qwen3.sh
```

#### 参数说明
【--input】

可以直接输入到数据集目录或具体文件，如果是目录，则处理全部文件, 支持 .parquet \ .csv \ .json \ .jsonl \ .txt \ .arrow 格式， 同一个文件夹下的数据格式需要保持一致

【--tokenizer-name-or-path】

tokenizer的路径，指向tokenizer所在的目录

【--output-prefix】

输出的目录及前缀名

【--handler-name】

当前预训练默认使用 GeneralPretrainHandler，支持的是预训练数据风格，提取数据的text列，格式如下：
```commandline
[
  {"text": "document"},
  {"other keys": "optional content"}
]
```
用户可结合具体数据处理需求添加新的Handler进行数据处理

【--json-keys】

从文件中提取的列名列表，默认为 text，可以为 text, input, title 等多个输入，结合具体需求及数据集内容使用，如：
```commandline
--json-keys text input output \
```

#### 处理结果
预训练数据集处理结果如下：
```commandline
./dataset/alpaca_qwen3_text_document.bin
./dataset/alpaca_qwen3_text_document.idx
```
预训练时，数据集路径--data-path参数传入 ./dataset/alpaca_qwen3_text_document 即可


## 代码适配
执行以下命令拉去MindSpore-Core-MS代码仓：
```commandline
git clone -b feature-0.2 https://gitcode.com/ascend/MindSpeed-Core-MS.git
```
基于MindSpeed-Core-MS代码仓可以进行代码拉取、代码一键适配等功能，请确保环境已完成以下配置：
* 所部署容器网络可用，python已安装
* git已完成配置，可以正常进行clone操作

执行以下命令进行一键适配：
```commandline
cd MindSpeed-Core-MS
source test_convert_llm.sh
```
根据以下命令设置环境变量：
```commandline
MindSpeed_Core_MS_PATH=$(pwd)
export PYTHONPATH=${MindSpeed_Core_MS_PATH}/msadapter/mindtorch:${MindSpeed_Core_MS_PATH}/Megatron-LM:${MindSpeed_Core_MS_PATH}/MindSpeed:${MindSpeed_Core_MS_PATH}/MindSpeed-LLM:${MindSpeed_Core_MS_PATH}/transformers/src/:$PYTHONPATH
```

## 训练任务拉起
### 配置训练参数
预训练脚本保存在 MindSpeed-LLM/tests/mindspore/0day/qwen3 中各模型文件夹下：pretrain_xxx_ms.sh

需根据实际情况修改路径和参数值：

示例：

MindSpeed-LLM/tests/mindspore/0day/qwen3/qwen3-0.6b/pretrain_qwen3_0point6_ms.sh

路径配置：包括权重保存路径、权重加载路径、词表路径、数据集路径
```commandline
# 根据实际情况配置权重保存、权重加载、词表、数据集路径
# 注意：提供的路径需要加双引号
CKPT_SAVE_DIR="./ckpt/qwen3"  # 训练完成后的权重保存路径
CKPT_LOAD_DIR="./model_weights/qwen3/"  # 权重加载路径，填入权重转换时保存的权重路径
DATA_PATH="./dataset/alpaca_qwen3_text_document"  # 数据集路径，填入数据预处理时保存的数据路径
TOKENIZER_MODEL="./model_from_hf/qwen3/tokenizer"  # 词表路径，填入下载的开源权重词表路径
```
【单机运行】
```commandline
GPUS_PER_NODE=8
MASTER_PORT=6000
MASTER_ADDR=locahost # 主节点IP
NNODES=1
NODE_RANK=0  
MASTER_PORT=9110
log_dir=msrun_log_pretrain # log输出路径
```
【多机运行】
```commandline
# 根据分布式集群实际情况配置分布式参数
GPUS_PER_NODE=8
MASTER_ADDR="your master node IP" #主节点IP
MASTER_PORT=6000
NNODES=4 # 集群里的节点数，以实际情况填写
NODE_RANK="current node id" # 当前节点RANK,主节点为0，其他可以使1,2..
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
```

### 启动训练任务
初始化环境变量：
```commandline
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```
环境变量配置完成后执行训练脚本：
```commandline
cd MindSpeed-LLM
bash ./tests/mindspore/0day/qwen3/qwen3-0.6b/pretrain_qwen3_0point6_ms.sh
```

