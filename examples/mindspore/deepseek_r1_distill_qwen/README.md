# MindSpeed MindSpore后端Deepseek R1蒸馏Qwen2.5 math 7B模型指南

Deepseek R1蒸馏Qwen2.5-Math-7B模型，该指导采用OpenR1-Math-220K数据集（已经过DeepSeek-R1蒸馏）进行微调，基于一台800T A2昇腾服务器进行。

## 动态图环境搭建

拉取镜像
`docker pull swr.cn-central-221.ovaijisuan.com/mindformers/cann8.2.rc1_mindspore2.6.0:20250520`

创建容器后续操作在容器中进行。/mnt/data/用于存放方案库，可以根据需要更改

```
docker run -it --name=msms --ipc=host --network=host --privileged=true \
        --device=/dev/davinci0 \
        --device=/dev/davinci1 \
        --device=/dev/davinci2 \
        --device=/dev/davinci3 \
        --device=/dev/davinci4 \
        --device=/dev/davinci5 \
        --device=/dev/davinci6 \
        --device=/dev/davinci7 \
        --device=/dev/davinci_manager \
        --device=/dev/devmm_svm \
        --device=/dev/hisi_hdc \
        -v /usr/local/sbin/:/usr/local/sbin/ \
        -v /etc/hccn.conf:/etc/hccn.conf \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /etc/vnpu.cfg:/etc/vnpu.cfg \
        -v /mnt/data/:/mnt/data/ \
        --pids-limit 409600 \
        --shm-size="500g" \
        swr.cn-central-221.ovaijisuan.com/mindformers/cann8.2.rc1_mindspore2.6.0:20250520 \
        /bin/bash

cd /mnt/data/
git clone https://gitee.com/ascend/MindSpeed-Core-MS.git
cd MindSpeed-Core-MS

```
按顺序执行如下命令
```
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source test_convert_llm.sh
```

## 数据集及模型准备
### 模型权重下载

Qwen25Math7B文件夹用于存放权重文件，可以修改
```
cd /mnt/data
mkdir Qwen25Math7B
pip install modelscope
modelscope download --model Qwen/Qwen2.5-Math-7B --local_dir /mnt/data/Qwen25Math7B

```
### OpenR1-Math-220K数据集下载
OpenR1Math220K文件夹用于存放数据，可以修改

```
cd /mnt/data
mkdir OpenR1Math220K
cd OpenR1Math220K
wget https://hf-mirror.com/datasets/open-r1/OpenR1-Math-220k/resolve/main/data/train-00000-of-00010.parquet
wget https://hf-mirror.com/datasets/open-r1/OpenR1-Math-220k/resolve/main/data/train-00001-of-00010.parquet
wget https://hf-mirror.com/datasets/open-r1/OpenR1-Math-220k/resolve/main/data/train-00002-of-00010.parquet
wget https://hf-mirror.com/datasets/open-r1/OpenR1-Math-220k/resolve/main/data/train-00003-of-00010.parquet
wget https://hf-mirror.com/datasets/open-r1/OpenR1-Math-220k/resolve/main/data/train-00004-of-00010.parquet
wget https://hf-mirror.com/datasets/open-r1/OpenR1-Math-220k/resolve/main/data/train-00005-of-00010.parquet
wget https://hf-mirror.com/datasets/open-r1/OpenR1-Math-220k/resolve/main/data/train-00006-of-00010.parquet
wget https://hf-mirror.com/datasets/open-r1/OpenR1-Math-220k/resolve/main/data/train-00007-of-00010.parquet
wget https://hf-mirror.com/datasets/open-r1/OpenR1-Math-220k/resolve/main/data/train-00008-of-00010.parquet
wget https://hf-mirror.com/datasets/open-r1/OpenR1-Math-220k/resolve/main/data/train-00009-of-00010.parquet

```

## 数据转换

将下载的.parquet格式数据进行处理，首先创建文件夹用于存放处理后的数据。

```
cd /mnt/data
mkdir OpenR1Math220K_handled
```
修改/mnt/data/MindSpeed-Core-MS/MindSpeed-LLM/examples/mcore/qwen25_math/data_convert_qwen25_math_pretrain.sh文件，改成如下。如果要处理多个parquet数据，指定数据文件夹即可。

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python ./preprocess_data.py \
    --input /mnt/data/JRMindSpeed/OpenR1Math220K/train-00000-of-00010.parquet \
    --tokenizer-name-or-path /mnt/data/JRMindSpeed/Qwen25Math7B/ \
    --output-prefix /mnt/data/JRMindSpeed/OpenR1Math220K_handled/sharegpt \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF \
    --handler-name SharegptStyleInstructionHandler \
    --prompt-type qwen_math_r1 \
	--map-keys '{"messages":"messages", "tags":{"role_tag": "role","content_tag": "content","user_tag": "user","assistant_tag": "assistant","system_tag": "system"}}'
```
执行如下命令运行脚本

```
cd /mnt/data/MindSpeed-Core-MS/MindSpeed-LLM
bash /mnt/data/MindSpeed-Core-MS/MindSpeed-LLM/examples/mcore/qwen25_math/data_convert_qwen25_math_pretrain.sh
```

## 原始权重转换
将原始的hg权重转换为megatron mcore权重，首先创建文件夹用于存放转换后的权重

```
cd /mnt/data
mkdir Qwen25Math7B_transfered
```
修改/mnt/data/MindSpeed-Core-MS/MindSpeed-LLM/examples/mcore/qwen25_math/ckpt_convert_qwen25_math_hf2mcore.sh,改成如下

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt.py \
       --use-mcore-models \
       --model-type GPT \
       --load-model-type hf \
       --save-model-type mg \
       --target-tensor-parallel-size 2 \
       --target-pipeline-parallel-size 4 \
       --add-qkv-bias \
       --load-dir /mnt/data/Qwen25Math7B/ \
       --save-dir /mnt/data/Qwen25Math7B_transfered/ \
       --tokenizer-model /mnt/data/Qwen25Math7B/tokenizer.json \
       --model-type-hf llama2 \
       --params-dtype bf16
```
执行如下命令运行脚本

```
cd /mnt/data/MindSpeed-Core-MS/MindSpeed-LLM
bash /mnt/data/MindSpeed-Core-MS/MindSpeed-LLM/examples/mcore/qwen25_math/ckpt_convert_qwen25_math_hf2mcore.sh
```
## 采用原始权重推理

方法一：生成式推理

修改/mnt/data/MindSpeed-Core-MS/MindSpeed-LLM/examples/mcore/qwen25_math/generate_qwen25_math_7b_ptd.sh，改成如下,日志路径可以根据需要修改：


```
#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CHECKPOINT="/mnt/data/Qwen25Math7B_transfered/"
TOKENIZER_PATH="/mnt/data/Qwen25Math7B/"

TP=2
PP=4
MBS=1
SEQ_LENGTH=4096

DISTRIBUTED_ARGS="
    --worker_num $WORLD_SIZE \
	--local_worker_num $NPUS_PER_NODE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --node_rank $NODE_RANK \
	--log_dir "/mnt/data/MindSpeed-Core-MS/MindSpeed-LLM/msrunlog" \
	--join=True
"

msrun $DISTRIBUTED_ARGS inference.py \
       --ai-framework mindspore \
	   --use-mcore-models \
       --input-layernorm-in-fp32 \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --num-layers 28 \
       --hidden-size 3584  \
       --num-attention-heads 28  \
       --ffn-hidden-size 18944 \
       --max-position-embeddings ${SEQ_LENGTH} \
       --seq-length ${SEQ_LENGTH} \
       --disable-bias-linear \
       --add-qkv-bias \
       --group-query-attention \
       --num-query-groups 4 \
       --swiglu \
       --normalization RMSNorm \
       --norm-epsilon 1e-6 \
       --position-embedding-type rope \
       --rotary-base 10000 \
       --make-vocab-size-divisible-by 1 \
       --padded-vocab-size 152064 \
       --micro-batch-size ${MBS} \
       --max-new-tokens 512 \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --tokenizer-not-use-fast \
       --hidden-dropout 0 \
       --attention-dropout 0 \
       --untie-embeddings-and-output-weights \
       --no-gradient-accumulation-fusion \
       --attention-softmax-in-fp32 \
       --seed 42 \
       --load ${CHECKPOINT} \
       --exit-on-missing-checkpoint
```
执行如下命令运行脚本

```
cd /mnt/data/MindSpeed-Core-MS/MindSpeed-LLM
bash /mnt/data/MindSpeed-Core-MS/MindSpeed-LLM/examples/mcore/qwen25_math/generate_qwen25_math_7b_ptd.sh
```


方法二：chat推理（暂不支持）

qwen25_math没有提供chat脚本，可以使用qwen25中的chat脚本。修改/mnt/data/MindSpeed-Core-MS/MindSpeed-LLM/examples/mcore/qwen25/chat_qwen25_7b_ptd.sh  ，改成如下
```
#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CHECKPOINT="/mnt/data/Qwen25Math7B_transfered/"
TOKENIZER_PATH="/mnt/data/Qwen25Math7B/"

TP=2
PP=4

SEQ_LENGTH=4096

DISTRIBUTED_ARGS="
    --worker_num $WORLD_SIZE \
	--local_worker_num $NPUS_PER_NODE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --node_rank $NODE_RANK \
	--log_dir "/mnt/data/MindSpeed-Core-MS/MindSpeed-LLM/msrunlog" \
	--join=True
"

msrun $DISTRIBUTED_ARGS inference.py \
	   --ai-framework mindspore \
	   --task chat \
       --prompt-type qwen \
       --use-mcore-models \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --num-layers 28 \
       --hidden-size 3584  \
       --num-attention-heads 28  \
       --ffn-hidden-size 18944 \
       --max-position-embeddings ${SEQ_LENGTH} \
       --seq-length ${SEQ_LENGTH} \
       --disable-bias-linear \
       --add-qkv-bias \
       --group-query-attention \
       --num-query-groups 4 \
       --swiglu \
       --use-fused-swiglu \
       --normalization RMSNorm \
       --norm-epsilon 1e-6 \
       --use-fused-rmsnorm \
       --position-embedding-type rope \
       --rotary-base 1000000 \
       --use-fused-rotary-pos-emb \
       --make-vocab-size-divisible-by 1 \
       --padded-vocab-size 152064 \
       --micro-batch-size 1 \
       --max-new-tokens 512 \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --tokenizer-not-use-fast \
       --hidden-dropout 0 \
       --attention-dropout 0 \
       --untie-embeddings-and-output-weights \
       --no-gradient-accumulation-fusion \
       --attention-softmax-in-fp32 \
       --seed 42 \
       --load ${CHECKPOINT} \
       --exit-on-missing-checkpoint
```
执行如下命令运行脚本

```
cd /mnt/data/MindSpeed-Core-MS/MindSpeed-LLM
bash /mnt/data/MindSpeed-Core-MS/MindSpeed-LLM/examples/mcore/qwen25/chat_qwen25_7b_ptd.sh
```
## 采用转换后数据进行全量微调
优于qwen2.5 math 7b没有提供全量微调脚本，因此可以使用qwen2.5 7b的全量微调脚本，两者的模型结构是一致的。
创建文件夹，用于存放微调后的权重
```
cd /mnt/data
mkdir Qwen25Math7B_tune
```

修改/mnt/data/MindSpeed-Core-MS/MindSpeed-LLM/examples/mcore/qwen25/tune_qwen25_7b_full_ptd.sh，改成如下：

```
#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CKPT_LOAD_DIR="/mnt/data/Qwen25Math7B_transfered/"
CKPT_SAVE_DIR="/mnt/data/Qwen25Math7B_tune/"
DATA_PATH="/mnt/data/OpenR1Math220K_handled/sharegpt"
TOKENIZER_PATH="/mnt/data/Qwen25Math7B/"


TP=2
PP=4
SEQ_LEN=4096
MBS=1
GBS=8

DISTRIBUTED_ARGS="
    --worker_num $WORLD_SIZE \
	--local_worker_num $NPUS_PER_NODE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --node_rank $NODE_RANK \
	--log_dir "/mnt/data/MindSpeed-Core-MS/MindSpeed-LLM/msrunTUNElog" \
	--join=True
"
TUNE_ARGS="
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --variable-seq-lengths \
    --tokenizer-not-use-fast \
    --prompt-type qwen \
"

GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers 28  \
    --hidden-size 3584  \
    --ffn-hidden-size 18944 \
    --num-attention-heads 28  \
    --max-position-embeddings ${SEQ_LEN} \
    --seq-length ${SEQ_LEN} \
    --disable-bias-linear \
    --add-qkv-bias \
    --group-query-attention \
    --num-query-groups 4 \
    --use-flash-attn \
    --swiglu \
    --use-fused-swiglu \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --use-fused-rmsnorm \
    --position-embedding-type rope \
    --rotary-base 1000000 \
    --use-fused-rotary-pos-emb \
    --untie-embeddings-and-output-weights \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 152064 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --train-iters 1500 \
    --lr 1.25e-6 \
    --lr-decay-style cosine \
    --min-lr 1.25e-7 \
    --lr-warmup-fraction 0.01 \
    --init-method-std 0.01 \
    --weight-decay 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --bf16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

CKPT_ARGS="
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --seed 1234 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 500 \
    --eval-interval 1500 \
    --eval-iters 0 \
    --log-throughput
"

msrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $CKPT_ARGS \
    $OUTPUT_ARGS \
    $TUNE_ARGS \
	--ai-framework mindspore \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    --distributed-backend nccl
```
执行如下命令运行脚本

```
cd /mnt/data/MindSpeed-Core-MS/MindSpeed-LLM
bash /mnt/data/MindSpeed-Core-MS/MindSpeed-LLM/examples/mcore/qwen25/tune_qwen25_7b_full_ptd.sh
```
## 微调后权重推理


和采用原始权重推理方法一致，generate生成式推理，或者chat推理（暂不支持），只需要在原来基础上修改CHECKPOINT权重路径，改为微调后的权重路径即可。


## 微调后权重转换
将微调后的权重转换为hg权重，转换后的hf权重会保存在原始权重文件夹下面的mg2hf文件夹中。

修改/mnt/data/MindSpeed-Core-MS/MindSpeed-LLM/examples/mcore/qwen25_math/ckpt_convert_qwen25_math_mcore2hf.sh,改成如下

```
# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置并行策略
python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --model-type-hf llama2 \
    --load-model-type mg \
    --save-model-type hf \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --add-qkv-bias \
    --load-dir /mnt/data/Qwen25Math7B_tune/ \
    --save-dir /mnt/data/Qwen25Math7B/  #填入原始HF模型路径，新权重会存于原始权重下的mg2hg文件夹中
```
执行如下命令运行脚本

```
cd /mnt/data/MindSpeed-Core-MS/MindSpeed-LLM
bash /mnt/data/MindSpeed-Core-MS/MindSpeed-LLM/examples/mcore/qwen25_math/ckpt_convert_qwen25_math_mcore2hf.sh
```