#!/bin/bash

export HCCL_CONNECT_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CPU_AFFINITY_CONF=1
export TASK_QUEUE_ENABLE=2
export STREAMS_PER_DEVICE=32
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_ASD_ENABLE=0

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"          # 原始 .parquet/.jsonl，训练初始化时自动预处理
TOKENIZER_PATH="your tokenizer path"
HF_CFG_DIR="your huggingface config dir"   # 含 config.json 等，用于 mcore->HF 自动转换
CKPT_LOAD_DIR="None"                        # 从头预训练填 None；从已有 HF 权重继续预训练填 HF 权重目录

TP=1
PP=1

LR=3e-4
MIN_LR=3e-5

MBS=6
GBS=48
SEQ_LENGTH=4096
TRAIN_ITERS=101726
SAVE_INTERVAL=29529

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# Dense GatedDeltaNet: pass NO MoE args; num_experts unset selects the dense
# MLP branch in qwen3_next_spec automatically.

OPTIMIZE_ARGS="
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer \
"

TRAIN_ARGS="
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --lr ${LR} \
    --lr-decay-style cosine \
    --min-lr ${MIN_LR} \
    --weight-decay 1e-1 \
    --lr-warmup-iters 1024 \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --seed 42 \
    --bf16 \
    --train-iters ${TRAIN_ITERS} \
    --seq-length ${SEQ_LENGTH} \
"

MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
"

# --full-attention-interval N: 1 full-attention layer every N layers, rest GDN
# => (N-1):1 GDN:attn. 3 here means 2:1.
GPT_ARGS="
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.qwen3_next_spec layer_spec \
    --qk-layernorm \
    --full-attention-interval 3 \
    --mamba-d-conv 4 \
    --mamba-expand 1 \
    --linear-key-head-dim 256 \
    --linear-num-key-heads 4 \
    --linear-num-value-heads 4 \
    --linear-value-head-dim 256 \
    --partial-rotary-factor 0.25 \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --num-layers 24 \
    --hidden-size 1024 \
    --ffn-hidden-size 2816 \
    --num-attention-heads 16 \
    --tokenizer-type PretrainedFromHF \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 32000 \
    --rotary-base 10000 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --rmsnorm-weight-in-fp32 \
    --add-rmsnorm-offset \
    --attention-softmax-in-fp32 \
    --no-gradient-accumulation-fusion \
    --group-query-attention \
    --num-query-groups 2 \
    --norm-epsilon 1e-06 \
    --mamba-chunk-size 64 \
    --use-triton-gdn \
    --loss-compute-mode default \
    --loss-chunk-size 1024 \
    --no-enable-linear-qkv \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0 \
    --handler-name GeneralPretrainHandler \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval ${SAVE_INTERVAL} \
    --eval-interval ${TRAIN_ITERS} \
    --eval-iters 0 \
    --no-load-optim \
    --no-load-rng \
"

# 训练合一（参考 docs/zh/pytorch/training/pretrain/mcore/train_from_hf.md）：
# 默认从头预训练（CKPT_LOAD_DIR=None），训练保存时自动将 Megatron 权重转为 HuggingFace 格式
# （--enable-mg2hf-convert），无需单独的权重转换脚本；数据用原始文件，训练时自动预处理。
# 若 CKPT_LOAD_DIR 指向 HuggingFace 权重，则同时自动做 HF->Megatron 加载转换以继续预训练。
HF2MG_ARGS=""
if [ "${CKPT_LOAD_DIR}" != "None" ]; then
    HF2MG_ARGS="--enable-hf2mg-convert"
fi
CKPT_ARGS="
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    --model-type-hf qwen3-next \
    --enable-mg2hf-convert \
    --hf-cfg-dir ${HF_CFG_DIR} \
    ${HF2MG_ARGS} \
"

mkdir -p logs
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $CKPT_ARGS \
    $OUTPUT_ARGS \
    $OPTIMIZE_ARGS \
    $TRAIN_ARGS \
    $MODEL_PARALLEL_ARGS \
    --distributed-backend nccl \
    --transformer-impl local \
    | tee logs/pretrain_gated_deltanet_340M_4K_A2_ptd.log
