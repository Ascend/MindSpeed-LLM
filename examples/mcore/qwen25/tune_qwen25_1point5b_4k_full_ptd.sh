#!/bin/bash
export HCCL_CONNECT_TIMEOUT=1200
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CKPT_LOAD_DIR="your model ckpt path"
CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"

TP=2
PP=4
MBS=8
GBS=8
SEQ_LEN=4096

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
    
GPT_ARGS="
    --finetune \
    --stage sft \
    --prompt-type qwen \
    --is-instruction-dataset \
    --no-pad-to-seq-lengths \
    --tokenizer-not-use-fast \
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers 28 \
    --hidden-size 1536 \
    --ffn-hidden-size 8960 \
    --num-attention-heads 12 \
    --group-query-attention \
    --num-query-groups 2 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --rotary-base 1000000 \
    --train-iters 5000 \
    --lr 7.5e-7 \
    --weight-decay 0.0 \
    --lr-decay-style cosine \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --add-qkv-bias \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.02 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --norm-epsilon 1e-06 \
    --swiglu \
    --use-flash-attn \
    --use-rotary-position-embeddings \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --seed 42 \
    --bf16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 5000 \
    --eval-interval 5000 \
    --eval-iters 0 \
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    | tee logs/tune_mcore_qwen25_1point5b_4k_full_ptd.log
