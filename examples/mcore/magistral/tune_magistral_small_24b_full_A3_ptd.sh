#!/bin/bash
export HCCL_CONNECT_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NPU_ASD_ENABLE=0

NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6006
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"


TP=8
PP=1
CP=1
MBS=1
GBS=32
SEQ_LENGTH=40960
TRAIN_ITERS=2000
CP_TYPE='ulysses_cp_algo'

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --use-distributed-optimizer \
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ${CP_TYPE} \
    --tokenizer-model ${TOKENIZER_PATH} \
    --tokenizer-type MagistralTokenizer \
    --max-position-embeddings ${SEQ_LENGTH} \
    --num-layers 40 \
    --hidden-size 5120 \
    --ffn-hidden-size 32768 \
    --num-attention-heads 32 \
    --seq-length ${SEQ_LENGTH} \
    --train-iters ${TRAIN_ITERS} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --use-flash-attn \
    --use-rotary-position-embeddings \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 131072 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --rotary-base 1000000000 \
    --use-fused-rotary-pos-emb \
    --normalization RMSNorm \
    --norm-epsilon 1e-5 \
    --use-fused-rmsnorm \
    --swiglu \
    --use-fused-swiglu \
    --hidden-dropout 0 \
    --attention-dropout 0 \
    --no-gradient-accumulation-fusion \
    --group-query-attention \
    --attention-softmax-in-fp32 \
    --num-query-groups 8 \
    --no-masked-softmax-fusion \
    --kv-channels 128 \
    --lr 1.25e-6 \
    --lr-decay-style constant \
    --min-lr 1.25e-7 \
    --lr-warmup-fraction 0 \
    --init-method-std 0.01 \
    --weight-decay 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --initial-loss-scale 4096 \
    --bf16 \
    --no-shared-storage
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
    --save-interval ${TRAIN_ITERS} \
    --eval-interval ${TRAIN_ITERS} \
    --eval-iters 0
"

TUNE_ARGS="
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --prompt-type magistral \
    --no-pad-to-seq-lengths \
    --tokenizer-not-use-fast \
    --calculate-per-token-loss
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $CKPT_ARGS \
    $OUTPUT_ARGS \
    $TUNE_ARGS \
    --distributed-backend nccl \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    | tee logs/tune_mcore_magistral_small_24b.log