#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CKPT_LOAD_DIR="your model ckpt path"
CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"

TP=8
PP=1
MBS=1
GBS=8
SEQ_LEN=4096
TRAIN_ITERS=2000
NUM_LAYERS=32

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

TUNE_ARGS="
    --finetune \
    --stage sft \
    --prompt-type llama3 \
    --reset-attention-mask \
    --is-instruction-dataset \
    --neat-pack \
"

GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --position-embedding-type rope \
    --rotary-base 500000 \
    --rope-scaling-type llama3 \
    --rope-scaling-factor 8.0 \
    --low-freq-factor 1.0 \
    --high-freq-factor 4.0 \
    --original-max-position-embeddings 8192 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --tokenizer-not-use-fast \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 128256 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --normalization RMSNorm \
    --norm-epsilon 1e-5 \
    --swiglu \
    --use-flash-attn \
    --use-fused-swiglu \
    --attention-dropout 0.0 \
    --init-method-std 0.02 \
    --hidden-dropout 0.0 \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --train-iters ${TRAIN_ITERS} \
    --lr 7.75e-7 \
    --min-lr 7.75e-8 \
    --lr-decay-style cosine \
    --weight-decay 1e-1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --seed 1234 \
    --bf16 \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --log-throughput \
    --eval-iters 0 \
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $TUNE_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    | tee logs/tune_llama31_8b_4k_full_pack.log
