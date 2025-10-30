#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_EXEC_TIMEOUT=1200

NPUS_PER_NODE=2
MASTER_ADDR=localhost
MASTER_PORT=6080
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_MODEL="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"

TP=2
PP=1

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"


TRAINING_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --use-mcore-models \
    --prompt-type llama2 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --micro-batch-size 1 \
    --global-batch-size 1 \
    --train-iters 2000 \
    --lr 1.0e-6 \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.00 \
    --weight-decay 0.0 \
    --clip-grad 1.0 \
    --swiglu \
    --position-embedding-type rope \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --hidden-dropout 0.0 \
    --normalization RMSNorm \
    --use-flash-attn \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --init-method-std 0.01 \
    --initial-loss-scale 1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-8 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --no-pad-to-seq-lengths \
    --rotary-base 10000 \
    --norm-epsilon 1e-05 \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 32000 \
    --vocab-size 32000 \
    --bf16 \
    --num-layers 80 \
    --hidden-size 8192 \
    --ffn-hidden-size 28672 \
    --num-attention-heads 64 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --use-rotary-position-embeddings \
    --group-query-attention \
    --num-query-groups 8 \
    --tokenizer-not-use-fast \
    --attention-dropout 0.0 \
    --seed 42 \
    --exit-on-missing-checkpoint
"

FINETUNE_ARGS="
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --qlora \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 10000 \
    --eval-iters 0 \
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $TRAINING_ARGS \
    $FINETUNE_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --load $CKPT_LOAD_DIR \
    --save $CKPT_SAVE_DIR \
    --log-throughput \
    --distributed-backend nccl \
    | tee logs/tune_llama2_70b_mocre_qlora.log
