#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True


NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6024
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"

TP=1
PP=2

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
    --is-instruction-dataset \
    --variable-seq-lengths \
    --prompt-type deepseek3 \
"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --use-mcore-models \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --position-embedding-type rope \
    --rotary-base 500000 \
    --normalization RMSNorm \
    --norm-epsilon 1e-5 \
    --swiglu \
    --micro-batch-size 1 \
    --global-batch-size 64 \
    --use-flash-attn \
    --rope-scaling-type llama3 \
    --rope-scaling-factor 8.0 \
    --rotary-percent 1.0 \
    --low-freq-factor 1.0 \
    --high-freq-factor 4.0 \
    --original-max-position-embeddings 8192 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --tokenizer-not-use-fast \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 128256 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.02 \
    --hidden-dropout 0.0 \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --lr 1e-7 \
    --train-iters 2000 \
    --lr-decay-style constant \
    --weight-decay 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --bf16 \
    --seed 42 \
    --vocab-size 128256 \
    --reuse-fp32-param \
    --use-distributed-optimizer \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 10 \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $TUNE_ARGS \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    | tee logs/tune_mcore_distill_llama_8b_full_8k.log