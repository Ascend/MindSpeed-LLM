#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NPU_ASD_ENABLE=0

GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=2
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

LORA_CHECKPOINT="your lora weight"
LOAD_CHECKPOINT_PATH="your init model load path"
SAVE_CHECKPOINT_PATH="your model ckpt save path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"
TP=8
PP=2

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --num-layers 80 \
    --hidden-size 8192 \
    --ffn-hidden-size 22016 \
    --num-attention-heads 64 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --tokenizer-not-use-fast \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 2 \
    --global-batch-size 128 \
    --make-vocab-size-divisible-by 1 \
    --lr 1.0e-6 \
    --train-iters 5000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --use-flash-attn \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.0e-7 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --initial-loss-scale 524288.0 \
    --no-gradient-accumulation-fusion \
    --load ${LOAD_CHECKPOINT_PATH}  \
    --lora-load ${LORA_CHECKPOINT} \
    --no-load-optim \
    --no-load-rng \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --swiglu \
    --finetune \
    --is-instruction-dataset \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-target-modules query_key_value dense dense_h_to_4h dense_4h_to_h \
    --bf16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10 \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save ${SAVE_CHECKPOINT_PATH} \
    | tee logs/tune_llama_65b.log

