#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DATA_PATH="your data path"
TOKENIZER_MODEL="your tokenizer path"
CKPT_LOAD_DIR="your model load ckpt path"
LORA_CKPT_DIR='your lora ckpt dir'
TP=1
PP=8

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --lora-load ${LORA_CKPT_DIR} \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --sequence-parallel \
    --use-fused-rotary-pos-emb \
    --npu-deterministic \
    --no-pad-to-seq-lengths \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --use-mcore-models \
    --micro-batch-size 1 \
    --global-batch-size 32 \

    --use-flash-attn \
    --use-rotary-position-embeddings \
    --use-fused-rmsnorm \
    --use-fused-swiglu \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --seq-length 2048 \
    --max-position-embeddings 8192 \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 128256 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --rotary-base 500000 \
    --normalization RMSNorm \
    --norm-epsilon 1e-5 \
    --swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --lr 5e-6 \
    --train-iters 200 \
    --lr-decay-style constant \
    --min-lr 0.0 \
    --weight-decay 0.0 \
    --lr-warmup-fraction 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --finetune
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0 \
    --prompt-type llama3 \
    --is-instruction-dataset
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 0 \
"

RL_ARGS="
    --stage dpo \
    --dpo-loss-type sigmoid \
    --is-pairwise-dataset
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $RL_ARGS \
    --distributed-backend nccl \
    --load ${CKPT_LOAD_DIR} \
    --save ${LORA_CKPT_DIR} \
    | tee logs/dpo_llama3_8b_lora_mcore.log
