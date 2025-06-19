#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_DETERMINISTIC=True
export ASCEND_LAUNCH_BLOCKING=1

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6014
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

echo "NODE_RANK ${NODE_RANK}"

CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_MODEL="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"
LORA_CKPT_DIR='your lora ckpt path'

TP=1
PP=8
EP=1
CP=1
CP_TYPE='megatron_cp_algo'
NUM_LAYERS=32

MOE_ARGS="
    --num-experts 8 \
    --expert-model-parallel-size ${EP} \
    --moe-router-topk 2 \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 0.02 \
    --moe-permutation-async-comm \
    --moe-layer-freq -1 \
    --first-k-dense-replace -1 \
"

GPT_ARGS="
    --save ${LORA_CKPT_DIR} \
    --lora-load ${LORA_CKPT_DIR} \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --variable-seq-lengths \
    --context-parallel-size ${CP} \
    --moe-token-dispatcher-type alltoall_seq \
    --use-mcore-models  \
    --disable-bias-linear \
    --seq-length 32768 \
    --max-position-embeddings 32768 \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 4096  \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --init-method-std 0.02 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --normalization RMSNorm \
    --prompt-type mixtral \
    --use-rotary-position-embeddings \
    --position-embedding-type rope \
    --no-check-for-nan-in-loss-and-grad \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --use-distributed-optimizer \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --group-query-attention \
    --num-query-groups 8 \
    --no-position-embedding \
    --vocab-size 32000 \
    --rotary-base 1000000 \
    --norm-epsilon 1e-5 \
    --no-masked-softmax-fusion \
    --use-fused-rotary-pos-emb \
    --use-flash-attn \
    --use-fused-swiglu \
    --use-fused-rmsnorm \

    --make-vocab-size-divisible-by 1 \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --load ${CKPT_LOAD_DIR} \
    --micro-batch-size 1 \
    --global-batch-size 16 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --initial-loss-scale 4096 \
    --lr 5.0e-6 \
    --train-iters 2000 \
    --lr-decay-style constant \
    --lr-decay-iters 1280 \
    --lr-warmup-iters 2 \
    --weight-decay 1e1 \
    --clip-grad 1.0 \
    --bf16 \
    --no-load-optim \
    --no-load-rng \
    --no-shared-storage  \
    --vocab-size 32000 \
    --finetune \
    --is-instruction-dataset \
"

DATA_ARGS="
    --data-path $DATA_PATH  \
    --split 100,0,0 \
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
  $MOE_ARGS \
  $GPT_ARGS \
  $DATA_ARGS \
  $OUTPUT_ARGS \
  $RL_ARGS \
  --distributed-backend nccl \
  | tee logs/dpo_mixtral_8x7b_lora_ptd.log  
