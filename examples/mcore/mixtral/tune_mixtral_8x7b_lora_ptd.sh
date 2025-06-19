#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_DETERMINISTIC=True

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6005
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

CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_MODEL="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"
LORA_CHECKPOINT="your lora ckpt path"

TP=1
PP=4
EP=1
CP=1
CP_TYPE='megatron_cp_algo'
NUM_LAYERS=32

MOE_ARGS="
    --num-experts 8 \
    --expert-model-parallel-size ${EP} \
    --moe-router-topk 1 \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 0.02 \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall_seq \
    --moe-router-pre-softmax \
    --moe-layer-freq -1 \
    --first-k-dense-replace -1 \
"

GPT_ARGS="
    --num-workers 16 \
    --use-mcore-models \
    --disable-bias-linear \
    --seq-length 32768 \
    --max-position-embeddings 32768 \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 4096  \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --init-method-std 0.01 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --normalization RMSNorm \
    --position-embedding-type rope \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --group-query-attention \
    --num-query-groups 8 \
    --vocab-size 32000 \
    --rotary-base 1000000 \
    --no-masked-softmax-fusion \
    --use-fused-rotary-pos-emb \
    --use-flash-attn \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --no-check-for-nan-in-loss-and-grad \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --make-vocab-size-divisible-by 1 \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --variable-seq-lengths \
    --use-distributed-optimizer \
    --context-parallel-size ${CP} \
    --context-parallel-algo  ${CP_TYPE} \
    --save ${CKPT_SAVE_DIR} \
    --load ${CKPT_LOAD_DIR} \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --lr-warmup-fraction 0.01 \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --lr 1e-5 \
    --train-iters 10 \
    --lr-decay-iters 1 \
    --lr-decay-style constant \
    --min-lr 1.0e-6 \
    --weight-decay 0 \
    --clip-grad 1.0 \
    --bf16 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --no-shared-storage \
    --prompt-type mixtral \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
"

FINETUNE_ARGS="
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-fusion \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
"

DATA_ARGS="
    --data-path ${DATA_PATH} \
    --split 100,0,0 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 1 \
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
  $MOE_ARGS \
  $GPT_ARGS \
  $DATA_ARGS \
  $OUTPUT_ARGS \
  $FINETUNE_ARGS \
  --log-throughput \
  --distributed-backend nccl \
  | tee logs/tune_mixtral_8x7b_lora_ptd.log 