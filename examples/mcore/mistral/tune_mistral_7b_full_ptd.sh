#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6002
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

CKPT_SAVE_DIR="your model save lora ckpt path"
DATA_PATH="your data path"
TOKENIZER_MODEL="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"

TP=1
PP=8
NUM_LAYERS=32

GPT_ARGS="
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --no-pad-to-seq-lengths \
    --tokenizer-not-use-fast \
    --prompt-type mistral \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --sliding-window 4096 \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --seq-length 32768 \
    --max-position-embeddings 32768 \
    --micro-batch-size 1 \
    --global-batch-size 32 \
    --make-vocab-size-divisible-by 1 \
    --lr 1.25e-6 \
    --train-iters 2000 \
    --lr-decay-style constant \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --use-fused-swiglu \
    --use-rotary-position-embeddings \
    --use-fused-rotary-pos-emb \
    --swiglu \
    --use-flash-attn \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --weight-decay 1e-1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 1 \
    --adam-beta2 0.999 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    --bf16 \
    --rotary-base 1000000 \
    --use-mcore-models \
    --transformer-impl local
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

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
  $GPT_ARGS \
  $DATA_ARGS \
  $OUTPUT_ARGS \
  --distributed-backend nccl \
  | tee logs/tune_mcore_mistral_7b_full_2000.log
