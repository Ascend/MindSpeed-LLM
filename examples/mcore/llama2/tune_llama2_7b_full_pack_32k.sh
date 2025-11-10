#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1


NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6065
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_MODEL="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"

TP=2
PP=1
CP=4

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"


GPT_ARGS="
    --no-check-for-nan-in-loss-and-grad \
    --use-mcore-models \
    --reuse-fp32-param \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --load ${CKPT_LOAD_DIR} \
    --context-parallel-algo ulysses_cp_algo \
    --ulysses-degree-in-cp 2 \
    --sequence-parallel \
    --attention-mask-type causal \
    --num-workers 16 \
    --cp-window-size 1 \
    --use-fused-rotary-pos-emb \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --use-cp-send-recv-overlap \
    --log-throughput \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --sequence-parallel \
    --use-distributed-optimizer \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 11008 \
    --num-attention-heads 32 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --seq-length 32768 \
    --max-position-embeddings 32768 \
    --micro-batch-size 1 \
    --global-batch-size 64 \
    --make-vocab-size-divisible-by 1 \
    --lr 1.0e-6 \
    --train-iters 2000 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --use-flash-attn \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --weight-decay 1e-1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 1 \
    --adam-beta2 0.999 \
    --adam-eps 1e-5 \
    --disable-bias-linear \
    --group-query-attention \
    --num-query-groups 8 \
    --bf16 \
    --is-instruction-dataset \
    --finetune \
    --stage sft \

"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0 \
    --reset-attention-mask \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 1000 \
    --eval-iters 0 \
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CKPT_SAVE_DIR \
    | tee logs/tune_llama2_7b_full_32k.log
