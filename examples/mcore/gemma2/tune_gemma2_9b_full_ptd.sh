#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6014
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_MODEL="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"

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
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --spec mindspeed_llm.tasks.models.spec.gemma2_spec layer_spec \
    --use-mcore-models \

    --use-fused-rmsnorm \
    --use-fused-rotary-pos-emb \
    --gelu-tanh \
    --post-norm \
    --query-pre-attn-scalar 256 \
    --output-logit-softcapping 30.0 \
    --interleave-sliding-window 4096 \
    --num-layers 42 \
    --num-layer-list 5,5,5,5,5,5,5,7 \
    --hidden-size 3584 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 16 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --micro-batch-size 1 \
    --global-batch-size 64 \
    --kv-channels 256 \
    --group-query-attention \
    --num-query-groups 8 \
    --make-vocab-size-divisible-by 1 \
    --lr 1e-6 \
    --train-iters 2000 \
    --lr-decay-style cosine \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.02 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --add-rmsnorm-offset \
    --norm-epsilon 1e-06 \
    --input-embeds-norm \
    --use-flash-attn \
    --use-distributed-optimizer \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --weight-decay 1e-1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --initial-loss-scale 1 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --vocab-size 256000 \
    --bf16 \
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --tokenizer-padding-side right \
    --no-pad-to-seq-lengths \
    --tokenizer-not-use-fast \
    --prompt-type gemma \
    --rotary-base 10000
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
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
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    | tee logs/tune_gemma2_9b_full_ptd.log
