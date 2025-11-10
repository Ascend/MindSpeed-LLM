#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_DETERMINISTIC=true

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6024
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

basepath=$(cd `dirname $0`; cd ../../../; pwd)

CKPT_SAVE_DIR="/data/ci/cache"
DATA_PATH="/data/ci/datasets/processed/gemma2-dataset-alpaca/alpaca_text_document"
TOKENIZER_MODEL="/data/ci/models/gemma2/hf/gemma2-9b-hf"
CKPT_LOAD_DIR="/data/ci/models/gemma2/mg/gemma2-9b-mg-tp8pp1-mcore-base"

TP=8
PP=1

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --spec mindspeed_llm.tasks.models.spec.gemma2_spec layer_spec \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --use-mcore-models \
    --transformer-impl local \
    --use-fused-rmsnorm \
    --use-fused-rotary-pos-emb \
    --gelu-tanh \
    --post-norm \
    --query-pre-attn-scalar 256 \
    --output-logit-softcapping 30.0 \
    --interleave-sliding-window 4096 \
    --num-layers 42 \
    --hidden-size 3584 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 16 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --kv-channels 256 \
    --group-query-attention \
    --num-query-groups 8 \
    --make-vocab-size-divisible-by 1 \
    --lr 1.25e-6 \
    --train-iters 15 \
    --lr-decay-style cosine \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
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
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --tokenizer-padding-side left \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --vocab-size 256000 \
    --log-throughput \
    --finetune \
    --bf16
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

torchrun $DISTRIBUTED_ARGS $basepath/pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --load ${CKPT_LOAD_DIR} \
