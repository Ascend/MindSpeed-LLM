#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_MODEL="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"
TP=1
PP=2

DISTRIBUTED_ARGS="
    --master_addr $MASTER_ADDR \
    --node_rank $NODE_RANK \
    --worker_num $WORLD_SIZE \
    --local_worker_num $NPUS_PER_NODE \
    --master_port $MASTER_PORT \
    --log_dir=msrun_log \
    --join=False \
    --cluster_time_out=300 \
    --bind_core=True \
"


GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --use-mcore-models \

    --use-fused-rmsnorm \
    --use-fused-rotary-pos-emb \
    --num-layers 18 \
    --hidden-size 2048 \
    --ffn-hidden-size 16384 \
    --num-attention-heads 8 \
    --group-query-attention \
    --num-query-groups 1 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --micro-batch-size 1 \
    --global-batch-size 256 \
    --kv-channels 256 \
    --make-vocab-size-divisible-by 1 \
    --lr 1.25e-6 \
    --train-iters 2000 \
    --lr-decay-style cosine \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --add-rmsnorm-offset \
    --norm-epsilon 1e-06 \
    --geglu \
    --input-embeds-norm \
    --use-flash-attn \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --use-distributed-optimizer \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --vocab-size 256000 \
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

msrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --ai-framework mindspore \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    | tee logs/train_gemma_2b_mcore.log