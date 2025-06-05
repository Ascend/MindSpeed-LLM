#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6015
NNODES=1
NODE_RANK=0
WORLD_SIZE=$((NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"

TP=1
PP=8

DISTRIBUTED_ARGS="
    --worker_num $WORLD_SIZE \
    --local_worker_num $NPUS_PER_NODE \
    --log_dir="msrun_log" \
    --join=True \
    --cluster_time_out=300 \
    --master_port $MASTER_PORT
"

DIST_ALGO="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
"

MODEL_ARGS="
    --use-mcore-models \
    --num-layers 32 \
    --num-attention-heads 32 \
    --ffn-hidden-size 11008 \
    --hidden-size 4096 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --position-embedding-type rope \
    --disable-bias-linear \
    --hidden-dropout 0.0 \
    --attention-dropout 0.0 \
    --attention-softmax-in-fp32 \
    --init-method-std 0.01 \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-not-use-fast
    --swiglu \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --make-vocab-size-divisible-by 1 \
    --untie-embeddings-and-output-weights \
"

FINETUNE_ARGS="
    --micro-batch-size 8 \
    --global-batch-size 256 \
    --train-iters 2000 \
    --lr 1e-6 \
    --min-lr 1e-7 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --weight-decay 0.0 \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --initial-loss-scale 1 \
    --use-flash-attn \
    --no-masked-softmax-fusion \
    --no-gradient-accumulation-fusion \
    --is-instruction-dataset \
    --variable-seq-lengths \
    --placeholder-token ки \
    --reward-tokens + - \
    --finetune \
    --stage prm \
    --bf16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000 \
    --eval-interval 500 \
    --eval-iters 200 \
    --no-load-optim \
    --no-load-rng \
    --load $CKPT_LOAD_DIR \
    --save $CKPT_SAVE_DIR
"

msrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $DIST_ALGO \
    $MODEL_ARGS \
    $FINETUNE_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --ai-framework mindspore \
    --distributed-backend nccl \
    | tee logs/train_prm_llama2_7b_full_ms.log