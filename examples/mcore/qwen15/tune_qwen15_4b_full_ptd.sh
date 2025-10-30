#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6005
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"
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
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --no-pad-to-seq-lengths \
    --tokenizer-not-use-fast \
    --prompt-type qwen \
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers 40 \
    --hidden-size 2560 \
    --ffn-hidden-size 6912 \
    --num-attention-heads 20 \
    --tokenizer-type PretrainedFromHF \
    --load ${CKPT_LOAD_DIR} \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length 8192 \
    --max-position-embeddings 32768 \
    --micro-batch-size 1 \
    --global-batch-size 64 \
    --make-vocab-size-divisible-by 1 \
    --lr 1.25e-6 \
    --min-lr 1.25e-7 \
    --train-iters 2000 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.02 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --use-flash-attn \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --use-fused-rotary-pos-emb \
    --use-distributed-optimizer \
    --use-rotary-position-embeddings \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --weight-decay 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --rotary-base 1000000 \
    --add-qkv-bias \
    --initial-loss-scale 8192 \
    --no-gradient-accumulation-fusion \
    --lr-warmup-fraction 0.01 \
    --no-load-optim \
    --no-load-rng \
    --padded-vocab-size 151936 \
    --bf16 \
    --seed 1234
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save ${CKPT_SAVE_DIR} \
    | tee logs/tune_mcore_qwen15_4b_full_2000.log
