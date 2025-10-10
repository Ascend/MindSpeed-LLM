#!/bin/bash
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=5600

NPUS_PER_NODE=8
MASTER_ADDR=localhost #主节点IP
MASTER_PORT=6010
NNODES=8
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"

TP=8
PP=8
NUM_LAYERS=62

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --recompute-method block \
    --recompute-granularity full \
    --recompute-num-layers ${NUM_LAYERS} \
    --num-layer-list  7,8,8,8,8,8,8,7 \
    --sequence-parallel \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 16384 \
    --ffn-hidden-size 53248 \
    --num-attention-heads 128 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length 8192 \
    --max-position-embeddings 131072 \
    --rotary-base 500000 \
    --rope-scaling-type llama3 \
    --rope-scaling-factor 8.0 \
    --low-freq-factor 1.0 \
    --high-freq-factor 4.0 \
    --original-max-position-embeddings 8192 \
    --make-vocab-size-divisible-by 1 \
    --untie-embeddings-and-output-weights \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --disable-bias-linear \
    --group-query-attention \
    --num-query-groups 16 \
    --vocab-size 128256 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --norm-epsilon 1e-5 \
    --use-fused-rmsnorm \
    --use-fused-rotary-pos-emb \
    --swiglu \
    --use-flash-attn \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --use-fused-swiglu \
    --no-gradient-accumulation-fusion \
    --no-shared-storage \
    --micro-batch-size 1 \
    --global-batch-size 512  \
    --lr 1.25e-6 \
    --train-iters 2000 \
    --lr-decay-style cosine \
    --min-lr 1.25e-7 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 4096.0 \
    --adam-beta2 0.95 \
    --adam-eps 1e-5 \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --lr-warmup-fraction 0.01 \
    --bf16 \
    --log-throughput \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 10 \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    | tee logs/train_llama31_mcore_200b_8k.log
