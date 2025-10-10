#!/bin/bash
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export MULTI_STREAM_MEMORY_REUSE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=5600

NPUS_PER_NODE=16
MASTER_ADDR=localhost #主节点IP
MASTER_PORT=6010
NNODES=8
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"

TP=16
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
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --use-cp-send-recv-overlap \
    --reuse-fp32-param \
    --swap-attention \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --use-distributed-optimizer \
    --context-parallel-algo megatron_cp_algo \
    --micro-batch-size 1 \
    --global-batch-size 16 \
    --sequence-parallel \
    --use-flash-attn \
    --use-rotary-position-embeddings \
    --use-fused-rotary-pos-emb \
    --use-fused-rmsnorm \
    --use-fused-swiglu \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --num-layers 14 \
    --hidden-size 16384 \
    --ffn-hidden-size 53248 \
    --num-attention-heads 128 \
    --group-query-attention \
    --num-query-groups 16 \
    --seq-length 131072 \
    --rope-scaling-type llama3 \
    --rope-scaling-factor 8.0 \
    --low-freq-factor 1.0 \
    --high-freq-factor 4.0 \
    --original-max-position-embeddings 8192 \
    --max-position-embeddings 131072 \
    --make-vocab-size-divisible-by 1 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --rotary-base 500000 \
    --normalization RMSNorm \
    --norm-epsilon 1e-5 \
    --swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --kv-head-repeat-before-uly-alltoall \
    --lr 1.25e-6 \
    --train-iters 2000 \
    --lr-decay-style cosine \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --no-shared-storage \
    --bf16 \
    --use-mcore-models \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-throughput \
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
    --save ${CKPT_SAVE_DIR} \
    | tee logs/train_llama31_mcore_50b_128k.log
