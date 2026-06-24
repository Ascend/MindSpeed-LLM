#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CPU_AFFINITY_CONF=1
export TASK_QUEUE_ENABLE=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=1800
export STREAMS_PER_DEVICE=32

NPUS_PER_NODE=8
MASTER_ADDR=localhost #主节点IP
MASTER_PORT=6211
NNODES=1
NODE_RANK=0
WORLD_SIZE=$((NPUS_PER_NODE*$NNODES))


CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"


TP=1
PP=1
EP=8
CP=8
CP_TYPE='ulysses_cp_algo'
NUM_LAYERS=1
SEQ_LEN=262144
SUB_SEQ_LEN=16384
MBS=1
GBS=8
TRAIN_ITERS=2000

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

MLA_ARGS="
    --multi-latent-attention \
    --qk-pos-emb-head-dim 64 \
    --qk-head-dim 128 \
    --q-lora-rank 1536 \
    --kv-lora-rank 512 \
    --v-head-dim 128 \
    --qk-layernorm \
    --enable-mla-scale-q-lora \
    --enable-mla-scale-kv-lora \
    --mla-fa-without-pad \
"

MOE_ARGS="
    --num-experts 16 \
    --num-zero-experts 8 \
    --moe-router-topk 12 \
    --moe-router-dtype fp32 \
    --moe-router-load-balancing-type softmax_topk \
    --moe-router-topk-scaling-factor 6.0 \
    --moe-ffn-hidden-size 2048 \
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall \
    --moe-aux-loss-coeff 0.001 \
"

MEM_ARGS="
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 1 \
"

GPT_ARGS="
    --disable-gloo-group \
    --transformer-impl transformer_engine \
    --spec mindspeed_llm.tasks.models.spec.longcat_spec layer_spec \
    --manual-gc \
    --manual-gc-interval 50 \
    --use-distributed-optimizer \
    --use-flash-attn \
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
    --expert-tensor-parallel-size 1 \
    --sequence-parallel \
    --context-parallel-size ${CP} \
    --context-parallel-algo ${CP_TYPE} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 3072 \
    --ffn-hidden-size 6144 \
    --num-attention-heads 64 \
    --kv-channels 64 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 131072 \
    --lr 1.25e-6 \
    --train-iters ${TRAIN_ITERS} \
    --lr-decay-style cosine \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --rotary-base 10000000 \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --distributed-timeout-minutes 120 \
    --seed 42 \
    --no-gradient-accumulation-fusion \
    --no-bias-dropout-fusion \
    --reset-attention-mask \
    --fix-sub-seq-length ${SUB_SEQ_LEN} \
    --no-shared-storage \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval ${TRAIN_ITERS} \
    --eval-interval ${TRAIN_ITERS} \
    --eval-iters 0 \
    --no-save-optim \
    --no-save-rng \
"


torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $MLA_ARGS \
    $MEM_ARGS \
    $MOE_ARGS \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    --distributed-backend nccl | tee logs/pretrain_longcat_flash_0.3b_256k_bf16_A5_ptd_acc.log
