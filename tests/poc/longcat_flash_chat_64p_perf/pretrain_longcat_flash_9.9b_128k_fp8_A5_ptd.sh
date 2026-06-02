#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CPU_AFFINITY_CONF=1,npu0:192-215,npu1:216-239,npu2:0-23,npu3:24-47,npu4:48-71,npu5:72-95,npu6:240-263,npu7:264-287
export TASK_QUEUE_ENABLE=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=3600
export STREAMS_PER_DEVICE=32

NPUS_PER_NODE=8
MASTER_ADDR=localhost #主节点IP，多机训练时请修改为实际主节点IP
MASTER_PORT=6066
NNODES=8              # 请根据实际节点数量修改
NODE_RANK=0           # 请根据实际节点Rank修改
WORLD_SIZE=$((NPUS_PER_NODE*$NNODES))


CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"


TP=1
PP=1
EP=64
CP=8
CP_TYPE='ulysses_cp_algo'
NUM_LAYERS=2
SEQ_LEN=131072
SUB_SEQ_LEN=16384
MBS=1
GBS=16
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
    --num-experts 128 \
    --num-zero-experts 64 \
    --moe-router-topk 12 \
    --moe-router-dtype fp32 \
    --moe-router-load-balancing-type softmax_topk \
    --moe-router-topk-scaling-factor 6.0 \
    --moe-ffn-hidden-size 2048 \
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall \
    --moe-aux-loss-coeff 0.001 \
    --fix-router \
"

MEM_ARGS="
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
"

FP8_ARGS="
    --fp8-format e4m3 \
    --fp8-recipe mxfp8 \
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
    --hidden-size 6144 \
    --ffn-hidden-size 12288 \
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
    --no-shared-storage \
"


torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $MLA_ARGS \
    $MEM_ARGS \
    $MOE_ARGS \
    $FP8_ARGS \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    --distributed-backend nccl | tee logs/pretrain_longcat_flash_9.9b_128k_fp8_A5_ptd_perf.log
