#!/bin/bash

export HCCL_CONNECT_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NPU_ASD_ENABLE=0

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6013
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

basepath=$(cd `dirname $0`; cd ../../../; pwd)

# please fill these path configurations
CKPT_LOAD_DIR="/data/ci/models/qwen2-moe/mg/qwen2-moe-t1p2e2-new-spec"
DATA_PATH="/data/ci/datasets/processed/qwen2_moe_dataset/enwiki_text_document"
TOKENIZER_PATH="/data/ci/models/qwen2-moe/hf/qwen2-moe-hf"
PROFILE_DIR="/data/ci/cache/qwen2_moe_profile"

TP=1
PP=2
EP=2
CP=2
SEQ_LENGTH=32768
TRAIN_ITERS=15
CP_TYPE='ulysses_cp_algo'
ROUTER_BALANCING_TYPE='pai_megatron_aux_loss'

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MOE_ARGS="
    --num-experts 4 \
    --moe-router-topk 2 \
    --n-shared-experts 2 \
    --shared-expert-gate \
    --moe-layer-freq -1 \
    --moe-router-load-balancing-type ${ROUTER_BALANCING_TYPE} \
    --moe-ffn-hidden-size 320 \
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --moe-alltoall-overlap-comm \
    --moe-token-dispatcher-type alltoall_seq \
    --moe-aux-loss-coeff 0.001 \
    --reuse-fp32-param \
"

OPTIMIZE_ARGS="
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather
"

TRAIN_ARGS="
    --spec mindspeed_llm.tasks.models.spec.qwen2_moe_spec layer_spec \
    --finetune \
    --micro-batch-size 1 \
    --global-batch-size 64 \
    --lr 1.25e-6 \
    --lr-decay-style cosine \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --seed 42 \
    --bf16 \
    --train-iters ${TRAIN_ITERS} \
    --seq-length ${SEQ_LENGTH} \
    --no-shared-storage
"

MODEL_PARALLEL_ARGS="
    --sequence-parallel \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ${CP_TYPE} \
"

GPT_ARGS="
    --use-mcore-models \
    --manual-gc \
    --manual-gc-interval 50 \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --num-layers 2 \
    --hidden-size 448 \
    --ffn-hidden-size 2368 \
    --num-attention-heads 16 \
    --tokenizer-type PretrainedFromHF \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --rotary-base 1000000 \
    --attention-softmax-in-fp32 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --attention-softmax-in-fp32 \
    --add-qkv-bias \
    --no-gradient-accumulation-fusion \
    --group-query-attention \
    --num-query-groups 4
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
    --no-load-optim \
    --no-load-rng
"

PROFILE_ARGS="
    --profile \
    --profile-step-start 5 \
    --profile-step-end 6 \
    --profile-ranks 0 \
    --profile-level level1 \
    --profile-with-cpu \
    --profile-record-shapes \
    --profile-save-path ${PROFILE_DIR} \
"

torchrun $DISTRIBUTED_ARGS $basepath/pretrain_gpt.py \
    ${GPT_ARGS} \
    ${DATA_ARGS} \
    ${MOE_ARGS} \
    ${OUTPUT_ARGS} \
    ${OPTIMIZE_ARGS} \
    ${TRAIN_ARGS} \
    ${PROFILE_ARGS} \
    ${MODEL_PARALLEL_ARGS} \
    --load ${CKPT_LOAD_DIR} \
    --log-throughput \
    --distributed-backend nccl

#!/bin/bash
rm -rf ${PROFILE_DIR}
