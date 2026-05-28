#!/bin/bash

export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export ACL_DEVICE_SYNC_TIMEOUT=7200
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_ALGO="alltoall=level0:NA;level1:pipeline"

NPUS_PER_NODE=8
MASTER_ADDR=localhost #主节点IP
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

basepath=$(cd `dirname $0`; cd ../../../; pwd)

TOKENIZER_PATH="/data/ci/models/deepseek4-flash/hf/DeepSeek-V4-Flash/"
CKPT_LOAD_DIR="/data/ci/models/deepseek4-flash/mg/deepseekv4-flash-tp1-pp1-ep8-16experts/"
DATA_PATH="/data/ci/datasets/origin/train-00000-of-00001-a09b74b3ef9c3b56.parquet"
OUTPUT_PREFIX="/data/ci/cache/deepseek4-flash-dataset/deepsee4-flash"

TP=1
PP=1
EP=8
CP=1
CP_TYPE='ulysses_cp_algo'
NUM_LAYERS=4
EXPERTS=16
SEQ_LEN=4096
MBS=1
GBS=8

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

DSA_ARGS="
    --enable-dsa-indexer \
    --index-n-heads 64 \
    --index-head-dim 128 \
    --index-topk 512 \
    --enable-mhc \
    --hc-mult 4 \
    --kv-compress \
    --norm-eps 1e-6 \
    --use-triton-sinkhorn \
    --use-triton-mhc \
    --use-triton-rmsnorm-without-weight \
"

MLA_ARGS="
    --multi-latent-attention \
    --qk-pos-emb-head-dim 64 \
    --qk-head-dim 512 \
    --q-lora-rank 1024 \
    --o-lora-rank 1024 \
    --kv-lora-rank 512 \
    --v-head-dim 128 \
    --qk-layernorm \
    --mla-fa-without-pad \
"

CA_ARGS="
    --use-g2-attention \
    --o-groups 8 \
    --g2-window-size 128 \
    --rope-head-dim 64 \
    --original-seq-len 65536 \
    --rope-factor 16 \
    --compress-rope-theta 160000.0 \
    --max-batch-size 4 \
    --compress-ratios 0 0 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 \
    --use-g2-indexer-loss \
"

MOE_ARGS="
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall \
    --moe-layer-freq 1 \
    --first-k-dense-replace -1 \
    --num-experts ${EXPERTS} \
    --moe-router-topk 6 \
    --moe-ffn-hidden-size 2048 \
    --moe-router-load-balancing-type none \
    --moe-router-group-topk 1 \
    --moe-router-num-groups 1 \
    --moe-router-topk-scaling-factor 1.5 \
    --seq-aux \
    --moe-aux-loss-coeff 0.001 \
    --moe-router-score-function sqrtsoftplus \
    --moe-router-enable-expert-bias \
    --moe-shared-expert-intermediate-size 2048 \
    --moe-router-dtype fp32 \
    --n-hash-layers 3 \
    --moe-permute-fusion \
    --moe-fb-overlap \
"

MTP_ARGS="
    --mtp-num-layers 1 \
    --mtp-loss-scaling-factor 0.3 \
"

MEM_ARGS="
    --mtp-mem-efficient-logits \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --swap-optimizer \
"

ROPE_ARGS="
    --beta-fast 32 \
    --beta-slow 1 \
    --rope-scaling-factor 40 \
    --rope-scaling-mscale 1.0 \
    --rope-scaling-mscale-all-dim 1.0 \
    --rope-scaling-original-max-position-embeddings 4096 \
    --rope-scaling-type yarn
"

GPT_ARGS="
    --transformer-impl local \
    --spec mindspeed_llm.tasks.models.spec.deepseek4_spec layer_spec \
    --mtp-spec mindspeed_llm.tasks.models.spec.deepseek4_spec mtp_spec \
    --manual-gc \
    --manual-gc-interval 5 \
    --use-distributed-optimizer \
    --use-flash-attn \
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
    --expert-tensor-parallel-size 1 \
    --sequence-parallel \
    --context-parallel-size ${CP} \
    --context-parallel-algo  ${CP_TYPE} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 4096 \
    --ffn-hidden-size 4096 \
    --num-attention-heads 64 \
    --tokenizer-type PretrainedFromHF  \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings 163840 \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --lr 1.0e-5 \
    --train-iters 15 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.02 \
    --hidden-dropout 0.0 \
    --position-embedding-type g2 \
    --normalization RMSNorm \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.0e-7 \
    --weight-decay 1e-2 \
    --lr-warmup-iters 5 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --initial-loss-scale 65536 \
    --vocab-size 129280 \
    --padded-vocab-size 129280 \
    --rotary-base 10000 \
    --norm-epsilon 1e-6 \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --distributed-timeout-minutes 120 \
    --no-shared-storage \
    --no-gradient-accumulation-fusion \
    --ckpt-format torch
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0 \
    --handler-name GeneralPretrainHandler \
    --output-prefix ${OUTPUT_PREFIX} \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --load $CKPT_LOAD_DIR \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 0 \
    --no-save-optim \
    --no-save-rng \
"

torchrun $DISTRIBUTED_ARGS $basepath/pretrain_deepseek4.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $MLA_ARGS \
    $ROPE_ARGS \
    $MOE_ARGS \
    $DSA_ARGS \
    $CA_ARGS \
    $MEM_ARGS \
    $MTP_ARGS \
    --finetune \
    --distributed-backend nccl
