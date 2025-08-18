#!/bin/bash
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_ALGO="alltoall=level0:NA;level1:pipeline"
export HCCL_BUFFSIZE=400
export HCCL_DETERMINISTIC=true
export ASCEND_LAUNCH_BLOCKING=1
export NCCL_DETERMINISTIC=1

basepath=$(cd `dirname $0`; cd ../../../../; pwd)
GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6066
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="/data/mindspore/st/test_ds3_pretrain/ckpt_tp2ep2pp2vpp1"
DATA_PATH="/data/mindspore/st/test_ds3_pretrain/dataset/dataset/enwiki_text_document"
TOKENIZER_PATH="/data/mindspore/st/test_ds3_pretrain/tokenizer"
CKPT_LOAD_DIR="/data/mindspore/st/test_ds3_pretrain/ckpt_tp2ep2pp2vpp1"

TP=2
PP=2
EP=2
CP=1
VPP=1
CP_TYPE='ulysses_cp_algo'
NUM_LAYERS=4
SEQ_LEN=4096
MBS=1
GBS=16


DISTRIBUTED_ARGS="
    --master_addr $MASTER_ADDR \
    --node_rank $NODE_RANK \
    --worker_num $WORLD_SIZE \
    --local_worker_num $GPUS_PER_NODE \
    --master_port $MASTER_PORT \
    --log_dir=msrun_log \
    --join=True \
    --cluster_time_out=300 \
    --bind_core=True \
"


MLA_ARGS="
    --multi-latent-attention \
    --qk-pos-emb-head-dim 64 \
    --qk-head-dim 128 \
    --q-lora-rank 1536 \
    --kv-lora-rank 512 \
    --v-head-dim 128 \
    --qk-layernorm \
"


MOE_ARGS="
    --first-k-dense-replace 1 \
    --moe-layer-freq 1 \
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --use-fused-moe-token-permute-and-unpermute \
    --moe-token-dispatcher-type alltoall \
    --n-shared-experts 1 \
    --num-experts 64 \
    --moe-router-topk 8 \
    --moe-intermediate-size 2048 \
    --moe-router-load-balancing-type noaux_tc \
    --n-group 8 \
    --topk-group 4 \
    --routed-scaling-factor 2.5 \
    --seq-aux \
    --norm-topk-prob \
    --moe-router-score-function sigmoid \
    --moe-router-enable-expert-bias \
"


MTP_ARGS="
    --mtp-num-layers 1 \
    --mtp-loss-scaling-factor 0.3 \
    --mtp-mem-efficient-logits \
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
    --no-gradient-accumulation-fusion \
    --spec mindspeed_llm.tasks.models.spec.deepseek_spec layer_spec \
    --reset-position-ids \
    --noop-layers 3 \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --no-shared-storage \
    --use-distributed-optimizer \
    --use-flash-attn \
    --use-mcore-models \
    --num-layers-per-virtual-pipeline-stage ${VPP} \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
    --sequence-parallel \
    --context-parallel-size ${CP} \
    --context-parallel-algo  ${CP_TYPE} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 7168 \
    --ffn-hidden-size 18432 \
    --num-attention-heads 128 \
    --tokenizer-type PretrainedFromHF  \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings 163840 \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --lr 1.0e-5 \
    --train-iters 10 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.02 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
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
    --lr-warmup-iters 0 \
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
    --reuse-fp32-param \
"


DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0 \
"


OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 200 \
    --eval-interval 2000 \
    --eval-iters 0 \
    --no-save-optim \
    --no-save-rng
"


msrun $DISTRIBUTED_ARGS $basepath/pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $MLA_ARGS \
    $ROPE_ARGS \
    $MOE_ARGS \
    $MTP_ARGS \
    --load $CKPT_LOAD_DIR \
    --distributed-backend nccl \
    --ai-framework mindspore \