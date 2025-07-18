#!/bin/bash
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
date
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_DETERMINISTIC=true  
export ASCEND_LAUNCH_BLOCKING=1  
export NCCL_DETERMINISTIC=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MS_ALLOC_CONF=enable_vmm:True

basepath=$(cd `dirname $0`; cd ../../../../; pwd)

MASTER_PORT=6001
MASTER_ADDR=localhost
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=8
NPUS_PER_NODE=8
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH="/data/mindspore/st/test_ds3_sft/finetune_dataset/alpaca"
TOKENIZER_PATH="/data/mindspore/st/test_ds3_sft/tokenizer"
CKPT_LOAD_DIR="/data/mindspore/st/test_ds3_sft/load"


DISTRIBUTED_ARGS="
    --local_worker_num $GPUS_PER_NODE \
    --worker_num $WORLD_SIZE \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --log_dir msrun_log/ \
    --join True \
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
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --use-fused-moe-token-permute-and-unpermute \
    --moe-token-dispatcher-type alltoall \
    --n-shared-experts 1 \
    --num-experts 128 \
    --moe-router-topk 8 \
    --moe-intermediate-size 2048 \
    --moe-router-load-balancing-type noaux_tc \
    --topk-group 4 \
    --n-group 8 \
    --routed-scaling-factor 2.5 \
    --seq-aux \
    --norm-topk-prob \
    --moe-router-score-function sigmoid \
    --moe-router-enable-expert-bias \
"


ROPE_ARGS="
    --beta-fast 32 \
    --beta-slow 1 \
    --rope-scaling-factor 40 \
    --rope-scaling-mscale 1.0 \
    --rope-scaling-mscale-all-dim  1.0 \
    --rope-scaling-original-max-position-embeddings 4096 \
    --rope-scaling-type yarn
"


GPT_ARGS="
    --spec mindspeed_llm.tasks.models.spec.deepseek_spec layer_spec \
    --num-layer-list 1,1 \
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 4 \
    --use-distributed-optimizer \
    --use-flash-attn \
    --shape-order BNSD \
    --use-mcore-models \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 2 \
    --expert-model-parallel-size 2\
    --sequence-parallel \
    --context-parallel-size 1 \
    --context-parallel-algo  ulysses_cp_algo \
    --num-layers 2 \
    --hidden-size 7168 \
    --ffn-hidden-size 18432 \
    --num-attention-heads 128 \
    --tokenizer-type PretrainedFromHF  \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length 4096 \
    --max-position-embeddings 163840 \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --make-vocab-size-divisible-by 1 \
    --lr 5e-6 \
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
    --min-lr 1e-6 \
    --weight-decay 0.1 \
    --lr-warmup-iters 1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 65536 \
    --vocab-size 129280 \
    --padded-vocab-size 129280 \
    --rotary-base 10000 \
    --norm-epsilon 1e-6 \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --distributed-timeout-minutes 120 \
    --no-gradient-accumulation-fusion \
    --reuse-fp32-param \
"

DATA_ARGS="
    --data-path ${DATA_PATH} \
    --split 100,0,0
"


OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 2000 \
    --eval-iters 0 \
    --no-save-optim \
    --no-save-rng \
    --load ${CKPT_LOAD_DIR} \
"


FINETUNE_ARGS="
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --prompt-type deepseek3 \
    "


CUSTOM_ARGS="
    --moe-router-bias-update-rate 0 \
    --num-workers 0 \
    --no-shared-storage \
    "


msrun $DISTRIBUTED_ARGS $basepath/posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $MLA_ARGS \
    $ROPE_ARGS \
    $MOE_ARGS \
    $FINETUNE_ARGS \
    $CUSTOM_ARGS \
    --distributed-backend nccl \
    --ai-framework mindspore \
