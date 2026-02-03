#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CPU_AFFINITY_CONF=1
export TASK_QUEUE_ENABLE=2
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_CONNECT_TIMEOUT=3600
export STREAMS_PER_DEVICE=32

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

basepath=$(cd `dirname $0`; cd ../../../; pwd)

DATA_PATH=/data/ci/datasets/processed/pretrain_dataset/alpaca_text_document
TOKENIZER_PATH=/data/ci/models/deepseek3/mg/deepseek-v3-mcore-tp1-pp2-ep4-16experts
CKPT_LOAD_DIR=/data/ci/models/deepseek32/mg/deepseek32-tp1-pp2-ep4-8expert

TP=1
PP=2
EP=4
CP=1
CP_TYPE='ulysses_cp_algo'
NUM_LAYERS=4
SEQ_LEN=2048
MBS=1
GBS=16

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MLA_ARGS="
    --multi-latent-attention \
    --qk-pos-emb-head-dim 64 \
    --qk-head-dim 128 \
    --q-lora-rank 1536 \
    --kv-lora-rank 512 \
    --v-head-dim 128 \
    --qk-layernorm \
    --mla-mm-split \
    --mla-fa-without-pad \
"

MOE_ARGS="
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall \
    --moe-permute-fusion \
    --first-k-dense-replace 0 \
    --moe-layer-freq 1 \
    --moe-shared-expert-intermediate-size 2048 \
    --num-experts 8 \
    --moe-router-topk 2 \
    --moe-ffn-hidden-size 2048 \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-router-num-groups 4 \
    --moe-router-group-topk 1 \
    --moe-router-topk-scaling-factor 2.5 \
    --moe-aux-loss-coeff 0.0001 \
    --moe-router-score-function sigmoid \
    --moe-router-enable-expert-bias \
    --moe-router-dtype fp32 \
"

MTP_ARGS="
    --mtp-num-layers 1 \
    --mtp-loss-scaling-factor 0.3 \
"

PIPELINE_ARGS="
    --moe-fb-overlap \
    --num-layers-per-virtual-pipeline-stage 1 \
"

MEM_ARGS="
    --mtp-mem-efficient-logits \
    --swap-optimizer \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
"

DSA_ARGS="
    --enable-dsa-indexer \
    --index-topk 1024 \
    --indexer-loss-coeff 1.0 \
    --init-norm-weight-in-fp32 \
"

OTHERS_ARGS="
    --tensorboard-dir ./tb \
    --no-shared-storage \
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
    --finetune \
    --transformer-impl local \
    --spec mindspeed_llm.tasks.models.spec.deepseek_spec layer_spec \
    --gemm-gradient-accumulation-fusion \
    --noop-layers 3 \
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
    --train-iters 15 \
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
    --eval-iters 0 \
    --no-save-optim \
    --no-save-rng
"

python -m torch.distributed.launch $DISTRIBUTED_ARGS $basepath/pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $MLA_ARGS \
    $PIPELINE_ARGS \
    $MEM_ARGS \
    $ROPE_ARGS \
    $MOE_ARGS \
    $DSA_ARGS \
    $OTHERS_ARGS \
    --load $CKPT_LOAD_DIR \
    --transformer-impl local \
    --distributed-backend nccl