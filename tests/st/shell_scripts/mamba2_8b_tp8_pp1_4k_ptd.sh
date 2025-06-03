#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=3600

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

basepath=$(cd `dirname $0`; cd ../../../; pwd)

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

echo "NODE_RANK ${NODE_RANK}"

DATA_PATH="/data/mamba_enwiki/mamba_enwiki_text_document"
CKPT_LOAD_DIR="/data/mamba2_tp8dp1_mbs2_gbs8"
TOKENIZER_PATH="/data/mamba-hf/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model"



TP=8
PP=1
NUM_LAYERS=56
SEQ_LEN=4096
MBS=2
GBS=8

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MAMBA_ARGS="
    --spec mindspeed_llm.tasks.models.spec.mamba_spec layer_spec \
    --reuse-fp32-param \
    --no-shared-storage \
    --use-distributed-optimizer \
    --use-flash-attn \
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --num-layers ${NUM_LAYERS} \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --mamba-ngroups 8 \
    --mamba-chunk-size 128 \
    --mamba-d-state 128 \
    --mamba-d-conv 4 \
    --mamba-expand 2 \
    --mamba-headdim 64 \
    --tokenizer-type  GPTSentencePieceTokenizer \
    --tokenizer-model ${TOKENIZER_PATH} \
    --hidden-size 4096 \
    --seq-length 4096 \
    --max-position-embeddings 163840 \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --train-iters 15 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.02 \
    --hidden-dropout 0.0 \
    --position-embedding-type none \
    --normalization RMSNorm \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --overlap-param-gather \
    --overlap-grad-reduce \
    --swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --lr 2.5e-5 \
    --min-lr 2.5e-6 \
    --lr-decay-style cosine \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 65536 \
    --rotary-base 10000 \
    --no-gradient-accumulation-fusion \
    --norm-epsilon 1e-6 \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --finetune \
    --log-throughput
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 500 \
    --eval-interval 500 \
    --eval-iters 0 \
    --load ${CKPT_LOAD_DIR} \
    --no-save-rng \
    --no-save-optim
"

torchrun $DISTRIBUTED_ARGS $basepath/pretrain_mamba.py \
    $MAMBA_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl