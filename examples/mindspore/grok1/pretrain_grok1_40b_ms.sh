#!/bin/bash
# To check the performance of a Dropless MoE model, we should run the model for at least 500 iterations or resume from trained checkpoints.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=4
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"

TP=8
PP=2
VPP=1
EP=2
CP=1
CP_TYPE=ulysses_cp_algo
NUM_LAYERS=12
SEQ_LEN=8192
MBS=1
GBS=128

DISTRIBUTED_ARGS="
    --master_addr $MASTER_ADDR \
    --node_rank $NODE_RANK \
    --worker_num $WORLD_SIZE \
    --local_worker_num $NPUS_PER_NODE \
    --master_port $MASTER_PORT \
    --log_dir=msrun_log \
    --join=False \
    --cluster_time_out=300 \
    --bind_core=True \
"


MOE_ARGS="
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall_seq \
    --moe-grouped-gemm \
    --num-experts 8 \
    --moe-router-load-balancing-type aux_loss \
    --moe-router-topk 2 \
    --moe-aux-loss-coeff 1e-2 \
    --embedding-multiplier-scale 78.38367176906169 \
    --output-multiplier-scale 0.5773502691896257 \
    --input-jitter \
    --moe-layer-freq -1 \
    --first-k-dense-replace -1 \
"

GPT_ARGS="
    --spec mindspeed_llm.tasks.models.spec.grok_spec layer_spec \
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers-per-virtual-pipeline-stage ${VPP} \
    --expert-model-parallel-size ${EP} \
    --sequence-parallel \
    --context-parallel-size ${CP} \
    --context-parallel-algo  ${CP_TYPE} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 6144 \
    --ffn-hidden-size 32768 \
    --num-attention-heads 48 \
    --group-query-attention \
    --num-query-groups 8 \
    --post-norm \
    --tokenizer-type PretrainedFromHF  \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings 8192 \
    --use-distributed-optimizer \
    --gemm-gradient-accumulation-fusion \
    --reuse-fp32-param \
    --use-flash-attn \
    --make-vocab-size-divisible-by 1 \
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
    --vocab-size 131072 \
    --rotary-base 10000 \
"

CKPT_ARGS="
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --seed 1234 \
    --save $CKPT_SAVE_DIR \
    --load $CKPT_LOAD_DIR \
"

TRAIN_ARGS="
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --lr 1.0e-5 \
    --train-iters 2000 \
    --lr-decay-iters 1280 \
    --lr-decay-style cosine \
    --min-lr 1.0e-6 \
    --weight-decay 0.1 \
    --lr-warmup-iters 2 \
    --clip-grad 1.0 \
    --bf16
"

DATA_ARGS="
    --no-shared-storage \
    --data-path $DATA_PATH \
    --split 949,50,1 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 0 \
"

msrun  $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $CKPT_ARGS \
    $TRAIN_ARGS \
    $MOE_ARGS \
    --distributed-backend nccl \
    --ai-framework mindspore \
    | tee logs/pretrain_grok1_mcore_40b.log