#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6011
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))

basepath=$(cd `dirname $0`; cd ../../../; pwd)

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

echo "NODE_RANK ${NODE_RANK}"

DATA_PATH="/data/ci/datasets/processed/qwen3_30b_dist/alpaca_text_document"
TOKENIZER_PATH="/data/ci/models/Qwen3-30B-A3B/hf/Qwen3-30B-A3B"
CKPT_LOAD_DIR="/data/ci/models/Qwen3-30B-A3B/mg/qwen3-30b-layer1-tp4pp1ep2"

TP=4
PP=1
EP=2
CP=2
CP_TYPE='ulysses_cp_algo'
NUM_LAYERS=1

MOE_ARGS="
    --num-experts 128 \
    --moe-router-topk 8 \
    --moe-router-load-balancing-type aux_loss \
	--moe-ffn-hidden-size 768 \
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall_seq \
    --moe-layer-freq -1 \
    --first-k-dense-replace -1 \
    --moe-aux-loss-coeff 0.001
"

OPTIMIZE_ARGS="
    --sequence-parallel \
    --use-distributed-optimizer \
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --no-masked-softmax-fusion \
    --gemm-gradient-accumulation-fusion \
    --recompute-method uniform \
    --recompute-granularity full \
    --recompute-num-layers 1
"

MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ${CP_TYPE}
"

TRAIN_ARGS="
    --micro-batch-size 1 \
    --global-batch-size 16 \
    --lr 1.25e-6 \
    --train-iters 15 \
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
    --bf16 \
    --seq-length 4096 \
    --no-shared-storage \
    --manual-gc \
    --manual-gc-interval 15
"

GPT_ARGS="
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --kv-channels 128 \
    --qk-layernorm \
    --norm-topk-prob \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --load ${CKPT_LOAD_DIR}
    --max-position-embeddings 32768 \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 2048 \
    --ffn-hidden-size 6144 \
    --num-attention-heads 32 \
    --tokenizer-type PretrainedFromHF \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --rotary-base 1000000 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --swiglu \
    --attention-softmax-in-fp32 \
    --no-gradient-accumulation-fusion \
    --group-query-attention \
    --num-query-groups 4
"

DATA_ARGS=(
    --data-path $DATA_PATH \
    --split 100,0,0
)

OUTPUT_ARGS=(
    --finetune \
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 10000 \
    --eval-iters 0 \
    --no-load-optim \
    --no-load-rng
)

torchrun ${DISTRIBUTED_ARGS[@]} $basepath/pretrain_gpt.py \
  ${MOE_ARGS[@]} \
  ${OPTIMIZE_ARGS[@]}\
  ${MODEL_PARALLEL_ARGS[@]}\
  ${TRAIN_ARGS[@]}\
  ${GPT_ARGS[@]} \
  ${DATA_ARGS[@]} \
  ${OUTPUT_ARGS[@]} \
  --transformer-impl local \
  --ckpt-format torch \
  --distributed-backend nccl
