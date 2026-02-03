#!/bin/bash
export HCCL_CONNECT_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export NPU_ASD_ENABLE=0

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6066
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CKPT_LOAD_DIR="your model ckpt path"
CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"

TP=1
PP=1
EP=8
CP=1

MBS=1
GBS=16
SEQ_LENGTH=8192
TRAIN_ITERS=2000
ROUTER_BALANCING_TYPE='softmax_topk'

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MOE_ARGS="
    --num-experts 256 \
    --num-zero-experts 128 \
    --moe-router-topk 12 \
    --moe-router-dtype fp32 \
    --moe-router-load-balancing-type ${ROUTER_BALANCING_TYPE} \
    --moe-router-topk-scaling-factor 6.0 \
	--moe-ffn-hidden-size 2048 \
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall \
    --moe-aux-loss-coeff 0.001 \
    --fix-router
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

OPTIMIZE_ARGS="
    --use-flash-attn \
    --sequence-parallel \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer \
"

TRAIN_ARGS="
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
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
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
    --expert-tensor-parallel-size 1 \
"

GPT_ARGS="
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.longcat_spec layer_spec \
    --qk-layernorm \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --num-layers 1 \
    --hidden-size 6144 \
    --ffn-hidden-size 12288 \
    --num-attention-heads 64 \
    --kv-channels 64 \
    --tokenizer-type PretrainedFromHF \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 131072 \
    --rotary-base 10000000 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --attention-softmax-in-fp32 \
    --no-gradient-accumulation-fusion \
    --transformer-impl transformer_engine \
    --no-bias-dropout-fusion
"

# FP8_ARGS="
#     --fp8-format e4m3 \
#     --fp8-recipe mxfp8 \
# "

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
    --profile-level level1  \
    --profile-with-cpu  \
    --profile-record-shapes  \
    --profile-save-path ./longcat_8k_bf16_profiling \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $MOE_ARGS \
    $MLA_ARGS \
    $OUTPUT_ARGS \
    $OPTIMIZE_ARGS \
    $TRAIN_ARGS \
    $MODEL_PARALLEL_ARGS \
    $PROFILE_ARGS \
    --distributed-backend nccl \
    --transformer-impl local \
    | tee logs/pretrain_mcore_longcat_ptd_8k.log