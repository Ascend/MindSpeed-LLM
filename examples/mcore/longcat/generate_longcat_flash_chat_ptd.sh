#!/bin/bash
export HCCL_CONNECT_TIMEOUT=6000
export HCCL_EXEC_TIMEOUT=5400
export HCCL_IF_BASE_PORT=48890
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_ASD_ENABLE=0
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1

NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6066
NNODES=8
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
TOKENIZER_PATH="./model_from_hf/longcat-flash-chat-hf/"
CKPT_LOAD_DIR="./model_weights/longcat-flash-chat-mcore/"
TP=16
PP=4
EP=32
CP=1

MBS=1
GBS=16
SEQ_LENGTH=4096
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
    --num-experts 512 \
    --num-zero-experts 256 \
    --moe-router-topk 12 \
    --moe-router-dtype fp32 \
    --moe-router-load-balancing-type ${ROUTER_BALANCING_TYPE} \
    --moe-router-topk-scaling-factor 6.0 \
	--moe-ffn-hidden-size 2048 \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall \
    --moe-aux-loss-coeff 0.001 \
    --moe-grouped-gemm \
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
    --gemm-gradient-accumulation-fusion \
    --swap-optimizer \
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
    --num-layers 28 \
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
    --no-bias-dropout-fusion \
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

torchrun $DISTRIBUTED_ARGS inference.py \
    $GPT_ARGS \
    $MOE_ARGS \
    $MLA_ARGS \
    $OPTIMIZE_ARGS \
    $MODEL_PARALLEL_ARGS \
    --load ${CKPT_LOAD_DIR} \
    --distributed-backend nccl \
    --transformer-impl local \
    | tee logs/generate_mcore_longcat_ptd.log