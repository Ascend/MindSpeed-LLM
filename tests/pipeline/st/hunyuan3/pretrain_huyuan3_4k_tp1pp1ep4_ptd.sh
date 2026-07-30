#！/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CPU_AFFINITY_CONF=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600
export TASK_QUEUE_ENABLE=2
export STREAMS_PER_DEVICE=32
export HCCL_IF_BASE_PORT=25809
export TORCH_DISABLE_GLOO=1

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DATA_PATH="/data/ci/datasets/processed/hunyuan3/hunyuan3_text_document"
TOKENIZER_PATH="/data/ci/models/hunyuan3/hf/hunyuan3_cfg/"
CKPT_LOAD_DIR="/data/ci/models/hunyuan3/mg/hunyuan3_tp1pp1ep4/"

TP=1
PP=1
EP=4
CP=1
CP_TYPE='ulysses_cp_algo'
NUM_LAYERS=2
SEQ_LEN=4096
MBS=1
GBS=16

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"


GQA_ARGS="
    --group-query-attention \
    --num-query-groups 8 \
    --qk-layernorm \
    --kv-channels 128 \
    --num-attention-heads 64 \
"

MOE_ARGS="
    --moe-layer-freq -1 \
    --moe-grouped-gemm \
    --num-experts 16 \
    --first-k-dense-replace 1 \
    --n-shared-experts 1 \
    --norm-topk-prob \
    --moe-ffn-hidden-size 1536 \
    --moe-router-topk 8 \
    --moe-router-enable-expert-bias \
    --moe-router-topk-scaling-factor 2.826 \
    --moe-router-num-groups 8 \
    --moe-router-group-topk 4 \
    --router-gating-in-fp32 \
    --moe-router-score-function sigmoid \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall \
    --moe-router-load-balancing-type none \
    --moe-aux-loss-coeff 0.0001 \
    --expert-tensor-parallel-size 1 \
    --moe-fb-overlap \
    --moe-permute-fusion \
"

OPTIMIZE_ARGS="
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer \
    --swap-optimizer \
"

GPT_ARGS="
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.bailing_spec layer_spec \
    --hidden-size 4096 \
    --ffn-hidden-size 13312 \
    --max-position-embeddings 262144 \
    --vocab-size 120832 \
    --padded-vocab-size 120832 \
    --swiglu \
    --disable-bias-linear \
    --normalization RMSNorm \
    --rotary-base 11158840 \
    --position-embedding-type rope \
    --untie-embeddings-and-output-weights \
    --make-vocab-size-divisible-by 1 \
    --no-masked-softmax-fusion \
    --bf16 \
    --norm-epsilon 1e-05 \
    --attention-dropout 0.0 \
    --init-method-std 0.02 \
    --hidden-dropout 0.0 \
    --num-layers ${NUM_LAYERS} \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
    --sequence-parallel \
    --tokenizer-type PretrainedFromHF  \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length ${SEQ_LEN} \
    --no-load-optim \
    --no-load-rng \
    --ckpt-format torch
"

TRAIN_ARGS="
    --finetune \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --train-iters 15 \
    --lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-decay-style cosine \
"

DATA_ARGS="
    --no-shared-storage \
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 0 \
    --no-save-optim \
    --no-save-rng \
    --load $CKPT_LOAD_DIR \
"

python -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_gpt.py \
    $OPTIMIZE_ARGS \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $GQA_ARGS \
    $MOE_ARGS \
    $TRAIN_ARGS \
    --distributed-backend nccl \
    --transformer-impl local
