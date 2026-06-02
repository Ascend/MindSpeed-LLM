#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CPU_AFFINITY_CONF=1,npu0:192-215,npu1:216-239,npu2:0-23,npu3:24-47,npu4:48-71,npu5:72-95,npu6:240-263,npu7:264-287
export TASK_QUEUE_ENABLE=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=3600
export STREAMS_PER_DEVICE=32

NPUS_PER_NODE=8
MASTER_ADDR=localhost #主节点 IP
MASTER_PORT=6001
NNODES=8
NODE_RANK=0
WORLD_SIZE=$((NPUS_PER_NODE*$NNODES))

time=$(date +%m%d_%H%M)

CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"


TP=1
PP=1
EP=64
CP=8
MBS=1
GBS=16
LAYER=30
EXPERTS=128
SEQ_LENGTH=131072
SUB_SEQ=32768
SEQ_LEN_KEY="$((SEQ_LENGTH / 1024))k"
if [[ "$SUB_SEQ" -eq -1 ]]; then
    SUB_SEQ_LABEL="0k"
else
    SUB_SEQ_LABEL="$((SUB_SEQ / 1024))k"
fi
TRAIN_ITERS=2000
CP_TYPE="kvallgather_cp_algo"
ROUTER_BALANCING_TYPE="aux_loss"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
    "

MOE_ARGS="
    --num-experts $EXPERTS \
    --moe-router-topk 8 \
    --moe-ffn-hidden-size 1536 \
    --moe-router-load-balancing-type $ROUTER_BALANCING_TYPE \
    --norm-topk-prob \
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall \
    --moe-layer-freq -1 \
    --first-k-dense-replace 0 \
    --moe-aux-loss-coeff 0.001 \
    --fix-router
    "

OPTIMIZE_ARGS="
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --sequence-parallel \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer \
    --use-cp-send-recv-overlap \
    --gemm-gradient-accumulation-fusion \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --manual-gc \
    --manual-gc-interval 50
    "

TRAIN_ARGS="
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
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
    --train-iters $TRAIN_ITERS \
    --seq-length $SEQ_LENGTH \
    --no-shared-storage
    "


MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --expert-model-parallel-size $EP \
    --context-parallel-size $CP \
    --context-parallel-algo $CP_TYPE \
    --expert-tensor-parallel-size 1 \
    --attention-mask-type causal
    "

GPT_ARGS="
    --disable-gloo-group \
    --hccl-op-mode "cp:6;ep:2;tp_ep_mp:2;dp:2;tp_cp:2;dp_cp:2;default_group:2" \
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --kv-channels 128 \
    --qk-layernorm \
    --tokenizer-name-or-path $TOKENIZER_PATH \
    --max-position-embeddings $SEQ_LENGTH \
    --num-layers $LAYER \
    --hidden-size 4096 \
    --ffn-hidden-size 12288 \
    --num-attention-heads 64 \
    --tokenizer-type PretrainedFromHF \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --rotary-base 1000000 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --attention-softmax-in-fp32 \
    --group-query-attention \
    --num-query-groups 4 \
    --use-fused-ring-attention-update \
    --swap-optimizer
    "

LAYOUT_ARGS="
    --fix-sub-seq-length $SUB_SEQ \
    --reset-attention-mask \
    --reset-position-ids \
    --variable-seq-lengths
    "

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
    "

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval $TRAIN_ITERS \
    --eval-interval $TRAIN_ITERS \
    --eval-iters 0 \
    --no-load-optim \
    --no-load-rng
    "

mkdir -p logs
python -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_gpt.py \
     $GPT_ARGS \
     $DATA_ARGS \
     $MOE_ARGS \
     $OUTPUT_ARGS \
     $OPTIMIZE_ARGS \
     $TRAIN_ARGS \
     $PRECISION_ARGS \
     $MODEL_PARALLEL_ARGS \
     $LAYOUT_ARGS \
    --transformer-impl transformer_engine \
    --load  ${CKPT_LOAD_DIR} \
    --save  ${CKPT_SAVE_DIR} \
    --distributed-backend nccl | tee logs/qwen3_76b_128k_bf16_A5_perf.log
