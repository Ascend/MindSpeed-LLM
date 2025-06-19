#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CPU_AFFINITY_CONF=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=3600
export TASK_QUEUE_ENABLE=2

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"

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
    --mamba-num-groups 8 \
    --mamba-chunk-size 128 \
    --mamba-state-dim 128 \
    --mamba-d-conv 4 \
    --mamba-expand 2 \
    --mamba-head-dim 64 \
    --tokenizer-type  GPTSentencePieceTokenizer \
    --tokenizer-model ${TOKENIZER_PATH} \
    --hidden-size 4096 \
    --hybrid-attention-ratio 0.08 \
    --hybrid-mlp-ratio 0.5 \
    --seq-length 4096 \
    --max-position-embeddings 163840 \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --train-iters 2000 \
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
    --lr-warmup-iters 0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 65536 \
    --rotary-base 10000 \
    --no-gradient-accumulation-fusion \
    --norm-epsilon 1e-6 \
    --no-load-optim \
    --no-load-rng \
    --bf16
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
    --save ${CKPT_SAVE_DIR} \
    --no-save-rng \
    --no-save-optim
"

python -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_mamba.py \
    $MAMBA_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    | tee logs/pretrain_mamba2_hybrid_8b_4k_ptd.log