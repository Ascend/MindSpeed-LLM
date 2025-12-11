#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=3600

# 删缓存 预编译，提高用例执行稳定性
rm -rf /root/.cache
python -c "import mindspeed; from mindspeed.op_builder import GMMOpBuilder; GMMOpBuilder().load()" &
python -c "import mindspeed; from mindspeed.op_builder import GMMV2OpBuilder; GMMV2OpBuilder().load()" &
python -c "import mindspeed; from mindspeed.op_builder import MatmulAddOpBuilder; MatmulAddOpBuilder().load()" &
python -c "import mindspeed; from mindspeed.op_builder import MoeTokenPermuteOpBuilder; MoeTokenPermuteOpBuilder().load()" &
python -c "import mindspeed; from mindspeed.op_builder import MoeTokenUnpermuteOpBuilder; MoeTokenUnpermuteOpBuilder().load()" &
python -c "import mindspeed; from mindspeed.op_builder import RotaryPositionEmbeddingOpBuilder; RotaryPositionEmbeddingOpBuilder().load()" &
python -c "import mindspeed; from mindspeed.op_builder import GroupMatmulAddOpBuilder; GroupMatmulAddOpBuilder().load()"

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


DATA_PATH="/data/ci/datasets/processed/qwen3_next_data/qwen3_next_aplaca_text_document"
CKPT_LOAD_DIR="/data/ci/models/qwen3_next/mg/qwen3_next_tp1dp1ep8_mbs1_gbs8"
TOKENIZER_PATH="/data/ci/models/qwen3_next/hf/Qwen3-Next-80B-A3B-hf"

TP=1
PP=1
EP=8
CP=1

MBS=1
GBS=8
SEQ_LENGTH=4096
TRAIN_ITERS=15
ROUTER_BALANCING_TYPE='aux_loss'

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MOE_ARGS="
    --num-experts 512 \
    --moe-router-topk 10 \
    --moe-router-load-balancing-type ${ROUTER_BALANCING_TYPE} \
    --n-shared-experts 1 \
    --shared-expert-gate \
    --moe-ffn-hidden-size 512 \
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall_seq \
    --moe-layer-freq 1 \
    --norm-topk-prob \
    --moe-aux-loss-coeff 0.001 \
    --topk-softmax-in-fp32 \
    --moe-router-pre-softmax \
"

OPTIMIZE_ARGS="
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer \
    --gemm-gradient-accumulation-fusion \
    --swap-optimizer \
    --enable-recompute-layers-per-pp-rank \
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 1 \
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
    --no-shared-storage \
    --train-iters 15 \
    --finetune \
    --log-throughput \
"

MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
"

GPT_ARGS="
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.qwen3_next_spec layer_spec \
    --qk-layernorm \
    --full-attention-interval 4 \
    --mamba-d-conv 4 \
    --mamba-expand 1 \
    --kv-channels 256 \
    --linear-key-head-dim 128 \
    --linear-num-key-heads 16 \
    --linear-num-value-heads 32 \
    --linear-value-head-dim 128 \
    --partial-rotary-factor 0.25 \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --num-layers 4 \
    --hidden-size 2048 \
    --ffn-hidden-size 5120 \
    --num-attention-heads 16 \
    --tokenizer-type PretrainedFromHF \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --rotary-base 10000000 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --no-enable-linear-qkv \
    --swiglu \
    --rmsnorm-weight-in-fp32 \
    --add-rmsnorm-offset \
    --attention-softmax-in-fp32 \
    --no-gradient-accumulation-fusion \
    --group-query-attention \
    --num-query-groups 2 \
    --norm-epsilon 1e-06 \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --mamba-chunk-size 64 \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

MTP_ARGS="
    --mtp-num-layers 1 \
    --mtp-loss-scaling-factor 0.3 \
    --mtp-mem-efficient-logits \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10 \
    --eval-interval 500 \
    --eval-iters 0 \
    --no-save-optim \
    --no-save-rng \
"

torchrun ${DISTRIBUTED_ARGS[@]} $basepath/pretrain_gpt.py \
    $MTP_ARGS \
    $GPT_ARGS \
    $DATA_ARGS \
    $MOE_ARGS \
    $OUTPUT_ARGS \
    $OPTIMIZE_ARGS \
    $TRAIN_ARGS \
    $MODEL_PARALLEL_ARGS \
    --load  ${CKPT_LOAD_DIR} \
    --distributed-backend nccl