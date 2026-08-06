#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_IF_BASE_PORT=25809
export CPU_AFFINITY_CONF=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600
export TASK_QUEUE_ENABLE=2

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=8
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CKPT_LOAD_DIR="your model ckpt path"
CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"

TP=4
PP=8
EP=2
CP=1
VPP=4
MBS=1
GBS=32
CP_TYPE='ulysses_cp_algo'
SEQ_LENGTH=32768
TRAIN_ITERS=2000
ROUTER_BALANCING_TYPE='aux_loss'

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

RECOMPUTE_ARGS="
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 4 \
    --use-cp-send-recv-overlap \
    --manual-gc \
    --manual-gc-interval 10
"

MOE_ARGS="
    --num-experts 128 \
    --moe-router-topk 8 \
    --moe-ffn-hidden-size 1536 \
    --moe-router-load-balancing-type ${ROUTER_BALANCING_TYPE} \
    --norm-topk-prob \
    --moe-token-dispatcher-type alltoall_seq \
    --moe-aux-loss-coeff 0.001 \
    --moe-permutation-async-comm \
    --moe-layer-freq 1 \
    --first-k-dense-replace 0
"

OPTIMIZE_ARGS="
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer \
    --reuse-fp32-param \
    --overlap-grad-reduce \
    --overlap-param-gather
"

TRAIN_ARGS="
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --lr 1.5e-5 \
    --lr-decay-style cosine \
    --min-lr 1e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.03 \
    --attention-dropout 0.05 \
    --init-method-std 0.01 \
    --hidden-dropout 0.05 \
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
   --context-parallel-size ${CP} \
   --context-parallel-algo ${CP_TYPE} \
    --num-layers-per-virtual-pipeline-stage ${VPP} \
    --attention-mask-type causal \
    --sequence-parallel \
    --expert-tensor-parallel-size 4
"

GPT_ARGS="
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --kv-channels 128 \
    --qk-layernorm \
    --norm-topk-prob \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --noop-layers 94,95 \
    --num-layers 96 \
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
    --num-query-groups 4
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --handler-name AlpacaStyleInstructionHandler \
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

TUNE_ARGS="
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --tokenizer-not-use-fast \
    --prompt-type qwen3 \
    --reset-attention-mask \
    --lora-ckpt-filter \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --neat-pack
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $TUNE_ARGS \
    $GPT_ARGS \
    $DATA_ARGS \
    $MOE_ARGS \
    $OUTPUT_ARGS \
    $OPTIMIZE_ARGS \
    $RECOMPUTE_ARGS \
    $TRAIN_ARGS \
    $MODEL_PARALLEL_ARGS \
    --distributed-backend nccl \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    | tee ./logs/tune_qwen3_235b_a22b_lora.log
