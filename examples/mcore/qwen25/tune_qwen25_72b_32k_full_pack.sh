#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=4
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CKPT_LOAD_DIR="your model ckpt path"
CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"

TP=8
PP=2
SEQ_LEN=32768
MBS=1
GBS=32
TRAIN_ITERS=2000
CP=4

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

TUNE_ARGS="
    --finetune \
    --stage sft \
    --prompt-type qwen \
    --reset-attention-mask \
    --is-instruction-dataset \
    --neat-pack \
    --padded-samples
"

MEMORY_ARGS="
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
"
GPT_ARGS="
    --context-parallel-size ${CP} \
    --context-parallel-algo ulysses_cp_algo \
    --sequence-parallel \
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers 80  \
    --reuse-fp32-param \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --hidden-size 8192  \
    --ffn-hidden-size 29568 \
    --num-attention-heads 64  \
    --max-position-embeddings ${SEQ_LEN} \
    --seq-length ${SEQ_LEN} \
    --disable-bias-linear \
    --add-qkv-bias \
    --group-query-attention \
    --num-query-groups 8 \
    --use-flash-attn \
    --swiglu \
    --use-fused-swiglu \
    --normalization RMSNorm \
    --norm-epsilon 1e-5 \
    --use-fused-rmsnorm \
    --position-embedding-type rope \
    --rotary-base 1000000 \
    --use-fused-rotary-pos-emb \
    --untie-embeddings-and-output-weights \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 152064 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --train-iters ${TRAIN_ITERS} \
    --lr 7.75e-7 \
    --lr-decay-style cosine \
    --min-lr 7.75e-8 \
    --lr-warmup-fraction 0.01 \
    --init-method-std 0.01 \
    --weight-decay 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --bf16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

CKPT_ARGS="
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --seed 1234 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 0 \
    --log-throughput
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $MEMORY_ARGS \
    $GPT_ARGS \
    $DATA_ARGS \
    $CKPT_ARGS \
    $OUTPUT_ARGS \
    $TUNE_ARGS \
    --distributed-backend nccl \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    | tee logs/tune_mcore_qwen25_72b_32k_full_pack.log