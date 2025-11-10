#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TASK_QUEUE_ENABLE=2

# Change for multinode config
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CKPT_LOAD_DIR="your model ckpt path"
CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"

TP=1
PP=2
MBS=1
GBS=128
SEQ_LEN=4096
TRAIN_ITERS=2000

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
ACC_ARGS="
    --reuse-fp32-param \
"
GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --num-layers 48 \
    --hidden-size 5120 \
    --ffn-hidden-size 13824 \
    --num-attention-heads 40 \
    --group-query-attention \
    --num-query-groups 8 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 152064 \
    --rotary-base 1000000 \
    --lr 1.25e-6 \
    --train-iters ${TRAIN_ITERS} \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --norm-epsilon 1e-5 \
    --swiglu \
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --add-qkv-bias \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --seed 42 \
    --bf16 \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
"
TUNE_ARGS="
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --tokenizer-not-use-fast \
    --prompt-type qwen \
    --padded-samples \
    --neat-pack \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-fusion \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2
    --reset-attention-mask
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval ${TRAIN_ITERS} \
    --eval-interval ${TRAIN_ITERS} \
    --eval-iters 0 \
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $ACC_ARGS \
    $OUTPUT_ARGS \
    $TUNE_ARGS \
    --distributed-backend nccl \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    | tee logs/tune_mcore_qwen25_14b_4k_lora_pack.log
