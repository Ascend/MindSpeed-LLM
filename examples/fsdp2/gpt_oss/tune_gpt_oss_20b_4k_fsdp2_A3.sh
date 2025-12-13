#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=2
export HCCL_CONNECT_TIMEOUT=1800
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TORCH_HCCL_ZERO_COPY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6499
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"
HF_PATH="huggingface model path"
CKPT_SAVE_DIR="ckpt save path"
FSDP2_PATH="./examples/fsdp2/gpt_oss/fsdp2_config.yaml"

MBS=1
GBS=16
SEQ_LENGTH=4096
TRAIN_ITERS=2000

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

FSDP2_ARGS="
    --use-torch-fsdp2 \
    --untie-embeddings-and-output-weights \
    --ckpt-format torch_dcp \
    --fsdp2-config-path $FSDP2_PATH \
    --init-from-hf-path $HF_PATH \
    --model-id gpt_oss \
"

OPTIMIZE_ARGS="
    --use-fused-rotary-pos-emb \
    --moe-grouped-gemm \
    --use-fused-rmsnorm \
"

TRAIN_ARGS="
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --lr 1.25e-6 \
    --lr-decay-style cosine \
    --min-lr 1.25e-7 \
    --weight-decay 0e0 \
    --lr-warmup-fraction 0.0 \
    --attention-dropout 0.0 \
    --init-method-std 0.02 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --initial-loss-scale 4096 \
    --seed 42 \
    --bf16 \
    --train-iters ${TRAIN_ITERS} \
    --seq-length ${SEQ_LENGTH} \
    --no-shared-storage
"

ROPE_ARGS="
    --beta-fast 32 \
    --beta-slow 1 \
    --rope-scaling-factor 32 \
    --rope-scaling-original-max-position-embeddings 4096 \
    --rope-scaling-type yarn
"

GPT_ARGS="
    --use-mcore-models \
    --add-qkv-bias \
    --interleave-sliding-window 128 \
    --kv-channels 64 \
    --position-embedding-type rope \
    --spec mindspeed_llm.tasks.models.spec.gpt_oss_spec layer_spec \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --num-layers 24 \
    --hidden-size 2880 \
    --num-attention-heads 64 \
    --tokenizer-type PretrainedFromHF \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 201088 \
    --rotary-base 150000 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --attention-softmax-in-fp32 \
    --norm-epsilon 1e-5 \
    --no-gradient-accumulation-fusion \
    --group-query-attention \
    --num-query-groups 8
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval ${TRAIN_ITERS} \
    --eval-interval ${TRAIN_ITERS} \
    --eval-iters 0 \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
"

TUNE_ARGS="
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --no-pad-to-seq-lengths
"

PREPROCESS_ARGS="
    --handler-name AlpacaStyleInstructionHandler \
    --tokenizer-type PretrainedFromHF \
    --workers 4 \
    --prompt-type gpt_oss \
    --enable-thinking true
"

torchrun $DISTRIBUTED_ARGS train_fsdp2.py \
    $FSDP2_ARGS \
    $OPTIMIZE_ARGS \
    $TRAIN_ARGS \
    $ROPE_ARGS \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $TUNE_ARGS \
    $PREPROCESS_ARGS \
   --save ${CKPT_SAVE_DIR} \
    --distributed-backend nccl \
    | tee logs/tune_gpt_oss_20b.log