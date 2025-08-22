# #!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Change for multinode config
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
TOKENIZER_PATH="your tokenizer model path"
CHECKPOINT="your model directory path"


DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT"

MOE_ARGS="
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall_seq \
    --first-k-dense-replace 1 \
    --moe-layer-freq 1 \
    --n-shared-experts 1 \
    --num-experts 128 \
    --moe-router-topk 8 \
    --moe-ffn-hidden-size 1408 \
    --moe-router-load-balancing-type none \
    --moe-router-num-groups 1 \
    --moe-router-group-topk 1 \
    --norm-topk-prob \
    --moe-router-score-function sigmoid \
    --moe-router-enable-expert-bias \
    --moe-router-dtype fp32 \
"

GPT_ARGS="
    --tensor-model-parallel-size 8  \
    --pipeline-model-parallel-size 1  \
    --use-mcore-models \
    --num-layers 46  \
    --hidden-size 4096  \
    --ffn-hidden-size 10944 \
    --seq-length 4096 \
    --group-query-attention \
    --kv-channels 128 \
    --num-query-groups 8 \
    --num-attention-heads 96  \
    --padded-vocab-size 151552 \
    --make-vocab-size-divisible-by 1 \
    --max-position-embeddings 131072 \
    --position-embedding-type rope \
    --rotary-percent 0.5 \
    --no-rope-fusion \
    --disable-bias-linear \
    --add-qkv-bias \
    --swiglu \
    --post-norm \
    --norm-epsilon 1e-05 \
    --hidden-dropout 0.0 \
    --attention-dropout 0.0 \
    --normalization RMSNorm \
    --max-new-tokens 256 \
    --micro-batch-size 1 \
    --tokenizer-type PretrainedFromHF  \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --untie-embeddings-and-output-weights \
    --attention-softmax-in-fp32 \
    --no-load-optim \
    --no-load-rng \
    --no-masked-softmax-fusion \
    --no-gradient-accumulation-fusion \
    --exit-on-missing-checkpoint \
    --seed 42 \
    --bf16 \
"

torchrun ${DISTRIBUTED_ARGS} inference.py \
         ${MOE_ARGS} \
         ${GPT_ARGS} \
         --load ${CHECKPOINT}  \
         --distributed-backend nccl \
         | tee logs/generate_glm45_moe_106b_mcore.log