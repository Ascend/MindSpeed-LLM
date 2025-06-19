#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# please fill these path configurations
CHECKPOINT="Your ckpt file path"
TOKENIZER_PATH="Your vocab file path"
DATA_PATH="Your data path (such as ./mmlu/test/)"
TASK="mmlu"

# distributed config
MASTER_ADDR=localhost
MASTER_PORT=6014
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TP=1
PP=8
EP=1
SEQ_LENGTH=4096
ROUTER_BALANCING_TYPE='softmax_topk'

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MOE_ARGS="
    --num-experts 64 \
    --moe-router-topk 8 \
    --n-shared-experts 8 \
    --shared-expert-gate \
    --moe-router-load-balancing-type ${ROUTER_BALANCING_TYPE} \
    --moe-intermediate-size 2560 \
    --moe-ffn-hidden-size 2560 \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type allgather \
    --moe-layer-freq -1 \
    --first-k-dense-replace -1 \
"

torchrun $DISTRIBUTED_ARGS evaluation.py \
        $MOE_ARGS \
       --use-mcore-models \
       --task-data-path $DATA_PATH \
       --task ${TASK} \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --seq-length ${SEQ_LENGTH} \
       --max-position-embeddings ${SEQ_LENGTH} \
       --max-new-tokens 2 \
       --num-layers 28  \
       --hidden-size 3584  \
       --ffn-hidden-size 18944 \
       --num-attention-heads 28  \
       --disable-bias-linear \
       --swiglu \
       --position-embedding-type rope \
       --load ${CHECKPOINT} \
       --normalization RMSNorm \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --tokenizer-not-use-fast \
       --micro-batch-size 1  \
       --exit-on-missing-checkpoint \
       --attention-softmax-in-fp32 \
       --input-layernorm-in-fp32 \
       --num-layer-list 3,3,3,3,4,4,4,4 \
       --no-load-rng \
       --no-load-optim \
       --untie-embeddings-and-output-weights \
       --add-qkv-bias \
       --make-vocab-size-divisible-by 1 \
       --padded-vocab-size 151936 \
       --rotary-base 1000000 \
       --no-gradient-accumulation-fusion \
       --no-masked-softmax-fusion \
       --seed 42 \
       --group-query-attention \
       --num-query-groups 4 \
       --no-chat-template \
       --seed 42 \
       | tee logs/eval_mcore_qwen2_57b_a14b.log