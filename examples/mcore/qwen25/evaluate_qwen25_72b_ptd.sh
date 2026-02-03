#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# modify script model path and tokenizer path
TOKENIZER_PATH="your tokenizer directory path"
CHECKPOINT="your model directory path"

# configure task and data path
DATA_PATH="./mmlu/test/"
TASK="mmlu"

# distributed config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

SEQ_LEN=32768
TP=8
PP=1

# configure generation parameters
python -m torch.distributed.launch $DISTRIBUTED_ARGS evaluation.py   \
       --task-data-path $DATA_PATH \
       --task $TASK\
       --no-chat-template \
       --seq-length ${SEQ_LEN} \
       --max-new-tokens 1 \
       --evaluation-batch-size 1 \
       --max-position-embeddings ${SEQ_LEN} \
       --tensor-model-parallel-size ${TP}  \
       --pipeline-model-parallel-size ${PP}  \
       --num-layers 80  \
       --hidden-size 8192  \
       --ffn-hidden-size 29568 \
       --num-attention-heads 64  \
       --group-query-attention \
       --num-query-groups 8 \
       --add-qkv-bias \
       --swiglu \
       --disable-bias-linear \
       --rotary-base 1000000 \
       --padded-vocab-size 152064 \
       --load ${CHECKPOINT}  \
       --normalization RMSNorm \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --tokenizer-not-use-fast \
       --norm-epsilon 1e-5 \
       --micro-batch-size 1  \
       --position-embedding-type rope \
       --exit-on-missing-checkpoint \
       --no-load-rng \
       --no-load-optim \
       --untie-embeddings-and-output-weights \
       --no-masked-softmax-fusion \
       --make-vocab-size-divisible-by 1 \
       --use-mcore-models \
       --use-rotary-position-embeddings \
       --use-fused-rmsnorm \
       --bf16 \
       --transformer-impl local \
       --seed 42 | tee logs/evaluation_mcore_qwen25_72b_${TASK}.log
