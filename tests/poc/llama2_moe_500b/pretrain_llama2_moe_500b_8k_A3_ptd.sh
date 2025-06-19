#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=2400

NPUS_PER_NODE=16
MASTER_ADDR=localhost #主节点IP
MASTER_PORT=6000
NNODES=16
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_MODEL="your tokenizer.model file path"
CKPT_LOAD_DIR="your model ckpt path"

TP=1
PP=16
CP=1
EP=16

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --use-mcore-models \
    --no-shared-storage \
    --reuse-fp32-param \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo megatron_cp_algo \
    --use-cp-send-recv-overlap \
    --log-throughput \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --num-layers-per-virtual-pipeline-stage 1 \
    --use-distributed-optimizer \
    --num-layers 64 \
    --hidden-size 4096 \
    --ffn-hidden-size 4096 \
    --num-attention-heads 32 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --micro-batch-size 1 \
    --global-batch-size 512 \
    --make-vocab-size-divisible-by 1 \
    --lr 1.0e-6 \
    --train-iters 5000 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --use-fused-rotary-pos-emb \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --use-flash-attn \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.0e-7 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 4096.0 \
    --adam-beta2 0.95 \
    --adam-eps 1e-5 \
    --disable-bias-linear \
    --group-query-attention \
    --num-query-groups 4 \
    --expert-model-parallel-size ${EP} \
    --num-experts 160 \
    --moe-grouped-gemm \
    --moe-token-dispatcher-type alltoall_seq \
    --moe-router-topk 5 \
    --moe-permutation-async-comm \
    --lr-warmup-fraction 0.01 \
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 4 \
    --enable-recompute-layers-per-pp-rank \
    --use-fused-moe-token-permute-and-unpermute \
    --recompute-in-advance \
    --fix-router \
    --bf16
"

DATA_ARGS="
    --data-path ${DATA_PATH} \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} pretrain_gpt.py \
    ${GPT_ARGS} \
    ${DATA_ARGS} \
    ${OUTPUT_ARGS} \
    --distributed-backend nccl \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    | tee logs/train_llama2_moe_500b_8k_A3_ptd.log
