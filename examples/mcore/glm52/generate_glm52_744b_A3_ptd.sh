#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600

NPUS_PER_NODE=16
MASTER_ADDR=localhost #主节点IP
MASTER_PORT=6000
NNODES=2
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
TOKENIZER_PATH="your tokenizer model path"
CHECKPOINT="your model directory path"

TP=2
PP=4
EP=8
NUM_LAYERS=78
SEQ_LEN=1024


DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"


MLA_ARGS="
    --multi-latent-attention \
    --qk-pos-emb-head-dim 64 \
    --qk-head-dim 192 \
    --q-lora-rank 2048 \
    --kv-lora-rank 512 \
    --v-head-dim 256 \
    --qk-layernorm \
    --enable-dsa-indexer \
    --init-norm-weight-in-fp32 \
    --index-topk 2048 \
    --index-n-heads 32 \
    --mla-mm-split \
    --use-fused-lightning-indexer-loss \
    --use-fused-lightning-indexer \
    --use-sparse-flash-attn \
    --enable-mla-absorb \
    --index-topk-freq 4 \
    --index-skip-topk-offset 3 \
    --index-share-for-mtp-iteration \
    --task chat \
"


MOE_ARGS="
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall \
    --first-k-dense-replace 3 \
    --moe-layer-freq 1 \
    --n-shared-experts 1 \
    --num-experts 256 \
    --moe-router-topk 8 \
    --moe-ffn-hidden-size 2048 \
    --moe-router-load-balancing-type none \
    --moe-router-num-groups 1 \
    --moe-router-group-topk 1 \
    --moe-router-topk-scaling-factor 2.5 \
    --moe-aux-loss-coeff 0.0001 \
    --seq-aux \
    --norm-topk-prob \
    --moe-router-score-function sigmoid \
    --moe-router-enable-expert-bias \
    --moe-router-dtype fp32 \
"


GPT_ARGS="
    --kv-channels 64 \
    --no-rope-fusion \
    --top-p 0.95 \
    --temperature 1.0 \
    --use-fused-rmsnorm \
    --use-fused-swiglu \
    --spec mindspeed_llm.tasks.models.spec.deepseek_spec layer_spec \
    --gemm-gradient-accumulation-fusion \
    --reuse-fp32-param \
    --shape-order BNSD \
    --use-mcore-models \
    --use-flash-attn \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layer-list 18,20,20,20 \
    --expert-model-parallel-size ${EP} \
    --expert-tensor-parallel-size 1 \
    --sequence-parallel \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 6144 \
    --ffn-hidden-size 12288 \
    --num-attention-heads 64 \
    --tokenizer-type PretrainedFromHF  \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings 1048576  \
    --micro-batch-size 1 \
    --make-vocab-size-divisible-by 1 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.02 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --use-rotary-position-embeddings \
    --swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --vocab-size 154880 \
    --padded-vocab-size 154880 \
    --rotary-base 8000000 \
    --norm-epsilon 1e-5 \
    --max-new-tokens 128 \
    --bf16 \
    --transformer-impl local \
    --ckpt-format torch
"

torchrun $DISTRIBUTED_ARGS inference.py \
    $GPT_ARGS \
    $MLA_ARGS \
    $MOE_ARGS \
    --load ${CHECKPOINT} \
    --distributed-backend nccl \
    | tee logs/generate_glm52_744b_signal.log
