#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_ALGO="alltoall=level0:NA;level1:pipeline"

NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=13333
NNODES=1
NODE_RANK=0

WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
echo "World Size: $WORLD_SIZE"

TOKENIZER_PATH="your tokenizer model path"
CHECKPOINT="your model directory path"

TP=1
PP=4
EP=4
CP=1
CP_TYPE='kvallgather_cp_algo'
NUM_LAYERS=44
SEQ_LEN=4096
MBS=1

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

DSA_ARGS="
    --enable-dsa-indexer \
    --index-n-heads 64 \
    --index-head-dim 128 \
    --index-topk 512 \
    --enable-mhc \
    --hc-mult 4 \
    --kv-compress \
    --norm-eps 1e-6 \
"

MLA_ARGS="
    --multi-latent-attention \
    --qk-pos-emb-head-dim 64 \
    --qk-head-dim 512 \
    --q-lora-rank 1024 \
    --o-lora-rank 1024 \
    --kv-lora-rank 512 \
    --v-head-dim 128 \
    --qk-layernorm \
    --mla-fa-without-pad \
"

CA_ARGS="
    --use-g2-attention \
    --o-groups 8 \
    --g2-window-size 128 \
    --rope-head-dim 64 \
    --original-seq-len 65536 \
    --rope-factor 16 \
    --compress-rope-theta 160000 \
    --max-batch-size 4 \
    --compress-ratios 0 0 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 \
"

MOE_ARGS="
    --moe-permute-fusion \
    --swiglu-limit 10.0 \
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --moe-layer-freq 1 \
    --moe-token-dispatcher-type alltoall \
    --first-k-dense-replace -1 \
    --num-experts 256 \
    --moe-router-topk 6 \
    --moe-ffn-hidden-size 2048 \
    --moe-router-load-balancing-type none \
    --moe-router-group-topk 1 \
    --moe-router-num-groups 1 \
    --moe-router-topk-scaling-factor 1.5 \
    --moe-router-score-function sqrtsoftplus \
    --moe-router-enable-expert-bias \
    --moe-shared-expert-intermediate-size 2048 \
    --moe-router-dtype fp32 \
    --n-hash-layers 3 \
"

MTP_ARGS="
    --mtp-num-layers 1 \
"

ROPE_ARGS="
    --beta-fast 32 \
    --beta-slow 1 \
    --rope-scaling-factor 40 \
    --rope-scaling-mscale 1.0 \
    --rope-scaling-mscale-all-dim 1.0 \
    --rope-scaling-original-max-position-embeddings 4096 \
    --rope-scaling-type yarn
"

GPT_ARGS="
    --noop-layers 43 \
    --transformer-impl local \
    --spec mindspeed_llm.tasks.models.spec.deepseek4_spec layer_spec \
    --mtp-spec mindspeed_llm.tasks.models.spec.deepseek4_spec mtp_spec \
    --manual-gc \
    --manual-gc-interval 50 \
    --use-flash-attn \
    --task chat \
    --prompt-type deepseek4 \
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
    --expert-tensor-parallel-size 1 \
    --context-parallel-size ${CP} \
    --context-parallel-algo  ${CP_TYPE} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 4096 \
    --ffn-hidden-size 4096 \
    --num-attention-heads 64 \
    --tokenizer-type PretrainedFromHF  \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings 163840 \
    --micro-batch-size ${MBS} \
    --make-vocab-size-divisible-by 1 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.02 \
    --hidden-dropout 0.0 \
    --position-embedding-type g2 \
    --normalization RMSNorm \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --vocab-size 129280 \
    --padded-vocab-size 129280 \
    --rotary-base 10000 \
    --norm-epsilon 1e-6 \
    --no-load-optim \
    --no-load-rng \
    --max-new-tokens 256 \
    --bf16 \
    --ckpt-format torch \
    --distributed-timeout-minutes 120 \
    --no-gradient-accumulation-fusion \
"

mkdir -p logs

torchrun $DISTRIBUTED_ARGS inference_deepseek4.py \
    --load ${CHECKPOINT} \
    $MTP_ARGS \
    $GPT_ARGS \
    $MLA_ARGS \
    $ROPE_ARGS \
    $MOE_ARGS \
    $DSA_ARGS \
    $CA_ARGS \
    --distributed-backend nccl 2>&1 | tee logs/generate_deepseek4_flash_4k_A3_ptd_16die.log
