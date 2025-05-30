#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_ALGO="alltoall=level0:NA;level1:pipeline"
export HCCL_BUFFSIZE=400

NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=9110
NNODES=32
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"

TP=2
PP=8
EP=16
CP=1
CP_TYPE='ulysses_cp_algo'
NUM_LAYERS=64
SEQ_LEN=4096
MBS=1
GBS=3840

DISTRIBUTED_ARGS="
    --master_addr $MASTER_ADDR \
    --node_rank $NODE_RANK \
    --worker_num $WORLD_SIZE \
    --local_worker_num $NPUS_PER_NODE \
    --master_port $MASTER_PORT \
    --log_dir=msrun_log \
    --join=False \
    --cluster_time_out=300 \
    --bind_core=True \
"

MLA_ARGS="
    --multi-head-latent-attention \
    --qk-rope-head-dim 64 \
    --qk-nope-head-dim 128 \
    --q-lora-rank 1536 \
    --kv-lora-rank 512 \
    --v-head-dim 128 \
    --qk-layernorm \
    --mla-mm-split \
    --mla-fa-without-pad \
"

MOE_ARGS="
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --use-fused-moe-token-permute-and-unpermute \
    --moe-token-dispatcher-type alltoall \
    --first-k-dense-replace 3 \
    --moe-layer-freq 1 \
    --n-shared-experts 1 \
    --num-experts 256 \
    --moe-router-topk 8 \
    --moe-intermediate-size 2048 \
    --moe-router-load-balancing-type noaux_tc \
    --n-group 8 \
    --topk-group 4 \
    --routed-scaling-factor 2.5 \
    --moe-aux-loss-coeff 0.0001 \
    --seq-aux \
    --norm-topk-prob \
    --moe-router-score-function sigmoid \
    --moe-router-enable-expert-bias \
    --moe-tp-extend-ep \
    --router-gating-in-fp32 \
"

MTP_ARGS="
    --mtp-num-layers 1 \
    --mtp-loss-scaling-factor 0.3 \
    --mtp-mem-efficient-logits \
"

ROPE_ARGS="
    --rope-scaling-beta-fast 32 \
    --rope-scaling-beta-slow 1 \
    --rope-scaling-factor 40 \
    --rope-scaling-mscale 1.0 \
    --rope-scaling-mscale-all-dim 1.0 \
    --rope-scaling-original-max-position-embeddings 4096 \
    --rope-scaling-type yarn
"

MEM_ARGS="
    --use-distributed-optimizer \
    --recompute-method uniform \
    --recompute-granularity full \
    --recompute-num-layers 1 \
"

GPT_ARGS="\
    --spec mindspeed_llm.tasks.models.spec.deepseek_spec layer_spec \
    --reset-position-ids \
    --noop-layers 61,62,63 \
    --no-shared-storage \
    --use-flash-attn \
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers-per-virtual-pipeline-stage ${VPP} \
    --expert-model-parallel-size ${EP} \
    --sequence-parallel \
    --context-parallel-size ${CP} \
    --context-parallel-algo  ${CP_TYPE} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 7168 \
    --ffn-hidden-size 18432 \
    --num-attention-heads 128 \
    --tokenizer-type PretrainedFromHF  \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings 163840 \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --lr 1.0e-5 \
    --train-iters 2000 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.02 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.0e-7 \
    --weight-decay 1e-2 \
    --lr-warmup-iters 500 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --initial-loss-scale 65536 \
    --vocab-size 129280 \
    --padded-vocab-size 129280 \
    --rotary-base 10000 \
    --norm-epsilon 1e-6 \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --distributed-timeout-minutes 120
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1 \
    --eval-interval 2000 \
    --eval-iters 0 \
    --no-save-optim \
    --no-save-rng
"

msrun $DISTRIBUTED_ARGS $basepath/pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $MLA_ARGS \
    $ROPE_ARGS \
    $MOE_ARGS \
    $MTP_ARGS \
    $MEM_ARGS \
    --distributed-backend nccl \
    --ai-framework mindspore \
    2>&1 | tee logs/pretrain_deepseek3_671b_4k_A3_ms.log
