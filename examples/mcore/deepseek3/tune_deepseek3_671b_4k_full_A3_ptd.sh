#!/bin/bash



export HCCL_CONNECT_TIMEOUT=3600
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

NPUS_PER_NODE=16
MASTER_ADDR=localhost #主节点IP
MASTER_PORT=6000
NNODES=16
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_LOAD_DIR="your model ckpt path"
CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"

TP=2
PP=16
EP=8
CP=1
CP_TYPE='ulysses_cp_algo'
NUM_LAYERS=64
SEQ_LEN=4096
MBS=1
GBS=1024

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
    --qk-head-dim 128 \
    --q-lora-rank 1536 \
    --kv-lora-rank 512 \
    --v-head-dim 128 \
    --qk-layernorm \
"

MOE_ARGS="
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --moe-tp-extend-ep \
    --moe-alltoall-overlap-comm \
    --moe-token-dispatcher-type alltoall_seq \
    --moe-permute-fusion \
    --first-k-dense-replace 3 \
    --moe-layer-freq 1 \
    --n-shared-experts 1 \
    --num-experts 256 \
    --moe-router-topk 8 \
    --moe-ffn-hidden-size 2048 \
    --moe-router-load-balancing-type none \
    --moe-router-group-topk 4 \
    --moe-router-num-groups 8 \
    --moe-router-topk-scaling-factor 2.5 \
    --seq-aux \
    --moe-aux-loss-coeff 0.001 \
    --norm-topk-prob \
    --moe-router-score-function sigmoid \
    --moe-router-enable-expert-bias \
"

MTP_ARGS="
    --mtp-num-layers 1 \
    --mtp-loss-scaling-factor 0.3 \
"

ROPE_ARGS="
    --beta-fast 32 \
    --beta-slow 1 \
    --rope-scaling-factor 40 \
    --rope-scaling-mscale 1.0 \
    --rope-scaling-mscale-all-dim  1.0 \
    --rope-scaling-original-max-position-embeddings 4096 \
    --rope-scaling-type yarn
"

GPT_ARGS="
    --spec mindspeed_llm.tasks.models.spec.deepseek_spec layer_spec \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --use-distributed-optimizer \
    --reuse-fp32-param \
    --use-flash-attn \
    --shape-order BNSD \
    --use-mcore-models \
    --noop-layers 47,62,63 \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
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
    --make-vocab-size-divisible-by 1 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
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
    --vocab-size 129280 \
    --padded-vocab-size 129280 \
    --rotary-base 10000 \
    --norm-epsilon 1e-6 \
    --distributed-timeout-minutes 120 \
"

TRAIN_ARGS="
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --train-iters 2000 \
    --lr 1.0e-5 \
    --lr-decay-style cosine \
    --init-method-std 0.02 \
    --min-lr 1.0e-7 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --initial-loss-scale 65536 \
    --bf16 \
"

CKPT_ARGS="
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --seed 1234 \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
"

DATA_ARGS="
    --no-shared-storage \
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 500 \
    --eval-interval 500 \
    --eval-iters 0 \
"

FINETUNE_ARGS="
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --no-pad-to-seq-lengths \
"

python -m torch.distributed.launch $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $CKPT_ARGS \
    $DATA_ARGS \
    $TRAIN_ARGS \
    $OUTPUT_ARGS \
    $MLA_ARGS \
    $ROPE_ARGS \
    $MTP_ARGS \
    $MOE_ARGS \
    $FINETUNE_ARGS \
    --distributed-backend nccl \
    | tee tune_deepseek3_671b_4k_full_A3_256_ptd.log