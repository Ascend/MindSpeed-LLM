#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your checkpoint save path"
DATA_PATH="your finetune dataset path"
TOKENIZER_MODEL="your tokenizer model path"
CKPT_LOAD_DIR="your checkpoint load path"

TP=1
PP=1
EP=8

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MLA_ARGS="
    --spec mindspeed_llm.tasks.models.spec.deepseek_spec layer_spec \
    --multi-latent-attention \
    --qk-pos-emb-head-dim 64 \
    --qk-head-dim 128 \
    --kv-lora-rank 512 \
    --v-head-dim 128 \
    --qk-layernorm \
"

MOE_ARGS="
    --moe-permute-fusion \
    --moe-grouped-gemm \
    --moe-alltoall-overlap-comm \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall_seq \
    --moe-permute-fusion \
    --first-k-dense-replace 1 \
    --moe-layer-freq 1 \
    --n-shared-experts 2 \
    --num-experts 64 \
    --moe-router-topk 6 \
    --moe-ffn-hidden-size 1408 \
    --moe-router-load-balancing-type pai_megatron_aux_loss \
    --moe-router-group-topk 1 \
    --moe-router-num-groups 8 \
    --moe-aux-loss-coeff 0.01 \
    --moe-router-topk-scaling-factor 1.0 \
    --seq-aux
"

ROPE_ARGS="
    --beta-fast 32 \
    --beta-slow 1 \
    --rope-scaling-factor  40 \
    --rope-scaling-mscale 0.707 \
    --rope-scaling-mscale-all-dim  0.707 \
    --rope-scaling-original-max-position-embeddings 4096 \
    --rope-scaling-type yarn
"

FITUNE_ARGS="
    --stage sft \
    --finetune \
    --is-instruction-dataset \
    --no-pad-to-seq-lengths \
    --prompt-type deepseek2-lite \
    --tokenizer-not-use-fast \
    "


GPT_ARGS="
    --shape-order BNSD \
    --load $CKPT_LOAD_DIR \
    --use-distributed-optimizer \
    --use-flash-attn \
    --use-mcore-models \
    --reuse-fp32-param \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
    --sequence-parallel \
    --num-layers 27 \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --hidden-size 2048 \
    --ffn-hidden-size 10944 \
    --num-attention-heads 16 \
    --tokenizer-type PretrainedFromHF  \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --num-workers 8 \
    --seq-length 4096 \
    --max-position-embeddings 163840 \
    --micro-batch-size 1 \
    --global-batch-size 768 \
    --make-vocab-size-divisible-by 1 \
    --lr 9e-6 \
    --train-iters 462240 \
    --lr-decay-style cosine \
    --lr-decay-iters 462240 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.008 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --swiglu \
    --dataloader-type cyclic \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 1 \
    --vocab-size 102400 \
    --padded-vocab-size 102400 \
    --rotary-base 10000 \
    --no-gradient-accumulation-fusion \
    --norm-epsilon 1e-6 \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 1000 \
    --eval-iters 0 \
    --no-save-optim \
    --no-save-rng
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $MLA_ARGS \
    $ROPE_ARGS \
    $MOE_ARGS \
    $FITUNE_ARGS \
    --distributed-backend nccl \
    --save $CKPT_SAVE_DIR \
    | tee ./logs/tune_deepseek2_lite_ptd_16p.log