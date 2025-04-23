# 注意： deepseek25_60b非全参公版模型，请使用随机初始化权重，请勿与deepseek25_236b全参huggingface权重转换减参权重配合使用。

#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=1200

NPUS_PER_NODE=8
MASTER_ADDR=localhost #主节点IP
MASTER_PORT=6000
NNODES=4
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"

TP=1
PP=4
EP=8
NUM_LAYERS=16

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MLA_ARGS="
    --multi-head-latent-attention \
    --qk-rope-head-dim 64 \
    --qk-nope-head-dim 128 \
    --q-lora-rank 1536 \
    --kv-lora-rank 512 \
    --v-head-dim 128 \
    --qk-layernorm \
"

MOE_ARGS="
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall \
    --use-fused-moe-token-permute-and-unpermute \
    --first-k-dense-replace 1 \
    --moe-layer-freq 1 \
    --n-shared-experts 2 \
    --num-experts 160 \
    --moe-router-topk 6 \
    --moe-intermediate-size 1536 \
    --moe-router-load-balancing-type group_limited_greedy \
    --topk-group 3 \
    --moe-aux-loss-coeff 0.003 \
    --moe-device-level-aux-loss-coeff 0.05 \
    --moe-comm-aux-loss-coeff 0.02 \
    --routed-scaling-factor 16.0 \
    --seq-aux
"

ROPE_ARGS="
    --rope-scaling-beta-fast 32 \
    --rope-scaling-beta-slow 1 \
    --rope-scaling-factor  40 \
    --rope-scaling-mscale 1.0 \
    --rope-scaling-mscale-all-dim 1.0 \
    --rope-scaling-original-max-position-embeddings 4096 \
    --rope-scaling-type yarn
"

GPT_ARGS="
    --spec mindspeed_llm.tasks.models.spec.deepseek_spec layer_spec \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --reuse-fp32-param \
    --no-shared-storage \
    --use-distributed-optimizer \
    --use-flash-attn \
    --shape-order BNSD \
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers-per-virtual-pipeline-stage 2 \
    --expert-model-parallel-size ${EP} \
    --sequence-parallel \
    --output-layer-slice-num 10 \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 5120 \
    --ffn-hidden-size 12288 \
    --num-attention-heads 128 \
    --tokenizer-type PretrainedFromHF  \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length 8192 \
    --max-position-embeddings 163840 \
    --micro-batch-size 1 \
    --global-batch-size 256 \
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
    --vocab-size 102400 \
    --padded-vocab-size 102400 \
    --rotary-base 10000 \
    --no-gradient-accumulation-fusion \
    --norm-epsilon 1e-6 \
    --no-load-optim \
    --no-load-rng \
    --bf16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 0 \
    --no-save-optim \
    --no-save-rng
"

python -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $MLA_ARGS \
    $ROPE_ARGS \
    $MOE_ARGS \
    --distributed-backend nccl \
    --save $CKPT_SAVE_DIR \
    --load $CKPT_LOAD_DIR \
    | tee logs/pretrain_deepseek25_60b_8k_ptd.log
