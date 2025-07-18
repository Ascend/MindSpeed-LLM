#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_MODEL="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"

TP=1
PP=1
CP=8
CP_TYPE="ulysses_cp_algo"
SEQ_LENGTH=32768
MBS=1
GBS=64
TRAIN_ITERS=2000

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MEMORY_ARGS="
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 5
"

MLA_ARGS="
    --multi-latent-attention \
    --qk-pos-emb-head-dim 32 \
    --qk-head-dim 64 \
    --q-lora-rank 768 \
    --kv-lora-rank 256 \
    --v-head-dim 64 \
    --qk-layernorm \
    --scale-depth 1.4 \
    --dim-model-base 256
"

LONG_FACTOR="1.0591234137867171,1.1241891283591912,1.2596935748670968,1.5380380402321725,2.093982484148734,3.1446935121267696,4.937952647693647,7.524541999994549,10.475458000005451,13.062047352306353,14.85530648787323,15.906017515851266,16.461961959767827,16.740306425132907,16.87581087164081,16.940876586213285"
SHORT_FACTOR="1.0591234137867171,1.1241891283591912,1.2596935748670968,1.5380380402321725,2.093982484148734,3.1446935121267696,4.937952647693647,7.524541999994549,10.475458000005451,13.062047352306353,14.85530648787323,15.906017515851266,16.461961959767827,16.740306425132907,16.87581087164081,16.940876586213285"

ROPE_ARGS="
    --rotary-percent 0.5 \
    --rope-scaling-original-max-position-embeddings 32768 \
    --rope-scaling-type longrope \
    --scale-emb 12 \
    --long-factor ${LONG_FACTOR} \
    --short-factor ${SHORT_FACTOR}
"

GPT_ARGS="
    --use-cp-send-recv-overlap \
    --use-distributed-optimizer \
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ${CP_TYPE} \
    --sequence-parallel \
    --output-layer-slice-num 8 \
    --use-flash-attn \
    --num-layers 62 \
    --hidden-size 2560 \
    --ffn-hidden-size 6400 \
    --num-attention-heads 40 \
    --tokenizer-type PretrainedFromHF  \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --seq-length ${SEQ_LENGTH} \
    --max-position-embeddings 32768 \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --lr 1.0e-5 \
    --train-iters ${TRAIN_ITERS} \
    --lr-decay-style cosine \
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
    --vocab-size 73448 \
    --padded-vocab-size 73448 \
    --rotary-base 10000 \
    --no-gradient-accumulation-fusion \
    --norm-epsilon 1e-5 \
    --tokenizer-not-use-fast \
    --spec mindspeed_llm.tasks.models.spec.minicpm_spec layer_spec \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --seed 1234
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval ${TRAIN_ITERS} \
    --eval-interval ${TRAIN_ITERS} \
    --eval-iters 0 \
    --no-save-optim \
    --no-save-rng
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $MLA_ARGS \
    $ROPE_ARGS \
    $MEMORY_ARGS \
    --distributed-backend nccl \
    --load $CKPT_LOAD_DIR \
    --save $CKPT_SAVE_DIR \
    | tee logs/pretrain_minicpm3_4b_32k_ptd_8p.log
