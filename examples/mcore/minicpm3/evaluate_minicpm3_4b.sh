#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_CONNECT_TIMEOUT=1200

# modify script model path and tokenizer path
TOKENIZER_PATH="your tokenizer model path"
CHECKPOINT="your model directory path"

# configure task and data path
DATA_PATH="./mmlu/test/"
TASK="mmlu"

# distributed config
MASTER_ADDR=localhost
MASTER_PORT=6011
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=2
DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
LONG_FACTOR="1.0591234137867171,1.1241891283591912,1.2596935748670968,1.5380380402321725,2.093982484148734,3.1446935121267696,4.937952647693647,7.524541999994549,10.475458000005451,13.062047352306353,14.85530648787323,15.906017515851266,16.461961959767827,16.740306425132907,16.87581087164081,16.940876586213285"
SHORT_FACTOR="1.0591234137867171,1.1241891283591912,1.2596935748670968,1.5380380402321725,2.093982484148734,3.1446935121267696,4.937952647693647,7.524541999994549,10.475458000005451,13.062047352306353,14.85530648787323,15.906017515851266,16.461961959767827,16.740306425132907,16.87581087164081,16.940876586213285"

torchrun $DISTRIBUTED_ARGS evaluation.py   \
    --task-data-path $DATA_PATH \
    --task $TASK\
    --tensor-model-parallel-size 1  \
    --pipeline-model-parallel-size 2  \
    --use-mcore-models \
    --use-flash-attn \
    --multi-latent-attention \
    --prompt-type minicpm3 \
    --spec mindspeed_llm.tasks.models.spec.minicpm_spec layer_spec \
    --qk-pos-emb-head-dim 32 \
    --qk-head-dim 64 \
    --q-lora-rank 768 \
    --kv-lora-rank 256 \
    --v-head-dim 64 \
    --qk-layernorm \
    --scale-depth 1.4 \
    --dim-model-base 256 \
    --rotary-percent 0.5 \
    --rope-scaling-type longrope \
    --rope-scaling-original-max-position-embeddings 32768 \
    --use-fused-rotary-pos-emb \
    --scale-emb 12 \
    --long-factor ${LONG_FACTOR} \
    --short-factor ${SHORT_FACTOR} \
    --num-layers 62 \
    --hidden-size 2560 \
    --ffn-hidden-size 6400 \
    --num-attention-heads 40 \
    --position-embedding-type rope \
    --norm-epsilon 1e-5 \
    --seq-length 32768 \
    --max-new-tokens 1 \
    --top-p 0.7 \
    --micro-batch-size 1 \
    --global-batch-size 16 \
    --max-position-embeddings 32768 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --swiglu \
    --load ${CHECKPOINT} \
    --tokenizer-type PretrainedFromHF  \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --tokenizer-not-use-fast \
    --bf16 \
    --normalization RMSNorm \
    --disable-bias-linear \
    --attention-softmax-in-fp32 \
    --no-load-optim \
    --no-load-rng \
    --no-masked-softmax-fusion \
    --no-gradient-accumulation-fusion \
    --exit-on-missing-checkpoint \
    --make-vocab-size-divisible-by 1 \
    | tee logs/evaluate_minicpm3_4b.log