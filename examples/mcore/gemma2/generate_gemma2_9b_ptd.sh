#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Change for multinode config
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
TOKENIZER_PATH="your tokenizer directory path"
CHECKPOINT="your model directory path"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT"

torchrun $DISTRIBUTED_ARGS inference.py \
       --tensor-model-parallel-size 8  \
       --pipeline-model-parallel-size 1  \
       --spec mindspeed_llm.tasks.models.spec.gemma2_spec layer_spec \
       --use-mcore-models \
       --gelu-tanh \
       --post-norm \
       --query-pre-attn-scalar 256 \
       --output-logit-softcapping 30.0 \
       --attn-logit-softcapping 50.0 \
       --interleave-sliding-window 4096 \
       --group-query-attention \
       --num-query-groups 8 \
       --load ${CHECKPOINT}  \
       --num-layers 42 \
       --hidden-size 3584  \
       --kv-channels 256 \
       --ffn-hidden-size 14336 \
       --num-attention-heads 16  \
       --position-embedding-type rope \
       --seq-length 8192 \
       --max-position-embeddings 8192 \
       --max-new-tokens 256 \
       --micro-batch-size 1 \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --tokenizer-not-use-fast \
       --normalization RMSNorm \
       --add-rmsnorm-offset \
       --norm-epsilon 1e-06 \
       --input-embeds-norm \
       --disable-bias-linear \
       --hidden-dropout 0 \
       --attention-dropout 0 \
       --attention-softmax-in-fp32 \
       --no-load-optim \
       --no-load-rng \
       --no-masked-softmax-fusion \
       --no-gradient-accumulation-fusion \
       --exit-on-missing-checkpoint \
       --make-vocab-size-divisible-by 1 \
       --vocab-size 256000 \
       --bf16 \
       --seed 42 \
       --transformer-impl local \
       | tee logs/generate_gemma2_9b_mcore.log
