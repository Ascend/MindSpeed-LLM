# #!/bin/bash
# export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Change for multinode config
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
TOKENIZER_PATH="./model_from_hf/GLM-4-32B-0414"
CHECKPOINT="./model_weights/GLM-4-32B-0414"


DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT"

torchrun ${DISTRIBUTED_ARGS} inference.py \
       --spec mindspeed_llm.tasks.models.spec.gemma2_spec layer_spec \
       --tensor-model-parallel-size 2  \
       --pipeline-model-parallel-size 4  \
       --num-layer-list 15,15,15,16 \
       --use-mcore-models \
       --num-layers 61  \
       --hidden-size 6144  \
       --ffn-hidden-size 23040 \
       --seq-length 8192 \
       --group-query-attention \
       --num-query-groups 2 \
       --num-attention-heads 48  \
       --padded-vocab-size 151552 \
       --make-vocab-size-divisible-by 1 \
       --max-position-embeddings 8192 \
       --position-embedding-type rope \
       --use-glm-rope \
       --rotary-percent 0.5 \
       --no-rope-fusion \
       --disable-bias-linear \
       --swiglu \
       --post-norm \
       --norm-epsilon 1e-05 \
       --hidden-dropout 0.0 \
       --attention-dropout 0.0 \
       --normalization RMSNorm \
       --max-new-tokens 256 \
       --micro-batch-size 1 \
       --load ${CHECKPOINT}  \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --untie-embeddings-and-output-weights \
       --attention-softmax-in-fp32 \
       --no-load-optim \
       --no-load-rng \
       --no-masked-softmax-fusion \
       --no-gradient-accumulation-fusion \
       --exit-on-missing-checkpoint \
       --seed 42 \
       --bf16 \
       --transformer-impl local \
       | tee logs/generate_GLM-4-32B-0414_mcore.log