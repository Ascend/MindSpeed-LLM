source examples/fsdp2/env_config.sh

export TORCH_HCCL_ZERO_COPY=1

NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6499
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
TIMESTAMP=$(date "+%Y-%m-%d_%H-%M-%S")

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS train_fsdp2.py examples/fsdp2/qwen3/pretrain_qwen3_32b_4k_fsdp2_A3.yaml | tee ./logs/fsdp2_qwen3_32b_pretrain_${TIMESTAMP}.log