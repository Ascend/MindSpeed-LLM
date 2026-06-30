source examples/fsdp2/env_config.sh

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6060
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
mkdir -p ./logs
torchrun $DISTRIBUTED_ARGS train_fsdp2.py tests/pipeline/st/glm52/pretrain_glm52_38b_4k_fsdp2_A3.yaml \
    --parallel.fsdp_size 8 \
    --parallel.ep_size 8 \
    --parallel.ep_fsdp_size 1
