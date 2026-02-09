source examples/fsdp2/env_config.sh

NPUS_PER_NODE=8
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
mkdir -p ./logs
torchrun $DISTRIBUTED_ARGS train_fsdp2.py \
     examples/fsdp2/gpt_oss/tune_gpt_oss_20b_a3b_4K_fsdp2.yaml \
     | tee logs/tune_gpt_oss_20b_${TIMESTAMP}.log
