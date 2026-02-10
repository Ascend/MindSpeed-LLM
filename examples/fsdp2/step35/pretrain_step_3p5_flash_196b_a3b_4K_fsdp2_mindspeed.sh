source examples/fsdp2/env_config.sh

export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export MULTI_STREAM_MEMORY_REUSE=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

NPUS_PER_NODE=16
MASTER_ADDR=localhost #主节点IP
MASTER_PORT=6499
NNODES=12
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS train_fsdp2.py examples/fsdp2/step35/pretrain_step_3p5_flash_196b_a3b_4K_fsdp2.yaml
