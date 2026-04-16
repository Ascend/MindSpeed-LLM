source examples/fsdp2/env_config.sh

NPUS_PER_NODE=16
MASTER_ADDR=localhost #主节点IP
MASTER_PORT=6499
NNODES=8
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
bash tests/tools/fsdp2/moe_hf_param_merge_experts.sh
torchrun $DISTRIBUTED_ARGS train_fsdp2.py examples/fsdp2/minimax_m27/pretrain_minimax_m2p7_229b_4K_fsdp2_A3.yaml | tee logs/pretrain_minimax_m2p7_229b_4K_fsdp2_A3_${TIMESTAMP}.log