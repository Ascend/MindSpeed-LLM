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
# Commonly used parameters are passed as CLI args here; see companion YAML for full config.
# CLI args take precedence over the YAML when both are set. All args can also be moved into the YAML if preferred.
torchrun $DISTRIBUTED_ARGS train_fsdp2.py examples/fsdp2/minimax_m27/pretrain_minimax_m2p7_229b_8K_fsdp2_A3.yaml \
    --model.model_name_or_path /home/data/MiniMax-M2.7/ \
    --parallel.fsdp_size 128 \
    --parallel.ep_size 128 \
    --parallel.ep_fsdp_size 1 \
    --training.per_device_train_batch_size 1 \
    --training.gradient_accumulation_steps 1 \
    --training.output_dir ./output \
    --optimization.moe_grouped_gemm True \
    --optimization.use_fused_rmsnorm True \
    --optimization.use_fused_rotary_pos_emb True \
    --optimization.use_flash_attn True \
    | tee logs/pretrain_minimax_m2p7_229b_8K_fsdp2_A3_${TIMESTAMP}.log
