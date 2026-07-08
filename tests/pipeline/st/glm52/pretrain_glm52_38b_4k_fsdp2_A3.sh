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
# Commonly used parameters are passed as CLI args here; see companion YAML for full config.
# CLI args take precedence over the YAML when both are set. All args can also be moved into the YAML if preferred.
torchrun $DISTRIBUTED_ARGS train_fsdp2.py tests/pipeline/st/glm52/pretrain_glm52_38b_4k_fsdp2_A3.yaml \
    --model.model_name_or_path /data/ci/models/glm52/hf/glm52_layer4 \
    --parallel.fsdp_size 8 \
    --parallel.ep_size 8 \
    --parallel.ep_fsdp_size 1 \
    --training.per_device_train_batch_size 1 \
    --training.gradient_accumulation_steps 1 \
    --training.output_dir ./output \
    --optimization.use_fused_rmsnorm True \
    --optimization.moe_grouped_gemm True \
    --optimization.use_fused_rotary_pos_emb True \
    --optimization.use_sparse_flash_attn True \
    --optimization.use_fused_lightning_indexer True \
    --optimization.use_fused_lightning_indexer_loss True \
    --optimization.indexer_loss_coeff 0.1
