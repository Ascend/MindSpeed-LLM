source examples/fsdp2/env_config.sh

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=12322
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# Commonly used parameters are passed as CLI args here; see companion YAML for full config.
# CLI args take precedence over the YAML when both are set. All args can also be moved into the YAML if preferred.
torchrun $DISTRIBUTED_ARGS train_fsdp2.py ./tests/st/shell_scripts/pretrain_qwen3_8b_4K_fsdp2.yaml \
    --model.model_name_or_path /data/ci/models/Qwen3-8B/hf/Qwen3-8B \
    --data.dataset '{"file_name": "/data/ci/datasets/origin/train-00000-of-00001-a09b74b3ef9c3b56.parquet"}' \
    --parallel.fsdp_size 8 \
    --parallel.ep_size 1 \
    --parallel.ep_fsdp_size 1 \
    --training.per_device_train_batch_size 2 \
    --training.gradient_accumulation_steps 1 \
    --training.output_dir ./output
