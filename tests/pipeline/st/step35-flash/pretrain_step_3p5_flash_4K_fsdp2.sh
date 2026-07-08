source examples/fsdp2/env_config.sh

export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export MULTI_STREAM_MEMORY_REUSE=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

NPUS_PER_NODE=8
MASTER_ADDR=localhost #主节点IP
MASTER_PORT=6499
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
torchrun $DISTRIBUTED_ARGS train_fsdp2.py tests/pipeline/st/step35-flash/pretrain_step_3p5_flash_4K_fsdp2.yaml \
    --model.model_name_or_path /data/ci/models/step3.5/mg/step3.5-flash-layer4/ \
    --parallel.fsdp_size 8 \
    --parallel.ep_size 4 \
    --parallel.ep_fsdp_size 2 \
    --training.per_device_train_batch_size 1 \
    --training.gradient_accumulation_steps 1 \
    --training.output_dir ./output
