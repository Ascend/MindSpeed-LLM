source examples/fsdp2/env_config.sh

export HCCL_CONNECT_TIMEOUT=3600
export CUDA_DEVICE_MAX_CONNECTIONS=2
export STREAMS_PER_DEVICE=32
export NPU_ASD_ENABLE=0
export TORCH_HCCL_ZERO_COPY=1

NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6499
NNODES=4
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
torchrun $DISTRIBUTED_ARGS train_fsdp2.py examples/fsdp2/gpt_oss/tune_gpt_oss_20b_128k_fsdp2_A3.yaml \
    --model.model_name_or_path /home/data/gpt-oss-20b-hf/ \
    --model.tokenizer_name_or_path /home/data/gpt-oss-20b-hf/ \
    --parallel.fsdp_size 64 \
    --parallel.ep_size 1 \
    --parallel.ep_fsdp_size 1 \
    --training.per_device_train_batch_size 1 \
    --training.gradient_accumulation_steps 1 \
    --training.output_dir ./output \
    --optimization.use_fused_rmsnorm True \
    --optimization.use_fused_rotary_pos_emb True \
    --optimization.use_flash_attn True \
    | tee logs/tune_gpt_oss_20b_128k_${TIMESTAMP}.log
