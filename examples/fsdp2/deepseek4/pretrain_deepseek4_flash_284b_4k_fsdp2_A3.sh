source examples/fsdp2/env_config.sh

NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6499
NNODES=16
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
torchrun $DISTRIBUTED_ARGS train_fsdp2.py examples/fsdp2/deepseek4/pretrain_deepseek4_flash_284b_4k_fsdp2_A3.yaml \
    --model.model_name_or_path /home/data/deepseek4-bf16-hf/ \
    --data.dataset '{"file_name": "your origin data path.example: /home/train-00000-of-a09b74b3ef9c3b56.parquet"}' \
    --parallel.fsdp_size 256 \
    --parallel.ep_size 128 \
    --parallel.ep_fsdp_size 2 \
    --training.per_device_train_batch_size 1 \
    --training.gradient_accumulation_steps 8 \
    --training.output_dir ./output \
    --optimization.use_fused_rmsnorm True \
    --optimization.use_fused_rotary_pos_emb True \
    --optimization.use_sparse_flash_attn True \
    --optimization.use_fused_lightning_indexer True \
    --optimization.use_fused_lightning_indexer_loss True \
    --optimization.use_ascend_mhc True \
    --optimization.use_triton_swiglu_limit True \
    | tee logs/pretrain_deepseek4_flash_284b_4k_fsdp2_A3_${TIMESTAMP}.log
