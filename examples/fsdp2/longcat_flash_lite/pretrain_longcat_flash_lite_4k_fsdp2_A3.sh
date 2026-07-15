source examples/fsdp2/env_config.sh

NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6499
NNODES=2
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

mkdir -p ./logs
bash tests/tools/fsdp2/longcat_flash_lite_moe_hf_weight_convert.sh
# Commonly used parameters are passed as CLI args here; see companion YAML for full config.
# CLI args take precedence over the YAML when both are set. All args can also be moved into the YAML if preferred.
torchrun $DISTRIBUTED_ARGS train_fsdp2.py examples/fsdp2/longcat_flash_lite/pretrain_longcat_flash_lite_4k_fsdp2_A3.yaml \
    --model.model_name_or_path ./fsdp_weights/LongCat-Flash-Lite-mergeExperts \
    --data.dataset '{"file_name": "./datasets/alpaca/train-00000-of-00001-a09b74b3ef9c3b56.parquet"}' \
    --parallel.fsdp_size 32 \
    --parallel.ep_size 32 \
    --training.per_device_train_batch_size 1 \
    --training.gradient_accumulation_steps 1 \
    --training.output_dir ./output \
    | tee logs/pretrain_longcat_flash_lite_4k_fsdp2_A3.log
