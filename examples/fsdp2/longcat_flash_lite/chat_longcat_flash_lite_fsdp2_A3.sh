source examples/fsdp2/env_config.sh

NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=42323
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
bash tests/tools/fsdp2/longcat_flash_lite_moe_hf_weight_convert.sh
torchrun $DISTRIBUTED_ARGS inference_fsdp2.py examples/fsdp2/longcat_flash_lite/pretrain_longcat_flash_lite_4k_fsdp2_A3.yaml \
    --parallel.fsdp_size 16 \
    --parallel.ep_size 16 \
    --inference.infer_backend huggingface \
    --inference.max_new_tokens: 512 \
    | tee logs/chat_longcat_flash_lite_fsdp2_A3.log
