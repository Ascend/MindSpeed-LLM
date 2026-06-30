source examples/fsdp2/env_config.sh

NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6060
NNODES=2
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
torchrun $DISTRIBUTED_ARGS inference_fsdp2.py ./examples/fsdp2/glm52/glm52_744b_4k_fsdp2_A3.yaml \
    --model.model_name_or_path "your hf model path" \
    --parallel.fsdp_size 16 \
    --parallel.ep_size 8 \
    --parallel.ep_fsdp_size 2 \
    --inference.infer_backend huggingface \
    --inference.max_new_tokens: 512 \
