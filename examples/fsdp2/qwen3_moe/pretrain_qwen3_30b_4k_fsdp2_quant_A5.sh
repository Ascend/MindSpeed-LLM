#!/bin/bash
source examples/fsdp2/env_config.sh

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6499
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
QUANT_ARGS="
    --model.quant_recipe_name mxfp8 \
    --model.enable_fsdp_low_precision_all_gather \
    --model.quant_converters quantize.linear.mx quantize.moe.mx \
    --parallel.efsdp_shard_placement_fn shard_by_dim_0 \
    --parallel.ep_dispatcher eager \
"
mkdir -p ./logs
bash tests/tools/fsdp2/moe_hf_param_merge_experts.sh
torchrun $DISTRIBUTED_ARGS train_fsdp2.py \
     examples/fsdp2/qwen3_moe/pretrain_qwen3_30b_4k_fsdp2_A5.yaml \
     $QUANT_ARGS\
     | tee logs/pretrain_qwen3_moe_30b_a3b_4K_fsdp2_${TIMESTAMP}.log