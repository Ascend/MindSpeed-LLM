#=============================================
# Date: 2026-08-18
# Description: Guards DeepSeek-V4-Flash FSDP2 pretrain base scenario (cp_size=1, default
#              attn implementation): HF weight loading, fsdp_modules sharding incl. MTP,
#              MoE ep config and recompute. Moved from CI gate (tests/st) to pipeline since
#              these features are already guarded at the gate by pretrain_deepseek4_flash_1k_fsdp2_cp2.sh.
# Remarks: Requires transformers==5.8.1, installed and restored automatically by this script.
#=============================================
set -e
set -o pipefail

source examples/fsdp2/env_config.sh

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6499
NNODES=1
NODE_RANK=0

echo "Installing transformers==5.8.1 ..."
pip install --no-cache-dir transformers==5.8.1
if [ $? -ne 0 ]; then
    echo "Failed to install transformers==5.8.1, exiting"
    exit 1
fi

TRAIN_EXIT_CODE=0

trap '
    echo "Restoring transformers to 5.2.0 ..."
    pip install --no-cache-dir transformers==5.2.0 || exit 1
    exit $TRAIN_EXIT_CODE
' EXIT

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
set +e
torchrun $DISTRIBUTED_ARGS train_fsdp2.py \
    ./tests/pipeline/st/deepseek4-flash/pretrain_deepseek4_flash_1k_fsdp2.yaml

TRAIN_EXIT_CODE=$?
set -e
