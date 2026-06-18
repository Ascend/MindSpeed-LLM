#!/bin/bash

# =============================================================================
# Merge per-expert MoE weights into fused gate_up_proj / down_proj (bf16).
#
# Usage:
#   bash moe_hf_param_merge_experts.sh
#
# Edit the three variables below before running:
#   MODEL_ID  - model family, one of: qwen3_moe | minimax_m27
#               (selects naming / module path / fused tensor rank)
#   LOAD_DIR  - input HF checkpoint, MUST already be bf16. For FP8 models
#               (e.g. minimax_m27) run the dequant step first, then point here.
#   SAVE_DIR  - output path for the merged checkpoint.
#
# =============================================================================

MODEL_ID=qwen3_moe
LOAD_DIR=./hf_weights/Qwen3-30B
SAVE_DIR=./model_weights/Qwen3-30B-mergeExperts
CONVERTER=./tests/tools/fsdp2/moe_hf_param_merge_experts.py

# Skip conversion when merged weights already exist.
if [[ -f "${SAVE_DIR}/model.safetensors.index.json" ]] && \
   ls "${SAVE_DIR}"/model-*.safetensors >/dev/null 2>&1; then
    echo "[skip] merged weights already exist at ${SAVE_DIR}, skip conversion."
    exit 0
fi

python "${CONVERTER}" \
    --model-id "${MODEL_ID}" \
    --load-dir "${LOAD_DIR}" \
    --save-dir "${SAVE_DIR}"
