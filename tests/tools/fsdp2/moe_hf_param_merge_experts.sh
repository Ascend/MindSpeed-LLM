#!/bin/bash

LOAD_DIR=./hf_weights/Qwen3-MoE
SAVE_DIR=./model_weights/Qwen3-MoE-mergeExperts
CONVERTER=./tests/tools/fsdp2/moe_hf_param_merge_experts.py

# Skip conversion when merged weights already exist.
if [[ -f "${SAVE_DIR}/model.safetensors.index.json" ]] && ls "${SAVE_DIR}"/model-*.safetensors >/dev/null 2>&1; then
    echo "[skip] merged weights already exist at ${SAVE_DIR}, skip conversion."
    exit 0
fi

python "${CONVERTER}" \
    --load-dir "${LOAD_DIR}" \
    --save-dir "${SAVE_DIR}"