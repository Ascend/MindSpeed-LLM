#!/bin/bash

LOAD_DIR=./hf_weights/LongCat-Flash-Lite
SAVE_DIR=./fsdp_weights/LongCat-Flash-Lite-mergeExperts
CONVERTER=./tests/tools/fsdp2/longcat_flash_lite_moe_hf_weight_convert.py

# Skip conversion when merged weights already exist.
if [[ -f "${SAVE_DIR}/model.safetensors.index.json" ]] && ls "${SAVE_DIR}"/model-*.safetensors >/dev/null 2>&1; then
    echo "[skip] merged weights already exist at ${SAVE_DIR}, skip conversion."
    exit 0
fi

python "${CONVERTER}" \
    --load-dir "${LOAD_DIR}" \
    --save-dir "${SAVE_DIR}"
