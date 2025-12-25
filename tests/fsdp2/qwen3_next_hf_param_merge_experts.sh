#!/bin/bash

python examples/fsdp2/qwen3-next/qwen3_next_hf_param_merge_experts.py \
        --load-dir  ./model_weights/Qwen3-Next-A3B \
        --save-dir  ./model_weights/Qwen3-Next-A3B-mergeExperts