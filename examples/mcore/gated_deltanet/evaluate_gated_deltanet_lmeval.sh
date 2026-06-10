#!/bin/bash
# 在训练保存的 HuggingFace 权重上，用 lm-evaluation-harness 跑下游任务评估。

# 离线环境（数据集/权重已在本地时）
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# please fill these path configurations
MODEL_PATH="Your HF weights path"
OUT_DIR="Your eval output path"

NUM_PROC=8
BATCH_SIZE=16
# 与本目录精度表一致的任务集（mmlu/gsm8k 在 340M 规模接近随机，按需添加）
TASKS="arc_easy,arc_challenge,hellaswag,lambada_openai,openbookqa,piqa"

accelerate launch --multi_gpu --num_processes ${NUM_PROC} \
    -m lm_eval \
    --model hf \
    --model_args pretrained=${MODEL_PATH},dtype=bfloat16,trust_remote_code=True \
    --tasks ${TASKS} \
    --batch_size ${BATCH_SIZE} \
    --output_path ${OUT_DIR} \
    --log_samples
