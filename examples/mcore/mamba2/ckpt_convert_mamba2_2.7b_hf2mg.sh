# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

python mindspeed_llm/tasks/checkpoint/convert_ckpt_mamba2.py \
    --load-model-type hf \
    --save-model-type mg \
    --load-dir ./ckpt/mamba2-hf \
    --save-dir ./ckpt/mamba2-tp1pp4 \
    --target-pp-size 4 \
    --target-tp-size 1 \
    --d-model 2560 \
    --mamba-d-state 128 \
    --mamba2-head-dim 64 \
    --mamba2-n-groups 1 \
