# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python mindspeed_llm/tasks/checkpoint/convert_ckpt_mamba2.py \
    --load-model-type mg \
    --save-model-type hf \
    --load-dir ./ckpt/mamba2-tp4pp2 \
    --save-dir ./ckpt/mamba2-tp2pp4 \
    --target-pp-size 4 \
    --target-tp-size 2 \
    --d-model 2560 \
    --mamba-d-state 128 \
    --mamba2-head-dim 64 \
    --mamba2-n-groups 1 \
