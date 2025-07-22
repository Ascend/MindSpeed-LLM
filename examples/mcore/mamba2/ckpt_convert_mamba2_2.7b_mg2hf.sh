# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

python mindspeed_llm/tasks/checkpoint/convert_ckpt_mamba2.py \
    --load-model-type mg \
    --save-model-type hf \
    --load-dir ./ckpt/mamba2-mcore \
    --save-dir ./ckpt/mamba2-hf \
    --d-model 2560 \
    --mamba-d-state 128 \
    --mamba2-head-dim 64 \
    --mamba2-n-groups 1 \

    # 注意，如果load权重不是训练后保存的权重，则需要增加如下配置参数
    # 数值仅供参考，具体请按需修改
    # --input-tp-rank 1 \
    # --input-pp-rank 2 \
    # --num-layers 64 \
