# 修改 ascend-toolkit 路径
export CUDA_DEVICE_MAX_CONNECTIONS=1
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置并行策略
python convert_ckpt_v2.py \
    --load-model-type mg \
    --save-model-type hf \
    --load-dir ./model_weights/qwen2.5_mcore/ \
    --save-dir ./model_from_hf/qwen2.5_7b_hf/ \
    --hf-cfg-dir ./model_from_hf/qwen2.5_7b_hf/ \
    --model-type-hf qwen25
