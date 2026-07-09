# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 设置并行策略
python convert_ckpt_v2.py \
    --load-model-type mg \
    --save-model-type hf \
    --model-type-hf gemma2 \
    --load-dir ./model_weights/gemma2_mcore/ \
    --save-dir ./model_from_hf/gemma2_hf/ \
    --hf-cfg-dir ./model_from_hf/gemma2_hf/
