# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 权重格式转换
python convert_ckpt_v2.py \
    --model-type-hf glm45-moe \
    --load-model-type mg \
    --save-model-type hf \
    --load-dir ./model_weights/glm45_air_mcore \
    --save-dir ./model_from_hf/glm45_air_hf \
    --first-k-dense-replace 1 \
    --mtp-num-layers 1
