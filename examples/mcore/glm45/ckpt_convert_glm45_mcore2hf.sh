# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 权重格式转换
python convert_ckpt_v2.py \
    --model-type-hf glm45 \
    --load-model-type mg \
    --save-model-type hf \
    --load-dir ./model_weights/glm45_mcore \
    --save-dir ./model_from_hf/glm45_hf \
    --hf-cfg-dir ./model_from_hf/GLM-4.5/ \
    --first-k-dense-replace 3 \
    --moe-grouped-gemm