# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 设置需要的权重转换参数
python convert_ckpt_v2.py \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 4 \
    --target-pipeline-parallel-size 2 \
    --load-dir ./model_from_hf/qwen2.5_7b_hf/ \
    --save-dir ./model_weights/qwen2.5_mcore/ \
    --model-type-hf qwen25  # --num-layer-list 11,13,19,21 参数根据需要添加
