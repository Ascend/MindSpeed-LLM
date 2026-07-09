# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 权重格式转换
python convert_ckpt_v2.py \
    --load-model-type hf \
    --save-model-type mg \
    --model-type-hf gemma2 \
    --target-tensor-parallel-size 8 \
    --target-pipeline-parallel-size 1 \
    --load-dir ./model_from_hf/gemma2_hf/ \
    --save-dir ./model_weights/gemma2_mcore/
