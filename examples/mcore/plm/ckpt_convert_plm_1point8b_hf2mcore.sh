# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1


# 权重格式转换
python convert_ckpt_v2.py \
    --load-model-type hf \
    --save-model-type mg \
    --model-type-hf plm \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --load-dir ./model_from_hf/plm \
    --save-dir ./model_weights/plm \
