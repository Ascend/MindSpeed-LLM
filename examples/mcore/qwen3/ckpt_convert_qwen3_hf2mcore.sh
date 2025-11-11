# 修改 ascend-toolkit 路径
export CUDA_DEVICE_MAX_CONNECTIONS=1
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt_v2.py \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 8 \
    --load-dir ./model_from_hf/qwen3_hf/ \
    --save-dir ./model_weights/qwen3_mcore/ \
    --model-type-hf qwen3