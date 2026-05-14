# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

python convert_ckpt_v2.py \
    --moe-grouped-gemm \
    --model-type-hf bailing_mini \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --target-expert-parallel-size 8 \
    --load-dir ./model_from_hf/ling_mini_v2-hf/ \
    --save-dir ./model_weights/bailing_mini-mcore/ \
