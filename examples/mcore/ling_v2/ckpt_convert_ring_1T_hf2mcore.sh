# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

python convert_ckpt_v2.py \
    --moe-grouped-gemm \
    --model-type-hf bailing_mini \
    --load-model-type hf \
    --save-model-type mg \
    --params-dtype bf16 \
    --target-tensor-parallel-size 8 \
    --target-pipeline-parallel-size 8 \
    --target-expert-parallel-size 8 \
    --moe-tp-extend-ep \
    --load-dir ./model_from_hf/ring_1T-hf/ \
    --save-dir ./model_weights/ring_1T-mcore/ \
