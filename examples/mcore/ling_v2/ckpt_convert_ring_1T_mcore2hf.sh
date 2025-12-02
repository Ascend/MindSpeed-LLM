# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

python convert_ckpt_v2.py \
    --moe-grouped-gemm \
    --model-type-hf bailing_mini \
    --load-model-type mg \
    --save-model-type hf \
    --params-dtype bf16 \
    --moe-tp-extend-ep \
    --load-dir ./model_weights/ring_1T-mcore/ \
    --save-dir ./model_from_hf/ring_1T_v2-hf/mg2hf/ \
    --spec mindspeed_llm.tasks.models.spec.bailing_spec layer_spec \