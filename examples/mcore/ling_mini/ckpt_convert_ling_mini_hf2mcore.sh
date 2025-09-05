# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

python convert_ckpt_v2.py \
    --use-mcore-models \
    --moe-grouped-gemm \
    --model-type-hf bailing_mini \
    --load-model-type hf \
    --save-model-type mg \
    --params-dtype bf16 \
    --mtp-num-layers 1 \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --target-expert-parallel-size 8 \
    --load-dir ./model_from_hf/ling_mini_v2-hf/ \
    --save-dir ./model_weights/bailing_mini-mcore/ \
    --tokenizer-model ./model_from_hf/ling_mini_v2-hf/ \
    --spec mindspeed_llm.tasks.models.spec.bailing_spec layer_spec \
