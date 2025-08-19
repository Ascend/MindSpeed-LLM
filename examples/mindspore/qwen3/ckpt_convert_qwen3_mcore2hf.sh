# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

python ./mindspeed_llm/mindspore/convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --load-dir ./model_weights/qwen3_mcore/ \
    --save-dir ./model_from_hf/qwen3_hf/ \
    --params-dtype bf16 \
    --model-type-hf qwen3 \
    --ai-framework mindspore