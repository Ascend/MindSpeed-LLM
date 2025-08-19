# 修改 ascend-toolkit 路径
export CUDA_DEVICE_MAX_CONNECTIONS=1
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python ./mindspeed_llm/mindspore/convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 2 \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --load-dir ./model_from_hf/qwen3_hf/ \
    --save-dir ./model_weights/qwen3_mcore/ \
    --tokenizer-model ./model_from_hf/qwen3_hf/tokenizer.json \
    --params-dtype bf16 \
    --model-type-hf qwen3 \
    --ai-framework mindspore