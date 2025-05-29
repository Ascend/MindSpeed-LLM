# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --load-dir ./model_from_hf/Qwen3-4B-Base/ \
    --save-dir ./model_weights/Qwen3-4B-mcore/ \
    --tokenizer-model ./model_from_hf/Qwen3-4B-Base/tokenizer.json \
    --model-type-hf qwen3 \
    --params-dtype bf16