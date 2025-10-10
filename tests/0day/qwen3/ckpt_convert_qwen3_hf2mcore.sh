# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 4 \
    --target-pipeline-parallel-size 2 \
    --load-dir ./model_from_hf/Qwen3-1.7B-Base/ \
    --save-dir ./model_weight/Qwen3-1.7B-mcore \
    --tokenizer-model ./model_from_hf/Qwen3-1.7B-Base/tokenizer.json \
    --model-type-hf qwen3 \
    --params-dtype bf16 \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec