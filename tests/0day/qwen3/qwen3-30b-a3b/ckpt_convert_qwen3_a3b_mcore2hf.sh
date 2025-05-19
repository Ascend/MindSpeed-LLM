# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --target-expert-parallel-size 1 \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --load-dir ./model_weights/qwen3_a3b_mcore/ \
    --save-dir ./model_from_hf/qwen3_a3b_hf/ \
    --model-type-hf qwen3-moe \
    --moe-grouped-gemm \
    --params-dtype bf16
