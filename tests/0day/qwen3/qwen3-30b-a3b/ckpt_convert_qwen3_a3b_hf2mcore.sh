# 修改 ascend-toolkit 路径
export CUDA_DEVICE_MAX_CONNECTIONS=1
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 8 \
    --target-expert-parallel-size 1 \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --load-dir ./model_from_hf/qwen3_a3b_hf/ \
    --save-dir ./model_weights/qwen3_a3b_mcore/ \
    --tokenizer-model ./model_from_hf/qwen3_a3b_hf/tokenizer.json \
    --model-type-hf qwen3-moe \
    --moe-grouped-gemm \
    --params-dtype bf16