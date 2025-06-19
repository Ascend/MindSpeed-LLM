# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 权重格式转换，设置需要的并行配置
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --use-mcore-models \
    --load-dir ./model_from_hf/MiniCPM-MoE-8x2B \
    --save-dir ./model_from_hf/MiniCPM-MoE-8x2B-mcore \
    --tokenizer-model ./model_from_hf/MiniCPM-MoE-8x2B/tokenizer.model \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 4 \
    --target-expert-parallel-size 2 \
    --model-type-hf minicpm-moe \
    --params-dtype bf16