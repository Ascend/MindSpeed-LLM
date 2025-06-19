# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 设置并行策略
python convert_ckpt.py \
    --use-mcore-models \
    --model-type-hf gemma \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 2 \
    --params-dtype bf16 \
    --load-dir ./model_from_hf/gemma_hf/ \
    --save-dir ./model_weights/gemma_mcore/ \
    --tokenizer-model ./model_from_hf/gemma_hf/tokenizer.model