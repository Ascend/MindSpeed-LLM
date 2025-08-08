# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 权重格式转换
python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --model-type-hf llama2 \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 8 \
    --target-pipeline-parallel-size 2 \
    --load-dir ./model_from_hf/Codellama-hf/ \
    --save-dir ./model_weights/Codellama-mcore/ \
    --tokenizer-model ./model_from_hf/Codellama-hf/tokenizer.model \
    --params-dtype bf16
