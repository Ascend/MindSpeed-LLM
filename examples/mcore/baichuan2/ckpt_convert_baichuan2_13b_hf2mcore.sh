# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 权重格式转换
python convert_ckpt.py \
    --use-mcore-models \
    --model-type-hf baichuan2 \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --params-dtype bf16 \
    --target-tensor-parallel-size 8 \
    --target-pipeline-parallel-size 1 \
    --load-dir ./model_from_hf/Baichuan2-13B_hf/ \
    --save-dir ./model_weights/Baichuan2-13B_mcore/ \
    --tokenizer-model ./model_from_hf/Baichuan2-13B_hf/tokenizer.model
