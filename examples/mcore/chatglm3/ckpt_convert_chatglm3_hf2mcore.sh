# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 权重格式转换
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 2 \
    --load-dir ./model_from_hf/chatglm3-hf/ \
    --save-dir ./model_weights/chatglm3-mcore/ \
    --tokenizer-model ./model_from_hf/chatglm3-hf/tokenizer.model \
    --use-mcore-models \
    --add-qkv-bias \
    --model-type-hf chatglm3
