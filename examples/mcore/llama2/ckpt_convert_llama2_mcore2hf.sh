# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 权重格式转换
python convert_ckpt_v2.py \
    --load-model-type mg \
    --save-model-type hf \
    --load-dir ./model_weights/Llama2-mcore/ \
    --save-dir ./model_from_hf/llama-2-7b-hf/ \
    --hf-cfg-dir ./model_from_hf/llama-2-7b-hf/ \
    --model-type-hf llama2
