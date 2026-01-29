# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 权重格式转换
python mindspeed_llm/tasks/checkpoint/convert_param.py \
          --cvt-type hf2mg \
          --model-name llama \
          --hf-dir ./model_from_hf/llama31-hf \
          --mg-dir /tmp/llama31-mcore-tp2d \
          --model-index-file ./model_from_hf/llama31-hf/model.safetensors.index.json \
          --tensor-model-parallel-size 8 \
          --pipeline-model-parallel-size 1 \
          --make-vocab-size-divisible-by 1 \
          --num-layers 32 \
          --tp-2d \
          --tp-x 4 \
          --tp-y 2