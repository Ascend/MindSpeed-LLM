# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 权重格式转换
python convert_ckpt_v2.py \
   --use-mcore-models \
   --model-type-hf phi3.5 \
   --load-model-type hf \
   --save-model-type mg \
   --target-tensor-parallel-size 1 \
   --target-pipeline-parallel-size 8 \
   --load-dir ./model_from_hf/Phi-3.5-mini-instruct/ \
   --save-dir ./model_weights/phi35_mini_mcore/ \
   --tokenizer-model ./model_from_hf/Phi-3.5-mini-instruct/tokenizer.model
