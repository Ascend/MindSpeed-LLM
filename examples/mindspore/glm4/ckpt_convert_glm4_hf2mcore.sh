# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 权重格式转换
python mindspeed_llm/mindspore/convert_ckpt.py \
   --model-type-hf chatglm3 \
   --model-type GPT \
   --load-model-type hf \
   --save-model-type mg \
   --target-tensor-parallel-size 2 \
   --target-pipeline-parallel-size 2 \
   --load-dir ./model_from_hf/glm4_hf \
   --save-dir ./model_weights/glm4_mcore \
   --tokenizer-model ./model_from_hf/glm4_hf/tokenizer.json \
   --add-qkv-bias \
   --use-mcore-models \
   --params-dtype bf16 \
   --ai-framework mindspore \