# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 权重格式转换
python convert_ckpt.py \
   --use-mcore-models \
   --model-type-hf gemma2 \
   --model-type GPT \
   --load-model-type hf \
   --save-model-type mg \
   --params-dtype bf16 \
   --post-norm \
   --spec mindspeed_llm.tasks.models.spec.gemma2_spec layer_spec \
   --target-tensor-parallel-size 8 \
   --target-pipeline-parallel-size 1 \
   --load-dir ./model_from_hf/gemma2_hf/ \
   --save-dir ./model_weights/gemma2_mcore/ \
   --tokenizer-model ./model_from_hf/gemma2_hf/tokenizer.json