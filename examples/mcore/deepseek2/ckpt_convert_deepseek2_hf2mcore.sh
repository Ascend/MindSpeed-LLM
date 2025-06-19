# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 权重格式转换，设置需要的并行策略
python convert_ckpt.py \
   --use-mcore-models \
   --moe-grouped-gemm \
   --model-type-hf deepseek2 \
   --model-type GPT \
   --load-model-type hf \
   --save-model-type mg \
   --params-dtype bf16 \
   --target-tensor-parallel-size 1 \
   --target-pipeline-parallel-size 1 \
   --target-expert-parallel-size 8 \
   --load-dir ./model_from_hf/deepseek2-hf/ \
   --save-dir ./model_weights/deepseek2-mcore/ \
   --tokenizer-model ./model_from_hf/deepseek2-hf/ \
   --spec mindspeed_llm.tasks.models.spec.deepseek_spec layer_spec
