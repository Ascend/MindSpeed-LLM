# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1


# 权重格式转换
python convert_ckpt.py \
   --use-mcore-models \
   --model-type-hf plm \
   --model-type GPT \
   --load-model-type hf \
   --save-model-type mg \
   --params-dtype bf16 \
   --target-tensor-parallel-size 1 \
   --target-pipeline-parallel-size 1 \
   --tokenizer-model ./model_from_hf/plm \
   --spec mindspeed_llm.tasks.models.spec.plm_spec layer_spec \
   --load-dir ./model_from_hf/plm \
   --save-dir ./model_weights/plm \

