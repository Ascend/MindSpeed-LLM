# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 权重格式转换
python convert_ckpt.py \
   --model-type-hf chatglm4 \
   --post-norm \
   --spec mindspeed_llm.tasks.models.spec.gemma2_spec layer_spec \
   --model-type GPT \
   --load-model-type hf \
   --save-model-type mg \
   --target-tensor-parallel-size 2 \
   --target-pipeline-parallel-size 4 \
   --load-dir./model_from_hf/GLM-4-9B-0414 \
   --save-dir ./model_weights/ \
   --tokenizer-model ./model_from_hf/GLM-4-9B-0414/tokenizer.json \
   --add-qkv-bias \
   --use-mcore-models \
   --params-dtype bf16
