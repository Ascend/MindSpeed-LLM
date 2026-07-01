# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 权重格式转换
python convert_ckpt.py \
   --model-type-hf glm4 \
   --post-norm \
   --spec mindspeed_llm.tasks.models.spec.gemma2_spec layer_spec \
   --model-type GPT \
   --load-model-type mg \
   --save-model-type hf \
   --target-tensor-parallel-size 1 \
   --target-pipeline-parallel-size 1 \
   --add-qkv-bias \
   --use-mcore-models \
   --params-dtype bf16 \
   --load-dir ./model_weights/glm4_9b_0414_mcore/ \
    --save-dir ./model_from_hf/glm4_9b_0414_hf/  # 需要填入原始HF模型路径，新权重会存于./model_from_hf/glm4_hf/mg2hf/
`
`
