# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置并行策略
python convert_ckpt.py \
    --use-mcore-models \
    --model-type-hf glm4 \
    --post-norm \
    --spec mindspeed_llm.tasks.models.spec.gemma2_spec layer_spec \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --params-dtype bf16 \
    --add-qkv-bias \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --load-dir ./model_weights/GLM-4-Z1-9B-0414 \
    --save-dir ./model_from_hf/GLM-4-Z1-9B-0414  # 需要填入原始HF模型路径，新权重会存于./model_from_hf/glm4_hf/mg2hf/
