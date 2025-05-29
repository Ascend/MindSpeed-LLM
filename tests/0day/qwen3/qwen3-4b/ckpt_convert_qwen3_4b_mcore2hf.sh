# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置并行策略
python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --model-type-hf qwen3 \
    --load-model-type mg \
    --save-model-type hf \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --load-dir ./model_weights/Qwen3-4B-mcore/ \
    --save-dir ./model_from_hf/Qwen3-4B-Base/  # 需要填入原始HF模型路径，新权重会存于./model_from_hf/Qwen3-4B-Base/mg2hg/
