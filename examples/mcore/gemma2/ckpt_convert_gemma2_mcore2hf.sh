# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 设置并行策略
python convert_ckpt.py \
    --use-mcore-models \
    --model-type-hf gemma2 \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --post-norm \
    --spec mindspeed_llm.tasks.models.spec.gemma2_spec layer_spec \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --load-dir ./model_weights/gemma2_mcore/ \
    --save-dir ./model_from_hf/gemma2_hf/  # 需要填入原始HF模型路径，新权重会存于./model_from_hf/gemma2_hf/mg2hg/