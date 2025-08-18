# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 设置并行策略
python mindspeed_llm/mindspore/convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --model-type-hf llama2 \
    --load-model-type mg \
    --save-model-type hf \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --add-qkv-bias \
    --ai-framework mindspore \
    --load-dir ./model_weights/qwen2.5_mcore/ \
    --save-dir ./model_from_hf/qwen2.5_7b_hf/  # 需要填入原始HF模型路径，新权重会存于./model_from_hf/qwen2.5_7b_hf/mg2hg/
