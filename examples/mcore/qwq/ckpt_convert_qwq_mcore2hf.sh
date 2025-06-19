# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 设置并行策略
python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --model-type-hf llama2 \
    --load-model-type mg \
    --save-model-type hf \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --add-qkv-bias \
    --load-dir ./model_weights/qwq_mcore/ \
    --save-dir ./model_from_hf/qwq_32b_hf/  # 需要填入原始HF模型路径，新权重会存于./model_from_hf/qwq_32b_hf/mg2hg/