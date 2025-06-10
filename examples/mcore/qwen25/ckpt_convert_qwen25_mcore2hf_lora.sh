# 修改 ascend-toolkit 路径
export CUDA_DEVICE_MAX_CONNECTIONS=1
source /usr/local/Ascend/ascend-toolkit/set_env.sh

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
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --load-dir ./model_weights/qwen25_mcore/ \
    --lora-load ./ckpt/qwen25_7b_lora \
    --save-dir ./model_from_hf/qwen25_7b_hf/  # 需要填入原始HF模型路径，新权重会存于./model_from_hf/qwen25_7b_hf/mg2hg/