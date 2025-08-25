export CUDA_DEVICE_MAX_CONNECTIONS=1
## 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置需要的权重转换参数
python convert_ckpt.py \
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.qwen2_moe_spec layer_spec \
    --model-type GPT \
    --model-type-hf qwen2-moe \
    --load-model-type mg \
    --save-model-type hf \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --add-qkv-bias \
    --moe-grouped-gemm \
    --params-dtype bf16 \
    --load-dir ./model_weights/qwen2_moe_mcore/ \
    --save-dir  ./model_from_hf/qwen2_moe_hf/ # 需要填入原始HF模型路径，新权重会存于./model_from_hf/qwen2_moe_hf/mg2hf/