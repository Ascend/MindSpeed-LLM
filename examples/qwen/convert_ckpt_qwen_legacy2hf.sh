# 修改modelling_qwen.py文件第39行，将：
# SUPPORT_FP16 = SUPPORT_CUDA and torch.cuda.get_device_capability(0)[0] >= 7
# 修改为：
# SUPPORT_FP16 = True

# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt.py \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --model-type-hf qwen \
    --add-qkv-bias \
    --load-dir ./model_weights/qwen-legacy/ \
    --save-dir ./model_from_hf/qwen-hf/  # 需要填入原始HF模型路径，新权重会存于./model_from_hf/qwen-hf/mg2hg/