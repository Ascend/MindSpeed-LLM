# 修改modelling_qwen.py文件第39行，将：
# SUPPORT_FP16 = SUPPORT_CUDA and torch.cuda.get_device_capability(0)[0] >= 7
# 修改为：
# SUPPORT_FP16 = True

# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt.py \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 8 \
    --load-dir ./model_from_hf/qwen-7b/ \
    --save-dir ./model_weights/qwen-7b-legacy/ \
    --tokenizer-model ./model_from_hf/qwen-7b/tokenizer.model \
	--model-type-hf qwen \
	--add-qkv-bias