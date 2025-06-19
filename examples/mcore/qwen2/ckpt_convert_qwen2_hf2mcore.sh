# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 设置需要的权重转换参数
python convert_ckpt.py \
       --use-mcore-models \
       --model-type GPT \
       --load-model-type hf \
       --save-model-type mg \
       --target-tensor-parallel-size 1 \
       --target-pipeline-parallel-size 1 \
       --add-qkv-bias \
       --load-dir ./model_from_hf/qwen2_hf/ \
       --save-dir ./model_weights/qwen2_mcore/ \
       --tokenizer-model ./model_from_hf/qwen2_hf/tokenizer.json \
       --model-type-hf llama2 \
       --params-dtype bf16 #  --num-layers-per-virtual-pipeline-stage 2 等参数根据模型需要添加
