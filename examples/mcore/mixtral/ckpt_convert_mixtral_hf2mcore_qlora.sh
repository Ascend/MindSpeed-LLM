export CUDA_DEVICE_MAX_CONNECTIONS=1

# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置需要的并行配置
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --params-dtype bf16 \
    --qlora-nf4 \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 1 \
    --target-expert-parallel-size 1 \
    --load-dir ./model_from_hf/Mixtral-hf/ \
    --save-dir ./model_weights/Mixtral-mcore/ \
    --tokenizer-model ./model_from_hf/Mixtral-hf/tokenizer.model \
    --use-mcore-models \
    --model-type-hf mixtral
