# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

python convert_ckpt.py \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 8 \
    --target-pipeline-parallel-size 8 \
    --load-dir ./model_from_hf/qwen15_hf/ \
    --save-dir ./model_weights/qwen15_mcore/ \
    --tokenizer-model ./model_from_hf/qwen15_hf/tokenizer.model \
    --use-mcore-models \
    --model-type-hf llama2 \
    --add-qkv-bias  # --num-layer-list 7,8,9,10,11,11,12,12  --params-dtype bf16 --num-layers-per-virtual-pipeline-stage 2 等参数根据模型需要添加

