# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 4 \
    --target-expert-parallel-size 8 \
    --load-dir ./model_from_hf/hunyuanLarge_hf/ \
    --save-dir ./model_weights/hunyuanLarge_mcore/ \
    --tokenizer-model ./Tencent-Hunyuan-Large/Hunyuan-A52B-Instruct \
    --model-type-hf hunyuan \
    --moe-grouped-gemm \
    --params-dtype bf16 \
    --spec mindspeed_llm.tasks.models.spec.hunyuan_spec layer_spec