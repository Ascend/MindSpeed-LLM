# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置需要的权重转换参数
python convert_ckpt_v2.py \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 4 \
    --target-pipeline-parallel-size 2 \
    --target-expert-parallel-size 1 \
    --load-dir ./model_from_hf/qwen3_8b_hf/ \
    --save-dir ./model_weights/qwen3_8b_mcore/ \
    --model-type-hf qwen3 \
