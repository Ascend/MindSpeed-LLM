# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt_v2.py \
    --load-model-type mg \
    --save-model-type hf \
    --noop-layers 94,95 \
    --load-dir ./model_weights/qwen3_moe_mcore/ \
    --save-dir ./model_from_hf/qwen3_moe_hf/ \
    --moe-grouped-gemm \
    --model-type-hf qwen3-moe

# 脚本配置仅供参考，具体配置请根据训练策略修改