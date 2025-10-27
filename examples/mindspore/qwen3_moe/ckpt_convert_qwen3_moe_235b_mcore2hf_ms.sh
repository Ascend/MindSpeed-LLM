# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python ./mindspeed_llm/mindspore/convert_ckpt_v2.py \
    --load-model-type mg \
    --save-model-type hf \
    --load-dir ./model_weights/qwen3_moe_ms/ \
    --save-dir ./model_from_hf/qwen3_moe_hf/ \
    --moe-grouped-gemm \
    --model-type-hf qwen3-moe \
    --ai-framework mindspore

# 脚本配置仅供参考，具体配置请根据训练策略修改