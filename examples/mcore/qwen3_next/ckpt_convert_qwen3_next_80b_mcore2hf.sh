# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt_v2.py \
    --load-model-type mg \
    --save-model-type hf \
    --load-dir ./model_weights/qwen3_next_mcore/ \
    --save-dir ./model_from_hf/qwen3_next_hf/ \
    --hf-dir ./origin_hf_cfg \
    --moe-grouped-gemm \
    --model-type-hf qwen3-next

# 脚本配置仅供参考，具体配置请根据训练策略修改，该模型暂不支持开启tp
# mtp层请根据需要配置--mtp-num-layers 1 \