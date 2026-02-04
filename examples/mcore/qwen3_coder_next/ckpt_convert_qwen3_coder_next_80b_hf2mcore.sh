# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt_v2.py \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --target-expert-parallel-size 64 \
    --load-dir ./model_from_hf/qwen3_coder_next_hf/ \
    --save-dir ./model_weights/qwen3_coder_next_mcore/ \
    --moe-grouped-gemm \
    --model-type-hf qwen3-next

# 脚本配置仅供参考，具体配置请根据训练策略修改，该模型暂不支持开启tp
# mtp层请根据需要配置--mtp-num-layers 1 \