# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt_v2.py \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 4 \
    --target-pipeline-parallel-size 4 \
    --target-expert-parallel-size 32 \
    --noop-layers 0,63 \
    --num-layers-per-virtual-pipeline-stage 2 \
    --load-dir ./model_from_hf/qwen3_moe_hf/ \
    --save-dir ./model_weights/qwen3_moe_mcore/ \
    --expert-tensor-parallel-size 1 \
    --moe-grouped-gemm \
    --model-type-hf qwen3-moe

# 脚本配置仅供参考，具体配置请根据训练策略修改