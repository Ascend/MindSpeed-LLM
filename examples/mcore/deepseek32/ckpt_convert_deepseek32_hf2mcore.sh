# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt_v2.py \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 8 \
    --target-expert-parallel-size 64 \
    --load-dir ./model_from_hf/deepseek32_hf/ \
    --save-dir ./model_weights/deepseek32_mcore/ \
    --moe-grouped-gemm \
    --mla-mm-split \
    --noop-layers 61,62,63 \
    --expert-tensor-parallel-size 1 \
    --mtp-num-layers 1 \
    --model-type-hf deepseek32

# 脚本配置仅供参考，具体并行度及参数配置需根据实际训练集群硬件和规模调整
# mtp层请根据需要配置--mtp-num-layers 1 \