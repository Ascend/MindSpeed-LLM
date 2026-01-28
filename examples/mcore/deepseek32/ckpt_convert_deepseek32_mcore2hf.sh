# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt_v2.py \
    --load-model-type mg \
    --save-model-type hf \
    --load-dir ./model_weights/deepseek32_mcore/ \
    --save-dir ./model_from_hf/deepseek32_hf/ \
    --moe-grouped-gemm \
    --mla-mm-split \
    --model-type-hf deepseek32 \
    --noop-layers 61,62,63 \
    --expert-tensor-parallel-size 1 \
    --mtp-num-layers 1 \
    --hf-cfg-dir ./origin_hf_cfg

# 脚本配置仅供参考，具体并行度及参数配置需根据实际训练集群硬件和规模调整
# mtp层请根据需要配置--mtp-num-layers 1 \