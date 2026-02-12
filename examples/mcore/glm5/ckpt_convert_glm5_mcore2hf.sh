# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt_v2.py \
    --load-model-type mg \
    --save-model-type hf \
    --expert-tensor-parallel-size 1 \
    --load-dir ./model_weights/glm5_mcore/ \
    --save-dir ./model_from_hf/glm5_hf/ \
    --moe-grouped-gemm \
    --mla-mm-split \
    --noop-layers 78,79 \
    --model-type-hf glm5 \
    --mtp-num-layers 1 \

# 脚本配置仅供参考，具体并行度及参数配置需根据实际训练集群硬件和规模调整
