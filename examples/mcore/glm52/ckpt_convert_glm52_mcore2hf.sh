# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt_v2.py \
    --load-model-type mg \
    --save-model-type hf \
    --expert-tensor-parallel-size 1 \
    --load-dir ./model_weights/glm52_mcore/ \
    --save-dir ./model_from_hf/glm52_hf/ \
    --moe-grouped-gemm \
    --mla-mm-split \
    --num-layer-list 10,12,8,8,8,12,12,8 \
    --model-type-hf glm5 \

# 脚本配置仅供参考，具体并行度及参数配置需根据实际训练集群硬件和规模调整
