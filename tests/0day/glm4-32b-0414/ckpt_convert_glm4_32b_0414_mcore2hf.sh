# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置并行策略
python convert_ckpt_v2.py \
    --model-type-hf glm4 \
    --load-model-type mg \
    --save-model-type hf \
    --load-dir ./model_weights/GLM-4-32B-0414 \
    --save-dir ./model_from_hf/GLM-4-32B-0414
