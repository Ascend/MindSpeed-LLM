# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 权重格式转换
python convert_ckpt_v2.py \
   --model-type-hf glm4_9b \
   --load-model-type mg \
   --save-model-type hf \
   --load-dir ./model_weights/glm4_9b_0414_mcore/ \
    --save-dir ./model_from_hf/glm4_9b_0414_hf/
`
`
