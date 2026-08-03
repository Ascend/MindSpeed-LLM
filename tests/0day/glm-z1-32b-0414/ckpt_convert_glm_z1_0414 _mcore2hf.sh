# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 权重格式转换
python convert_ckpt_v2.py \
   --model-type-hf glm4 \
   --load-model-type mg \
   --save-model-type hf \
   --num-layer-list 15,15,15,16 \
   --load-dir ./model_weights/glm-z1-mcore \
   --save-dir ./model_from_hf/GLM-Z1-32B-0414/
