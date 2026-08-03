# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 权重格式转换
python convert_ckpt_v2.py \
   --model-type-hf glm4_9b \
   --load-model-type hf \
   --save-model-type mg \
   --target-tensor-parallel-size 2 \
   --target-pipeline-parallel-size 4 \
   --load-dir ./model_from_hf/GLM-4-Z1-9B-0414 \
   --save-dir ./model_weights/GLM-4-Z1-9B-0414 \
