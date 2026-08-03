# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 权重格式转换
python convert_ckpt_v2.py \
   --model-type-hf glm4 \
   --load-model-type hf \
   --save-model-type mg \
   --target-tensor-parallel-size 2 \
   --target-pipeline-parallel-size 4 \
   --num-layer-list 15,15,15,16 \
   --load-dir ./model_from_hf/GLM-4-32B-Base-0414/ \
   --save-dir ./model_weights/GLM-4-32B-Base-0414 \
