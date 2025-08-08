# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 权重格式转换
python convert_ckpt_v2.py \
   --model-type-hf glm45-moe \
   --post-norm \
   --load-model-type hf \
   --save-model-type mg \
   --target-tensor-parallel-size 8 \
   --target-pipeline-parallel-size 1 \
   --target-expert-parallel-size 1 \
   --first-k-dense-replace 1 \
   --load-dir ./model_from_hf/GLM-4.5-moe-106B \
   --save-dir ./model_weights/ \

