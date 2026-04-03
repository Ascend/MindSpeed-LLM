# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 权重格式转换
python convert_ckpt_v2.py \
   --load-dir ./model_from_hf/GLM-4.5/ \
   --save-dir ./model_weights/hf2mg-GLM-4.5 \
   --model-type-hf glm45 \
   --load-model-type hf \
   --save-model-type mg \
   --target-tensor-parallel-size 4 \
   --target-pipeline-parallel-size 4 \
   --target-expert-parallel-size 1 \
   --first-k-dense-replace 3 \
   --moe-grouped-gemm \
