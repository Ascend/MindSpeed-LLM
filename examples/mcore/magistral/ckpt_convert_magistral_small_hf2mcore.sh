# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 权重格式转换
python convert_ckpt_v2.py \
   --model-type-hf magistral \
   --load-model-type hf \
   --save-model-type mg \
   --params-dtype bf16 \
   --target-tensor-parallel-size 8 \
   --target-pipeline-parallel-size 1 \
   --load-dir ./model_from_hf/magistral_small/ \
   --save-dir ./model_weights/magistrall_small_mcore/
   # --num-layer-list 17,20,22,21 等参数根据模型需求添加

