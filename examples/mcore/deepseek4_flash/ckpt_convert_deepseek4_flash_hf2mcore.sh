# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt_v2.py \
  --load-model-type hf \
  --save-model-type mg \
  --model-type-hf deepseek4 \
  --load-dir ./model_from_hf/deepseek4_flash_hf/ \
  --save-dir ./model_weights/deepseek4_flash_mcore/ \
  --target-tensor-parallel-size 2 \
  --target-pipeline-parallel-size 4 \
  --target-expert-parallel-size 32 \
  --ckpt-expert-tensor-parallel-size 1 \
  --noop-layers 43 \
  --mtp-num-layers 1 \
  --moe-grouped-gemm