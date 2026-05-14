# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt_v2.py \
  --load-model-type mg \
  --save-model-type hf \
  --model-type-hf deepseek4 \
  --load-dir ./model_weights/deepseek4_flash_mcore/ \
  --save-dir ./model_from_hf/deepseek4_flash_hf/ \
  --noop-layers 43 \
  --mtp-num-layers 1 \
  --moe-grouped-gemm \
  --expert-tensor-parallel-size 1 \

# 当前仅支持开启 gemm 并且 etp=1 的情景
# 如果使用base模型，请将--model-type-hf 设置为 deepseek4_base
