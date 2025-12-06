# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt_v2.py \
    --load-model-type mg \
    --save-model-type hf \
    --load-dir ./model_weights/bailing_mini-mcore/ \
    --save-dir ./model_from_hf/ling_mini_v2-hf/mg2hf/ \
    --hf-dir ./origin_hf_cfg \
    --moe-grouped-gemm \
    --model-type-hf bailing_mini \
    --mtp-num-layers 1 \

# 脚本配置仅供参考，具体配置请根据训练策略修改，该模型暂不支持开启tp
# mtp层请根据需要配置--mtp-num-layers 1 \