# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt_v2.py \
    --moe-grouped-gemm \
    --model-type-hf deepseek2-lite \
    --load-model-type mg \
    --save-model-type hf \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --target-expert-parallel-size 1 \
    --load-dir ./model_weights/deepseek2_lite_mcore/ \
    --save-dir ./model/deepseek2_lite/ \
    --hf-dir ./origin_hf_cfg \