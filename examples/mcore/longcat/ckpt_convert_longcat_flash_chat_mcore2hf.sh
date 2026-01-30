# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt_v2.py \
    --model-type-hf longcat \
    --load-model-type mg \
    --save-model-type hf \
    --target-tensor-parallel-size 16 \
    --target-pipeline-parallel-size 4 \
    --target-expert-parallel-size 32 \
    --expert-tensor-parallel-size 1 \
    --moe-grouped-gemm \
    --load-dir ./model_weights/longcat-flash-chat-mcore/ \
    --save-dir ./model_from_hf/longcat-flash-chat-hf/ \
    --hf-dir ./origin_hf_cfg \