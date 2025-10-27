# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python ./mindspeed_llm/mindspore/convert_ckpt_v2.py \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 2 \
    --target-expert-parallel-size 8 \
    --num-layers-per-virtual-pipeline-stage 2 \
    --load-dir ./model_from_hf/qwen3_moe_hf/ \
    --save-dir ./model_weights/qwen3_moe_ms/ \
    --moe-grouped-gemm \
    --model-type-hf qwen3-moe \
    --ai-framework mindspore

# 脚本配置仅供参考，具体配置请根据训练策略修改
# 若num_hidden_layers不能被（target-pipeline-parallel-size * num-layers-per-virtual-pipeline-stage）整除，需要添加--noop-layers参数
# 假设num_hidden_layers是94，target-pipeline-parallel-size是4，num-layers-per-virtual-pipeline-stage是8，则需添加--noop-layers 94，95 \