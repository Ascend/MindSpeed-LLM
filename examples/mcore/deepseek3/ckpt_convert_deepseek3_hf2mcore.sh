# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python examples/mcore/deepseek3/convert_ckpt_deepseek3.py \
    --moe-grouped-gemm \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 8 \
    --target-expert-parallel-size 32 \
    --load-dir ./model_from_hf/deepseek3-bf16-hf \
    --save-dir ./model_weights/deepseek3-mcore \
    --num-layers 64 \
    --num-nextn-predict-layers 1 \
    --num-layers-per-virtual-pipeline-stage 2 \
    --noop-layers 47,62,63
    # --num-layer-list, --moe-tp-extend-ep 等参数根据任务需要进行配置
    # 当前分支不支持dualpipe，仅在微调、推理、评估任务中使用转换后权重
