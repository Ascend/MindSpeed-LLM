# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

python examples/mcore/deepseek3/convert_ckpt_deepseek3_mcore2hf.py \
    --moe-grouped-gemm \
    --source-tensor-parallel-size 2 \
    --source-pipeline-parallel-size 8 \
    --source-expert-parallel-size 32 \
    --load-dir ./model_weights/deepseek3-mcore \
    --save-dir ./model_from_hf/deepseek3-hf \
    --num-layers 64 \
    --mtp-num-layers 1 \
    --num-layers-per-virtual-pipeline-stage 2 \
    --noop-layers 47,62,63 \
    # --num-layer-list, --moe-tp-extend-ep 等参数根据任务需要进行配置
