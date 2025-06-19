# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

python examples/mcore/deepseek3/convert_ckpt_deepseek3_mcore2hf.py \
    --source-tensor-parallel-size 1 \
    --source-pipeline-parallel-size 4 \
    --source-expert-parallel-size 8 \
    --load-dir ./model_weights/deepseek3-mcore \   
    --save-dir ./model_from_hf/deepseek3-hf \
    --num-layers 61 \
    --first-k-dense-replace 3 \
    --num-layer-list 16,15,15,15 \
    --lora-r 8 \
    --lora-alpha 16 \

    # --load-dir 指定lora权重路径，此权重包含base权重和lora权重
    # --num-layer-list, --noop-layers, --num-layers-per-virtual-pipeline-stage等参数根据任务需要进行配置
    # 如果需要将独立的lora权重合并到base中，参见readme