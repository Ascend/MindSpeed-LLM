export CUDA_DEVICE_MAX_CONNECTIONS=1
# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 权重格式转换
python convert_ckpt_v2.py \
    --load-model-type hf \
    --save-model-type mg \
    --model-type-hf phi3.5-moe \
    --target-tensor-parallel-size 4 \
    --target-pipeline-parallel-size 4 \
    --target-expert-parallel-size 1 \
    --load-dir ./model_from_hf/Phi3.5-MoE-instruct-hf \
    --save-dir ./model_weights/phi3.5-moe-mcore \
    --moe-grouped-gemm
#    --num-layers-per-virtual-pipeline-stage 2 \     当转换pretain使用的权重时，增加该参数
