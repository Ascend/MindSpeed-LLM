export CUDA_DEVICE_MAX_CONNECTIONS=1
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt_v2.py \
    --load-model-type mg \
    --save-model-type hf \
    --model-type-hf phi3.5-moe \
    --load-dir ./model_weights/phi3.5-moe-mcore \
    --save-dir ./model_from_hf/Phi3.5-MoE-instruct-hf \
    --hf-cfg-dir ./model_from_hf/Phi3.5-MoE-instruct-hf \
    --moe-grouped-gemm
