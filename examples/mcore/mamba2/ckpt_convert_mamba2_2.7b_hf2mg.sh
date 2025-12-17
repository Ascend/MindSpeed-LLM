# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

python convert_ckpt_v2.py \
    --load-model-type hf \
    --save-model-type mg \
    --load-dir ./ckpt/mamba2-hf \
    --save-dir ./ckpt/mamba2-tp1pp4 \
    --target-pipeline-parallel-size 1 \
    --target-tensor-parallel-size 1 \
    --mamba-d-model 2560 \
    --mamba-d-state 128 \
    --mamba-head-dim 64 \
    --mamba-n-groups 1 \
    --model-type-hf 'mamba2'
