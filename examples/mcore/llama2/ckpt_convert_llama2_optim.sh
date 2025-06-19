# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 权重格式转换，设置需要的并行配置，--num-layers-per-virtual-pipeline-stage 5，--params-dtype bf16 结合需要使用
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type optim \
    --load-dir ./ckpt/llama2/ \
    --target-tensor-parallel-size 4 \
    --target-pipeline-parallel-size 2 \
    --save-dir ./ckpt/llama2-7b-tp4pp2_optim_test/ \
    --use-mcore-models \
    --model-type-hf llama2
