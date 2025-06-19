# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

python convert_ckpt.py \
    --use-mcore-models \
    --model-type-hf deepseek2-lite \
    --model-type GPT \
    --load-model-type optim \
    --params-dtype bf16 \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 4 \
    --target-expert-parallel-size 2 \
    --moe-grouped-gemm \
    --spec mindspeed_llm.tasks.models.spec.deepseek_spec layer_spec \
    --load-dir /data/optim_ckpt/deepseek-lite-tp1pp2ep4/ \
    --save-dir /data/optim_ckpt/deepseek2_lite_pp4_ep2_optim_base/
