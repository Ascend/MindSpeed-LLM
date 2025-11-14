# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

python convert_ckpt.py \
    --use-mcore-models \
    --model-type-hf plm \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --params-dtype bf16 \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --spec mindspeed_llm.tasks.models.spec.plm_spec layer_spec \
    --load-dir ./model_weights/plm \
    --save-dir ./model_from_hf/plm
