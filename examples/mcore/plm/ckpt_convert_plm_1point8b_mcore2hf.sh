# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

python convert_ckpt_v2.py \
    --load-model-type mg \
    --save-model-type hf \
    --model-type-hf plm \
    --load-dir ./model_weights/plm \
    --save-dir ./model_from_hf/plm \
    --hf-cfg-dir ./model_from_hf/plm
