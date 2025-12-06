# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

python convert_ckpt_v2.py \
    --model-type-hf magistral \
    --load-model-type mg \
    --save-model-type hf \
    --params-dtype bf16 \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --hf-dir ./origin_hf_cfg \
    --load-dir ./model_weights/magistral_mcore/ \
    --save-dir ./model_from_hf/magistral_hf/  # 需要填入原始HF模型路径，新权重会存于./model_from_hf/magistral_hf/mg2hf/