# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

python convert_ckpt_v2.py \
   --use-mcore-models \
   --model-type-hf phi3.5 \
   --load-model-type mg \
   --save-model-type hf \
   --target-tensor-parallel-size 1 \
   --target-pipeline-parallel-size 1 \
   --load-dir ./model_weights/phi35_mini_mcore/ \
   --save-dir ./model_from_hf/Phi-3.5-mini-instruct/ \
   --hf-dir ./origin_hf_cfg \
