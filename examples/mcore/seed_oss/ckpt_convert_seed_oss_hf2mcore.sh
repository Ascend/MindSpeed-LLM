# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 设置需要的权重转换参数
python convert_ckpt_v2.py \
       --use-mcore-models \
       --load-model-type hf \
       --save-model-type mg \
       --target-tensor-parallel-size 4 \
       --target-pipeline-parallel-size 4 \
       --add-qkv-bias \
       --load-dir /home/hf_weights/Seed-OSS-36B-Base \
       --save-dir /home/mg_weights/Seed-OSS-36B-mcore \
       --model-type-hf seed-oss