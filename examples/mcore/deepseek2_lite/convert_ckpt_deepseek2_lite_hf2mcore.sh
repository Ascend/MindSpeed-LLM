# 请按照您的真实环境修改 set_env.sh 路径
# 按照您的实际需要修改目录信息并完成对应的TP、PP、EP的参数配置

source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt_v2.py \
   --moe-grouped-gemm \
   --model-type-hf deepseek2-lite \
   --load-model-type hf \
   --save-model-type mg \
   --target-tensor-parallel-size 1 \
   --target-pipeline-parallel-size 1 \
   --target-expert-parallel-size 8 \
   --load-dir /data/deepseek2_lite_hf \
   --save-dir ./model_weights/deepseek2_lite_mcore/ \
