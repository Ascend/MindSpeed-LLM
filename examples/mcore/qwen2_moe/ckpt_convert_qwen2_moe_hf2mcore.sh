export CUDA_DEVICE_MAX_CONNECTIONS=1
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置需要的权重转换参数
python convert_ckpt.py \
       --use-mcore-models \
       --model-type GPT \
       --load-model-type hf \
       --save-model-type mg \
       --target-tensor-parallel-size 1 \
       --target-pipeline-parallel-size 4 \
       --target-expert-parallel-size 1 \
       --add-qkv-bias \
       --load-dir ./model_from_hf/qwen2_moe_hf/ \
       --save-dir ./model_weights/qwen2_moe_mcore/ \
       --spec mindspeed_llm.tasks.models.spec.qwen2_moe_spec layer_spec \
       --tokenizer-model ./model_from_hf/qwen2_moe_hf/tokenizer.json \
       --model-type-hf qwen2-moe \
       --moe-grouped-gemm \
       --params-dtype bf16