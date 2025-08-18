# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 设置需要的权重转换参数
python mindspeed_llm/mindspore/convert_ckpt.py \
       --use-mcore-models \
       --model-type GPT \
       --load-model-type hf \
       --save-model-type mg \
       --target-tensor-parallel-size 4 \
       --target-pipeline-parallel-size 2 \
       --add-qkv-bias \
       --load-dir ./model_from_hf/qwen2.5_7b_hf/ \
       --save-dir ./model_weights/qwen2.5_mcore/ \
       --tokenizer-model ./model_from_hf/qwen2.5_7b_hf/tokenizer.json \
       --model-type-hf llama2 \
       --params-dtype bf16 \
       --ai-framework mindspore \
