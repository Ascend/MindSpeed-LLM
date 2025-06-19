# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 权重格式转换
python convert_ckpt.py \
   --use-mcore-models \
   --model-type-hf llama2 \
   --model-type GPT \
   --load-model-type hf \
   --save-model-type mg \
   --params-dtype bf16 \
   --target-tensor-parallel-size 1 \
   --target-pipeline-parallel-size 4 \
   --num-layers-per-virtual-pipeline-stage 1 \
   --load-dir ./model_from_hf/internlm3_8b_hf/ \
   --save-dir ./model_weights/internlm3-8b-mcore-tp1-pp4-vpp1/ \
   --tokenizer-model ./model_from_hf/internlm3_8b_hf/tokenizer.model
