# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

python convert_ckpt.py \
    --use-mcore-models \
    --model-type-hf llama2 \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --params-dtype bf16 \
    --target-tensor-parallel-size 8 \
    --target-pipeline-parallel-size 2 \
    --load-dir ./model_from_hf/Yi-hf/ \
    --save-dir ./model_weights/Yi-mcore/ \
    --tokenizer-model ./model_from_hf/Yi-hf/tokenizer.model
