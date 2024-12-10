# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 权重格式转换，设置需要的并行配置
python convert_ckpt.py \
    --use-mcore-models \
    --model-type-hf minicpm3 \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --params-dtype bf16 \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 2 \
    --load-dir ./model_from_hf/MiniCPM3-4B-hf/ \
    --save-dir ./model_weights/MiniCPM3-4B_mcore-tp1pp2 \
    --spec mindspeed_llm.tasks.models.spec.minicpm_spec layer_spec \
    --tokenizer-model ./model_from_hf/MiniCPM3-4B-hf/tokenizer.model
