# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --load-dir ./model_weights/qwen3_8b_mcore/ \
    --lora-load ./ckpt/qwen3_8b_lora \
    --save-dir ./model_from_hf/qwen3_hf/ \
    --model-type-hf qwen3 \
