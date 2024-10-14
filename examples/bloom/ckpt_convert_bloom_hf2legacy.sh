# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

#设置你需要的并行策略，--params-dtype bf16 结合需要使用
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 8 \
    --target-pipeline-parallel-size 1 \
    --load-dir ./model_from_hf/Bloom-hf/ \
    --save-dir ./model_weights/Bloom-legacy/ \
    --tokenizer-model None \
    --model-type-hf bloom \
    --add-qkv-bias \
    --add-dense-bias
