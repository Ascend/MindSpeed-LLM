# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./finetune_dataset

python ./preprocess_data.py \
    --input ./dataset/orca_rlhf.jsonl \
    --tokenizer-name-or-path ./model_from_hf/qwen3_moe_hf/ \
    --output-prefix ./finetune_dataset/orca_rlhf \
    --handler-name AlpacaStylePairwiseHandler \
    --tokenizer-type PretrainedFromHF \
    --workers 4 \
    --log-interval 1000 \
    --prompt-type qwen3 \
    --map-keys '{"prompt":"question", "query":"", "system":"system"}'
