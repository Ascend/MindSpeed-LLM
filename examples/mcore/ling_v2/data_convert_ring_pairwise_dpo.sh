# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./dpo_dataset

python ./preprocess_data.py \
        --input ./dpo_dataset_orca/orca_rlhf.jsonl \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path ./model_from_hf/ring_1T_v2-hf/ \
        --output-prefix ./dpo_dataset_orca/orca_rlhf \
        --workers 1 \
        --log-interval 1000 \
        --overwrite-cache \
        --handler-name AlpacaStylePairwiseHandler \
        --prompt-type bailing_mini \
        --map-keys '{"prompt":"question", "query":"", "system":"system"}'