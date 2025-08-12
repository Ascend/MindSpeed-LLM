# 请按照您的真实环境修改 set_env.sh 路径
source /home/poc_test/ascend-toolkit/set_env.sh
mkdir -p ./dataset

python ./preprocess_data.py \
        --input /data/dpo_en.json \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-not-use-fast \
        --tokenizer-name-or-path /data/chatglm3-6b-base \
        --output-prefix dataset/dpo_en \
        --workers 4 \
        --log-interval 1000 \
        --trust-remote-code \
        --handler-name SharegptStylePairwiseHandler \
        --prompt-type chatglm3 