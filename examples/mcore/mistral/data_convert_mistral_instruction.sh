# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./finetune_dataset

python ./preprocess_data.py \
    --input ./dataset/alpaca_gpt4_data_zh.json \
    --tokenizer-name-or-path ./model_from_hf/mistral-7b-hf/ \
    --output-prefix ./finetune_dataset/mistral/alpaca \
    --handler-name AlpacaStyleInstructionHandler \
    --json-keys text \
    --workers 4 \
    --log-interval 1000  \
    --prompt-type mistral \
    --tokenizer-type PretrainedFromHF
    # --map-keys '{"prompt":"instruction","query":"input","response":"output"}' # 默认值，可不传