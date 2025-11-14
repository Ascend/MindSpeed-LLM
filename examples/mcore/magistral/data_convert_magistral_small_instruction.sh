# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./finetune_dataset

python ./preprocess_data.py \
    --input ./dataset/alpaca_gpt4_data_zh.json \
    --tokenizer-model ./model_from_hf/tekken.json \
    --output-prefix ./finetune_dataset/magistral/alpaca \
    --handler-name AlpacaStyleInstructionHandler \
    --tokenizer-type MagistralTokenizer \
    --workers 4 \
    --log-interval 1000  \
    --prompt-type magistral \
    --enable-thinking true
    
    # --map-keys '{"prompt":"instruction","query":"input","response":"output"}' # 默认值，可不传