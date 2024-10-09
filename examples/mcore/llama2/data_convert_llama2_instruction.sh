# 请根据 examples/README.md 下 “数据集准备及处理” 章节下载 Alpaca 数据集
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./finetune_dataset

python ./preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./model_from_hf/Llama2-hf/ \
    --output-prefix ./finetune_dataset/alpaca \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF \
    --overwrite-cache \
    --handler-name AlpacaStyleInstructionHandler \
    --prompt-type llama2 \
    # --map-keys '{"prompt":"instruction","query":"input","response":"output"}' # 默认值，可不传