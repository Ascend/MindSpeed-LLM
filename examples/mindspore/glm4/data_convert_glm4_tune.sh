# 请根据 examples/README.md 下 “数据集准备及处理” 章节下载 Alpaca 数据集
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./finetune_dataset

python ./preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/glm4_hf/ \
        --output-prefix ./dataset/glm4_9b_hf/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-not-use-fast \
        --handler-name AlpacaStyleInstructionHandler \
        --prompt-type glm4 \
        --seq-length 32768 \
        --pack \
        --append-eod \
