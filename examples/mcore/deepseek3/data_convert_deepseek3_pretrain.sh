# 请根据 examples/README.md 下 “数据集准备及处理” 章节下载 Enwiki 数据集（一般取第一条即可）
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./dataset

python ./preprocess_data.py \
    --input ./dataset/train-00000-of-00042-d964455e17e96d5a.parquet \
    --tokenizer-name-or-path ./model_from_hf/deepseek3-hf/ \
    --tokenizer-type PretrainedFromHF \
    --handler-name GeneralPretrainHandler \
    --output-prefix ./dataset/enwiki \
    --json-keys text \
    --workers 4 \
    --log-interval 1000 \
    --append-eod