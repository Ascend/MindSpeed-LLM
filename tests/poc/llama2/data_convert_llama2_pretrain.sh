# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./dataset

python ./preprocess_data.py \
  --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
  --tokenizer-name-or-path ./model_from_hf/llama-2-7b-hf/ \
  --tokenizer-type PretrainedFromHF \
  --handler-name GeneralPretrainHandler \
  --output-prefix ./dataset/enwiki \
  --json-keys text \
  --workers 4 \
  --log-interval 1000
