# please source your set_env.sh in your cann path
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./dataset

# For feature reset-attention-mask , we use --append-eod to pretrain.
python ./preprocess_data.py \
   --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
   --tokenizer-name-or-path ./model_from_hf/llama3_hf/ \
   --output-prefix ./dataset/enwiki \
   --workers 4 \
   --log-interval 1000  \
   --tokenizer-type PretrainedFromHF \
   --append-eod
