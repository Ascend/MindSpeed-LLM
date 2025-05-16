# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./dataset/mamba2-hf/

python ./preprocess_data.py \
        --input ./dataset/train-00000-of-00042-d964455e17e96d5a.parquet \
        --tokenizer-model ./model_from_hf/mamba2-hf/mamba2_2.7b_from_8b.model \
        --output-prefix ./dataset/mamba2-hf/enwiki \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type GPTSentencePieceTokenizer