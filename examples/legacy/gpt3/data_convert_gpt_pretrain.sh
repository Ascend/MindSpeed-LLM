# 请根据 docs/pytorch/models/dense_model.md 下 “社区BUG列表” 章节下载 gpt2-vocab.json，gpt2-merges.txt 文件
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./dataset

# 处理成训练数据
python ./preprocess_data.py \
    --input ./dataset/ \
    --output-prefix ./dataset/gpt_text_sentence \
    --tokenizer-type GPT2BPETokenizer \
    --vocab-file ./vocab_file/gpt2-vocab.json \
    --merge-file ./vocab_file/gpt2-merges.txt \
    --append-eod \
    --workers 4 \
    --log-interval 1000