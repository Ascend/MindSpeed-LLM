# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir -p ./finetune_dataset/glm45/

python ./preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path /home/hf_weights/GLM-4.5/ \
        --output-prefix ./finetune_dataset/glm45/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF \
        --handler-name AlpacaStyleInstructionHandler \
        --overwrite-cache \
        --prompt-type glm4_moe