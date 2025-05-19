# 过程奖励数据集

## 常用的过程奖励数据集

- [MATH-SHEPHERD](https://huggingface.co/datasets/zhuzilin/Math-Shepherd)

## 数据集下载

MATH-SHEPHERD 数据集下载可以基于网页直接下载，也可以基于命令行下载，比如：

```shell
mkdir dataset
cd dataset/
wget https://huggingface.co/datasets/zhuzilin/Math-Shepherd/resolve/main/data/train-00000-of-00002.parquet
cd ..
```

## 数据集处理

### 数据预处理

```bash
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir -p ./dataset

python ./preprocess_data.py \
        --input ./data/math_shepherd/train-00000-of-00002.parquet \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-not-use-fast \
        --tokenizer-name-or-path ./model_from_hf/Llama2-hf \
        --output-prefix ./prm_dataset/math-shepherd \
        --workers 4 \
        --log-interval 1000 \
        --handler-name AlpacaStyleProcessRewardHandler \
        --seq-length 4096 \
        --placeholder-token ки \
        --reward-tokens + - \
```

【--input】

可以直接输入到数据集目录或具体文件，如果是目录，则处理全部文件, 支持 .parquet \ .csv \ .json \ .jsonl \ .txt \ .arrow 格式， 同一个文件夹下的数据格式需要保持一致.

【--handler-name】

PRM微调数据预处理Alpaca风格数据集时，应指定为AlpacaStyleProcessRewardHandler。

【--placeholder-token】

微调数据prompt中每个推理步骤间的分割占位符，默认为"ки"。

【--reward-tokens】

微调数据label中表示每个推理步骤是否正确的奖励hard标签token，"+"代表当前推理步骤是正确的，"-"代表当前推理步骤是错误的。

启动脚本
MindSpeed-LLM过程奖励数据集处理脚本命名风格及启动方法为：

```bash
# Legacy
# 命名及启动：examples/mcore/model_name/data_convert_xxx_process_reward.sh
bash examples/legacy/llama2/data_convert_llama2_process_reward.sh
过程奖励微调数据集处理结果如下：

./prm_dataset/math-shepherd_packed_attention_mask_document.bin
./prm_dataset/math-shepherd_packed_attention_mask_document.idx
./prm_dataset/math-shepherd_packed_input_ids_document.bin
./prm_dataset/math-shepherd_prm_packed_input_ids_document.idx
./prm_dataset/math-shepherd_prm_packed_labels_document.bin
./prm_dataset/math-shepherd_prm_packed_labels_document.idx
```
