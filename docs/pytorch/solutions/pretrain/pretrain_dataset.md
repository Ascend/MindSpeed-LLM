# 预训练数据集处理

## 常用的预训练数据集

- [Alpaca数据集](https://huggingface.co/datasets/tatsu-lab/alpaca)
- [Enwiki数据集](https://huggingface.co/datasets/lsb/enwiki20230101)
- [C4数据集](https://huggingface.co/datasets/allenai/c4)
- [ChineseWebText](https://huggingface.co/datasets/CASIA-LM/ChineseWebText)

## 数据集下载

数据集下载可以基于网页直接下载，也可以基于命令行下载，比如：

```shell
mkdir dataset
cd dataset/
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..
```

## 数据集处理

### 预训练数据集处理方法

```shell
# 请按照您的真实环境 source set_env.sh 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./dataset

python ./preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./model_from_hf/llama-2-7b-hf \
    --tokenizer-type PretrainedFromHF \
    --handler-name GeneralPretrainHandler \
    --output-prefix ./dataset/alpaca_llama2_7b \
    --json-keys text \
    --workers 4 \
    --log-interval 1000  
```

MindSpeed-LLM预训练数据集处理脚本命名风格及启动方法为:

```shell
# 命名及启动：examples/mcore/model_name/data_convert_xxx_pretrain.sh
bash examples/mcore/llama2/data_convert_llama2_pretrain.sh
```

### 参数说明

【--input】

可以直接输入到数据集目录或具体文件，如果是目录，则处理全部文件, 支持.parquet/.csv/.json/.jsonl/.txt/.arrow格式， 同一个文件夹下的数据格式需要保持一致 

【--tokenizer-type】

说明使用tokenizer类别，参数值为PretrainedFromHF时，词表路径填写模型目录即可。否则需要配置`--tokenizer-model`参数，指定分词器模型的路径，路径具体到tokenizer.model文件

【--tokenizer-name-or-path】

词表路径，当tokenizer类别为PretrainedFromHF时，只需配置到目标模型的tokenizer所在目录即可

【--output-prefix】

转换后输出的数据集文件的文件名前缀

【--handler-name】

当前预训练默认使用 `GeneralPretrainHandler`，支持的是预训练数据风格，提取数据的`text`列，格式如下：

```shell
[
  {"text": "document"},
  {"other keys": "optional content"}
]
```

用户可结合具体数据处理需求添加新的Handler进行数据处理 

【--json-keys】

从文件中提取的列名列表，默认为 `text`，可以为 `text`, `input`, `title` 等多个输入，结合具体需求及数据集内容使用，如：

```shell
--json-keys text input output \
```

【--workers】

同时进行数据集处理的进程数

【--n-subs】

数据预处理并行加速参数。当需要预处理的数据集比较大时，可以通过并行处理进行加速，方法为设置参数`--n-subs`，通过该参数设置并行处理数量。在数据预处理过程会将原始数据集切分为`n-subs`个子集，对子集进行并行处理，然后合并，从而实现加速。建议预处理数据集超过GB级别时加上该参数。

### 处理结果

预训练数据集处理结果如下：

```shell
./dataset/alpaca_llama2_7b_text_document.bin
./dataset/alpaca_llama2_7b_text_document.idx
```

预训练时，数据集路径`--data-path`参数传入 `./dataset/alpaca_llama2_7b_text_document` 即可