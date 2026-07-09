# Pretraining Dataset Processing

## Common Pretraining Datasets

- [Alpaca Dataset](https://huggingface.co/datasets/tatsu-lab/alpaca)
- [Enwiki Dataset](https://huggingface.co/datasets/lsb/enwiki20230101)
- [C4 Dataset](https://huggingface.co/datasets/allenai/c4)
- [ChineseWebText](https://huggingface.co/datasets/CASIA-LM/ChineseWebText)

## Dataset Download

You can download datasets directly from a web page or using the CLI. For example:

```shell
mkdir dataset
cd dataset/
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..
```

## Dataset Processing

### Pretraining Dataset Processing Method

```shell
source /usr/local/Ascend/cann/set_env.sh # Change this to the actual Toolkit package path.
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

The naming convention and startup command for the MindSpeed LLM pretraining dataset processing scripts are as follows:

```shell
# Naming and startup: examples/mcore/model_name/data_convert_xxx_pretrain.sh
bash examples/mcore/llama2/data_convert_llama2_pretrain.sh
```

### Parameters

`--input`

You can point this parameter to a dataset directory or a specific file. If you pass a directory, the tool processes all files. It supports `.parquet`, `.csv`, `.json`, `.jsonl`, `.txt`, and `.arrow` formats. Data in the same folder must use the same format.

`--tokenizer-type`

This parameter specifies the tokenizer type. When the value is `PretrainedFromHF`, you only need to point the vocabulary path to the model directory. Otherwise, configure the `--tokenizer-model` parameter to specify the tokenizer model path down to the `tokenizer.model` file.

`--tokenizer-name-or-path`

This parameter sets the vocabulary path. When the tokenizer type is `PretrainedFromHF`, you only need to specify the directory that contains the tokenizer for the target model.

`--output-prefix`

This parameter sets the file prefix for the converted dataset output.

`--handler-name`

The default pretraining handler is `GeneralPretrainHandler`. It supports pretraining data formats and extracts the `text` column. The format is as follows:

```shell
[
  {"text": "document"},
  {"other keys": "optional content"}
]
```

You can add new handlers to process data based on your specific requirements.

`--json-keys`

This parameter lists the column names to extract from the file. The default is `text`. You can specify multiple columns such as `text`, `input`, and `title`, depending on your requirements and dataset content. For example:

```shell
--json-keys text input output \
```

`--workers`

This parameter sets the number of processes that handle the dataset simultaneously.

`--n-subs`

This parameter accelerates preprocessing for large datasets. When you need to preprocess a large dataset, set `--n-subs` to enable parallel processing. The preprocessing step splits the original dataset into `n-subs` subsets, processes the subsets in parallel, and then merges the results. Add this parameter when the dataset is larger than 1 GB.

### Processing Results

The pretraining dataset processing results are as follows:

```shell
./dataset/alpaca_llama2_7b_text_document.bin
./dataset/alpaca_llama2_7b_text_document.idx
```

For pretraining, pass `./dataset/alpaca_llama2_7b_text_document` as the value of the `--data-path` parameter.
