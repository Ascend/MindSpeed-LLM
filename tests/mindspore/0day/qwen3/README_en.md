# Model Script Notes

The models in `tests/mindspore/0day/qwen3` currently support only the basic functionality required for the day-0 release. They are still in internal testing and have not completed sufficient performance testing or acceptance checks. In practice, undiscovered issues may still exist. A formal release will follow after sufficient validation.

## Weight Conversion

### Usage Instructions

It supports converting Hugging Face weights to MCore weights for training, fine-tuning, and other tasks.

### Launching the Script

Use the [Hugging Face-to-Megatron script](https://gitcode.com/ascend/MindSpeed-LLM/blob/master/convert_ckpt.py) in the Qwen3 model directory.

```commandline
# Dense model.
bash tests/mindspore/0day/qwen3/ckpt_convert_qwen3_dense_hf2mcore.sh

# Sparse model.
bash tests/mindspore/0day/qwen3/ckpt_convert_qwen3_moe_hf2mcore.sh
```

### Related Parameters

`--target-tensor-parallel-size`

Tensor parallelism degree. The default value is `1`.

`--target-pipeline-parallel-size`

Pipeline parallelism degree. The default value is `1`.

`--target-expert-parallel-size`

Expert parallelism degree. The default value is `1`.

`--load-dir`

Hugging Face weights that have already been dequantized to the BF16 data format.

`--save-dir`

The save path for the converted Megatron-format weights.

`--num-layers`

Number of model layers. If you configure empty-operation layers, set `num-layers` to the total number of layers plus the number of empty-operation layers.

`--spec`

For Qwen3 models, pass `mindspeed_llm.tasks.models.spec.qwen3_spec` as `layer_spec` here.

`--model-type-hf`

For Qwen3 sparse models, pass `qwen3`. For dense models, pass `qwen3-moe`.

`--moe-grouped-gemm`

Whether to enable MoE grouped matrix multiplication optimization.

## Data Preprocessing

### Common Training Datasets

[Alpaca dataset](https://huggingface.co/datasets/tatsu-lab/alpaca)

[Enwiki dataset](https://huggingface.co/datasets/lsb/enwiki20230101)

[C4 dataset](https://huggingface.co/datasets/allenai/c4)

[ChineseWebText](https://huggingface.co/datasets/CASIA-LM/ChineseWebText)

### Dataset Download

You can download the dataset directly from the web page or from the CLI. For example:

```commandline
mkdir dataset
cd dataset/
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..
```

### Dataset Processing

#### Dataset Processing Method

Run the following script to process the data:

```commandline
bash ./tests/mindspore/0day/qwen3/data_convert_qwen3.sh
```

#### Parameters

`--input`

You can provide a dataset directory or a specific file directly. If you provide a directory, the script processes all files in it. Supported formats include `.parquet`, `.csv`, `.json`, `.jsonl`, `.txt`, and `.arrow`. Data files in the same folder must use the same format.

`--tokenizer-name-or-path`

The tokenizer path, which points to the directory that contains the tokenizer.

`--output-prefix`

The output directory and prefix.

`--handler-name`

The default pretraining handler is `GeneralPretrainHandler`. It supports the pretraining data style and extracts the `text` column from the data, as shown below:

```commandline
[
  {"text": "document"},
  {"other keys": "optional content"}
]
```

You can add new handlers to process data based on specific requirements.

`--json-keys`

The list of column names to extract from the file. The default is `text`, but you can specify multiple inputs such as `text`, `input`, and `title`, depending on your needs and the dataset content. For example:

```commandline
--json-keys text input output \
```

#### Processing Results

The preprocessing results for the training dataset are as follows:

```commandline
./dataset/alpaca_qwen3_text_document.bin
./dataset/alpaca_qwen3_text_document.idx
```

For pretraining, pass `./dataset/alpaca_qwen3_text_document` to the `--data-path` parameter.

## Code Adaptation

Run the following command to clone the MindSpore-Core-MS repository:

```commandline
git clone -b feature-0.2 https://gitcode.com/ascend/MindSpeed-Core-MS.git
```

Using the MindSpore-Core-MS repository, you can clone the code and perform one-click code adaptation. Ensure that the environment is configured as follows:

* The deployment container has network access, and Python is installed.
* Git is configured and can run clone operations normally.

Run the following command for one-click adaptation:

```commandline
cd MindSpeed-Core-MS
source test_convert_llm.sh
```

Set the environment variables with the following command:

```commandline
MindSpeed_Core_MS_PATH=$(pwd)
export PYTHONPATH=${MindSpeed_Core_MS_PATH}/msadapter/mindtorch:${MindSpeed_Core_MS_PATH}/Megatron-LM:${MindSpeed_Core_MS_PATH}/MindSpeed:${MindSpeed_Core_MS_PATH}/MindSpeed-LLM:${MindSpeed_Core_MS_PATH}/transformers/src/:$PYTHONPATH
```

## Training Job Launch

### Configuring Training Parameters

The pretraining scripts are stored in each model folder under `MindSpeed-LLM/tests/mindspore/0day/qwen3`: `pretrain_xxx_ms.sh`.

You need to modify the paths and parameter values based on actual conditions.

Example:

`MindSpeed-LLM/tests/mindspore/0day/qwen3/qwen3-0.6b/pretrain_qwen3_0point6_ms.sh`

Path configuration includes the checkpoint save path, checkpoint load path, tokenizer path, and dataset path.

```commandline
# Configure the checkpoint save path, checkpoint load path, tokenizer path, and dataset path based on actual conditions.
# Note: Enclose the provided paths in double quotes.
CKPT_SAVE_DIR="./ckpt/qwen3"  # Path where the weights are saved after training completes.
CKPT_LOAD_DIR="./model_weights/qwen3/"  # Weight load path. Enter the path where the weights were saved during conversion.
DATA_PATH="./dataset/alpaca_qwen3_text_document"  # Dataset path. Enter the path where the processed dataset was saved.
TOKENIZER_MODEL="./model_from_hf/qwen3/tokenizer"  # Tokenizer path. Enter the tokenizer path from the downloaded open-source weights.
```

`Single-node run`

```commandline
GPUS_PER_NODE=8
MASTER_PORT=6000
MASTER_ADDR=localhost # Master node IP address.
NNODES=1
NODE_RANK=0
MASTER_PORT=9110
log_dir=msrun_log_pretrain # Log output path.
```

`Multi-node run`

```commandline
# Configure the distributed parameters based on the actual distributed cluster.
GPUS_PER_NODE=8
MASTER_ADDR="your master node IP address" # Master node IP address.
MASTER_PORT=6000
NNODES=4 # Number of nodes in the cluster. Enter the actual value.
NODE_RANK="current node id" # Current node rank. The master node is 0. Others can be 1, 2, and so on.
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
```

### Launching the Training Job

Initialize the environment variables:

```commandline
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

After you finish configuring the environment variables, run the training script:

```commandline
cd MindSpeed-LLM
bash ./tests/mindspore/0day/qwen3/qwen3-0.6b/pretrain_qwen3_0point6_ms.sh
```
