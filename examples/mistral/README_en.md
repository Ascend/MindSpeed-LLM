# Mistral-7B-32K

<p align="left">
        <b><a href="README.md">简体中文</a> </b> |
        <b>English</b> 
</p>

# Table of Contents

- [Mistral](#mistral)
- [Table of Contents](#table-of-contents)
- [7B](#7B-32K)
  - [Hardware-Requirements](#hardware-requirements)
  - [Preparation](#preparation)
  - [Model-Training](#model-training)
  - [Model-Inference](#model-inference)
  - [Model-Evaluation](#model-evaluation)

# 7B-32K

## Hardware-Requirements

Minimum hardware requirements for training:

| Hardware |  Configuration  |
| :------: |:---------------:|
|   NPU   | 8 x Ascend NPUs |

Recommended hardware configuration for inference:

| Hardware |  Configuration  |
| :------: | :-------------: |
|   NPU   | 8 x Ascend NPUs |

## Preparation

1. Clone the code repository to the local server

    ```shell
    git clone https://gitee.com/ascend/ModelLink.git
    cd ModelLink
    git checkout 1.0.RC2
    cd ..
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    git checkout core_r0.6.0
    cp -r megatron ../ModelLink/
    cd ..
    cd ModelLink
    mkdir logs
    mkdir model_from_hf
    mkdir dataset
    mkdir ckpt
    ```

2. Set up the environment

    ```bash
    # python3.8
    conda create -n test python=3.8
    conda activate test

    # Install torch and torch_npu
    pip install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl
    pip install torch_npu-2.1.0*-cp38-cp38m-linux_aarch64.whl
    pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

    # modify the path according to your own  ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # install MindSpeed
    git clone https://gitee.com/ascend/MindSpeed.git
    cd MindSpeed
    git checkout 2b0edd2
    pip install -r requirements.txt 
    pip3 install -e .
    cd ..

    # install other packages
    pip install -r requirements.txt 
    ```

3. Download the pre-trained weights and vocabulary for Mistral-7B from [here](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/tree/main). (It is recommended to only download weights in safetensors format)

    ```shell
    #!/bin/bash
    cd ./model_from_hf/
    git lfs install
    git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
    cd ..
    ```

4. Weight conversion

    HuggingFace weights --> Megatron weights with any parallel slicing strategy
    ***(This scenario is generally used to train open-source HuggingFace models on Megatron)***

    ```bash
    # Modify the ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # HF to tp8-pp1
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader llama2_hf \
        --saver megatron \
        --load-dir ./model_from_hf/Mistral-7B-Instruct-v0.2/ \
        --save-dir ./model_weights/Mistral-7B-Instruct-v0.2-tp8-pp1/ \
        --tokenizer-model ./model_from_hf/Mistral-7B-Instruct-v0.2/tokenizer.model \
        --target-tensor-parallel-size 8 \
        --target-pipeline-parallel-size 1
    ```

    Any Megatron weights with parallel slicing strategy --> HuggingFace weights
    ***(This scenario is generally used to convert the trained megatron model back to the HuggingFace format)***

    ```bash
    # Modify the ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # tp8-pp1 to HF
    python tools/checkpoint/convert_ckpt.py \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --model-type GPT \
        --loader megatron \
        --saver megatron \
        --save-model-type save_huggingface_llama \
        --load-dir ./model_weights/Mistral-7B-Instruct-v0.2-tp8-pp1/ \
        --save-dir ./model_from_hf/Mistral-7B-Instruct-v0.2/    # <-- Fill in the original HF model path here, new weights will be saved in ./model_from_hf/Mistral-7B-Instruct-v0.2/mg2hg/
    ```

## Model-Training

Prepare dataset

Download the datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet), save to ModelLink/dataset/ directory.

```shell
# download datasets
cd ./dataset
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..
# process datasets
mkdir ./dataset/Mistral-7B/
python ./tools/preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./model_from_hf/Mistral-7B-Instruct-v0.2/ \
    --output-prefix ./dataset/Mistral-7B/alpaca \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF
```

Configure Mistral-7B pre-training script: ***examples/mistral/pretrain_mistral_7b_ptd.sh***

```shell
# Set the ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# Configure according to the actual vocabulary, dataset, and model parameter save path
DATA_PATH="./dataset/Mistral-7B/alpaca_text_document"
TOKENIZER_MODEL="./model_from_hf/Mistral-7B-Instruct-v0.2/"
CKPT_SAVE_DIR="./ckpt/Mistral-7B-Instruct-v0.2-tp8-pp1/"

# Configure distributed parameters according to the actual distributed cluster
GPUS_PER_NODE=8
MASTER_ADDR="your master node IP"
MASTER_PORT=6000
NNODES=1
NODE_RANK="current node id"
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

# Training parallel strategy
TP=8
PP=1
```

Start Mistral-7B pre-training script: ***examples/pretrain_mistral_7b_ptd.sh***

```shell
bash examples/mistral/pretrain_mistral_7b_ptd.sh
```

**Note**: 
1. If using multi machine training, it is necessary to set up multi machine data sharing, and non primary nodes can read the primary node data through data sharing. Alternatively, directly copy the data generated by the master node to non master nodes.
2. The hyperparameters for training in the pretrain_mistral_7b_ptd.sh script need to be adjusted according to actual situations. For example, the global-batch-size needs to be set larger during pre-training to achieve better results, such as 256.

Fine-Tuning

Prepare fine-tuning dataset
Download the fine-tuning datasets from [here](https://huggingface.co/datasets/silk-road/alpaca-data-gpt4-chinese/blob/main/Alpaca_data_gpt4_zh.jsonl)

```shell
# download datasets
mkdir finetune_dataset
cd ./finetune_dataset
wget https://huggingface.co/datasets/silk-road/alpaca-data-gpt4-chinese/blob/main/Alpaca_data_gpt4_zh.jsonl
cd ..

# process datasets  
mkdir ./finetune_dataset/Mistral-7B/
python ./tools/preprocess_data.py \
    --input ./finetune_dataset/Alpaca_data_gpt4_zh.jsonl \
    --output-prefix ./finetune_dataset/Mistral-7B/alpaca \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ./model_from_hf/Mistral-7B-Instruct-v0.2/ \
    --append-eod \
    --tokenizer-not-use-fast \
    --handler-name GeneralInstructionHandler \
    --workers 4
```

Supervised Fine-Tuning

The configuration script for full parameters fine-tuning  is basically the same as that for pretrain shell. *The difference is that the dataset and the training parameter is-instruction-dataset are added.*

Add the fine-tuning parameter `--finetune` and add pretrained-weight load parameter `--load`, so that fine-tuning starts from the first step.

```shell
DATA_PATH="./finetune_dataset/Mistral-7B/alpaca"
CKPT_PATH="./ckpt/Mistral-7B-Instruct-v0.2-tp8-pp1/"
--load ${CKPT_PATH} \
--finetune \
--is-instruction-dataset
```


## Model-Inference

First, configure the inference script: ***examples/mistral/generate_mistral_7b_ptd.sh***

```bash
# Execute set_env.sh according to your own ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# Modify the model weight path and tokenizer path
CHECKPOINT="./model_weights/Mistral-7B-Instruct-v0.2-tp8-pp1/"
TOKENIZER_MODEL="./model_from_hf/Mistral-7B-Instruct-v0.2/"

# Modify according to the actual loaded model weight the parallel configuration
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=8
TP=8
PP=1

# Note
This model used in this document is an L1 model that requires instruction compliance training and needs to be used with templates. The basic operations are the same as above, only the startup entry has changed:
--inference-prompt-type mixtral
```

Then you can start it directly

```bash
bash examples/mistral/generate_mistral_7b_ptd.sh
```

An example of inference is as follows:
![Inference](https://gitee.com/ascend/ModelLink/raw/master/sources/images/mistral/generate_demo.png)

## Model-Evaluation

Evaluate the model using the MMLU dataset. Dataset download path [here](https://huggingface.co/datasets/cais/mmlu).
Configure the evaluation script: ***examples/mistral/evaluate_mistral_7b_ptd.sh***

```bash
# Ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# Modify the model parameter path and tokenizer path
CHECKPOINT="./model_weights/Mistral-7B-Instruct-v0.2-tp8-pp1/"
TOKENIZER_MODEL="./model_from_hf/Mistral-7B-Instruct-v0.2/"

# Configure tasks and dataset paths
DATA_PATH="./mmlu/data/test/"
TASK="mmlu"
```

Start the evaluation

```bash
bash examples/mistral/evaluate_mistral_7b_ptd.sh
```

The evaluation results are as follows

| Dataset | Dataset | Refer Accuracy | Ours |
| :-----: | :-----: | :------------: | :---: |
|  MMLU  |  14042  |   0.563   |   0.563   |
