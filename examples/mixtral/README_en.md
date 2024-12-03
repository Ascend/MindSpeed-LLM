# Mixtral

<p align="left">
        <b><a href="README.md">简体中文</a> </b> |
        <b>English</b> 
</p>

# Table of Contents

- [Mixtral](#mixtral)
- [Table of Contents](#table-of-contents)
- [8x7B](#8x7b)
  - [Hardware-Requirements](#hardware-requirements)
  - [Preparation](#preparation)
  - [Model-Training](#model-training)
  - [Model-Performance](#model-performance)
    - [Throughput](#throughput)
  - [Model-Inference](#model-inference)
  - [Model-Evaluation](#model-evaluation)

# 8x7B

## Hardware-Requirements

Minimum hardware requirements for training:

| Hardware |  Configuration   |
| :------: |:----------------:|
|   NPU   | 32 x Ascend NPUs |

Recommended hardware configuration for inference:

| Hardware |  Configuration  |
| :------: | :-------------: |
|   NPU   | 8 x Ascend NPUs |

## Preparation

1. Clone the code repository to the local server

    ```shell
    git clone https://gitee.com/ascend/MindSpeed-LLM.git
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    git checkout -f bcce6f
    cp -r megatron ../MindSpeed-LLM/
    cd ..
    cd MindSpeed-LLM
	git checkout 1.0
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
    git checkout 224ae35e8fc96778f957029d1371ddb623452a50
    pip install -r requirements.txt 
    pip3 install -e .
    cd ..

    # install other packages
    pip install -r requirements.txt 
    ```

3. Download the pre-trained weights and vocabulary for Mixtral-8x7B from [here](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/tree/main). (It is recommended to only download weights in safetensors format)

    ```shell
    #!/bin/bash
    cd ./model_from_hf/
    git lfs install
    git clone https://huggingface.co/mistralai/Mixtral-8x7B-v0.1
    mv Mixtral-8x7B-v0.1 Mixtral-8x7B
    cd ..
    ```

4. Weight conversion

    HuggingFace weights --> Megatron weights with any parallel slicing strategy
    ***(This scenario is generally used to train open-source HuggingFace models on Megatron)***

    ```bash
    # Modify the ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    
    # HF to tp8-pp4-ep1
    python tools/checkpoint/util.py \
        --model-type GPT \
        --loader mixtral_hf \
        --saver mixtral \
        --load-dir ./model_from_hf/Mixtral-8x7B/ \
        --save-dir ./model_weights/Mixtral-8x7B-v0.1-tp1-tp8-pp4/ \
        --tokenizer-model ./model_from_hf/Mixtral-8x7B/tokenizer.model \
        --target-tensor-parallel-size 8 \
        --target-pipeline-parallel-size 4 \
        --target-expert-parallel-size 1
    ```

    Any Megatron weights with parallel slicing strategy --> Any Megatron weights with parallel slicing strategy
    ***(This scenario is generally used to reconfigure the sliced model weights, such as training on a dual-node 16-card EP2-PP8 strategy, and then wanting to infer on a single-node 8-card TP8)***

    ```bash
    # Modify the ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    
    # tp8-pp4-ep1 to tp8-pp1-ep1
    python tools/checkpoint/util.py \
        --model-type GPT \
        --loader mixtral_mg \
        --saver mixtral \
        --load-dir ./model_weights/Mixtral-8x7B-v0.1-tp8-pp4-ep1/ \
        --save-dir ./model_weights/Mixtral-8x7B-v0.1-tp8-pp1-ep1/ \
        --target-tensor-parallel-size 8 \
        --target-pipeline-parallel-size 1 \
        --target-expert-parallel-size 1 
    ```

    Any Megatron weights with parallel slicing strategy --> HuggingFace weights
    ***(This scenario is generally used to convert the trained megatron model back to the HuggingFace format)***

    ```bash
    # Modify the ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    
    # tp8-pp4-ep1 to HF
    python tools/checkpoint/util.py \
        --model-type GPT \
        --loader mixtral_mg \
        --saver mixtral \
        --save-model-type huggingface \
        --load-dir ./model_weights/Mixtral-8x7B-v0.1-tp8-pp4-ep1/ \
        --save-dir ./model_from_hf/Mixtral-8x7B/    # <-- Fill in the original HF model path here, new weights will be saved in ./model_from_hf/Mixtral-8x7B/mg2hg/
    ```

## Model-Training

1. Prepare dataset

    Download the datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet), save to MindSpeed-LLM/dataset/ directory.

    ```shell
    # download datasets
    cd ./dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    # process datasets
    mkdir ./dataset/Mixtral-8x7B/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/Mixtral-8x7B/ \
        --output-prefix ./dataset/Mixtral-8x7B/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF
    ```

    Configure Mixtral-8x7B pre-training script: ***examples/mixtral/pretrain_mixtral_8x7b_ptd.sh***

    ```shell
    # Set the ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # Configure according to the actual vocabulary, dataset, and model parameter save path
    DATA_PATH="./dataset/Mixtral-8x7B/alpaca_text_document"
    TOKENIZER_MODEL="./model_from_hf/Mixtral-8x7B/"
    CKPT_SAVE_DIR="./ckpt/Mixtral-8x7B/"

    # Configure distributed parameters according to the actual distributed cluster
    GPUS_PER_NODE=8
    MASTER_ADDR="your master node IP"
    MASTER_PORT=6000
    NNODES=4
    NODE_RANK="current node id"
    WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

    # Training parallel strategy
    TP=8
    PP=4
    EP=1
    ```

    Start Mixtral-8x7B pre-training script: ***examples/mixtral/pretrain_mixtral_8x7b_ptd.sh***

    ```shell
    bash examples/mixtral/pretrain_mixtral_8x7b_ptd.sh
    ```

    **Note**: If using multi machine training, and no data sharing configuration on the mechines, it's necessary to add the parameter `--no-shared-storage`. This parameter will determine whether non master nodes need to load data based on distributed parameters, and check the corresponding cache and generated data.

2. Fine-Tuning

    Prepare fine-tuning dataset
    Download the fine-tuning datasets from [here](https://huggingface.co/datasets/silk-road/alpaca-data-gpt4-chinese/blob/main/Alpaca_data_gpt4_zh.jsonl)

    ```shell
    # download datasets
    mkdir finetune_dataset
    cd ./finetune_dataset
    wget https://huggingface.co/datasets/silk-road/alpaca-data-gpt4-chinese/blob/main/Alpaca_data_gpt4_zh.jsonl
    cd ..

    # process datasets  
    mkdir ./finetune_dataset/Mixtral-8x7B/
    python ./tools/preprocess_data.py \
        --input ./finetune_dataset/Alpaca_data_gpt4_zh.jsonl \
        --output-prefix ./finetune_dataset/Mixtral-8x7B/alpaca \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path ./model_from_hf/Mixtral-8x7B/ \
        --append-eod \
        --tokenizer-not-use-fast \
        --handler-name GeneralInstructionHandler \
        --workers 4
    ```

3. Supervised Fine-Tuning

    The configuration script for full parameters fine-tuning  is basically the same as that for pretrain shell. *The difference is that the dataset and the training parameter is-instruction-dataset are added.*

    Add the fine-tuning parameter `--finetune` and add pretrained-weight load parameter `--load`, so that fine-tuning starts from the first step.

    ```shell
    DATA_PATH="./finetune_dataset/Mixtral-8x7B/alpaca"
    CKPT_PATH="./ckpt/Mixtral-8x7B/"
        --load ${CKPT_PATH} \
        --finetune \
        --is-instruction-dataset
    ```

## Model-Performance

### Throughput

Comparison of Mixtral-8x7B performance on 4 nodes and 32 chips with tp8 pp4:

|  Device  |    Model    | Iterations | Sample Throughput (samples/step) | Tokens Throughput (tokens/s/p) | Single Step Iteration Time (s/step) |
| :-------: | :----------: | :--------: |:--------------------------------:|:------------------------------:|:-----------------------------------:|
|   NPUs   | Mixtral-8x7B |    1000    |               0.47               |              487               |                16.81                |
| Reference | Mixtral-8x7B |    1000    |               0.59               |              610               |                13.41                |

## Model-Inference

First, configure the inference script: ***tasks/inference/generate_mixtral_8x7b_ptd.sh***

```bash
# Execute set_env.sh according to your own ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# Modify the model weight path and tokenizer path
CHECKPOINT="./model_weights/Mixtral-8x7B-v0.1-tp8-pp1-ep1/"
TOKENIZER_MODEL="./model_from_hf/Mixtral-8x7B/"

# Modify according to the actual loaded model weight the parallel configuration
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=8
TP=8
PP=1

# Note
The Mixtral-8x7B-v0.1 model used in this document is an L0 model, only with continuation ability, inference does not involve any templates and is prone to repetition or non-stop answering.

If you want to have better human-machine dialogue capabilities, please use the Mixtral-8x7B-Instruct-v0.1 model. This model requires instruction compliance training and needs to be used with templates. The basic operations are the same as above, only the startup entry has changed:
torchrun $DISTRIBUTED_ARGS inference.py
```

Then you can start it directly

```bash
bash tasks/inference/generate_mixtral_8x7b_ptd.sh
```

An example of inference is as follows:
![Inference](../../sources/images/mixtral/generate_demo.png)

## Model-Evaluation

Evaluate the model using the MMLU dataset. Dataset download path [here](https://huggingface.co/datasets/cais/mmlu).
Configure the evaluation script: ***tasks/evaluation/evaluate_mixtral_8x7b_ptd.sh***

```bash
# Ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# Modify the model parameter path and tokenizer path
TOKENIZER_PATH="./model_from_hf/Mixtral-8x7B/"                                           #tokenizer path
CHECKPOINT="./model_weights/Mixtral-8x7B-v0.1-tp8-pp1-ep1"                                         #model path

# Configure tasks and dataset paths
DATA_PATH="./mmlu/data/test/"
TASK="mmlu"
```

Start the evaluation

```bash
bash tasks/evaluation/evaluate_mixtral_8x7b_ptd.sh
```

The evaluation results are as follows

| Dataset | Dataset | Refer Accuracy | Ours |
| :-----: | :-----: | :------------: | :---: |
|  MMLU  |  14042  |     0.658     | 0.660 |
