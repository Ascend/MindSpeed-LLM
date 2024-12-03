# Bloom

<p align="left">
        <b><a href="README.md">简体中文</a></b> |
        <b>English</b> 
    </p>
</p>


# Bloom-7B

## Training

Here's a hardware summary of pre-training Bloom-7B:

| Hardware |                      Value                      |
| :------: | :---------------------------------------------: |
|   NPU    |               8 x Ascend NPUs                   |

### Script

1. Clone the repository to your local server:

    ```shell
    git clone https://gitee.com/ascend/MindSpeed-LLM.git
    cd MindSpeed-LLM
    git checkout 1.0.RC2
    cd ..
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    git checkout core_r0.6.0
    cp -r megatron ../MindSpeed-LLM/
    cd ..
    cd MindSpeed-LLM
    mkdir logs
    mkdir model_from_hf
    mkdir dataset
    mkdir ckpt
    ```

2. Build environment

    ```bash
    # python3.8
    conda create -n test python=3.8
    conda activate test

    # install torch and torch_npu 
    pip install torch-2.1.0-cp38-cp38m-linux_aarch64.whl
    pip install torch_npu-2.1.0.XXX-cp38-cp38m-linux_aarch64.whl
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

3. Prepare pretrained weights
    Download the Bloom-7B checkpoint from [here](https://huggingface.co/bigscience/bloom-7b1/tree/main)

    ```shell
    mkdir ./model_from_hf/Bloom-7B/
    cd ./model_from_hf/Bloom-7B/
    cd tokenizer
    wget https://huggingface.co/bigscience/bloom/resolve/main/special_tokens_map.json
    wget https://huggingface.co/bigscience/bloom/resolve/main/tokenizer.json
    wget https://huggingface.co/bigscience/bloom/resolve/main/tokenizer_config.json
    ...
    cd ../../
    ```

4. Weights convert

    HuggingFace weights --> Megatron weights
    ***(This scenario is generally used to train open-source HuggingFace models on Megatron)***

    ```shell
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader loader_bloom_hf \
        --saver saver_megatron \
        --target-tensor-parallel-size 8 \
        --target-pipeline-parallel-size 1 \
        --load-dir ./model_from_hf/Bloom-7B/ \
        --save-dir ./model_weights/Bloom-7B-v0.1-tp8-pp1/ \
        --tokenizer-model None 
    ```

    Any Megatron weights with parallel slicing strategy --> Any Megatron weights with parallel slicing strategy
    ***(This scenario is generally used to convert the trained megatron model back to the HuggingFace format)***

    ```shell
    # Modify the ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader megatron \
        --saver megatron \
        --save-model-type save_huggingface_llama \
        --load-dir ./model_weights/Bloom-7B-v0.1-tp8-pp1/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --embed-layernorm \
        --save-dir ./model_from_hf/Bloom-7B/   # <-- Fill in the original HF model path here, new weights will be saved in ./model_from_hf/Bloom-7B/mg2hg/
    ```

5. Prepare dataset

    Download the Bloom-7B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

    ```shell
    # download datasets
    cd dataset/
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..

    # prepare datasets
    mkdir ./dataset/Bloom-7B/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/Bloom-7B/ \
        --output-prefix ./dataset/Bloom-7B/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF
    ```

6. Config Bloom-7B pre-training script(Bloom-7B does not support Flash Attention) : examples/bloom/pretrain_bloom_ptd_7B.sh

    ```shell
    # modify the script according to your own  ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    CKPT_SAVE_DIR="./ckpt/Bloom-7B/"
    DATA_PATH="./dataset/Bloom-7B/alpaca_text_document"
    TOKENIZER_PATH="./model_from_hf/Bloom-7B/"
    CKPT_LOAD_DIR="./model_weights/Bloom-7B-v0.1-tp8-pp1/"
    ```

7. Launch Bloom-7B  pre-training script: examples/bloom/pretrain_bloom_ptd_7B.sh

    ```shell
    bash examples/bloom/pretrain_bloom_ptd_7B.sh 
    ```
    **Note**: If using multi machine training, and no data sharing configuration on the mechines, it's necessary to add the parameter `--no-shared-storage`. This parameter will determine whether non master nodes need to load data based on distributed parameters, and check the corresponding cache and generated data.




## Inference Bloom-7B

Config Bloom-7B inference script: examples/bloom/generate_bloom_7b_ptd.sh

```bash
# modify the script according to your own ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
# modify script model path and tokenizer path
CHECKPOINT="./model_weights/Bloom-7B-Base-v0.1-tp8-pp1/"
TOKENIZER_PATH="./model_from_hf/Bloom-7B-Base/"
```

Launch Bloom-7B inference script: examples/bloom/generate_bloom_7b_ptd.sh

```bash
bash examples/bloom/generate_bloom_7b_ptd.sh
```

Some inference samples are as follows:

![Inference](https://gitee.com/ascend/MindSpeed-LLM/raw/master/sources/images/bloom/bloom7b-generate.png)

## Evaluation Bloom-7B

Config Bloom-7B evaluation script: examples/bloom/evaluate_bloom_7B_ptd.sh

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# modify script model path and tokenizer path
CHECKPOINT="./model_weights/Bloom-7B-Base-v0.1-tp8-pp1/"
TOKENIZER_PATH="./model_from_hf/Bloom-7B-Base/"
# configure task and data path
DATA_PATH="your dataset path"
TASK="your task"
```

Launch Bloom-7B evaluation script:

```bash
bash examples/bloom/evaluate_bloom_7B_ptd.sh
```

Evaluation results

|  dataset | subject_num | question_num | reference_acc |NPU acc|
|:---:|:-----------:|:------------:|:-------------:|:---:|
| MMLU |     57      |    14042     |    0.254    |0.251|



# Bloom-176B

## Training

BLOOM model is from: [A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100).

> Scao, Teven Le, et al. "Bloom: A 176b-parameter open-access multilingual language model." arXiv preprint arXiv:2211.05100 (2022).

Here's a hardware summary of pre-training Bloom-176B:

| **Hardware** | **Value**       |
| ------------ | --------------- |
| NPU          | 12x8 Atlas 900 A2 PODc |


## Script

1. Clone the repository to your local server

    ```shell
    git clone https://gitee.com/ascend/MindSpeed-LLM.git
    cd MindSpeed-LLM
    git checkout 1.1
    cd ..
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    git checkout core_r0.6.0
    cp -r megatron ../MindSpeed-LLM/
    cd ..
    cd MindSpeed-LLM
    mkdir logs
    mkdir model_from_hf
    mkdir dataset
    mkdir ckpt
    ```

2. Build enviroment

    ```bash
    # python3.8
    conda create -n test python=3.8
    conda activate test

    # install torch and torch_npu 
    pip install torch-2.1.0-cp38-cp38m-linux_aarch64.whl
    pip install torch_npu-2.1.0.XXX-cp38-cp38m-linux_aarch64.whl
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

3. Prepare pretrained weights

    Download the Bloom-176B tokensizer from [here](https://huggingface.co/bigscience/bloom/tree/main).

    ```shell
    mkdir ./model_from_hf/Bloom-176B/
    cd ./model_from_hf/Bloom-176B/
    wget https://huggingface.co/bigscience/bloom/resolve/main/special_tokens_map.json
    wget https://huggingface.co/bigscience/bloom/resolve/main/tokenizer.json
    wget https://huggingface.co/bigscience/bloom/resolve/main/tokenizer_config.json
    ...
    cd ../../
    ```

4. Weights convert

    HuggingFace weights --> Megatron weights
    ***(This scenario is generally used to train open-source HuggingFace models on Megatron)***

    ```shell
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader loader_bloom_hf \
        --saver saver_megatron \
        --target-tensor-parallel-size 8 \
        --target-pipeline-parallel-size 5 \
        --load-dir ./model_from_hf/Bloom-176B/ \
        --save-dir ./model_weights/Bloom-176B-v0.1-pt8-pp5/ \
        --tokenizer-model None \
        --params-dtype bf16  
    ```

    Any Megatron weights with parallel slicing strategy --> Any Megatron weights with parallel slicing strategy
    ***(This scenario is generally used to convert the trained megatron model back to the HuggingFace format)***

    ```shell
    # Modify the ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader megatron \
        --saver megatron \
        --save-model-type save_huggingface_llama \
        --load-dir ./model_weights/Bloom-176B-v0.1-pt8-pp5/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --embed-layernorm \
        --params-dtype bf16 \
        --save-dir ./model_from_hf/Bloom-176B/   # <-- Fill in the original HF model path here, new weights will be saved in ./model_from_hf/Bloom-176B/mg2hg/
    ```

5. Prepare dataset

    Download the bloom-176b datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)    

    ```shell
    # download datasets
    cd dataset/
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..

    # process datasets  
    mkdir ./dataset/Bloom-176B/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/Bloom-176B/ \
        --output-prefix ./dataset/Bloom-176B/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF
    ```

6. Config Bloom-176B pre-training script(Bloom-176B does not support Flash Attention): examples/bloom/pretrain_bloom_176b.sh

    ```shell
    # modify MASTER_ADDR to the IP address of the master node in the cluster.
    # the master node is localhost, and the other nodes are the IP address of the master node
    MASTER_ADDR=localhost

    # modify the rank number of a node. The rank number of the master node is 0, and the rank number of other nodes increases in ascending order.
    NODE_RANK=0

    # modify the datasets path and tokenizer path
    TOKENIZER_NAME_OR_PATH=./model_from_hf/Bloom-176B/
    DATA_PATH=./dataset/Bloom-176B/alpaca_text_document
    ```

7. Launch Bloom-176B pre-training script: examples/bloom/pretrain_bloom_176b.sh

    Run the examples/bloom/pretrain_bloom_176b.sh on all nodes in the cluster.

    ```shell
    bash examples/bloom/pretrain_bloom_176b.sh
    ```
    **Note**: If using multi machine training, and no data sharing configuration on the mechines, it's necessary to add the parameter `--no-shared-storage`. This parameter will determine whether non master nodes need to load data based on distributed parameters, and check the corresponding cache and generated data.



## Inference Bloom 176B

Config Bloom-176B inference script: examples/bloom/generate_bloom_176b_ptd.sh

```bash
# modify the script according to your own ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
# modify script model path and tokenizer path
CHECKPOINT="./model_weights/Bloom-176B-v0.1-tp8-pp5/"
TOKENIZER_PATH="./model_from_hf/Bloom-176B/"
```

Launch Bloom-176B inference script: examples/bloom/generate_bloom_176b_ptd.sh

Bloom-176b needs 5 machines to inference, so you need to convert a new model, set
tp=8, pp=5

```bash
bash examples/bloom/generate_bloom_176b_ptd.sh
```

Some inference samples are as follows:

![Inference](https://gitee.com/ascend/MindSpeed-LLM/raw/master/sources/images/bloom/bloom176b-generate.png)

## Evaluation Bloom 176B

Config Bloom-176B evaluation script: examples/bloom/evaluate_bloom_176B_ptd.sh

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# modify script model path and tokenizer path
CHECKPOINT="./model_weights/Bloom-176B-v0.1-tp8-pp5/"
TOKENIZER_PATH="./model_from_hf/Bloom-176B/"
# configure task and data path
DATA_PATH="your dataset path"
TASK="your task"
```

Launch Bloom-176B evaluation script:

```bash
bash examples/bloom/evaluate_bloom_176B_ptd.sh
```

Evaluation results

|  dataset |reference_acc |NPU acc|
|:---:|:-------------:|:---:|
| boolq |  /    |0.645|
