# GPT3  $\color{black}{\rm\tiny{【model}}$ $\color{black}{\rm\tiny{contributed}}$ $\color{black}{\rm\tiny{by}}$ $\color{black}{\rm\tiny{Community】}}$

<p align="left">
        <b>English</b> |
        <b><a href="README_en.md">English</a> </b> 
</p>

# Contents

- [GPT3](#GPT3)
- [Contents](#contents)
- [GPT3-175B](#GPT3-175B)
  - [Training-175B](#training)
    - [Script](#script)

# GPT3-175B

## Training

Here is a hardware summary of pre-trianing GPT3-175B:

| Hardware |       Value       |
| :--: | :-------------: |
|    NPU   | 128 x Ascend NPUs |

### Script

1. Clone repository to your local server:

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
    mkdir vocab_file
    mkdir dataset
    ```

2. Build environment

    ```bash
    # python3.8
    conda create -n test python=3.8
    conda activate test

    # install torch and torch_npu
    pip install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl
    pip install torch_npu-2.1.0*-cp38-cp38m-linux_aarch64.whl
    pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

    # modify ascend-toolkit path
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

3. Prepare dataset and vocab file for pretrain
    3.1 Prepare dataset
    
    Download the GPT raw dataset from [here](https://huggingface.co/datasets/wikipedia/tree/main/data/20220301.en)
    ```shell
    # download enwiki raw data
    # There are 41 files in total, we can just select part to make our datasets.
    cd ./dataset
    wget https://huggingface.co/datasets/wikipedia/blob/main/data/20220301.en/train-00000-of-00041.parquet
    wget https://huggingface.co/datasets/wikipedia/blob/main/data/20220301.en/train-00001-of-00041.parquet
    wget https://huggingface.co/datasets/wikipedia/blob/main/data/20220301.en/train-00002-of-00041.parquet
    wget https://huggingface.co/datasets/wikipedia/blob/main/data/20220301.en/train-00003-of-00041.parquet
    wget https://huggingface.co/datasets/wikipedia/blob/main/data/20220301.en/train-00004-of-00041.parquet
    wget https://huggingface.co/datasets/wikipedia/blob/main/data/20220301.en/train-00005-of-00041.parquet
    wget https://huggingface.co/datasets/wikipedia/blob/main/data/20220301.en/train-00006-of-00041.parquet
    wget https://huggingface.co/datasets/wikipedia/blob/main/data/20220301.en/train-00007-of-00041.parquet
    cd ..

    # download vocab file and merge table
    cd vocab_file
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
    cd ..

    # process formal dataset
    python ./tools/preprocess_data.py \
        --input ./dataset/ \
        --output-prefix ./dataset/gpt_text_sentence \
        --tokenizer-type GPT2BPETokenizer \
        --vocab-file ./vocab_file/gpt2-vocab.json \
        --merge-file ./vocab_file/gpt2-merges.txt \
        --append-eod \
        --workers 4 \
        --log-interval 1000
    ```

    3.2 pre-training in ptd mode
    Config GPT3-175B PTD pre-training script: examples/gpt3/pretrain_gpt3_175B_ptd.sh

    ```shell
    # modify ascend-toolkit path according to your own config
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    
    # modify config according to your own actual situation
    VOCAB_FILE="./vocab_file/gpt2-vocab.json"   # vocab file for training
    MERGE_FILE="./vocab_file/gpt2-merges.txt"   # BPE merge file for training
    DATA_PATH="./dataset/gpt_text_sentence"  # dataset path
    ```

   Launch GPT3-175B PTD pre-training script: examples/gpt3/pretrain_gpt3_175B_ptd.sh

    ```shell
    bash examples/gpt3/pretrain_gpt3_175B_ptd.sh
    ```

