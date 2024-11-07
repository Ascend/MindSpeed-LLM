# ChatGLM
<p align="left">
        <b><a href="README.md">简体中文</a></b> |
        <b>English</b> 
</p>

#  Contents

- [ChatGLM3](#ChatGLM3)
- [Contents](#contents)
- [ChatGLM3-6B](#ChatGLM3-6b)
  - [Training-6B](#training)
    - [Script](#script)
  - [Inference-6B](#inference-6b)
  - [Evaluation-6B](#evaluation-6b)

# ChatGLM3-6B

## Training

Here's a hardware summary of pre-training  ChatGLM3-6B:

| Hardware |                      Value                      |
| :------: | :---------------------------------------------: |
|   NPU    |               8 x Ascend NPUs                   |

### Script

1. Clone the repository to your local server:

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
    pip install -e .
    cd ..
    
    # install other packages
    pip install -r requirements.txt 
    ```
3. Prepare pretrained weights and tokenizer
    Download the ChatGLM3-6B checkpoint from [here](https://huggingface.co/THUDM/chatglm3-6b/tree/main)

    ```shell
    #!/bin/bash
    mkdir ./model_from_hf/chatglm3_6b_hf/
    cd ./model_from_hf/chatglm3_6b_hf/
    wget https://huggingface.co/THUDM/chatglm3-6b-base/resolve/main/config.json
    wget https://huggingface.co/THUDM/chatglm3-6b-base/resolve/main/configuration_chatglm.py
    wget https://huggingface.co/THUDM/chatglm3-6b-base/resolve/main/modeling_chatglm.py
    wget https://huggingface.co/THUDM/chatglm3-6b-base/resolve/main/pytorch_model-00001-of-00007.bin
    wget https://huggingface.co/THUDM/chatglm3-6b-base/resolve/main/pytorch_model-00002-of-00007.bin
    wget https://huggingface.co/THUDM/chatglm3-6b-base/resolve/main/pytorch_model-00003-of-00007.bin
    wget https://huggingface.co/THUDM/chatglm3-6b-base/resolve/main/pytorch_model-00004-of-00007.bin
    wget https://huggingface.co/THUDM/chatglm3-6b-base/resolve/main/pytorch_model-00005-of-00007.bin
    wget https://huggingface.co/THUDM/chatglm3-6b-base/resolve/main/pytorch_model-00006-of-00007.bin
    wget https://huggingface.co/THUDM/chatglm3-6b-base/resolve/main/pytorch_model-00007-of-00007.bin
    wget https://huggingface.co/THUDM/chatglm3-6b-base/resolve/main/pytorch_model.bin.index.json
    wget https://huggingface.co/THUDM/chatglm3-6b-base/resolve/main/quantization.py
    wget https://huggingface.co/THUDM/chatglm3-6b-base/resolve/main/tokenization_chatglm.py
    wget https://huggingface.co/THUDM/chatglm3-6b-base/resolve/main/tokenizer.model
    wget https://huggingface.co/THUDM/chatglm3-6b-base/resolve/main/tokenizer_config.json
    cd ../../
    ```
4. weight conversion in ptd mode

    4.1 Convert weights from HuggingFace format to Megatron format 
    ***（This scenario is generally used to enable the open-source HuggingFace model to be trained on Megatron）***

    ```bash
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # convert to ptd weights
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader chatglm3_hf \
        --saver megatron \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 2 \
        --load-dir ./model_from_hf/chatglm3_6b_hf/ \
        --save-dir ./model_weights/chatglm3_6b_tp1pp2/ \
        --tokenizer-model ./model_from_hf/chatglm3_6b_hf/tokenizer.model \
        --add-qkv-bias
    ```

    Note: The --target-tensor-parallel-size of chatglm3 is related to the multi_query_attention configuration in the config.json, and the multi_query_attention set here is 2.

    4.2 Any Megatron weights with parallel slicing strategy --> Any Megatron weights with parallel slicing strategy

    ```shell
    # Modify the ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader megatron \
        --saver megatron \
        --save-model-type save_huggingface_chatglm3 \
        --load-dir ./model_weights/chatglm3_6b_tp1pp2/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --add-qkv-bias \
        --save-dir ./model_from_hf/chatglm3_6b_hf/     # <-- Fill in the original HF model path here, new weights will be saved in ./model_from_hf/chatglm3_6b_hf/mg2hg/
    ```

5. pre-training

    5.1 Prepare dataset 

    Download the ChatGLM3-6B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

    ```shell
    # download datasets
    cd ./dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    
    # process datasets  
    mkdir ./dataset/chatglm3_6b_hf/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/chatglm3_6b_hf/ \
        --output-prefix ./dataset/chatglm3_6b_hf/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF
    ```

    5.2 pre-training using ptd mode
    Config ChatGLM3-6B pre-training script: examples/chatglm3/pretrain_chatglm3_6B_8K.sh

    ```shell
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    
    # modify config according to your own actual situation
    LOAD_CHECKPOINT_PATH="./model_weights/chatglm3_6b_tp1pp2/"
    SAVE_CHECKPOINT_PATH="./ckpt/chatglm3_6b_hf/"
    TOKENIZER_PATH="./model_from_hf/chatglm3_6b_hf/"  #tokenizer path
    DATA_PATH="./dataset/chatglm3_6b_hf/alpaca_text_document"  #processed dataset
    ```

    Multi-machine training requires the addition of parameter --overlap-grad-reduce

    Launch ChatGLM3-6B  pre-training script: examples/chatglm3/pretrain_chatglm3_6B_8K.sh

    ```shell
    bash examples/chatglm3/pretrain_chatglm3_6B_8K.sh
    ```
    **Note**: If using multi machine training, and no data sharing configuration on the mechines, it's necessary to add the parameter `--no-shared-storage`. This parameter will determine whether non master nodes need to load data based on distributed parameters, and check the corresponding cache and generated data.

6. fine-tuning

    6.1 Prepare fine-tuning dataset
    Download the alpaca datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

    ```shell
    # download datasets
    mkdir finetune_dataset
    cd ./finetune_dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    
    # process datasets  
    mkdir ./finetune_dataset/chatglm3-6b-hf/
    python ./tools/preprocess_data.py \
        --input ./finetune_dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/chatglm3_6b_hf/ \
        --output-prefix ./finetune_dataset/chatglm3-6b-hf/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF \
        --handler-name GeneralInstructionHandler \
        --append-eod
    ```

    6.2 Full Parameters Fine-Tuning
    The configuration script for full parameters fine-tuning  is basically the same as that for pretrain_chatglm3_6B_8K.sh.*The difference is that the dataset and the training parameter is-instruction-dataset are added.*

    Add the fine-tuning parameter `--finetune` so that fine-tuning starts from the first step. Use --tokenizer-padding-side left. 

    ```bash
    DATA_PATH="./finetune_dataset/chatglm3-6b-hf/alpaca"
    TOKENIZER_PATH="./model_from_hf/chatglm3_6b_hf/"
    CKPT_LOAD_DIR="./model_weights/chatglm3_6b_tp1pp2/"
        --load ${CKPT_LOAD_DIR} \
        --finetune \
        --is-instruction-dataset \
        --tokenizer-padding-side left \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-not-use-fast \
    ```

    Launch ChatGLM3-6B finetune script: examples/chatglm3/tune_chatglm3_6B_8K.sh

    ```shell
    bash examples/chatglm3/tune_chatglm3_6B_8K.sh
    ```


## Inference

We support Inference for text generation with ChatGLM3_6B.
Inference different from pre-training, such as we need to Load pre-training checkpoint and the length of the output samples:

Config ChatGLM3-6B inference script: examples/chatglm3/generate_chatglm3_6B.sh

```shell
# modify the model weight path and tokenizer path
CHECKPOINT="./model_weights/chatglm3_6b_tp1pp2/"
TOKENIZER_PATH="./model_from_hf/chatglm3_6b_hf/"
```

Launch ChatGLM3-6B inference script.

```shell
bash ./examples/chatglm3/generate_chatglm3_6B.sh
```

Some inference samples are as follows:
![ChatGLM3-6B-generate.png](https://gitee.com/ascend/ModelLink/raw/master/sources/images/chatglm3/ChatGLM3-6B-generate.png)

## Evaluation

Use mmlu benchmark to evaluate our model. MMLU benchmark Download [here](https://github.com/FranxYao/chain-of-thought-hub/tree/main/MMLU/data/test).

Config chatglm3-6b evaluation script: examples/chatglm3/evaluate_chatglm3_6B.sh

```bash
# ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Modify the model parameter path and vocabulary path
TOKENIZER_PATH="./model_from_hf/chatglm3_6b_hf/"  # vocabulary path
CHECKPOINT="./model_weights/chatglm3_6b_tp2pp4/"  # parameter path

# Configure the task type and dataset path
DATA_PATH="./mmlu/data/test/"
TASK="mmlu"
```

Launch chatglm3-6b evaluation

```bash
bash examples/chatglm3/evaluate_chatglm3_6B.sh
```

| Task | Subset | Question | OpenSource | NPU |
|:---:|:---:|:---:|:-----------------------------------------:|:------:|
| MMLU | 57 | 14042 | [61.4](https://github.com/THUDM/ChatGLM3) |  61.5  |