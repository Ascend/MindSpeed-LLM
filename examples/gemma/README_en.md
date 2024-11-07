# Gemma  $\color{black}{\rm\tiny{【model}}$ $\color{black}{\rm\tiny{contributed}}$ $\color{black}{\rm\tiny{by}}$ $\color{black}{\rm\tiny{Ascend】}}$
<p align="left">
        <b><a href="README.md">简体中文</a></b> |
        <b>English</b> 
</p>

#  Contents

- [Gemma-2B](#Gemma-2B)
  - [Training](#training)
    - [Script](#script)
  - [Inference](#Inference)
  - [Evaluation](#Evaluation)

- [Gemma-7B](#Gemma-7B)
  - [Training](#training)
    - [Script](#script)
  - [Inference](#Inference)
  - [Evaluation](#Evaluation)

# Gemma-2B

## Training

Here's a hardware summary of pre-training  Gemma-2B:

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
    pip install torch-2.2.0-cp38-cp38m-manylinux2014_aarch64.whl
    pip install torch_npu-2.2.0*-cp38-cp38m-linux_aarch64.whl
    pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

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

    Download the Gemma-2B checkpoint from [here](https://huggingface.co/google/gemma-2b/tree/main)

    ```bash
    mkdir ./model_from_hf/Gemma-2B/
    cd ./model_from_hf/Gemma-2B/
    wget https://huggingface.co/google/gemma-2b/resolve/main/config.json
    wget https://huggingface.co/google/gemma-2b/resolve/main/generation_config.json
    wget https://huggingface.co/google/gemma-2b/resolve/main/model-00001-of-00002.safetensors
    wget https://huggingface.co/google/gemma-2b/resolve/main/model-00002-of-00002.safetensors
    wget https://huggingface.co/google/gemma-2b/resolve/main/model.safetensors.index.json
    wget https://huggingface.co/google/gemma-2b/resolve/main/special_tokens_map.json
    wget https://huggingface.co/google/gemma-2b/resolve/main/tokenizer.json
    wget https://huggingface.co/google/gemma-2b/resolve/main/tokenizer.model
    wget https://huggingface.co/google/gemma-2b/resolve/main/tokenizer_config.json
    cd ../../
    ```
4. Weights convert

    Convert weights from huggingface format to megatron format
    ***(This scenario is generally used to train open-source HuggingFace models on Megatron)***

    ```bash
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader gemma_hf \
        --saver megatron \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 2 \
        --load-dir ./model_from_hf/Gemma-2B/ \
        --save-dir ./model_weights/Gemma-2B-v0.1-tp1-pp2/ \
        --tokenizer-model ./model_from_hf/Gemma-2B/tokenizer.model
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
        --load-dir ./model_weights/Gemma-2B-v0.1-tp1-pp2/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --save-dir ./model_from_hf/Gemma-2B/   # Fill in the original HF model path here, new weights will be saved in ./model_from_hf/Gemma-2B/mg2hg/
    ```
5. Prepare dataset

    Download the Gemma-2B datasets from [here](https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered/resolve/main/wikipedia-cn-20230720-filtered.json)

    ```shell
    # download datasets
    cd ./dataset
    wget https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered/resolve/main/wikipedia-cn-20230720-filtered.json
    cd ..

    # process datasets  
    mkdir ./dataset/Gemma-2B/
    python ./tools/preprocess_data.py \
        --input ./dataset/wikipedia-cn-20230720-filtered.json \
        --output-prefix ./dataset/Gemma-2B/wikipedia_cn \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path ./model_from_hf/Gemma-2B/ \
        --json-key completion \
        --workers 16 \
        --log-interval 1000
    ```
6. pre-training

    Config Gemma-2B pre-training script: examples/gemma/pretrain_gemma_2b_ptd.sh

    ```shell
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # modify config according to your own actual situation
    CKPT_SAVE_DIR="./ckpt/Gemma-2B/"
    TOKENIZER_MODEL="./model_from_hf/Gemma-2B/"  #tokenizer path
    DATA_PATH="./dataset/Gemma-2B/wikipedia_cn_completion_document"  #processed dataset
    CKPT_LOAD_DIR="./model_weights/Gemma-2B-v0.1-tp1-pp2/"
    ```

    Launch Gemma-2B pre-training script: examples/gemma/pretrain_gemma_2b_ptd.sh

    ```shell
    bash examples/gemma/pretrain_gemma_2b_ptd.sh 
    ```
    **Note**: If using multi machine training, and no data sharing configuration on the mechines, it's necessary to add the parameter `--no-shared-storage`. This parameter will determine whether non master nodes need to load data based on distributed parameters, and check the corresponding cache and generated data.


7. fine-tuning

    7.1 Prepare fine-tuning dataset
    
    Download the fine-tuning datasets from [here](https://huggingface.co/datasets/fnlp/moss-003-sft-data/tree/main)
    
    ```bash
    mkdir finetune_dataset
    cd ./finetune_dataset
    wget https://huggingface.co/datasets/fnlp/moss-003-sft-data/resolve/main/moss-003-sft-no-tools.jsonl.zip  --no-check-certificate
    unzip moss-003-sft-no-tools.jsonl.zip
    cd ..

    # process datasets  
    mkdir ./finetune_dataset/Gemma-2B/  
    python tools/preprocess_data.py \
        --input ./finetune_dataset/moss-003-sft-no-tools.jsonl \
        --output-prefix ./finetune_dataset/Gemma-2B/moss \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path ./model_from_hf/Gemma-2B/ \
        --tokenizer-not-use-fast \
        --handler-name MOSSInstructionHandler
    ```
   
    7.2 Full Parameters Fine-Tuning

    The configuration script for full parameters fine-tuning  is basically the same as that for pretrain_gemma_2b_ptd.sh.*The difference is that the dataset and the training parameter is-instruction-dataset are added.*
    
    Add the fine-tuning parameter `--finetune` so that fine-tuning starts from the first step.
    ```bash
    CKPT_SAVE_DIR="./ckpt/Gemma-2B/"
    DATA_PATH="./finetune_dataset/Gemma-2B/moss"
    TOKENIZER_PATH="./model_from_hf/Gemma-2B/"
    CKPT_LOAD_DIR="./model_weights/Gemma-2B-v0.1-tp1-pp2/" 
    
        --finetune \
        --is-instruction-dataset \
        --tokenizer-not-use-fast \
    ```


## Inference

Config Gemma-2B inference script: examples/gemma/generate_gemma_2b_ptd.sh

```bash
# ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
# modify script model path and tokenizer path
CHECKPOINT="./model_weights/Gemma-2B-v0.1-tp1-pp2/"
TOKENIZER_PATH="./model_from_hf/Gemma-2B/"
```

Launch Gemma-2B inference script: examples/gemma/generate_gemma_2b_ptd.sh

```bash
bash examples/gemma/generate_gemma_2b_ptd.sh
```

## Evaluation

We use the [MMLU benchmark](https://huggingface.co/datasets/cais/mmlu) to evaluate our model.

Config Gemma-2b evaluation script: examples/gemma/evaluate_gemma_2b_ptd.sh

```bash
# ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Modify the model parameter path and vocabulary path
TOKENIZER_PATH="./model_from_hf/Gemma-2B/"  # vocabulary path
CHECKPOINT="./model_weights/Gemma-2B-v0.1-tp1-pp2/"  # parameter path

# Configure the task type and dataset path
DATA_PATH="./mmlu/data/test/"
TASK="mmlu"
```

Launch Gemma-2B evaluation

```bash
bash examples/gemma/evaluate_gemma_2b_ptd.sh
```

| Task | Subset | Question | OpenSource | NPU  |
|:---:|:---:|:---:|:----------:|:----:|
| MMLU | 57 | 14042 |    39.7    | 39.4 |


# Gemma-7B

## Training

Here's a hardware summary of pre-training  Gemma-7B:

| Hardware |                      Value                      |
| :------: | :---------------------------------------------: |
|   NPU    |               8 x Ascend NPUs                   |

### Script

1. Clone the repository to your local server:

    ```shell
    git clone https://gitee.com/ascend/ModelLink.git
    cd ModelLink
    git checkout 1.1
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
    pip install torch-2.2.0-cp38-cp38m-manylinux2014_aarch64.whl
    pip install torch_npu-2.2.0*-cp38-cp38m-linux_aarch64.whl
    pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

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
    Download the Gemma-7B checkpoint from [here](https://huggingface.co/Gemma/Gemma-7B/tree/main)

    ```bash
    mkdir ./model_from_hf/Gemma-7B/
    cd ./model_from_hf/Gemma-7B/
    wget https://huggingface.co/google/gemma-7b/resolve/main/config.json
    wget https://huggingface.co/google/gemma-7b/resolve/main/generation_config.json
    wget https://huggingface.co/google/gemma-7b/resolve/main/model-00001-of-00004.safetensors
    wget https://huggingface.co/google/gemma-7b/resolve/main/model-00002-of-00004.safetensors
    wget https://huggingface.co/google/gemma-7b/resolve/main/model-00003-of-00004.safetensors
    wget https://huggingface.co/google/gemma-7b/resolve/main/model-00004-of-00004.safetensors
    wget https://huggingface.co/google/gemma-7b/resolve/main/model.safetensors.index.json
    wget https://huggingface.co/google/gemma-7b/resolve/main/special_tokens_map.json
    wget https://huggingface.co/google/gemma-7b/resolve/main/tokenizer.json
    wget https://huggingface.co/google/gemma-7b/resolve/main/tokenizer.model
    wget https://huggingface.co/google/gemma-7b/resolve/main/tokenizer_config.json
    cd ../../
    ```
4. Weights convert

    Convert weights from huggingface format to megatron format
    ***(This scenario is generally used to train open-source HuggingFace models on Megatron)***

    ```bash
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader gemma_hf \
        --saver megatron \
        --target-tensor-parallel-size 8 \
        --target-pipeline-parallel-size 1 \
        --load-dir ./model_from_hf/Gemma-7B/ \
        --save-dir ./model_weights/Gemma-7B-v0.1-tp8-pp1/ \
        --tokenizer-model ./model_from_hf/Gemma-7B/tokenizer.model
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
        --load-dir ./model_weights/Gemma-7B-v0.1-tp8-pp1/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --save-dir ./model_from_hf/Gemma-7B/   # Fill in the original HF model path here, new weights will be saved in ./model_from_hf/Gemma-7B/mg2hg/
    ```
5. Prepare dataset

    Download the Gemma-7B datasets from [here](https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered/resolve/main/wikipedia-cn-20230720-filtered.json)

    ```shell
    # download datasets
    cd ./dataset
    wget https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered/resolve/main/wikipedia-cn-20230720-filtered.json
    cd ..

    # process datasets  
    mkdir ./dataset/Gemma-7B/
    python ./tools/preprocess_data.py \
        --input ./dataset/wikipedia-cn-20230720-filtered.json \
        --output-prefix ./dataset/Gemma-7B/wikipedia_cn \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path ./model_from_hf/Gemma-7B/ \
        --json-key completion \
        --workers 16 \
        --log-interval 1000
    ```
6. pre-training

    Config Gemma-7B pre-training script: examples/gemma/pretrain_gemma_7b_ptd.sh

    ```shell
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # modify config according to your own actual situation
    CKPT_SAVE_DIR="./ckpt/Gemma-7B/"
    TOKENIZER_MODEL="./model_from_hf/Gemma-7B/"  #tokenizer path
    DATA_PATH="./dataset/Gemma-7B/wikipedia_cn_completion_document"  #processed dataset
    CKPT_LOAD_DIR="./model_weights/Gemma-7B-v0.1-tp8-pp1/"
    ```

    Launch Gemma-7B pre-training script: examples/gemma/pretrain_gemma_7b_ptd.sh

    ```shell
    bash examples/gemma/pretrain_gemma_7b_ptd.sh 
    ```
    **Note**: If using multi machine training, and no data sharing configuration on the mechines, it's necessary to add the parameter `--no-shared-storage`. This parameter will determine whether non master nodes need to load data based on distributed parameters, and check the corresponding cache and generated data.


7. fine-tuning

    7.1 Prepare fine-tuning dataset
    
    Download the fine-tuning datasets from [here](https://huggingface.co/datasets/fnlp/moss-003-sft-data/tree/main)
    
    ```bash
    mkdir finetune_dataset
    cd ./finetune_dataset
    wget https://huggingface.co/datasets/fnlp/moss-003-sft-data/resolve/main/moss-003-sft-no-tools.jsonl.zip  --no-check-certificate
    unzip moss-003-sft-no-tools.jsonl.zip
    cd ..

    # process datasets  
    mkdir ./finetune_dataset/Gemma-7B/  
    python tools/preprocess_data.py \
        --input ./finetune_dataset/moss-003-sft-no-tools.jsonl \
        --output-prefix ./finetune_dataset/Gemma-7B/moss \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path ./model_from_hf/Gemma-7B/ \
        --tokenizer-not-use-fast \
        --handler-name MOSSInstructionHandler
    ```
   
    7.2 Full Parameters Fine-Tuning

    The configuration script for full parameters fine-tuning  is basically the same as that for pretrain_gemma_7b_ptd.sh.*The difference is that the dataset and the training parameter is-instruction-dataset are added.*
    
    Add the fine-tuning parameter `--finetune` so that fine-tuning starts from the first step.
    ```bash
    CKPT_SAVE_DIR="./ckpt/Gemma-7B/"
    DATA_PATH="./finetune_dataset/Gemma-7B/moss"
    TOKENIZER_PATH="./model_from_hf/Gemma-7B/"
    CKPT_LOAD_DIR="./model_weights/Gemma-7B-v0.1-tp8-pp1/" 
    
        --finetune \
        --is-instruction-dataset \
        --tokenizer-not-use-fast \
    ```
   
    7.3 Lora Fine-Tuning

    The Lora fine-tuning script is configured by adding the following lora parameters to the pretrain_gemma_7b_ptd.sh script:

    ```bash
        --lora-target-modules query_key_value dense dense_h_to_4h dense_4h_to_h \
        --lora-r 16 \
        --lora-alpha 32 \
    ```

    If the vocabulary is changed, add the following parameters:

    ```bash
        --lora-modules-to-save word_embeddings output_layer \
    ```
   
    The following parameters are added to the resumable training capability of Lora:

    ```bash
        --load ${ORIGIN_CHECKPOINT}  \
        --lora-load ${LORA_CHECKPOINT} \
    ```
   
    Launch Gemma-7B lora fine tune script: examples/finetune/tune_gemma_7b_ptd.sh

    ```shell
    bash examples/gemma/tune_gemma_7b_ptd.sh 
    ```


## Inference

Config Gemma-7B inference script: examples/gemma/generate_gemma_7b_ptd.sh

```bash
# ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
# modify script model path and tokenizer path
CHECKPOINT="./model_weights/Gemma-7B-v0.1-tp8-pp1/"
TOKENIZER_PATH="./model_from_hf/Gemma-7B/"
```

Config Gemma-7B lora inference script: examples/gemma/generate_gemma_7b_lora_ptd.sh

```bash
# modify lora model path
CHECKPOINT_LORA="your lora model directory path"
```

Launch Gemma-7B inference script: examples/gemma/generate_gemma_7b_ptd.sh

```bash
bash examples/gemma/generate_gemma_7b_ptd.sh
```

Launch Gemma-7B lora inference script: examples/gemma/generate_gemma_7b_lora_ptd.sh

```bash
bash examples/gemma/generate_gemma_7b_lora_ptd.sh
```

Some lora inference samples are as follows:
![Inference](https://gitee.com/ascend/ModelLink/raw/master/sources/images/gemma/gemma-7b-lora-inference.jpg)

## Evaluation

We use the [MMLU benchmark](https://huggingface.co/datasets/cais/mmlu) to evaluate our model.

Config Gemma-7B evaluation script: examples/gemma/evaluate_gemma_7b_ptd.sh

```bash
# ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Modify the model parameter path and vocabulary path
TOKENIZER_PATH="./model_from_hf/Gemma-7B/"  # vocabulary path
CHECKPOINT="./model_weights/Gemma-7B-v0.1-tp8-pp1/"  # parameter path

# Configure the task type and dataset path
DATA_PATH="./mmlu/data/test/"
TASK="mmlu"
```

Launch Gemma-7B evaluation

```bash
bash examples/gemma/evaluate_gemma_7b_ptd.sh
```

| Task | Subset | Question | OpenSource | NPU  |
|:---:|:---:|:---:|:----------:|:----:|
| MMLU | 57 | 14042 |    52.2    | 52.2 |