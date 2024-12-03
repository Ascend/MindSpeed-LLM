# LLaMA3  $\color{black}{\rm\tiny{【model}}$ $\color{black}{\rm\tiny{contributed}}$ $\color{black}{\rm\tiny{by}}$ $\color{black}{\rm\tiny{Ascend】}}$
<p align="left">
        <b><a href="README.md">简体中文</a></b> |
        <b>English</b> 
</p>

#  Contents

- [LLaMA3](#llama)
- [Contents](#contents)
- [LLAMA3-8B](#llama3-8b)
  - [Training](#training)
    - [Script](#script)
  - [Inference-8B](#inference-8b)
  - [Evaluation-8B](#evaluation-8b)
- [Contents](#contents)
- [LLAMA3-70B](#llama3-70b)
  - [Training](#training)
    - [Script](#script)
  - [Inference-70B](#inference-70b)
  - [Evaluation-70B](#evaluation-70b)

# LLAMA3-8B

## Training

Here's a hardware summary of pre-training  LLAMA3-8B:

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

    *Note that if you want to train with the weight from huggingface, please run fix a deepspeed loading checkpointing bug by modified `if zero_sd_list is None` as `if zero_sd_list is None or len(zero_sd_list) == 0` in the `_load_zero_checkpoint` function of `<deepspeed-installed-path>/runtime/engine.py`*

    ```text
    # original deepspeed/runtime/engine.py, about #Lines2746-2748
    zero_sd_list = self._get_all_zero_checkpoints(load_dir, tag)
    if zero_sd_list is None:
        return False
    
    # modified
    zero_sd_list = self._get_all_zero_checkpoints(load_dir, tag)
    if zero_sd_list is None or len(zero_sd_list) == 0:
        return False
    ```
3. Prepare pretrained weights and tokenizer
    Download the LLAMA3-8B checkpoint from [here](https://huggingface.co/unsloth/llama-3-8B/tree/main)

    ```shell
    #!/bin/bash
    mkdir ./model_from_hf/llama-3-8b-hf/
    cd ./model_from_hf/llama-3-8b-hf/
    wget https://huggingface.co/meta-llama/Meta-Llama-3-8B/raw/main/config.json
    wget https://huggingface.co/meta-llama/Meta-Llama-3-8B/raw/main/generation_config.json
    wget https://huggingface.co/meta-llama/Meta-Llama-3-8B/raw/main/model-00001-of-00004.safetensors
    wget https://huggingface.co/meta-llama/Meta-Llama-3-8B/raw/main/model-00002-of-00004.safetensors
    wget https://huggingface.co/meta-llama/Meta-Llama-3-8B/raw/main/model-00003-of-00004.safetensors
    wget https://huggingface.co/meta-llama/Meta-Llama-3-8B/raw/main/model-00004-of-00004.safetensors
    wget https://huggingface.co/meta-llama/Meta-Llama-3-8B/raw/main/model.safetensors.index.json
    wget https://huggingface.co/meta-llama/Meta-Llama-3-8B/raw/main/special_tokens_map.json
    wget https://huggingface.co/meta-llama/Meta-Llama-3-8B/raw/main/tokenizer.json
    wget https://huggingface.co/meta-llama/Meta-Llama-3-8B/raw/main/tokenizer_config.json
    cd ../../
    ```
4. weight conversion in ptd mode

    *Note that if you want to use the weight from huggingface, please run the weight conversion script first. The following uses llama-3-8b model weight conversion in ptd as an example.*

    ```bash
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    
    # convert to ptd weights
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader llama2_hf \
        --saver megatron \
        --target-tensor-parallel-size 8 \
        --target-pipeline-parallel-size 1 \
        --load-dir ./model_from_hf/llama-3-8b-hf/ \
        --save-dir ./model_weights/llama-3-8b-hf-v0.1-tp8-pp1/ \
        --tokenizer-model ./model_from_hf/llama-3-8b-hf/tokenizer.json
    ```

    Any Megatron weights with parallel slicing strategy --> Any Megatron weights with parallel slicing strategy

    ```shell
    # Modify the ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader megatron \
        --saver megatron \
        --save-model-type save_huggingface_llama \
        --load-dir ./model_weights/llama-3-8b-hf-v0.1-tp8-pp1/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --save-dir ./model_from_hf/llama-3-8b-hf/  # <-- Fill in the original HF model path here, new weights will be saved in ./model_from_hf/llama-3-8b-hf/mg2hg/
    ```

    Weight conversion is suitable for pre-training, fine-tuning, inference and evaluation. Adjust the parameters `target-tensor-parallel-size` and `target-pipeline-parallel-size` according to different tasks.
5. pre-training

    5.1 Prepare dataset

    Download the LLAMA3-8B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

    ```shell
    # download datasets
    cd ./dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..

    # process datasets  
    mkdir ./dataset/llama-3-8b-hf/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/llama-3-8b-hf/ \
        --output-prefix ./dataset/llama-3-8b-hf/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF
    ```

    5.2 pre-training using ptd mode
    Config LLAMA3-8B pre-training script: examples/llama3/pretrain_llama3_8b_ptd.sh

    ```shell
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # modify config according to your own actual situation
    CKPT_SAVE_DIR="./ckpt/llama-3-8b-hf/"
    TOKENIZER_MODEL="./model_from_hf/llama-3-8b-hf/"  #tokenizer path
    DATA_PATH="./dataset/llama-3-8b-hf/alpaca_text_document"  #processed dataset
    CKPT_LOAD_DIR="./model_weights/llama-3-8b-hf-v0.1-tp8-pp1/" #weight path
    ```

    Multi-machine training requires the addition of parameter --overlap-grad-reduce

    Launch LLAMA3-8B  pre-training script: examples/llama3/pretrain_llama3_8b_ptd.sh

    ```shell
    bash examples/llama3/pretrain_llama3_8b_ptd.sh 
    ```
    **Note**: If using multi machine training, and no data sharing configuration on the mechines, it's necessary to add the parameter `--no-shared-storage`. This parameter will determine whether non master nodes need to load data based on distributed parameters, and check the corresponding cache and generated data.

6. fine-tuning

    6.1 Prepare fine-tuning dataset
    
    Download the LLAMA3-8B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

    ```shell
    # download datasets
    mkdir finetune_dataset
    cd ./finetune_dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    
    # process datasets  
    mkdir ./finetune_dataset/llama-3-8b-hf/
    python ./tools/preprocess_data.py \
        --input ./finetune_dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/llama-3-8b-hf/ \
        --output-prefix ./finetune_dataset/llama-3-8b-hf/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF \
        --handler-name GeneralInstructionHandler \
        --append-eod
    ```

    6.2 Full Parameters Fine-Tuning
    
    The configuration script for full parameters fine-tuning  is basically the same as that for pretrain_llama3_8b_ptd.sh.*The difference is that the dataset and the training parameter is-instruction-dataset are added.*
    Add the fine-tuning parameter `--finetune` so that fine-tuning starts from the first step.

    ```bash
    DATA_PATH="./finetune_dataset/llama-3-8b-hf/alpaca"
    TOKENIZER_PATH="./model_from_hf/llama-3-8b-hf/"
    CKPT_PATH="./ckpt/llama-3-8b-hf/"
        --load ${CKPT_PATH} \
        --finetune \
        --is-instruction-dataset \
        --tokenizer-not-use-fast \
    ```



## Inference-8B

Config llama3-8B inference script: examples/llama3/generate_llama3_8b_ptd.sh

```bash
# modify the script according to your own ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
# modify script model path and tokenizer path
CHECKPOINT="./model_weights/llama-3-8b-hf-v0.1-tp8-pp1"
TOKENIZER_PATH="./model_from_hf/llama-3-8b-hf/"
```

Launch llama3-8B inference script: examples/llama3/generate_llama3_8b_ptd.sh

```bash
bash examples/llama3/generate_llama3_8b_ptd.sh
```

**When want to chat with Llama3，please download Instruct/Chat weights，and convert to Megatron weights. Then launch the chat script: examples/llama3/generate_llama3_8b_chat_ptd.sh**

Download the LLAMA3-8B-Instruct weight from [here](https://huggingface.co/unsloth/llama-3-8b-Instruct)

Chat example：
```
You >> 你好
MindSpeed-LLM:
你好！我是你的中文语言模型，很高兴和你交流！有什么想聊的题目或问题，请随时问我！

You >> 介绍一下llama3

MindSpeed-LLM:
LLaMA 3 is a large-scale language model developed by Meta AI, a leading artificial intelligence research organization. It's a significant upgrade to the previous LLaMA models, offering improved performance and capabilities.

Here are some key features and facts about LLaMA 3:

Scale: LLaMA 3 is a massive model, consisting of 13.5 billion parameters, making it one of the largest language models in the world.
Architecture: It's based on the Transformer architecture, which is a type of neural network designed specifically for natural language processing tasks.
Training data: LLaMA 3 was trained on a massive dataset of text from the internet, including books, articles, and websites. This training data allows the model to learn about various topics, styles, and language patterns.
Language understanding: LLaMA 3 is capable of understanding natural language, including nuances, idioms, and figurative language. It can also recognize and generate text in multiple languages.
Conversational capabilities: The model can engage in conversations, answering questions, and responding to prompts. It can also generate text based on a given topic or theme.
Creative writing: LLaMA
You >> 能用中文回答吗

MindSpeed-LLM:

LLaMA 3 是 Meta AI 开发的一种大规模语言模型，是之前 LLama 模型的升级版本，提供了改进的性能和能力。

以下是 LLama 3 的一些关键特点和事实：

规模：LLaMA 3 是一个庞大的模型，具有 13.5 亿参数，世界上最大的语言模型之一。
架构：它基于 Transformer 架构，这是一种专门为自然语言处理任务设计的神经网络。
训练数据：LLaMA 3 在互联网上的大量文本数据上进行了训练，包括书籍、文章和网站。这使得模型能够学习各种主题、风格和语言模式。
语言理解：LLaMA 3 能够理解自然语言，包括 nuance、idiom 和 figurative language。它也能识别和生成多种语言的文本。
对话能力：模型能够参与对话，回答问题和响应提示。它也能基于给定的主题或主题生成文本。
创作写作：LLa
```


## Evaluation-8B

We use MMLU benchmark to evaluate our model. Benchmark Download [here](https://huggingface.co/datasets/cais/mmlu).
Config llama3-8B evaluation script: examples/llama3/evaluate_llama3_8b_ptd.sh

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# modify script model path and tokenizer path
TOKENIZER_PATH="./model_from_hf/llama-3-8b-hf/"  #tokenizer path
CHECKPOINT="./model_weights/llama-3-8b-hf-v0.1-tp8-pp1"  #model path
# configure task and data path
DATA_PATH="./mmlu/data/test/"
TASK="mmlu"
```

Launch llama3-8B evaluation script:

```bash
bash examples/llama3/evaluate_llama3_8b_ptd.sh
```

Evaluation results

|  dataset | subject_num | question_num | reference_acc |NPU acc|
|:---:|:-----------:|:------------:|:-------------:|:---:|
| MMLU |     57      |    14042     |    0.666     |0.653|

# LLAMA3-70B

## Training

Here's a hardware summary of pre-training  LLAMA3-70B:

| Hardware |                      Value                      |
| :------: | :---------------------------------------------: |
|   NPU    |               64 x Ascend NPUs                   |

### Script

1. Clone the repository to your local server:

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

    *Note that if you want to train with the weight from huggingface, please run fix a deepspeed loading checkpointing bug by modified `if zero_sd_list is None` as `if zero_sd_list is None or len(zero_sd_list) == 0` in the `_load_zero_checkpoint` function of `<deepspeed-installed-path>/runtime/engine.py`*

    ```text
    # original deepspeed/runtime/engine.py, about #Lines2746-2748
    zero_sd_list = self._get_all_zero_checkpoints(load_dir, tag)
    if zero_sd_list is None:
        return False
    
    # modified
    zero_sd_list = self._get_all_zero_checkpoints(load_dir, tag)
    if zero_sd_list is None or len(zero_sd_list) == 0:
        return False
    ```
3. Prepare pretrained weights and tokenizer
    Download the LLAMA3-70B checkpoint from [here](https://huggingface.co/v2ray/Llama-3-70B/tree/main)

    ```shell
    #!/bin/bash
    mkdir ./model_from_hf/llama-3-70b-hf/
    cd ./model_from_hf/llama-3-70b-hf/
    wget https://huggingface.co/v2ray/Llama-3-70B/raw/main/config.json
    wget https://huggingface.co/v2ray/Llama-3-70B/raw/main/generation_config.json
    wget https://huggingface.co/v2ray/Llama-3-70B/raw/main/model-00001-of-00030.safetensors
    wget https://huggingface.co/v2ray/Llama-3-70B/raw/main/model-00002-of-00030.safetensors
    wget https://huggingface.co/v2ray/Llama-3-70B/raw/main/model-00003-of-00030.safetensors
    wget https://huggingface.co/v2ray/Llama-3-70B/raw/main/model-00004-of-00030.safetensors
    ...
    wget https://huggingface.co/v2ray/Llama-3-70B/raw/main/model-00030-of-00030.safetensors
    wget https://huggingface.co/v2ray/Llama-3-70B/raw/main/model.safetensors.index.json
    wget https://huggingface.co/v2ray/Llama-3-70B/raw/main/special_tokens_map.json
    wget https://huggingface.co/v2ray/Llama-3-70B/raw/main/tokenizer.json
    wget https://huggingface.co/v2ray/Llama-3-70B/raw/main/tokenizer_config.json
    cd ../../
    ```
4. weight conversion in ptd mode

    *Note that if you want to use the weight from huggingface, please run the weight conversion script first. The following uses llama-3-70b model weight conversion in ptd as an example.*

    ```bash
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    
    # convert to ptd weights
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader llama2_hf \
        --saver megatron \
        --target-tensor-parallel-size 8 \
        --target-pipeline-parallel-size 8 \
        --load-dir ./model_from_hf/llama-3-70b-hf/ \
        --save-dir ./model_weights/llama-3-70b-hf-v0.1-tp8-pp8/ \
        --tokenizer-model ./model_from_hf/llama-3-70b-hf/tokenizer.json
    ```

    Any Megatron weights with parallel slicing strategy --> Any Megatron weights with parallel slicing strategy

    ```shell
    # Modify the ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader megatron \
        --saver megatron \
        --save-model-type save_huggingface_llama \
        --load-dir ./model_weights/llama-3-70b-hf-v0.1-tp8-pp8/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --save-dir ./model_from_hf/llama-3-70b-hf/  # <-- Fill in the original HF model path here, new weights will be saved in ./model_from_hf/llama-3-70b-hf/mg2hg/
    ```

    Weight conversion is suitable for pre-training, fine-tuning, inference and evaluation. Adjust the parameters `target-tensor-parallel-size` and `target-pipeline-parallel-size` according to different tasks.
5. pre-training

    5.1 Prepare dataset

    Download the LLAMA3-70B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

    ```shell
    # download datasets
    cd ./dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..

    # process datasets  
    mkdir ./dataset/llama-3-70b-hf/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/llama-3-70b-hf/ \
        --output-prefix ./dataset/llama-3-70b-hf/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF
    ```

    5.2 pre-training using ptd mode
    Config LLAMA3-70B pre-training script: examples/llama3/pretrain_llama3_70b_ptd.sh

    ```shell
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # modify config according to your own actual situation
    CKPT_SAVE_DIR="./ckpt/llama-3-70b-hf/"
    TOKENIZER_MODEL="./model_from_hf/llama-3-70b-hf/"  #tokenizer path
    DATA_PATH="./dataset/llama-3-70b-hf/alpaca_text_document"  #processed dataset
    CKPT_LOAD_DIR="./model_weights/llama-3-70b-hf-v0.1-tp8-pp8/" #weight path
    ```

    Multi-machine training requires the addition of parameter --overlap-grad-reduce

    Launch LLAMA3-70B  pre-training script: examples/llama3/pretrain_llama3_70b_ptd.sh

    ```shell
    bash examples/llama3/pretrain_llama3_70b_ptd.sh 
    ```
    **Note**: If using multi machine training, and no data sharing configuration on the mechines, it's necessary to add the parameter `--no-shared-storage`. This parameter will determine whether non master nodes need to load data based on distributed parameters, and check the corresponding cache and generated data.

6. fine-tuning

    6.1 Prepare fine-tuning dataset
    
    Download the LLAMA3-70B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

    ```shell
    # download datasets
    mkdir finetune_dataset
    cd ./finetune_dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    
    # process datasets  
    mkdir ./finetune_dataset/llama-3-70b-hf/
    python ./tools/preprocess_data.py \
        --input ./finetune_dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/llama-3-70b-hf/ \
        --output-prefix ./finetune_dataset/llama-3-70b-hf/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF \
        --handler-name GeneralInstructionHandler \
        --append-eod
    ```

    6.2 Full Parameters Fine-Tuning
    
    The configuration script for full parameters fine-tuning  is basically the same as that for pretrain_llama3_70b_ptd.sh.*The difference is that the dataset and the training parameter is-instruction-dataset are added.*
    Add the fine-tuning parameter `--finetune` so that fine-tuning starts from the first step.

    ```bash
    DATA_PATH="./finetune_dataset/llama-3-70b-hf/alpaca"
    TOKENIZER_PATH="/model_from_hf/llama-3-70b-hf/"
    CKPT_PATH="./ckpt"
        --load ${CKPT_PATH} \
        --finetune \
        --is-instruction-dataset \
        --tokenizer-not-use-fast \
    ```



## Inference-70B

Config llama3-70B inference script: examples/llama3/generate_llama3_70b_ptd.sh

```bash
# modify the script according to your own ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
# modify script model path and tokenizer path
CHECKPOINT="./model_weights/llama-3-70b-hf-v0.1-tp8-pp1"
TOKENIZER_PATH="./model_from_hf/llama-3-70b-hf/"
```

Launch llama3-70B inference script: examples/llama3/generate_llama3_70b_ptd.sh

```bash
bash examples/llama3/generate_llama3_70b_ptd.sh
```


## Evaluation-70B

We use MMLU benchmark to evaluate our model. Benchmark Download [here](https://huggingface.co/datasets/cais/mmlu).
Config llama3-70B evaluation script: examples/llama3/evaluate_llama3_70b_ptd.sh

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# modify script model path and tokenizer path
TOKENIZER_PATH="./model_from_hf/llama-3-70b-hf/"  #tokenizer path
CHECKPOINT="./model_weights/llama-3-70b-hf-v0.1-tp8-pp1"  #model path
# configure task and data path
DATA_PATH="./mmlu/data/test/"
TASK="mmlu"
```

Launch llama3-70B evaluation script:

```bash
bash examples/llama3/evaluate_llama3_70b_ptd.sh
```

Evaluation results

|  dataset | subject_num | question_num | reference_acc |NPU acc|
|:---:|:-----------:|:------------:|:-------------:|:-------:|
| MMLU |     57      |    14042     |    0.795     |  0.783  |
