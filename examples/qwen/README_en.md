# Qwen
<p align="left">
        <b><a href="README.md">简体中文</a></b> |
        <b>English</b> 
</p>

#  Contents

- [Qwen-7B](#Qwen-7B)
  - [Training](#training)
  - [Script](#script)
  - [Inference](#Inference)
  - [Evaluation](#Evaluation)

- [Qwen-14B](#Qwen-14B)
  - [Training](#training)
  - [Script](#script)
  - [Inference](#Inference)
  - [Evaluation](#Evaluation)

- [Qwen-72B](#Qwen-72B)
  - [Training](#training)
  - [Script](#script)
  - [Inference](#Inference)
  - [Evaluation](#Evaluation)

# Qwen-7B

## Training

Here's a hardware summary of pre-training  Qwen-7B:

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
    Download the Qwen-7B checkpoint from [here](https://huggingface.co/Qwen/Qwen-7B/tree/main)

    ```bash
    mkdir ./model_from_hf/Qwen-7B/
    cd ./model_from_hf/Qwen-7B/
    wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/cache_autogptq_cuda_256.cpp
    wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/cache_autogptq_cuda_kernel_256.cu
    wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/config.json
    wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/configuration_qwen.py
    wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/cpp_kernels.py
    wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/generation_config.json
    wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/model-00001-of-00008.safetensors
    wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/model-00002-of-00008.safetensors
    wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/model-00003-of-00008.safetensors
    wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/model-00004-of-00008.safetensors
    wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/model-00005-of-00008.safetensors
    wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/model-00006-of-00008.safetensors
    wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/model-00007-of-00008.safetensors
    wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/model-00008-of-00008.safetensors
    wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/model.safetensors.index.json
    wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/modeling_qwen.py
    wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/qwen.tiktoken
    wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/qwen_generation_utils.py
    wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/tokenization_qwen.py
    wget https://huggingface.co/Qwen/Qwen-7B/resolve/main/tokenizer_config.json
    cd ../../
    ```

    Modify line 39 in the modelling_qwen.py file, changing:

    ```python
    SUPPORT_FP16 = SUPPORT_CUDA and torch.cuda.get_device_capability(0)[0] >= 7
    ```

    to：

    ```python
    SUPPORT_FP16 = True
    ```
4. Weights convert

    Convert weights from huggingface format to megatron format
    ***(This scenario is generally used to train open-source HuggingFace models on Megatron)***

    ```bash
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader qwen_hf \
        --saver megatron \
        --target-tensor-parallel-size 8 \
        --load-dir ./model_from_hf/Qwen-7B/ \
        --save-dir ./model_weights/Qwen-7B-v0.1-tp8-pp1/ \
        --tokenizer-model ./model_from_hf/Qwen-7B/qwen.tiktoken \
        --add-qkv-bias
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
        --save-model-type save_huggingface_qwen \
        --load-dir ./model_weights/Qwen-7B-v0.1-tp8-pp1/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --add-qkv-bias \
        --save-dir ./model_from_hf/Qwen-7B/   # Fill in the original HF model path here, new weights will be saved in ./model_from_hf/Qwen-7B/mg2hg/
    ```
5. Prepare dataset

    Download the Qwen-7B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

    ```shell
    # download datasets
    cd ./dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    
    # process datasets  
    mkdir ./dataset/Qwen-7B/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/Qwen-7B/ \
        --output-prefix ./dataset/Qwen-7B/alpaca \
        --tokenizer-type PretrainedFromHF \
        --seq-length 8192 \
        --workers 4 \
        --log-interval 1000
    ```
6. pre-training

    Config Qwen-7B pre-training script: examples/qwen/pretrain_qwen_7b_ptd.sh

    ```shell
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    
    # modify config according to your own actual situation
    CKPT_SAVE_DIR="./ckpt/Qwen-7B/"
    TOKENIZER_MODEL="./model_from_hf/Qwen-7B/"  #tokenizer path
    DATA_PATH="./dataset/Qwen-7B/alpaca_text_document"  #processed dataset
    CKPT_LOAD_DIR="./model_weights/Qwen-7B-v0.1-tp8-pp1/"
    ```

    Config Qwen-7B pre-training script: examples/qwen/pretrain_qwen_7b_ptd.sh

    ```shell
    bash examples/qwen/pretrain_qwen_7b_ptd.sh 
    ```
    **Note**: If using multi machine training, and no data sharing configuration on the mechines, it's necessary to add the parameter `--no-shared-storage`. This parameter will determine whether non master nodes need to load data based on distributed parameters, and check the corresponding cache and generated data.


## Inference

Config qwen-7b inference script: examples/qwen/generate_qwen_7b_ptd.sh

```bash
# ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
# modify script model path and tokenizer path
CHECKPOINT="./model_weights/Qwen-7B-v0.1-tp8-pp1/"
TOKENIZER_PATH="./model_from_hf/Qwen-7B/"
```

Launch qwen-7b inference script: examples/qwen/generate_qwen_7b_ptd.sh

```bash
bash examples/qwen/generate_qwen_7b_ptd.sh
```

**Note**: If using multi machine training, it is necessary to set up multi machine data sharing, and non primary nodes can read the primary node data through data sharing. Alternatively, directly copy the data generated by the master node to non master nodes.

Some inference samples are as follows:
![Inference](https://gitee.com/ascend/ModelLink/raw/master/sources/images/qwen/qwen_7b_inference.png)

## Evaluation

We use the [CEval benchmark](https://huggingface.co/datasets/ceval/ceval-exam) and [MMLU benchmark](https://huggingface.co/datasets/cais/mmlu) to evaluate our model.

Config qwen-7b evaluation script: examples/qwen/evaluate_qwen_7b_ptd.sh

```bash
# ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Modify the model parameter path and vocabulary path
TOKENIZER_PATH="./model_from_hf/Qwen-7B/"  # vocabulary path
CHECKPOINT="./model_weights/Qwen-7B-v0.1-tp8-pp1/"  # parameter path

# Configure the task type and dataset path
DATA_PATH="./mmlu/data/test/"  # "./ceval/val/" for ceval task
TASK="mmlu"  # "ceval" for ceval task
```

Launch qwen-7b evaluation

```bash
bash examples/qwen/evaluate_qwen_7b_ptd.sh
```

| Task | Subset | Question | OpenSource | NPU |
|:---:|:---:|:---:|:---:|:---:|
| CEval | 52 | 1346 | 63.5 | 62.5 |
| MMLU | 57 | 14042 | 58.2 | 58.1 |

# Qwen-14B

## Training

Here's a hardware summary of pre-training  Qwen-14B:

| Hardware |      Value      |
| :------: | :-------------: |
|   NPU   | 8 x Ascend NPUs |

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
    pip install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl
    pip install torch_npu-2.1.0*-cp38-cp38m-linux_aarch64.whl
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
   Download the Qwen-14B checkpoint from [here](https://huggingface.co/Qwen/Qwen-14B/tree/main)

   ```bash
   mkdir ./model_from_hf/Qwen-14B/
   cd ./model_from_hf/Qwen-14B/
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/cache_autogptq_cuda_256.cpp
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/cache_autogptq_cuda_kernel_256.cu
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/config.json
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/configuration_qwen.py
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/cpp_kernels.py
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/generation_config.json
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/model-00001-of-00015.safetensors
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/model-00002-of-00015.safetensors
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/model-00003-of-00015.safetensors
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/model-00004-of-00015.safetensors
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/model-00005-of-00015.safetensors
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/model-00006-of-00015.safetensors
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/model-00007-of-00015.safetensors
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/model-00008-of-00015.safetensors
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/model-00009-of-00015.safetensors
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/model-00010-of-00015.safetensors
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/model-00011-of-00015.safetensors
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/model-00012-of-00015.safetensors
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/model-00013-of-00015.safetensors
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/model-00014-of-00015.safetensors
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/model-00015-of-00015.safetensors
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/model.safetensors.index.json
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/modeling_qwen.py
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/qwen.tiktoken
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/qwen_generation_utils.py
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/tokenization_qwen.py
   wget https://huggingface.co/Qwen/Qwen-14B/resolve/main/tokenizer_config.json
   cd../../
   ```

   Modify line 39 in the modelling_qwen.py file, changing:

   ```python
   SUPPORT_FP16 = SUPPORT_CUDA and torch.cuda.get_device_capability(0)[0] >= 7
   ```

   to：

   ```python
   SUPPORT_FP16 = True
   ```
4. Weights convert

   Convert weights from huggingface format to megatron format
   ***(This scenario is generally used to train open-source HuggingFace models on Megatron)***

    ```bash
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
   
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader qwen_hf \
        --saver megatron \
        --target-tensor-parallel-size 8 \
        --load-dir ./model_from_hf/Qwen-14B/ \
        --save-dir ./model_weights/Qwen-14B-v0.1-tp8-pp1/ \
        --tokenizer-model ./model_from_hf/Qwen-14B/qwen.tiktoken \
        --add-qkv-bias
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
        --save-model-type save_huggingface_qwen \
        --load-dir ./model_weights/Qwen-14B-v0.1-tp8-pp1/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --add-qkv-bias \
        --save-dir ./model_from_hf/Qwen-14B/   # Fill in the original HF model path here, new weights will be saved in ./model_from_hf/Qwen-14B/mg2hg/
    ```
5. Prepare dataset

    Download the Qwen-14B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

    ```shell
    # download datasets
    cd ./dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
   
    # process datasets  
    mkdir ./dataset/Qwen-14B/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/Qwen-14B/ \
        --output-prefix ./dataset/Qwen-14B/alpaca \
        --tokenizer-type PretrainedFromHF \
        --seq-length 2048 \
        --workers 4 \
        --log-interval 1000
   ```
6. pre-training

   Config Qwen-14B pre-training script: examples/qwen/pretrain_qwen_14b_ptd.sh

   ```shell
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
   
    # modify config according to your own actual situation
    CKPT_SAVE_DIR="./ckpt/Qwen-14B/"
    TOKENIZER_MODEL="./model_from_hf/Qwen-14B/"  #tokenizer path
    DATA_PATH="./dataset/Qwen-14B/alpaca_text_document"  #processed dataset
    CKPT_LOAD_DIR="./model_weights/Qwen-14B-v0.1-tp8-pp1/"
   ```

   Launch Qwen-14B pre-training script: examples/qwen/pretrain_qwen_14b_ptd.sh

   ```shell
    bash examples/qwen/pretrain_qwen_14b_ptd.sh 
   ```
    **Note**: If using multi machine training, and no data sharing configuration on the mechines, it's necessary to add the parameter `--no-shared-storage`. This parameter will determine whether non master nodes need to load data based on distributed parameters, and check the corresponding cache and generated data.


## Inference

Config qwen-14b inference script: examples/qwen/generate_qwen_14b_ptd.sh

```bash
# ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
# modify script model path and tokenizer path
CHECKPOINT="./model_weights/Qwen-14B-v0.1-tp8-pp1/"
TOKENIZER_PATH="./model_from_hf/Qwen-14B/"
```

Launch qwen-14b inference script: examples/qwen/generate_qwen_14b_ptd.sh

```bash
bash examples/qwen/generate_qwen_7b_ptd.sh
```

Some inference samples are as follows:
![Inference](https://gitee.com/ascend/ModelLink/raw/master/sources/images/qwen/qwen_14b_inference.png)


## Evaluation

We use the [CEval benchmark](https://huggingface.co/datasets/ceval/ceval-exam) and [MMLU benchmark](https://huggingface.co/datasets/cais/mmlu) to evaluate our model. 

Config qwen-14b evaluation script: examples/qwen/evaluate_qwen_14b_ptd.sh

```bash
# ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Modify the model parameter path and vocabulary path
TOKENIZER_PATH="./model_from_hf/Qwen-14B/"  # vocabulary path
CHECKPOINT="./model_weights/Qwen-14B-v0.1-tp8-pp1/"  # parameter path

# Configure the task type and dataset path
DATA_PATH="./mmlu/data/test/"  # "./ceval/val/" for ceval task
TASK="mmlu"  # "ceval" for ceval task
```

Launch qwen-14b evaluation

```bash
bash examples/qwen/evaluate_qwen_14b_ptd.sh
```

| Task | Subset | Question | OpenSource | NPU |
|:---:|:---:|:---:|:---:|:---:|
| CEval | 52 | 1346 | 72.1 | 71.1 |
| MMLU | 57 | 14042 | 66.3 | 66.1 |

# Qwen-72B

## Training

Here's a hardware summary of pre-training  Qwen-72B:

| Hardware | Seq-length |       Value       |
| :------: |:----------:|:-----------------:|
|   NPU    |     8k     | 64 x Ascend NPUs  |
|   NPU    |    32k     | 320 x Ascend NPUs |

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
    pip install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl
    pip install torch_npu-2.1.0*-cp38-cp38m-linux_aarch64.whl
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
   Download the Qwen-72B checkpoint from [here](https://huggingface.co/Qwen/Qwen-72B/tree/main)

   ```bash
   mkdir ./model_from_hf/Qwen-72B/
   cd ./model_from_hf/Qwen-72B/
   wget https://huggingface.co/Qwen/Qwen-72B/resolve/main/cache_autogptq_cuda_256.cpp
   wget https://huggingface.co/Qwen/Qwen-72B/resolve/main/cache_autogptq_cuda_kernel_256.cu
   wget https://huggingface.co/Qwen/Qwen-72B/resolve/main/config.json
   wget https://huggingface.co/Qwen/Qwen-72B/resolve/main/configuration_qwen.py
   wget https://huggingface.co/Qwen/Qwen-72B/resolve/main/cpp_kernels.py
   wget https://huggingface.co/Qwen/Qwen-72B/resolve/main/generation_config.json
   wget https://huggingface.co/Qwen/Qwen-72B/resolve/main/model-00001-of-000082.safetensors
   ...
   cd ../../
   ```

   Modify line 39 in the modelling_qwen.py file, changing:

   ```python
   SUPPORT_FP16 = SUPPORT_CUDA and torch.cuda.get_device_capability(0)[0] >= 7
   ```

   to：

   ```python
   SUPPORT_FP16 = True
   ```
4. Weights convert

   Convert weights from huggingface format to megatron format
   ***(This scenario is generally used to train open-source HuggingFace models on Megatron)***

    ```bash
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
   
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader qwen_hf \
        --saver megatron \
        --target-tensor-parallel-size 8 \
        --load-dir ./model_from_hf/Qwen-72B/ \
        --save-dir ./model_weights/Qwen-72B-v0.1-tp8-pp1/ \
        --tokenizer-model ./model_from_hf/Qwen-72B/qwen.tiktoken \
        --add-qkv-bias
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
        --save-model-type save_huggingface_qwen \
        --load-dir ./model_weights/Qwen-72B-v0.1-tp8-pp1/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --add-qkv-bias \
        --save-dir ./model_from_hf/Qwen-72B/    # Fill in the original HF model path here, new weights will be saved in ./model_from_hf/Qwen-72B/mg2hg/
    ```
5. Prepare dataset

    Download the Qwen-72B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

    ```shell
    # download datasets
    cd ./dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..


    # process datasets  
    mkdir ./dataset/Qwen-72B/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/Qwen-72B/ \
        --output-prefix ./dataset/Qwen-72B/alpaca \
        --tokenizer-type PretrainedFromHF \
        --seq-length 8192 \
        --workers 4 \
        --log-interval 1000
    ```
6. pre-training

    Config Qwen-72B pre-training script: examples/qwen/pretrain_qwen_72b_ptd.sh

    ```shell
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    
    # modify config according to your own actual situation
    CKPT_SAVE_DIR="./ckpt/Qwen-72B/"
    TOKENIZER_MODEL="./model_from_hf/Qwen-72B/"  #tokenizer path
    DATA_PATH="./dataset/Qwen-72B/alpaca_text_document"  #processed dataset
    CKPT_LOAD_DIR="./model_weights/Qwen-72B-v0.1-tp8-pp1/"
    ```

    To use a 32K sequence, turn on the re-computation feature and change the value of seq-length to 32768. The parameter configuration is as follows:

    ```shell
        --seq-length 32768 \
        --recompute-granularity full \
        --recompute-method block \
        --recompute-num-layers 80 \
    ```

    Launch Qwen-72B pre-training script: examples/qwen/pretrain_qwen_72b_ptd.sh

    ```shell
    bash examples/qwen/pretrain_qwen_72b_ptd.sh 
    ```
    **Note**: If using multi machine training, and no data sharing configuration on the mechines, it's necessary to add the parameter `--no-shared-storage`. This parameter will determine whether non master nodes need to load data based on distributed parameters, and check the corresponding cache and generated data.



## Inference

Config qwen-72b inference script: examples/qwen/generate_qwen_72b_ptd.sh

```bash
# ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
# modify script model path and tokenizer path
CHECKPOINT="./model_weights/Qwen-72B-v0.1-tp8-pp1/"
TOKENIZER_PATH="./model_from_hf/Qwen-72B/"
```

Launch qwen-72b inference script: examples/qwen/generate_qwen_72b_ptd.sh

```bash
bash examples/qwen/generate_qwen_72b_ptd.sh
```

Some inference samples are as follows:
![Inference](https://gitee.com/ascend/ModelLink/raw/master/sources/images/qwen/qwen_72b_inference.png)

## Evaluation

We use the [CEval benchmark](https://huggingface.co/datasets/ceval/ceval-exam) and [MMLU benchmark](https://huggingface.co/datasets/cais/mmlu) to evaluate our model. 

Config qwen-72b evaluation script: examples/qwen/evaluate_qwen_72b_ptd.sh

```bash
# ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Modify the model parameter path and vocabulary path
TOKENIZER_PATH="./model_from_hf/Qwen-72B/"  # vocabulary path
CHECKPOINT="./model_weights/Qwen-72B-v0.1-tp8-pp1/"  # parameter path

# Configure the task type and dataset path
DATA_PATH="./mmlu/data/test/"  # "./ceval/val/" for ceval task
TASK="mmlu"  # "ceval" for ceval task
```

Launch qwen-72b evaluation

```bash
bash examples/qwen/evaluate_qwen_72b_ptd.sh
```

| Task | Subset | Question | OpenSource | NPU |
|:---:|:---:|:---:|:---:|:---:|
| CEval | 52 | 1346 | 83.3 | 81.8 |
| MMLU | 57 | 14042 | 77.4 | 74.6 |
