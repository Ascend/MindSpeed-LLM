# Intern-LM
<p align="left">
        <b><a href="README.md">简体中文</a></b> |
        <b>English</b> 
    </p>
</p>

#  Contents

- [Contents](#contents)
- [Internlm-7B](#internlm-7b)
  - [Training](#training)
    - [Script](#script)
  - [Inference](#Inference)
  - [Evaluation](#Evaluation)
- [Contents](#contents)
- [Internlm-65B](#internlm-65b)
  - [Training](#training)
    - [Script](#script)

# InternLM-7B

## Training

Here's a hardware summary of pre-training InternLM-7B:

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
    pip install torch-2.1.0-cp38-cp38m-linux_aarch64.whl
    pip install torch_npu-2.1.0.XXX-cp38-cp38m-linux_XXX.whl
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

3. Download the Internlm-7B tokenizer model and file from [here](https://huggingface.co/internlm/internlm-7b/tree/main) 

    ```shell
    mkdir ./model_from_hf/internlm-7b/
    cd ./model_from_hf/internlm-7b/
    wget https://huggingface.co/internlm/internlm-7b/resolve/main/config.json
    wget https://huggingface.co/internlm/internlm-7b/resolve/main/generation_config.json
    wget https://huggingface.co/internlm/internlm-7b/resolve/main/special_tokens_map.json
    wget https://huggingface.co/internlm/internlm-7b/resolve/main/tokenization_internlm.py
    wget https://huggingface.co/internlm/internlm-7b/resolve/main/tokenizer.model
    wget https://huggingface.co/internlm/internlm-7b/resolve/main/tokenizer_config.json
    cd ../../
    ```

4. Prepare dataset. Download the Internlm-7B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

    ```shell
    cd dataset/
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    ```

    ```shell
    #!/bin/bash
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    mkdir ./dataset/internlm-7b/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/internlm-7b/ \
        --output-prefix ./dataset/internlm-7b/alpaca \
        --workers 4 \
        --log-interval 1000  \
        --tokenizer-type PretrainedFromHF  \
        --handler-name AlpacaPretrainHandler  \
        --tokenizer-not-use-fast \
        --append-eod
    ```

5. Weights convert

    In order to adapt to the internlm-7B model, the following script is used to convert the model pre-training weights.
    ***(This scenario is generally used to train open-source HuggingFace models on Megatron)***

    ```shell
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader llama2_hf \
        --saver megatron \
        --target-tensor-parallel-size 8 \
        --target-pipeline-parallel-size 1 \
        --load-dir ./model_from_hf/internlm-7b/ \
        --save-dir ./model_weights/internlm-7b-v0.1-tp8-pp1/ \
        --tokenizer-model ./model_from_hf/internlm-7b/tokenizer.model \
        --add-qkv-bias \
        --add-dense-bias
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
        --load-dir ./model_weights/internlm-7b-v0.1-tp8-pp1/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --add-qkv-bias \
        --add-dense-bias \
        --save-dir ./model_from_hf/internlm-7b/    # <-- Fill in the original HF model path here, new weights will be saved in ./model_from_hf/internlm-7b/mg2hg/
    ```

6. Config Internlm-7B pre-training script.

    ```shell
    # modify the script according to your own  ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    # modify script orign dataset path according to your own dataset path
    CKPT_SAVE_DIR="./ckpt/internlm-7b/"
    CKPT_LOAD_DIR="./model_weights/internlm-7b-v0.1-tp8-pp1/"
    TOKENIZER_PATH="./model_from_hf/internlm-7b/tokenizer.model"  #tokenizer path
    DATA_PATH="./dataset/internlm-7b/alpaca_text_document" #processed dataset
    ```

7. Launch Internlm-7B pre-training script.

    ```shell
    bash examples/intern/pretrain_internlm_7b_ptd.sh 
    ```
    **Note**: If using multi machine training, and no data sharing configuration on the mechines, it's necessary to add the parameter `--no-shared-storage`. This parameter will determine whether non master nodes need to load data based on distributed parameters, and check the corresponding cache and generated data.



#### Inference

Inference script`</a>`：
examples/intern/generate_lnternlm_7b_ptd.sh

```
bash ./examples/intern/generate_lnternlm_7b_ptd.sh
```

Inference case:
![Internlm-7b-inference](https://gitee.com/ascend/ModelLink/raw/master/sources/images/intern/intern_7B_inference.png)

#### Evaluation

Evaluating the model using the MMLU dataset. dataset [download](https://huggingface.co/datasets/cais/mmlu)

Evaluation script: examples/intern/evaluate_internlm_7B_ptd.sh

```
bash  examples/intern/evaluate_internlm_7B_ptd.sh
```

The evaluation performance of Internlm-7B in **Ascend NPU**:

| Task    | Model     | NPU  | Benchmark |
|-----------------------------------------------------|-----------|------|------|
| MMLU | Internlm-7B  | 48.7 | [51.0](https://huggingface.co/internlm/internlm-7b) | 

# InternLM-65B

## Training

Here's a hardware summary of pre-training InternLM-65B:

| Hardware |                      Value                      |
| :------: | :---------------------------------------------: |
|   NPU    |               32 x Ascend NPUs                   |

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
    pip install torch-2.1.0-cp38-cp38m-linux_aarch64.whl
    pip install torch_npu-2.1.0.XXX-cp38-cp38m-linux_XXX.whl
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

3. Download tokenizer model and file from [here](https://huggingface.co/internlm/internlm-7b/tree/main) 

    ```shell
    mkdir ./model_from_hf/internlm-65b/
    cd ./model_from_hf/internlm-65b/
    wget https://huggingface.co/internlm/internlm-7b/resolve/main/config.json
    wget https://huggingface.co/internlm/internlm-7b/resolve/main/generation_config.json
    wget https://huggingface.co/internlm/internlm-7b/resolve/main/special_tokens_map.json
    wget https://huggingface.co/internlm/internlm-7b/resolve/main/tokenization_internlm.py
    wget https://huggingface.co/internlm/internlm-7b/resolve/main/tokenizer.model
    wget https://huggingface.co/internlm/internlm-7b/resolve/main/tokenizer_config.json
    cd ../../
    ```

4. Prepare dataset. Download the Internlm-65B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet) 

    ```shell
    cd dataset/
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    ```

    ```shell
    #!/bin/bash
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    mkdir ./dataset/internlm-65b/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/internlm-65b/ \
        --output-prefix ./dataset/internlm-65b/alpaca \
        --workers 4 \
        --log-interval 1000  \
        --tokenizer-type PretrainedFromHF  \
        --handler-name AlpacaPretrainHandler  \
        --tokenizer-not-use-fast \
        --append-eod
    ```

5. Config Internlm-65B pre-training script.

    ```shell
    # modify the script according to your own  ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    # modify script orign dataset path according to your own dataset path
    CKPT_SAVE_DIR="./ckpt/internlm-65b/"
    TOKENIZER_PATH="./model_from_hf/internlm-65b/tokenizer.model"  #tokenizer path
    DATA_PATH="./dataset/internlm-65b/alpaca_text_document"  #processed dataset
    ```

6. Launch Internlm-65B pre-training script.

    ```shell
    bash examples/intern/pretrain_internlm_65b_ptd.sh 
    ```
    **Note**: If using multi machine training, and no data sharing configuration on the mechines, it's necessary to add the parameter `--no-shared-storage`. This parameter will determine whether non master nodes need to load data based on distributed parameters, and check the corresponding cache and generated data.

