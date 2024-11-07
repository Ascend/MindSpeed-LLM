# BaiChuan
<p align="left">
        <b><a href="README.md">简体中文</a></b> |
        <b>English</b> 
    </p>
</p>


#  Contents

- [Baichuan-7B](#contents)
  - [Training](#pre-training)
    - [Script](#script)
  - [Inference](#Inference)
  - [Evaluation](#Evaluation)

- [Baichuan-13B](#contents)
  - [Training](#pre-training)
    - [Script](#script)
  - [Lora](#Lora)
  - [Inference](#Inference)
  - [Evaluation](#Evaluation)


# Baichuan-7B

## Training

Here's a hardware summary of pre-training Baichuan-7B:

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
    Download the Baichuan-7B checkpoint from [here](https://huggingface.co/baichuan-inc/Baichuan-7B/tree/main) 

    ```shell
    mkdir ./model_from_hf/Baichuan-7B/
    cd ./model_from_hf/Baichuan-7B/
    wget https://huggingface.co/baichuan-inc/Baichuan-7B/resolve/main/config.json
    wget https://huggingface.co/baichuan-inc/Baichuan-7B/resolve/main/configuration_baichuan.py
    wget https://huggingface.co/baichuan-inc/Baichuan-7B/resolve/main/generation_config.json
    wget https://huggingface.co/baichuan-inc/Baichuan-7B/resolve/main/handler.py
    wget https://huggingface.co/baichuan-inc/Baichuan-7B/resolve/main/modeling_baichuan.py
    wget https://huggingface.co/baichuan-inc/Baichuan-7B/resolve/main/pytorch_model.bin
    wget https://huggingface.co/baichuan-inc/Baichuan-7B/resolve/main/special_tokens_map.json
    wget https://huggingface.co/baichuan-inc/Baichuan-7B/resolve/main/tokenization_baichuan.py
    wget https://huggingface.co/baichuan-inc/Baichuan-7B/resolve/main/tokenizer.model
    wget https://huggingface.co/baichuan-inc/Baichuan-7B/resolve/main/tokenizer_config.json
    cd ../../
    ```

4. Weights convert

    In order to adapt to the Baichuan-7B model, the following script is used to convert the model pre-training weights.
    ***(This scenario is generally used to train open-source HuggingFace models on Megatron)***

    ```shell
    # modify the ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader llama2_hf \
        --saver megatron \
        --target-tensor-parallel-size 8 \
        --target-pipeline-parallel-size 1 \
        --load-dir ./model_from_hf/Baichuan-7B/ \
        --save-dir ./model_weights/Baichuan-7B-v0.1-tp8-pp1/ \
        --tokenizer-model ./model_from_hf/Baichuan-7B/tokenizer.model \
        --w-pack True  
    ```

    Any Megatron weights with parallel slicing strategy --> Any Megatron weights with parallel slicing strategy
    ***(This scenario is generally used to convert the trained megatron model back to the HuggingFace format)***

    ```shell
    # Modify the ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python tools/checkpoint/convert_ckpt.py --model-type GPT \
        --loader megatron \
        --saver megatron \
        --save-model-type save_huggingface_llama \
        --load-dir ./model_weights/Baichuan-7B-v0.1-tp8-pp1/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --w-pack True \
        --save-dir ./model_from_hf/Baichuan-7B/   # <-- Fill in the original HF model path here, new weights will be saved in ./model_from_hf/Baichuan-7B/mg2hg/
    ```

5. Prepare dataset

    Download the Baichuan-7B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet) 

    ```shell
    # download datasets
    cd ./dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..

    # process datasets          
    mkdir ./dataset/Baichuan-7B/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/Baichuan-7B/ \
        --output-prefix ./dataset/Baichuan-7B/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF
    ```

6. Config Baichuan-7B pre-training script : examples/baichuan/pretrain_baichuan_ptd_7B.sh

    ```shell
    # modify the script according to your own  ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    CKPT_SAVE_DIR="./ckpt/Baichuan-7B/"
    DATA_PATH="./dataset/Baichuan-7B/alpaca_text_document"
    TOKENIZER_MODEL="./model_from_hf/Baichuan-7B/tokenizer.model"
    CKPT_LOAD_DIR="./model_weights/Baichuan-7B-v0.1-tp8-pp1/"
    ```

7. Launch Baichuan-7B  pre-training script: examples/baichuan/pretrain_baichuan_ptd_7B.sh

    ```shell
    bash examples/baichuan/pretrain_baichuan_ptd_7B.sh 
    ```
    **Note**: If using multi machine training, and no data sharing configuration on the mechines, it's necessary to add the parameter `--no-shared-storage`. This parameter will determine whether non master nodes need to load data based on distributed parameters, and check the corresponding cache and generated data.



## Inference

Config Baichuan-7B inference script: examples/baichuan/generate_baichuan_7b_ptd.sh

```bash
# modify the script according to your own ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
# modify script model path and tokenizer path
CHECKPOINT="./model_weights/Baichuan-7B-v0.1-tp8-pp1/"
TOKENIZER_PATH="./model_from_hf/Baichuan-7B/"
```

Launch Baichuan-7B inference script: examples/baichuan/generate_baichuan_7b_ptd.sh

```bash
bash examples/baichuan/generate_baichuan_7b_ptd.sh
```

Some inference samples are as follows:

![Inference](https://gitee.com/ascend/ModelLink/raw/master/sources/images/baichuan/baichuan_7B_inference.png)

## Evaluation

We use the boolq benchmark to evaluate our model. Benchmark [Download](https://huggingface.co/datasets/boolq).

```shell
# config origin weight and vocab file path
CHECKPOINT=<origin-ckpt-path>
TOKENIZER_PATH=<tokenizer-path>
# config tasks and dataset path
DATA_PATH="./boolq/"
TASK="boolq"
```

```shell
bash ./examples/baichuan/evaluate_baichuan_7B_ptd.sh
```

<table>
  <thead>
    <tr>
      <th>Task</th>
      <th>Subset</th>
      <th>Model</th>
      <th>NPU</th>
      <th>OpenSource</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://huggingface.co/datasets/boolq">Boolq</a></td>
      <td>test</td>
      <th>Baichuan 7B</th>
      <td>0.69</td>
      <td><a href="https://hub.opencompass.org.cn/dataset-detail/BoolQ">0.67</a></td>
    </tr>
  </tbody>
</table>

# Baichuan-13B

## Training

Here's a hardware summary of pre-training Baichuan-13B:

| Hardware |                      Value                      |
| :------: | :---------------------------------------------: |
|   NPU    |               8 x Ascend NPUs               |



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
    pip install torch_npu-2.1.0.XXX-cp38-cp38m-linux_aarch64.whl
    pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

    # modify the path according to your own  ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    #install Mindspeed
    git clone https://gitee.com/ascend/MindSpeed.git
    cd MindSpeed
    git checkout 2b0edd2
    pip install -r requirements.txt 
    pip3 install -e .
    cd ..

    # install other packages
    pip install -r requirements.txt 
    ```

    **Note:** If the error message "'AttributeError: 'BaichuanTokenizer' object has no attribute'sp_model'" is displayed during the script execution, run the following command to rectify the error:

    ```shell
    pip install transformers==4.32.0 --force
    ```

3. Prepare pretrained weights


    Download the Baichuan-13B checkpoint from [here](https://huggingface.co/baichuan-inc/Baichuan-13B-Base/tree/main) 
    
    ```shell
    mkdir ./model_from_hf/Baichuan-13B/
    cd ./model_from_hf/Baichuan-13B/
    wget https://huggingface.co/baichuan-inc/Baichuan-13B-Base/resolve/main/config.json
    wget https://huggingface.co/baichuan-inc/Baichuan-13B-Base/resolve/main/configuration_baichuan.py
    wget https://huggingface.co/baichuan-inc/Baichuan-13B-Base/resolve/main/generation_config.json
    wget https://huggingface.co/baichuan-inc/Baichuan-13B-Base/resolve/main/modeling_baichuan.py
    wget https://huggingface.co/baichuan-inc/Baichuan-13B-Base/resolve/main/pytorch_model-00001-of-00003.bin
    wget https://huggingface.co/baichuan-inc/Baichuan-13B-Base/resolve/main/pytorch_model-00002-of-00003.bin
    wget https://huggingface.co/baichuan-inc/Baichuan-13B-Base/resolve/main/pytorch_model-00003-of-00003.bin
    wget https://huggingface.co/baichuan-inc/Baichuan-13B-Base/resolve/main/pytorch_model.bin.index.json
    wget https://huggingface.co/baichuan-inc/Baichuan-13B-Base/resolve/main/quantizer.py
    wget https://huggingface.co/baichuan-inc/Baichuan-13B-Base/resolve/main/special_tokens_map.json
    wget https://huggingface.co/baichuan-inc/Baichuan-13B-Base/resolve/main/tokenization_baichuan.py
    wget https://huggingface.co/baichuan-inc/Baichuan-13B-Base/resolve/main/tokenizer_config.json
    wget https://huggingface.co/baichuan-inc/Baichuan-13B-Base/resolve/main/tokenizer.model
    cd ../../
    ```

4. Weights convert

    In order to adapt to the baichuan-13B model, the following script is used to convert the model pre-training weights.

    ***(This scenario is generally used to train open-source HuggingFace models on Megatron)***

    ```shell
    mkdir baichuan-13B-mt

    # modify the ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader llama2_hf \
        --saver megatron \
        --target-tensor-parallel-size 8 \
        --load-dir ./model_from_hf/Baichuan-13B/ \
        --save-dir ./model_weights/Baichuan-13B-Base-v0.1-tp8-pp1/ \
        --tokenizer-model ./model_from_hf/Baichuan-13B/tokenizer.model \
        --params-dtype bf16 \
        --w-pack True  
    ```

    Any Megatron weights with parallel slicing strategy --> Any Megatron weights with parallel slicing strategy
    ***(This scenario is generally used to convert the trained megatron model back to the HuggingFace format)***

    ```shell
    # Modify the ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python tools/checkpoint/convert_ckpt.py --model-type GPT \
        --loader megatron \
        --saver megatron \
        --save-model-type save_huggingface_llama \
        --load-dir ./model_weights/Baichuan-13B-Base-v0.1-tp8-pp1/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --w-pack True \
        --save-dir ./model_from_hf/Baichuan-13B/   # <-- Fill in the original HF model path here, new weights will be saved in ./model_from_hf/Baichuan-13B/mg2hg/
    ```

5. Prepare dataset
    Download the Baichuan-13B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet) 

    ```shell
    cd ./dataset/
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    
    mkdir ./dataset/Baichuan-13B/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/Baichuan-13B/ \
        --output-prefix ./dataset/Baichuan-13B/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF 
    ```

6. Config Baichuan-13B pre-training script(Baichuan-13B does not support Flash Attention): examples/baichuan/pretrain_baichuan_ptd_13B.sh

    ```shell
    # modify the script according to your own  ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    CKPT_SAVE_DIR="./ckpt/Baichuan-13B/"
    DATA_PATH="./dataset/Baichuan-13B/alpaca_text_document"
    TOKENIZER_MODEL="./model_from_hf/Baichuan-13B/tokenizer.model"
    CKPT_LOAD_DIR="./model_weights/Baichuan-13B-Base-v0.1-tp8-pp1/" 
    ```

7. Launch Baichuan-13B pre-training script: examples/baichuan/pretrain_baichuan_ptd_13B.sh

    ```bash
    bash examples/baichuan/pretrain_baichuan_ptd_13B.sh
    ```
    **Note**: If using multi machine training, and no data sharing configuration on the mechines, it's necessary to add the parameter `--no-shared-storage`. This parameter will determine whether non master nodes need to load data based on distributed parameters, and check the corresponding cache and generated data.
   



## Inference

Config baichuan-13B inference script: examples/baichuan/generate_baichuan_13b_ptd.sh

```bash
# modify the script according to your own ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
# modify script model path and tokenizer path
CHECKPOINT="./model_weights/Baichuan-13B-Base-v0.1-tp8-pp1/"
TOKENIZER_PATH="./model_from_hf/Baichuan-13B/"
```

Launch baichuan-13B inference script: examples/baichuan/generate_baichuan_13b_ptd.sh

```bash
bash examples/baichuan/generate_baichuan_13b_ptd.sh
```

Some inference samples are as follows:
![Inference](https://gitee.com/ascend/ModelLink/raw/master/sources/images/baichuan/baichuan_13B_inference.png)

## Evaluation

We use the boolq benchmark to evaluate our model. Benchmark [Download](https://huggingface.co/datasets/boolq).

```shell
# config origin weight and vocab file path
CHECKPOINT=<origin-ckpt-path>
TOKENIZER_PATH=<tokenizer-path>
# config tasks and dataset path
DATA_PATH="./boolq/"
TASK="boolq"
```

```shell
bash ./examples/baichuan/evaluate_baichuan_13B_ptd.sh
```

<table>
  <thead>
    <tr>
      <th>Task</th>
      <th>Subset</th>
      <th>Model</th>
      <th>NPU</th>
      <th>OpenSource</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://huggingface.co/datasets/boolq">Boolq</a></td>
      <td>test</td>
      <th>Baichuan 13B</th>
      <td>0.747</td>
      <td><a href="https://hub.opencompass.org.cn/dataset-detail/BoolQ">0.736</a></td>
    </tr>
  </tbody>
</table>
