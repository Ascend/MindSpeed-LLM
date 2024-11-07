# LLaMA-7B/13B

<p align="left">
        <b><a href="README.md">简体中文</a></b> |
        <b>English</b>
</p>

- [LLaMA-7B/13B](#llama-7b13b)
  - [Training](#training)
    - [Script](#script)
  - [Inference](#Inference)
    - [Script](#script)
  - [Evaluation with Numerous Benchmarks](#Evaluation-with-Numerous-Benchmarks)
- [LLaMA-33B/65B](#llama-65b)
  - [Training](#pre-training)
    - [Datasets](#datasets)
    - [Script](#script-1)
  - [Inference](#Inference)
    - [Script](#script)
  - [Evaluation with Numerous Benchmarks](#Evaluation-with-Numerous-Benchmarks)
- [Citation](#citation)

## Training

Here's a hardware summary of pre-training LLaMA-7B/13B:

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
    # modify the path according to your own  ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    # install mindspeed
    git clone https://gitee.com/ascend/MindSpeed.git
    cd MindSpeed
    git checkout 2b0edd2
    pip install -r requirements.txt
    pip3 install -e .
    cd ..

    # install other packages
    pip install -r requirements.txt 
    ```

3. Download the LLaMA-7B/13B tokenizer model file and checkpoint.

    Download the LLaMA-7B checkpoint from [here](https://huggingface.co/ruibin-wang/llama-7b-hf/tree/main)

    ```shell
    cd ./model_from_hf
    # you must install git-lfs
    git clone https://huggingface.co/ruibin-wang/llama-7b-hf
    cd ..
    ```

    Download the LLaMA-13B checkpoint from [here](https://huggingface.co/ruibin-wang/llama-13b-hf/tree/main) 

    ```shell
    cd ./model_from_hf
    # you must install git-lfs
    git clone https://huggingface.co/ruibin-wang/llama-13b-hf
    cd ..
    ```

4. Weights convert

    In order to adapt to the LLaMA-7B/13B model, the following script is used to convert the model pre-training weights.
    ***(This scenario is generally used to train open-source HuggingFace models on Megatron)***

    LLaMA-7B

    ```shell
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader llama2_hf \
        --saver megatron \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 8 \
        --load-dir ./model_from_hf/llama-7b-hf/ \
        --save-dir ./model_weights/llama-7b-hf-v0.1-tp1-pp8/ \
        --tokenizer-model ./model_from_hf/llama-7b-hf/tokenizer.model
    ```

    LLaMA-13B

    ```shell
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    # Single machine with 8p
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader llama2_hf \
        --saver megatron \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 8 \
        --load-dir ./model_from_hf/llama-13b-hf/ \
        --save-dir ./model_weights/llama-13b-hf-v0.1-tp1-pp8/ \
        --tokenizer-model ./model_from_hf/llama-13b-hf/tokenizer.model
    ```

    Any Megatron weights with parallel slicing strategy --> Any Megatron weights with parallel slicing strategy
    ***(This scenario is generally used to convert the trained megatron model back to the HuggingFace format)***

    LLaMA-7B

    ```shell
    # Modify the ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader megatron \
        --saver megatron \
        --save-model-type save_huggingface_llama \
        --load-dir ./model_weights/llama-7b-hf-v0.1-tp1-pp8/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --save-dir ./model_from_hf/llama-7b-hf/  # <-- Fill in the original HF model path here, new weights will be saved in ./model_from_hf/llama-7b-hf/mg2hg/
    ```

    LLaMA-13B

    ```shell
    # Modify the ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader megatron \
        --saver megatron \
        --save-model-type save_huggingface_llama \
        --load-dir ./model_weights/llama-13b-hf-v0.1-tp1-pp8/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --save-dir ./model_from_hf/llama-13b-hf/  # <-- Fill in the original HF model path here, new weights will be saved in ./model_from_hf/llama-13b-hf/mg2hg/
    ```

    Weight conversion is suitable for pre-training, fine-tuning, inference and evaluation. Adjust the parameters `target-tensor-parallel-size` and `target-pipeline-parallel-size` according to different tasks.
5. Pretrain

    5.1 Prepare pretrain dataset
    Download the LLaMA-7B/13B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

    ```shell
    cd dataset/
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    ```

    Preprocess dataset

    LLaMA-7B

    ```shell
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    mkdir ./dataset/llama-7b-hf/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/llama-7b-hf/ \
        --output-prefix ./dataset/llama-7b-hf/alpaca \
        --workers 4 \
        --log-interval 1000  \
        --tokenizer-type PretrainedFromHF  
    ```

    LLaMA-13B

    ```shell
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    mkdir ./dataset/llama-7b-hf/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/llama-13b-hf/ \
        --output-prefix ./dataset/llama-7b-hf/alpaca \
        --workers 4 \
        --log-interval 1000  \
        --tokenizer-type PretrainedFromHF  
    ```

    5.2 Config LLaMA-7B/13B pre-training script.

    LLaMA-7B

    ```shell
    # modify the script according to your own  ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    # modify script orign dataset path according to your own dataset path
    TOKENIZER_MODEL="./model_from_hf/llama-7b-hf/tokenizer.model"
    DATA_PATH="./dataset/llama/alpaca_text_document"  #processed dataset
    LOAD_CHECKPOINT_PATH="./model_weights/llama-7b-hf-v0.1-tp1-pp8"
    SAVE_CHECKPOINT_PATH="./ckpt/llama-7b-hf/"
    ```

    LLaMA-13B

    ```shell
    # modify the script according to your own  ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    # modify script orign dataset path according to your own dataset path
    TOKENIZER_MODEL="./model_from_hf/llama-13b-hf/tokenizer.model"  
    DATA_PATH="./dataset/llama-13b-hf/alpaca_text_document"  #processed dataset
    LOAD_CHECKPOINT_PATH="./model_weights/llama-13b-hf-v0.1-tp1-pp8"
    SAVE_CHECKPOINT_PATH="./ckpt/llama-13b-hf/"
    ```

    5.3 Launch LLaMA-7B/13B pre-training script.

    **Note**: If using multi machine training, and no data sharing configuration on the mechines, it's necessary to add the parameter `--no-shared-storage`. This parameter will determine whether non master nodes need to load data based on distributed parameters, and check the corresponding cache and generated data.

    LLaMA-7B

    ```shell
    bash examples/llama/pretrain_llama_7b_ptd.sh
    ```

    LLaMA-13B

    ```shell
    # 8p
    bash examples/llama/pretrain_llama_13b_ptd.sh 
    ```

6. Pretrain

    6.1 Prepare fine-tune dataset
    Download the LLaMA-7B/13B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet) 

    ```shell
    mkdir finetune_dataset
    cd ./finetune_dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    ```

    Preprocess instruction dataset

    LLaMA-7B

    ```shell
    mkdir ./finetune_dataset/llama-7b-hf/
    python ./tools/preprocess_data.py \
        --input ./finetune_dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/llama-7b-hf/ \
        --output-prefix ./finetune_dataset/llama-7b-hf/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF \
        --handler-name GeneralInstructionHandler \
        --append-eod
    ```

    LLaMA-13B

    ```shell
    mkdir ./finetune_dataset/llama-13b-hf/
    python ./tools/preprocess_data.py \
        --input ./finetune_dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/llama-13b-hf/ \ 
        --output-prefix ./finetune_dataset/llama-13b-hf/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF \
        --handler-name GeneralInstructionHandler \
        --append-eod
    ```

    6.2 Config LLaMA-7B/13B fine-tune script.

    LLaMA-7B

    ```shell
    # modify the script according to your own  ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    # modify script orign dataset path according to your own dataset path
    TOKENIZER_PATH="./model_from_hf/llama-7b-hf/"  #tokenizer path
    DATA_PATH="./finetune_dataset/llama-7b-hf/alpaca"  #processed dataset
    LORA_CHECKPOINT="your lora weight"
    LOAD_CHECKPOINT_PATH="./model_weights/llama-13b-hf-v0.1-tp1-pp8"
    SAVE_CHECKPOINT_PATH="./ckpt/llama-7b-hf/"
    ```

    LLaMA-13B

    ```shell
    # modify the script according to your own  ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    # modify script orign dataset path according to your own dataset path
    TOKENIZER_PATH="./model_from_hf/llama-13b-hf/"  #tokenizer path
    DATA_PATH="./finetune_dataset/llama-13b-hf/alpaca"  #processed dataset
    LORA_CHECKPOINT="your lora weight"
    LOAD_CHECKPOINT_PATH="your init model load path"
    SAVE_CHECKPOINT_PATH="your model ckpt save path"
    ```

    Add the fine-tuning parameter `--finetune` so that fine-tuning starts from the first step.

    6.3 Launch LLaMA-7B/13B fine-tune script.

    LLaMA-7B

    ```shell
    bash examples/llama/tune_llama_7b_ptd.sh
    ```

    LLaMA-13B

    ```shell
    # 8p
    bash examples/llama/tune_llama_13b_ptd.sh 
    ```



## Inference

We support ModelLink Inference for text generation with LLaMA-7B and LLaMA-13B.
Inference different from pre-training, such as we need to Load pre-training checkpoint and the length of the output samples:

Config LLaMA-7B inference script `examples/llama/generate_llama_7b_ptd.sh` and LLaMA-13B inference script `examples/llama/generate_llama_13b_ptd.sh`.

```shell
# modify the model weight path and tokenizer path
CHECKPOINT=<checkpoint-path>
TOKENIZER_PATH=<tokenizer-path>
```

LLaMA-7B:

```shell
bash ./examples/llama/generate_llama_7b_ptd.sh
```

LLaMA-13B:

```shell
bash ./examples/llama/generate_llama_13b_ptd.sh
```

Some inference samples are as follows:

LLaMA-7B:

![llama-7B_generate.png](https://gitee.com/ascend/ModelLink/raw/master/sources/images/llama/llama-7B_generate.png)

LLaMA-13B:

![llama-13B_generate.png](https://gitee.com/ascend/ModelLink/raw/master/sources/images/llama/llama-13B_generate.png)

## Evaluation with Numerous Benchmarks

We use boolq benchmark to evaluate our model. Benchmark Download [here](https://huggingface.co/datasets/boolq).

Config LLaMA-7B evaluation script `examples/llama/evaluate_llama_7B_ptd.sh` and LLaMA-13B evaluation script `examples/llama/evaluate_llama_13B_ptd.sh`:

Modify checkpoint path, vocab path, dataset path and task:

```shell
CHECKPOINT=<checkpoint-path>
TOKENIZER_PATH=<tokenizer-path>
DATA_PATH="./boolq/data/test/"
TASK="boolq"
```

Change the max new tokens:

```shell
--max-new-tokens 1 
```

```text
# Note that, a deepspeed bug needs to be fixed during evaluation：
# Comment out line 671 in the file `<deepspeed-installed-path>/runtime/pipe/engine.py`：
# self.total_loss += self.loss.detach()
```

Start evaluation:

```shell
bash examples/llama/evaluate_llama_7B_ptd.sh
bash examples/llama/evaluate_llama_13B_ptd.sh
```

The evaluation performance of LLaMA-7B/13B in **Ascend NPU**:

| Task    | Model     | NPU  | Benchmark                                               |
|---------|-----------|------|---------------------------------------------------------|
| [Boolq](https://huggingface.co/datasets/boolq) | LLaMA-7B  | 74.6 | [75.4](https://hub.opencompass.org.cn/dataset-detail/BoolQ) | 
| [Boolq](https://huggingface.co/datasets/boolq) | LLaMA-13B | 79.6 | [78.7](https://hub.opencompass.org.cn/dataset-detail/BoolQ)   |

# LLaMA-33B/65B

This directory contains some of the scripts that were used to produce the results in the ModelLink. These scripts is to show the example how to run llama-65B in terminal.

LLaMA model is from: [LLaMA: OPen and Efficient Foundation Language Models](https://arxiv.org/pdf/2302.13971v1.pdf)

>Touvron, Hugo, et al. "LLaMA: OPen and Efficient Foundation Language Models." arXiv preprint arXiv:2302.13971 (2023).

## Training

LLaMA's model performace is better than GPT3 with less parameters. The 33B/65B LLaMA model is comparable to Google's Chinchilla-70B and Palm-540B.

Here's a hardware summary of training llama:

| Hardware |      Value       |
| :------: |:----------------:|
|   NPU    | 32 x Ascend NPUs |

### Datasets

The model was trained using alpaca datasets.

### Script

1. Clone the repository to your local server

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

2. Install ModelLink requirement environment.

    ```bash
    # python3.8
    conda create -n test python=3.8
    conda activate test
    # install torch and torch_npu
    wget https://download.pytorch.org/whl/torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl
    pip install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl
    pip install torch_npu-2.1.0*-cp38-cp38m-linux_aarch64.whl
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

3. Download tokenizer model file and checkpoint

    llama-33B checkpoint

    ```shell
    cd ./model_from_hf/

    # make sure you have git-lfs installed (https://git-lfs.com)
    git lfs install
    git clone https://huggingface.co/pinkmanlove/llama-33b-hf
    cd ..
    # revise "LLaMATokenizer" as "LLaMTokenizer" in tokenizer_config.json
    ```

    llama-65B checkpoint

    ```shell
    cd ./model_from_hf

    # make sure you have git-lfs installed (https://git-lfs.com)
    git lfs install
    git clone https://huggingface.co/pinkmanlove/llama-65b-hf
    cd ..
    # revise "LLaMATokenizer" as "LLaMTokenizer" in tokenizer_config.json
    ```

4. Weights convert

    In order to adapt to llama-33B/65B model, the following script is used to convert the model pre-training weights
    ***(This scenario is generally used to train open-source HuggingFace models on Megatron)***

    llama-33B

    ```shell
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader llama2_hf \
        --saver megatron \
        --target-tensor-parallel-size 4 \
        --target-pipeline-parallel-size 4 \
        --load-dir ./model_from_hf/llama-33b-hf/ \
        --save-dir ./model_weights/llama-33b-hf-v0.1-tp4-pp4/ \
        --tokenizer-model ./model_from_hf/llama-33b-hf/tokenizer.model
    ```

    llama-65B

    ```shell
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader llama2_hf \
        --saver megatron \
        --target-tensor-parallel-size 8 \
        --target-pipeline-parallel-size 4 \
        --load-dir ./model_from_hf/llama-65b-hf/ \
        --save-dir ./model_weights/llama-65b-hf-v0.1-tp8-pp4/ \
        --tokenizer-model ./model_from_hf/llama-65b-hf/tokenizer.model
    ```

    Any Megatron weights with parallel slicing strategy --> Any Megatron weights with parallel slicing strategy
    ***(This scenario is generally used to convert the trained megatron model back to the HuggingFace format)***
    LLaMA-33B

    ```shell
    # Modify the ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader megatron \
        --saver megatron \
        --save-model-type save_huggingface_llama \
        --load-dir /model_weights/llama-33b-hf-v0.1-tp4-pp4/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --save-dir  ./model_from_hf/llama-33b-hf/  # <-- Fill in the original HF model path here, new weights will be saved in ./model_from_hf/llama-33b-hf/mg2hg/
    ```

    LLaMA-65B

    ```shell
    # Modify the ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader megatron \
        --saver megatron \
        --save-model-type save_huggingface_llama \
        --load-dir /model_weights/llama-65b-hf-v0.1-tp8-pp4/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --save-dir ./model_from_hf/llama-65b-hf/   # <-- Fill in the original HF model path here, new weights will be saved in ./model_from_hf/llama-65b-hf/mg2hg/
    ```

    Weight conversion is suitable for pre-training, fine-tuning, inference and evaluation. Adjust the parameters `target-tensor-parallel-size` and `target-pipeline-parallel-size` according to different tasks.

5. Pretrain

    5.1 Prepare pretrain dataset

    ```shell
    cd dataset/
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    ```

    LLaMA-33B

    ```shell
    mkdir ./dataset/llama-33b-hf/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/llama-33b-hf \
        --output-prefix ./dataset/llama-33b-hf/alpaca \
        --workers 4 \
        --log-interval 1000  \
        --tokenizer-type PretrainedFromHF 
    ```

    LLaMA-65B

    ```shell
    mkdir ./dataset/llama-65b-hf/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/llama-65b-hf \
        --output-prefix ./dataset/llama-65b-hf/alpaca \
        --workers 4 \
        --log-interval 1000  \
        --tokenizer-type PretrainedFromHF 
    ```

    5.2 Config llama-33B/65B pre-training script :

    Config llama-33B pre-training script `./examples/llama/pretrain_llama_33B_ptd_32p.sh`:

    ```bash
    # modify the script according to your own conda and ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # modify script orign dataset path according to your own dataset path
    TOKENIZER_MODEL="./model_from_hf/llama-33b-hf/tokenizer.model"
    DATA_PATH="./dataset/llama-33b-hf/alpaca_text_document"
    LOAD_CHECKPOINT_PATH="./model_weights/llama-33b-hf-v0.1-tp4-pp4/"
    SAVE_CHECKPOINT_PATH="./ckpt/llama-33b-hf/"
    ```

    Config llama-65B pre-training script `./examples/llama/pretrain_llama_65b_ptd.sh`:

    ```bash
    # modify the script according to your own conda and ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # modify script orign dataset path according to your own dataset path
    TOKENIZER_MODEL=./model_from_hf/llama-65b-hf/tokenizer.model
    DATA_PATH="./dataset/llama-65b-hf/alpaca_text_document"
    LOAD_CHECKPOINT_PATH="./model_weights/llama-65b-hf-v0.1-tp8-pp4/"
    SAVE_CHECKPOINT_PATH="./ckpt/llama-65b-hf/"
    ```

    5.3 Launch  pre-training script:

    **Note**: If using multi machine training, and no data sharing configuration on the mechines, it's necessary to add the parameter `--no-shared-storage`. This parameter will determine whether non master nodes need to load data based on distributed parameters, and check the corresponding cache and generated data.

    Launch llama-33B pre-training script : ModelLink/examples/llama/pretrain_llama_33B_ptd_32p.sh

    ```bash
    bash examples/llama/pretrain_llama_33B_ptd_32p.sh
    ```

    Launch llama-65B pre-training script : ModelLink/examples/llama/pretrain_llama_65b_ptd.sh

    ```bash
    bash examples/llama/pretrain_llama_65b_ptd.sh
    ```

    Config llama-33B/65B pre-training script for multinode (Launch llama-65B pre-training script on each machine):

    ```shell
    MASTER_ADDR=localhost
    MASTER_PORT=6001
    NNODES=4
    NODE_RANK=0
    ```

6. Finetune

    6.1 Prepare fine-tune dataset

    ```shell
    mkdir finetune_dataset
    cd ./finetune_dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    ```

    LLaMA-33B

    ```shell
    mkdir ./finetune_dataset/llama-33b-hf/
    python ./tools/preprocess_data.py \
        --input ./dataset_llama2/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/llama-33b-hf/ \ 
        --output-prefix ./finetune_dataset/llama-33b-hf/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF \
        --handler-name GeneralInstructionHandler \
        --append-eod
    ```

    LLaMA-65B

    ```shell
    mkdir ./finetune_dataset/llama-65b-hf/
    python ./tools/preprocess_data.py \
        --input ./dataset_llama2/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/llama-65b-hf/  \
        --output-prefix ./finetune_dataset/llama-65b-hf/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF \
        --handler-name GeneralInstructionHandler \
        --append-eod
    ```

    6.2 Config llama-33B/65B fine-tune script :
    Config llama-33B fine-tune script `./examples/llama/tune_llama_33B_ptd_32p.sh`:

    ```bash
    # modify the script according to your own conda and ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    
    # modify script orign dataset path according to your own dataset path
    TOKENIZER_PATH="./model_from_hf/llama-33b-hf/"  #tokenizer path
    DATA_PATH="./finetune_dataset/llama-33b-hf/alpaca"  #processed dataset
    LORA_CHECKPOINT="your lora weight"
    LOAD_CHECKPOINT_PATH="./model_weights/llama-33b-hf-v0.1-tp4-pp4/"
    SAVE_CHECKPOINT_PATH="./ckpt/llama-33b-hf/"
    ```

    Config llama-65B fine-tune script `./examples/llama/tune_llama_65b_ptd.sh`:

    ```bash
    # modify the script according to your own conda and ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    
    # modify script orign dataset path according to your own dataset path
    TOKENIZER_PATH="./model_from_hf/llama-65b-hf/"  #tokenizer path
    DATA_PATH="./finetune_dataset/llama-65b-hf/alpaca"  #processed dataset
    LORA_CHECKPOINT="your lora weight"
    LOAD_CHECKPOINT_PATH="./model_weights/llama-65b-hf-v0.1-tp8-pp4/"
    SAVE_CHECKPOINT_PATH="./ckpt/llama-65b-hf/"
    ```

    Add the fine-tuning parameter `--finetune` so that fine-tuning starts from the first step.

    6.3 Launch fine-tune script:

    Launch llama-33B pre-training script : ModelLink/examples/llama/tune_llama_33B_ptd_32p.sh

    ```bash
    bash examples/llama/tune_llama_33B_ptd_32p.sh
    ```

    Launch llama-65B pre-training script : ModelLink/examples/llama/tune_llama_65b_ptd.sh

    ```bash
    bash examples/llama/tune_llama_65b_ptd.sh
    ```

    Config llama-33B/65B pre-training script for multinode (Launch llama-65B pre-training script on each machine):

    ```shell
    MASTER_ADDR=localhost
    MASTER_PORT=6001
    NNODES=4
    NODE_RANK=0
    ```



## Inference

We support ModelLink Inference for text generation with LLaMA-33B and LLaMA-65B.
Inference different from pre-training, such as we need to Load pre-training checkpoint and the length of the output samples:

Config LLaMA-33B inference script `examples/llama/generate_llama_33b_ptd.sh`.

Config LLaMA-65B inference script `examples/llama/generate_llama_65b_ptd.sh`.

```shell
# modify the model weight path and tokenizer path
CHECKPOINT=<checkpoint-path>
TOKENIZER_PATH=<tokenizer-path>
```

LLaMA-33B:

```shell
bash ./examples/llama/generate_llama_33b_ptd.sh
```

LLaMA-65B:

```shell
bash ./examples/llama/generate_llama_65b_ptd.sh
```

Some inference samples are as follows:

LLaMA-33B:

![llama-13B_generate.png](https://gitee.com/ascend/ModelLink/raw/master/sources/images/llama/llama33B_generate.png)

LLaMA-65B:

![llama-65B_generate.png](https://gitee.com/ascend/ModelLink/raw/master/sources/images/llama/llama-65B_generate.png)

## Evaluation with Numerous Benchmarks

We use Boolq benchmark to evaluate our model. Benchmark Download [here](https://huggingface.co/datasets/boolq).

Config LLaMA-33B evaluation script: examples/llama/evaluate_llama_33B_ptd.sh

Config LLaMA-65B evaluation script: examples/llama/evaluate_llama_65B_ptd.sh

Modify checkpoint path, vocab path, dataset path and task:

```shell
CHECKPOINT=<checkpoint-path>
TOKENIZER_PATH=<tokenizer-path>
DATA_PATH="./boolq/data/test/"
TASK="boolq"
```

Change the max new tokens:

```shell
--max-new-tokens 1 
```

```shell
# start evaluation
# evaluate llama-33B
bash examples/llama/evaluate_llama_33B_ptd.sh
# evaluate llama-65B
bash examples/llama/evaluate_llama_65B_ptd.sh
```

The evaluation performance of LLaMA-33B/65B in **Ascend NPU**:

| Task                                           | Model     | NPU  | Benchmark |
|------------------------------------------------|-----------|------|-----------|
| [Boolq](https://huggingface.co/datasets/boolq) | LLaMA-33B | 83.2 | [83.1](https://paperswithcode.com/sota/question-answering-on-boolq) |
| [Boolq](https://huggingface.co/datasets/boolq) | LLaMA-65B | 85.7 | [86.6](https://paperswithcode.com/sota/question-answering-on-boolq) |

## Citation

You may also consider original work in your reference:

```shell
@article{Touvron2023llama,
  title={LLaMA: OPen and Efficient Foundation Language Models},
  author={Hugo Touvron*, Thibaut Lavril*, Gautier Izacard*, Xavier Martinet, Marie-Anne Lachaux, Timothee Lacroix, Baptiste Roziere, Naman Goyal,
  Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave*, Guillaume Lample*},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}}
```
