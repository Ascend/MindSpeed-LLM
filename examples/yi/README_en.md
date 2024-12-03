# Yi $\color{black}{\rm\tiny{【Model}}$ $\color{black}{\rm\tiny{contributed}}$ $\color{black}{\rm\tiny{by}}$ $\color{black}{\rm\tiny{community】}}$
<p align="left">
        <b><a href="README.md">简体中文</a></b> |
        <b>English</b> 
    </p>
</p>


#  Contents
- [Yi-34B](#yi-34b)
  - [Training](#training)
    - [Script](#script)
  - [Inference](#inference)
  - [Evaluation](#evaluation)




# Yi-34B

## Training

Here's a hardware summary of pre-training Yi-34B:

| Hardware |                      Value                      |
| :------: | :---------------------------------------------: |
|   NPU    |               16 x Ascend NPUs                   |

### Script

1. Clone the repository to your local server:

    ```bash
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
    # 1).python3.8
    conda create -n test python=3.8
    conda activate test
    
    # 2).install torch and torch_npu 
    pip install torch-2.2.0-cp38-cp38m-linux_aarch64.whl
    pip install torch_npu-2.2.0.XXX-cp38-cp38m-linux_aarch64.whl
    pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl
    
    # 3).modify the path according to your own  ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    
    # 4).install MindSpeed
    git clone https://gitee.com/ascend/MindSpeed.git
    cd MindSpeed
    git checkout 2b0edd2
    pip install -r requirements.txt 
    
    pip3 install -e .
    cd ..
    
    # 5).install other packages
    pip install -r requirements.txt 
    ```
    
3. Prepare pretrained weights

    Download the Yi-34B checkpoint from [here](https://huggingface.co/yi/Yi-34b-hf/tree/main) 

    ```shell
    mkdir ./model_from_hf/Yi-34B/
    cd ./model_from_hf/Yi-34B/
    wget https://huggingface.co/01-ai/Yi-34B/resolve/main/config.json
    wget https://huggingface.co/01-ai/Yi-34B/resolve/main/generation_config.json
    wget https://huggingface.co/01-ai/Yi-34B/resolve/main/pytorch_model-00001-of-00007.bin
    wget https://huggingface.co/01-ai/Yi-34B/resolve/main/pytorch_model-00002-of-00007.bin
    wget https://huggingface.co/01-ai/Yi-34B/resolve/main/pytorch_model-00003-of-00007.bin
    wget https://huggingface.co/01-ai/Yi-34B/resolve/main/pytorch_model-00004-of-00007.bin
    wget https://huggingface.co/01-ai/Yi-34B/resolve/main/pytorch_model-00005-of-00007.bin
    wget https://huggingface.co/01-ai/Yi-34B/resolve/main/pytorch_model-00006-of-00007.bin
    wget https://huggingface.co/01-ai/Yi-34B/resolve/main/pytorch_model-00007-of-00007.bin
    wget https://huggingface.co/01-ai/Yi-34B/resolve/main/pytorch_model.bin.index.json
    wget https://huggingface.co/01-ai/Yi-34B/resolve/main/tokenizer.json
    wget https://huggingface.co/01-ai/Yi-34B/resolve/main/tokenizer.model
    wget https://huggingface.co/01-ai/Yi-34B/resolve/main/tokenizer_config.json
    cd ../../
    ```

4. Weights convert

    4.1 In order to adapt to the Yi-34B model, the following script is used to convert the model pre-training weights.
    ***(This scenario is generally used to train open-source HuggingFace models on Megatron)***

    ```shell
    # modify the ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
      
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --loader llama2_hf \
        --saver megatron \
        --target-tensor-parallel-size 8 \
        --target-pipeline-parallel-size 2 \
        --load-dir ./model_from_hf/Yi-34B/ \
        --save-dir ./model_weights/Yi-34B-Base-v0.1-tp8-pp2/ \
        --tokenizer-model ./model_from_hf/Yi-34B/tokenizer.model \
        --params-dtype bf16
    ```
    For inference or evaluation tasks, set the `--target-pipeline-parallel-size` value to `1` and change the `pp2` value to `pp1` in the `--save-dir` value.

    4.2 Any Megatron weights with parallel slicing strategy --> Any Megatron weights with parallel slicing strategy
    ***(This scenario is generally used to convert the trained megatron model back to the HuggingFace format)***

    ```shell
    # Modify the ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python tools/checkpoint/convert_ckpt.py --model-type GPT \
        --loader megatron \
        --saver megatron \
        --save-model-type save_huggingface_llama \
        --load-dir ./model_weights/Yi-34B-Base-v0.1-tp8-pp2/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --save-dir ./model_from_hf/Yi-34B/   # <-- Fill in the original HF model path here, new weights will be saved in ./model_from_hf/Yi-34B/mg2hg/
    ```

5. Pre-training
   
    5.1 Prepare dataset

    Download the Yi-34B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet) 

    ```shell
    # download datasets
    cd ./dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
   
    # process datasets          
    mkdir ./dataset/Yi-34B/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/Yi-34B/ \
        --output-prefix ./dataset/Yi-34B/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF
    ```
    5.2 pre-training
    
    Config Yi-34B pre-training script : examples/yi/pretrain_yi_34b_ptd_16p.sh

    ```shell
    # modify the script according to your own  ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
   
    CKPT_SAVE_DIR="./ckpt/Yi-34B/"
    DATA_PATH="./dataset/Yi-34B/alpaca_text_document"
    TOKENIZER_MODEL="./model_from_hf/Yi-34B/tokenizer.model"
    CKPT_LOAD_DIR="./model_weights/Yi-34B-v0.1-tp8-pp2/"
    ```

    Launch Yi-34B  pre-training script: examples/yi/pretrain_yi_34b_ptd_16p.sh

    ```shell
    bash examples/yi/pretrain_yi_ptd_34B.sh 
    ```
    **Note**: If using multi machine training, and no data sharing configuration on the mechines, it's necessary to add the parameter `--no-shared-storage`. This parameter will determine whether non master nodes need to load data based on distributed parameters, and check the corresponding cache and generated data.

6. fine-tuning

    6.1 Prepare fine-tuning dataset

    Download the fine-tuning datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)


    ```shell
    # download datasets
    mkdir finetune_dataset
    cd ./finetune_dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    
    # process datasets  
    mkdir ./finetune_dataset/Yi-34B/
    python ./tools/preprocess_data.py \
      --input ./finetune_dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
      --tokenizer-name-or-path ./model_from_hf/Yi-34B/ \
      --output-prefix ./finetune_dataset/Yi-34B/alpaca \
      --workers 4 \
      --log-interval 1000 \
      --tokenizer-type PretrainedFromHF \
      --handler-name GeneralInstructionHandler \
      --append-eod
    ```
    
    6.2 Full Parameters Fine-Tuning yi_34B
    
    The configuration script for full parameters fine-tuning  is basically the same as that for pretrain_yi_34b_ptd_16p.sh.*The difference is that the dataset and the training parameter is-instruction-dataset are added.*
    
    Add the fine-tuning parameter `--finetune` so that fine-tuning starts from the first step.
    
    ```bash
    CKPT_SAVE_DIR="./ckpt/Yi-34B/"
    DATA_PATH="./finetune_dataset/Yi-34B/alpaca"
    TOKENIZER_PATH="./model_from_hf/Yi-34B/"
    CKPT_LOAD_DIR="./model_weights/Yi-34B-Base-v0.1-tp8-pp2/"
    
      --finetune \
      --is-instruction-dataset \
      --tokenizer-type PretrainedFromHF \
      --tokenizer-name-or-path ${TOKENIZER_PATH} \
      --tokenizer-not-use-fast \
    ``` 


## Inference

Config Yi-34B inference script: examples/yi/generate_yi_34b_ptd.sh

```bash
# modify the script according to your own ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
# modify script model path and tokenizer path
CHECKPOINT="./model_weights/Yi-34B-v0.1-tp8-pp1/"
TOKENIZER_PATH="./model_from_hf/Yi-34B/"
```

Launch Yi-34B inference script: examples/yi/generate_yi_34b_ptd.sh

```bash
bash examples/yi/generate_yi_34b_ptd.sh
```

Some inference samples are as follows:
![Inference](https://gitee.com/ascend/MindSpeed-LLM/raw/master/sources/images/yi/yi-34b-generate.png)

## Evaluation

We use the boolq benchmark to evaluate our model. Benchmark [Download](https://huggingface.co/datasets/cais/mmlu).

```shell
# config origin weight and vocab file path
CHECKPOINT="./model_weights/Yi-34B-v0.1-tp8-pp1/"
TOKENIZER_PATH="./model_from_hf/Yi-34B/"
# config tasks and dataset path
DATA_PATH="./mmlu/"
TASK="mmlu"
```

```shell
bash ./examples/yi/evaluate_yi_34b_ptd.sh
```

<table>
  <thead>
    <tr>
      <th>Task</th>
      <th>Model</th>
      <th>NPU</th>
      <th>OpenSource</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://huggingface.co/datasets/cais/mmlu">MMLU</a></td>
      <th>Yi 34B</th>
      <td>75.8</td>
      <td><a href="https://hub.opencompass.org.cn/dataset-detail/MMLU">76.3</a></td>
    </tr>
  </tbody>
</table>
