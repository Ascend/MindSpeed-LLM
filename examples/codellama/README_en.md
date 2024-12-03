# CodeLlama $\color{black}{\rm\tiny{【Model}}$ $\color{black}{\rm\tiny{contributed}}$ $\color{black}{\rm\tiny{by}}$ $\color{black}{\rm\tiny{Community】}}$ 
<p align="left">
        <b><a href="README.md">简体中文</a></b> |
        <b>English</b> 
    </p>
</p>


#  Contents

- [CodeLlama-34B](#codellama-34b)
  - [Training](#training)
    - [Script](#script)
  - [Inference](#inference)
  - [Evaluation](#evaluation)




# CodeLlama-34B

## Training

Here's a hardware summary of pre-training CodeLlama-34B:

| Hardware |                      Value                      |
| :------: | :---------------------------------------------: |
|   NPU    |               16 x Ascend NPUs                   |

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
    pip install torch-2.2.0-cp38-cp38m-linux_aarch64.whl
    pip install torch_npu-2.2.0.XXX-cp38-cp38m-linux_aarch64.whl
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

    Download the CodeLlama-34B checkpoint from [here](https://huggingface.co/codellama/CodeLlama-34b-hf/tree/main) 

    ```shell
    mkdir ./model_from_hf/CodeLlama-34B/
    cd ./model_from_hf/CodeLlama-34B/
    wget https://huggingface.co/codellama/CodeLlama-34b-hf/resolve/main/config.json
    wget https://huggingface.co/codellama/CodeLlama-34b-hf/resolve/main/generation_config.json
    wget https://huggingface.co/codellama/CodeLlama-34b-hf/resolve/main/pytorch_model-00001-of-00007.bin
    wget https://huggingface.co/codellama/CodeLlama-34b-hf/resolve/main/pytorch_model-00002-of-00007.bin
    wget https://huggingface.co/codellama/CodeLlama-34b-hf/resolve/main/pytorch_model-00003-of-00007.bin
    wget https://huggingface.co/codellama/CodeLlama-34b-hf/resolve/main/pytorch_model-00004-of-00007.bin
    wget https://huggingface.co/codellama/CodeLlama-34b-hf/resolve/main/pytorch_model-00005-of-00007.bin
    wget https://huggingface.co/codellama/CodeLlama-34b-hf/resolve/main/pytorch_model-00006-of-00007.bin
    wget https://huggingface.co/codellama/CodeLlama-34b-hf/resolve/main/pytorch_model-00007-of-00007.bin
    wget https://huggingface.co/codellama/CodeLlama-34b-hf/resolve/main/pytorch_model.bin.index.json
    wget https://huggingface.co/codellama/CodeLlama-34b-hf/resolve/main/special_tokens_map.json
    wget https://huggingface.co/codellama/CodeLlama-34b-hf/resolve/main/tokenizer.json
    wget https://huggingface.co/codellama/CodeLlama-34b-hf/resolve/main/tokenizer.model
    wget https://huggingface.co/codellama/CodeLlama-34b-hf/resolve/main/tokenizer_config.json
    cd ../../
    ```

4. Weights convert

    4.1 In order to adapt to the CodeLlama-34B model, the following script is used to convert the model pre-training weights.
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
        --load-dir ./model_from_hf/CodeLlama-34B/ \
        --save-dir ./model_weights/CodeLlama-34B-Base-v0.1-tp8-pp2/ \
        --tokenizer-model ./model_from_hf/CodeLlama-34B/tokenizer.model \
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
        --load-dir ./model_weights/CodeLlama-34B-Base-v0.1-tp8-pp2/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --save-dir ./model_from_hf/CodeLlama-34B/   # <-- Fill in the original HF model path here, new weights will be saved in ./model_from_hf/CodeLlama-34B/mg2hg/
    ```

5. Pre-training
   
    5.1 Prepare dataset

    Download the CodeLlama-34B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet) 

    ```shell
    # download datasets
    cd ./dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..

    # process datasets          
    mkdir ./dataset/CodeLlama-34B/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/CodeLlama-34B/ \
        --output-prefix ./dataset/CodeLlama-34B/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF
    ```

    5.2 Pre-training
    
    Config CodeLlama-34B pre-training script : examples/codellama/pretrain_codellama_34b_ptd_16p.sh

    ```shell
    # modify the script according to your own  ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    CKPT_SAVE_DIR="./ckpt/CodeLlama-34B/"
    DATA_PATH="./dataset/CodeLlama-34B/alpaca_text_document"
    TOKENIZER_MODEL="./model_from_hf/CodeLlama-34B/tokenizer.model"
    CKPT_LOAD_DIR="./model_weights/CodeLlama-34B-v0.1-tp8-pp2/"
    ```

    Launch CodeLlama-34B  pre-training script: examples/codellama/pretrain_codellama_34b_ptd_16p.sh

    ```shell
    bash examples/codellama/pretrain_codellama_34b_ptd_16p.sh 
    ```
    **Note**: If using multi machine training, it is necessary to set up multi machine data sharing, and non primary nodes can read the primary node data through data sharing. Alternatively, directly copy the data generated by the master node to non master nodes.

6. Fine-tuning

    6.1 Prepare fine-tuning dataset

    Download the fine-tuning datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

    ```shell
    # download datasets
    mkdir finetune_dataset
    cd ./finetune_dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..

    # process datasets  
    mkdir ./finetune_dataset/CodeLlama-34B/
    python ./tools/preprocess_data.py \
        --input ./finetune_dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/CodeLlama-34B/ \
        --output-prefix ./finetune_dataset/CodeLlama-34B/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF \
        --handler-name GeneralInstructionHandler \
        --append-eod
    ```

    6.2 Full Parameters Fine-Tuning

    The configuration script for full parameters fine-tuning  is basically the same as that for pretrain_codellama_34b_ptd_16p.sh.*The difference is that the dataset and the training parameter `is-instruction-dataset` and `padded-vocab-size 32000` are added.*

    Add the fine-tuning parameter `--finetune` so that fine-tuning starts from the first step.

    ```bash
    DATA_PATH="./finetune_dataset/CodeLlama-34B/alpaca"
    TOKENIZER_PATH="./model_from_hf/CodeLlama-34B/"
    CKPT_SAVE_DIR="./ckpt/CodeLlama-34B/"
    CKPT_LOAD_DIR="./model_weights/CodeLlama-34B-Base-v0.1-tp8-pp2/" 
        --finetune \
        --is-instruction-dataset \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path ${TOKENIZER_PATH} \
        --tokenizer-not-use-fast \
        --padded-vocab-size 32000 \
    ```



## Inference

Config CodeLlama-34B inference script: examples/codellama/generate_codellama_34b_ptd.sh

```bash
# modify the script according to your own ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
# modify script model path and tokenizer path
CHECKPOINT="./model_weights/CodeLlama-34B-v0.1-tp8-pp1/"
TOKENIZER_PATH="./model_from_hf/CodeLlama-34B/"
```

Launch CodeLlama-34B inference script: examples/codellama/generate_codellama_34b_ptd.sh

```bash
bash examples/codellama/generate_codellama_34b_ptd.sh
```

Some inference samples are as follows:

![Inference](https://gitee.com/ascend/MindSpeed-LLM/raw/master/sources/images/codellama/codellama-34b-generate.png)

## Evaluation

We use the boolq benchmark to evaluate our model. Benchmark [Download](https://github.com/openai/human-eval/blob/master/data/HumanEval.jsonl.gz).

```shell
# config origin weight and vocab file path
CHECKPOINT=<origin-ckpt-path>
TOKENIZER_PATH=<tokenizer-path>
# config tasks and dataset path
DATA_PATH="./human_eval/"
TASK="human_eval"
```

```shell
bash ./examples/codellama/evaluate_codellama_34b_ptd.sh
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
      <td><a href="https://huggingface.co/datasets/openai_humaneval">human_eval</a></td>
      <th>CodelLlama 34B</th>
      <td>0.4878</td>
      <td><a href="https://paperswithcode.com/sota/code-generation-on-humaneval">0.488</a></td>
    </tr>
  </tbody>
</table>
