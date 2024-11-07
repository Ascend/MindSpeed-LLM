# Aquila-7B

<p align="left">
        <b><a href="README.md">简体中文</a></b> |
        <b>English</b> 
    </p>
</p>

- [Aquila-7B](#aquila-7b)
  - [Training](#training)
    - [Script](#script)
  - [Inference](#inference)
  - [Evaluation with Benchmark](#evaluation-with-benchmark)

## Training

Here's a hardware summary of pre-training Aquila-7B:

| Hardware | Device |      Value      |
| :------: | :----: | :-------------: |
|   NPU   |  A+K  | 8 x Ascend NPUs |

### Script

1. Clone the repository to your local server and switch to modellink 1.1 branch:

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

2. Build conda environment

    ```bash
    # python3.8
    conda create -n test python=3.8
    conda activate test
    # install torch, torch_npu and apex
    pip install torch==2.1.0
    pip install torch_npu-2.1.0.postxxxx-cp38-cp38-xxxx_aarch64.whl
    pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

    # source the set_env.sh file based on your host settings(you may need to change the path)
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    # use git to clone the MindSpeed source code, enter the directory, then install mindspeed package by source code
    git clone https://gitee.com/ascend/MindSpeed.git
    cd MindSpeed/
    git checkout 2b0edd2
    pip install -r requirements.txt
    pip install -e .
    cd ..

    # install other packages
    pip install -r requirements.txt
    ```

3. Download the Aquila-7B model, config, and tokenizer from [here](https://huggingface.co/BAAI/Aquila-7B/tree/main)

    save to ModelLink/model_from_hf/Aquila7B/ directory.


    Prepare dataset.
    
    step1: Download the datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet), save to ModelLink/dataset/ directory.
    
    ```shell
    cd dataset/
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    ```
    
    step2: use Aquila-7B specified tokenizer to pre-process data:
    
    ```shell
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    mkdir ./dataset/Aquila-7B/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/Aquila-7B/ \
        --output-prefix ./dataset/Aquila-7B/alpaca \
        --workers 4 \
        --log-interval 1000  \
        --tokenizer-type PretrainedFromHF
    ```

4. Weights convert

    HuggingFace weights --> Megatron weights
    ***(This scenario is generally used to train open-source HuggingFace models on Megatron)***

    ```shell
    # please modify the path to set_env.sh based on your environment.
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --load-dir ./model_from_hf/Aquila-7B/ \
        --save-dir ./model_weights/Aquila-7B-v0.1-tp8-pp1/ \
        --loader llama2_hf \
        --saver megatron \
        --target-tensor-parallel-size 8 \
        --tokenizer-model ./model_from_hf/Aquila-7B/tokenizer.json
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
        --load-dir ./model_weights/Aquila-7B-v0.1-tp8-pp1/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --save-dir ./model_from_hf/Aquila-7B/   # <-- Fill in the original HF model path here, new weights will be saved in ./model_from_hf/Aquila-7B/mg2hg/
    ```

5. Config Aquila-7B pre-training script.

    Config the environment variables in aquila pretrain script

    ```shell
    # set dataset path, CKPT load path for loading weights, and the tokenizer path
    TOKENIZER_PATH="./model_from_hf/Aquila-7B/"  #tokenizer path
    DATA_PATH="./dataset/Aquila-7B/alpaca_text_document"  #processed dataset
    CKPT_LOAD_DIR="./model_weights/Aquila-7B-v0.1-tp8-pp1/"   # pointing to the converted model weights
    CKPT_SAVE_DIR="./ckpt/Aquila-7B/"                   # pointing to the path to save checkpoints
    ```

    *Note that if you do not load weights for pre-training, you can ignore CKPT_LOAD_DIR, and remove the `--load` parameter from the training script, and vice versa*
    *If you do not want to save weights during pre-training, you can ignore CKPT_SAVE_DIR, and remove the `--save $CKPT_SAVE_DIR` parameter from the training script, and vice versa*
    *When you want to save checkpoint and load it in future pre-training, just follow the above "save" and "load" suggestions.*

6. Launch Aquila-7B pre-training script.

    Before running the pre-training script, please execute the set_env.sh script first to setup environment variables. Alternatively, you can do this inside aquila pre-training script.

    ```shell
    # you may need to change the path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    ```

    Start pre-training Aquila-7B model:

    ```shell
    bash examples/aquila/pretrain_aquila_7b_ptd.sh
    ```

    **Note**: If using multi machine training, and no data sharing configuration on the mechines, it's necessary to add the parameter `--no-shared-storage`. This parameter will determine whether non master nodes need to load data based on distributed parameters, and check the corresponding cache and generated data.



## Inference

We support MindSpeed Inference for text generation with Aquila 7B model.

Inference is different from pre-training because it requires loading the pre-trained model weights. Therefore, we need to complete the aforementioned model weight conversion task first, then configure the Aquila-7B Inference shell script `examples/aquila/generate_aquila_7b_ptd.sh`. "CKPT_LOAD_DIR" must point to the converted weights directory, and "TOKENIZER_PATH" must point to the directory which contains Aquila vocabulary files -- in our example, it is "./model_from_hf/Aquila-7B/". In your operation, please fill in correct value based on your actual scenario.

```shell
# please change to actual values
CKPT_LOAD_DIR="./model_weights/Aquila-7B-v0.1-tp8-pp1/"
TOKENIZER_PATH="./model_from_hf/Aquila-7B/"
```

Start Aquila-7B Inference:

```shell
bash ./examples/aquila/generate_aquila_7b_ptd.sh
```

Sample results of Aquila-7B Inference:

![aquila-7B_generate.png](https://gitee.com/ascend/ModelLink/raw/master/sources/images/aquila/aquila_7B_generate.png)

## Evaluation with Benchmark

We use BoolQ benchmark to evaluate our model. You can [go to the BoolQ Benchmark page](https://github.com/google-research-datasets/boolean-questions) and find the [dataset](https://storage.cloud.google.com/boolq/dev.jsonl), download it and save it. For example, save to "ModelLink/boolq/test" directory

Evaluation task is similar to inference task too，it also requires loading the pre-trained model weights. Please note that the model structure parameters used in converting weights should be consistent with those used in running the evaluation task.

After weight conversion is complete, we configure the Aquila-7B evaluation script `examples/aquila/evaluate_aquila_7b_ptd.sh`. We need to correctly specify the path to load weights, the path to tokenizer and vocab, and so on (the following example is for reference only)

```shell
CKPT_LOAD_DIR="./model_weights/Aquila-7B-v0.1-tp8-pp1/"
TOKENIZER_PATH="./model_from_hf/Aquila-7B/"
EVAL_DATA_PATH="./boolq/test"
TASK="boolq"
```

Start evaluation task

```shell
bash ./examples/aquila/evaluate_aquila_7b_ptd.sh
```

Sample Aquila-7B performance running in **Ascend NPU**:

| Task                                                                   | Model     | NPU   | Benchmark |
| ---------------------------------------------------------------------- | --------- | ----- | --------- |
| [BoolQ](https://github.com/google-research-datasets/boolean-questions) | Aquila-7B | 77.3% |           |
