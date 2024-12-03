# LLaMA  $\color{black}{\rm\tiny{【Model}}$ $\color{black}{\rm\tiny{contributed}}$ $\color{black}{\rm\tiny{by}}$ $\color{black}{\rm\tiny{Ascend】}}$ 
<p align="left">
        <b><a href="README.md">简体中文</a></b> |
        <b>English</b> 
</p>

#  Contents

- [LLaMA2-7B](#contents)
  - [Training](#pre-training)
  - [Script](#script)
  - [Performance](#performance)
    - [Machine performance](#machine-performance)
  - [Inference](#Inference-7B)
  - [Evaluation](#Evaluation-7B)

- [LLaMA2-13B](#contents)
  - [Training](#pre-training)
    - [Script](#script)
    - [Performance](#performance)
      - [Machine performance](#machine-performance)
    - [Inference](#Inference)
    - [Evaluation](#Evaluation)

- [LLaMA2-34B/70B](#LLaMA2-34b/70b)
  - [Training](#training-2)
    - [Script](#script-2)
    - [Performance](#performance-2)
      - [Machine performance](#machine-performance-2)
  - [Inference](#inference-2)
  - [Evaluation](#Evaluation-2)

# LLAMA2-7B

## Training

Here's a hardware summary of pre-training  LLAMA2-7B:

| Hardware |                      Value                      |
| :------: | :---------------------------------------------: |
|   NPU    |               8 x Ascend NPUs                   |

### Script

1. Clone the repository to your local server:

    ```shell
    git clone https://gitee.com/ascend/MindSpeed-LLM.git 
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    git checkout -f bcce6f
    cp -r megatron ../MindSpeed-LLM/
    cd ..
    cd MindSpeed-LLM
	git checkout 1.0
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
    git checkout 224ae35e8fc96778f957029d1371ddb623452a50
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
    Download the LLAMA2-7B checkpoint from [here](https://huggingface.co/daryl149/llama-2-7b-hf/tree/main)

    ```shell
      #!/bin/bash
      mkdir ./model_from_hf/llama-2-7b-hf/
      cd ./model_from_hf/llama-2-7b-hf/
      wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/config.json
      wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/generation_config.json
      wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/pytorch_model-00001-of-00002.bin
      wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/pytorch_model-00002-of-00002.bin
      wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/pytorch_model.bin.index.json
      wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/special_tokens_map.json
      wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/tokenizer.json
      wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/tokenizer.model
      wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/tokenizer_config.json
      cd ../../
    ```
4. weight conversion in ptd mode

   *Note that if you want to use the weight from huggingface, please run the weight conversion script first. The following uses llama-2-7b model weight conversion in ptd as an example.*

   ```bash
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # convert to ptd weights
    python tools/checkpoint/util.py \
        --model-type GPT \
        --loader llama2_hf \
        --saver megatron \
        --target-tensor-parallel-size 8 \
        --target-pipeline-parallel-size 1 \
        --load-dir ./model_from_hf/llama-2-7b-hf/ \
        --save-dir ./model_weights/llama-2-7b-hf-v0.1-tp8-pp1/ \
        --tokenizer-model ./model_from_hf/llama-2-7b-hf/tokenizer.model
    ```

    Any Megatron weights with parallel slicing strategy --> Any Megatron weights with parallel slicing strategy

    ```shell
    # Modify the ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python tools/checkpoint/util.py \
        --model-type GPT \
        --loader megatron \
        --saver megatron \
        --save-model-type save_huggingface_llama \
        --load-dir ./model_weights/llama-2-7b-hf-v0.1-tp8-pp1/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --save-dir ./model_from_hf/llama-2-7b-hf/  # <-- Fill in the original HF model path here, new weights will be saved in ./model_from_hf/llama-2-7b-hf/mg2hg/
    ```
    
    Weight conversion is suitable for pre-training, fine-tuning, inference and evaluation. Adjust the parameters `target-tensor-parallel-size` and `target-pipeline-parallel-size` according to different tasks.
1. pre-training

    5.1 Prepare dataset

    Download the LLAMA2-7B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

    ```shell
    # download datasets
    cd ./dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..

    # process datasets  
    mkdir ./dataset/llama-2-7b-hf/
    python ./tools/preprocess_data.py \
      --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
      --tokenizer-name-or-path ./model_from_hf/llama-2-7b-hf/ \
      --output-prefix ./dataset/llama-2-7b-hf/alpaca \
      --workers 4 \
      --log-interval 1000 \
      --tokenizer-type PretrainedFromHF
    ```

    5.2 pre-training using ptd mode
    Config LLAMA2-7B pre-training script: examples/llama2/pretrain_llama2_7b_ptd.sh

1. pre-training

      # modify config according to your own actual situation
      CKPT_SAVE_DIR="./ckpt/llama-2-7b-hf/"
      TOKENIZER_MODEL="./model_from_hf/llama-2-7b-hf/tokenizer.model"  #tokenizer path
      DATA_PATH="./dataset/llama-2-7b-hf/alpaca_text_document"  #processed dataset
    ```

    Download the LLAMA2-7B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

	Launch LLAMA2-7B  pre-training script: examples/llama2/pretrain_llama2_7b_ptd.sh
   
   ```shell
    bash examples/llama2/pretrain_llama2_7b_ptd.sh 
   ```

    **Note**: If using multi machine training, and no data sharing configuration on the mechines, it's necessary to add the parameter `--no-shared-storage`. This parameter will determine whether non master nodes need to load data based on distributed parameters, and check the corresponding cache and generated data.   

6. fine-tuning

    6.1 Prepare fine-tuning dataset
    Download the LLAMA2-7B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

    ```shell
    # download datasets
    mkdir finetune_dataset
    cd ./finetune_dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..

    # process datasets  
    mkdir ./finetune_dataset/llama-2-7b-hf/
    python ./tools/preprocess_data.py \
      --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
      --tokenizer-name-or-path ./model_from_hf/llama-2-7b-hf/ \
      --output-prefix ./finetune_dataset/llama-2-7b-hf/alpaca \
      --workers 4 \
      --log-interval 1000 \
      --tokenizer-type PretrainedFromHF \
      --handler-name GeneralInstructionHandler \
      --append-eod
    ```

   6.2 Full Parameters Fine-Tuning
   The configuration script for full parameters fine-tuning  is basically the same as that for pretrain_llama2_7b_ptd.sh.*The difference is that the dataset and the training parameter is-instruction-dataset are added.*

   Add the fine-tuning parameter `--finetune` so that fine-tuning starts from the first step.

   ```bash
   DATA_PATH="./finetune_dataset/llama-2-7b-hf/alpaca"
   TOKENIZER_PATH="./model_from_hf/llama-2-7b-hf/"
   CKPT_PATH="./ckpt/llama-2-7b-hf/"
   --load ${CKPT_PATH} \
   --finetune \
   --is-instruction-dataset \
   --tokenizer-type PretrainedFromHF \
   --tokenizer-name-or-path ${TOKENIZER_PATH} \
   --tokenizer-not-use-fast \
   ```

   6.3 Lora Fine-Tuning
   The Lora fine-tuning script is configured by adding the following lora parameters to the pretrain_llama2_7b_ptd.sh script:

   ```bash
       --lora-target-modules query_key_value dense proj dense_4h_to_h \
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
   
   Launch LLAMA2-7B lora fine tune script: tasks/finetune/tune_llama2_7b_ptd.sh
   
   ```shell
    bash tasks/finetune/tune_llama2_7b_ptd.sh 
   ```

### Performance

#### Machine performance

The performance of LLaMA2-7B in **Ascend NPU** and **Reference**:

| Device | Model       | total Iterations | throughput rate (samples/s/p) | throughput rate (tokens/s/p) | single-step time (s/step) |
| :------: | :-----------: | :----------------: | :-----------------------------: | :----------------------------: | :-------------------------: |
| NPUs   | LLaMA2-7B | 1024             | 5.19                      | 2730                      | 3.08                   |
| Reference   | LLaMA2-7B | 1024             | 5.63                      | 2884                       | 2.84                   |



## Inference-7B
Config llama2-7B inference script: tasks/inference/generate_llama2_7b_ptd.sh
```bash
# modify the script according to your own ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
# modify script model path and tokenizer path
CHECKPOINT="./model_weights/llama-2-7b-hf-v0.1-tp8-pp1"
TOKENIZER_PATH="./model_from_hf/llama-2-7b-hf/"
TOKENIZER_MODEL="./model_from_hf/llama-2-7b-hf/tokenizer.model"
```
Config llama2-7B lora inference script: tasks/inference/generate_llama2_7b_lora_ptd.sh
```bash
# modify lora model path
CHECKPOINT_LORA="your lora model directory path"
```
Launch llama2-7B inference script: tasks/inference/generate_llama2_7b_ptd.sh
```bash
bash examples/llama2/generate_llama2_7b_ptd.sh
```
Launch llama2-7B lora inference script: tasks/inference/generate_llama2_7b_lora_ptd.sh
```bash
bash examples/llama2/generate_llama2_7b_lora_ptd.sh
```

Some inference samples are as follows:
![Inference](../../sources/images/llama2/llama2-7B-generate.png)

## Evaluation-7B
We use MMLU benchmark to evaluate our model. Benchmark Download [here](https://huggingface.co/datasets/cais/mmlu). 
Config llama2-7B evaluation script: tasks/evaluation/evaluate_llama2_7B_ptd.sh

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# modify script model path and tokenizer path
TOKENIZER_PATH="./model_from_hf/llama-2-7b-hf/"  #tokenizer path
CHECKPOINT="./model_weights/llama-2-7b-hf-v0.1-tp8-pp1"  #model path
# configure task and data path
DATA_PATH="./mmlu/data/test/"
TASK="mmlu"
```

Launch llama2-7B evaluation script:

```bash
bash tasks/evaluation/evaluate_llama2_7B_ptd.sh
```

Evaluation results

```text
                           subject_name  question_n   acc_ref   acc_npu  score_diff
17                     public_relations         110  0.563636  0.554545      0.009091
44                         econometrics         114  0.368421  0.377193      0.008772
30               electrical_engineering         145  0.503448  0.510345      0.006897
5                       world_religions         171  0.701754  0.707602      0.005848
25               high_school_us_history         204  0.647059  0.651961      0.004902
45                          human_aging         223  0.596413  0.600897      0.004484
38                            marketing         234  0.709402  0.713675      0.004274
55            high_school_world_history         237  0.620253  0.624473      0.004219
31           high_school_microeconomics         238  0.420168  0.424370      0.004202
7                             nutrition         306  0.503268  0.500000      0.003268
56                  high_school_biology         310  0.541935  0.545161      0.003226
20                           philosophy         311  0.569132  0.565916      0.003215
24               elementary_mathematics         378  0.291005  0.293651      0.002646
22               high_school_psychology         545  0.645872  0.647706      0.001835
12                     professional_law        1534  0.339635  0.340939      0.001304
13                        miscellaneous         783  0.679438  0.678161      0.001277
6                       moral_scenarios         895  0.221229  0.222346      0.001117
37  high_school_government_and_politics         193  0.694301  0.694301      0.000000
54                           prehistory         324  0.555556  0.555556      0.000000
53                    us_foreign_policy         100  0.700000  0.700000      0.000000
39                high_school_geography         198  0.626263  0.626263      0.000000
40                     security_studies         245  0.522449  0.522449      0.000000
41                high_school_chemistry         203  0.408867  0.408867      0.000000
52                   clinical_knowledge         265  0.513208  0.513208      0.000000
49              professional_psychology         612  0.482026  0.482026      0.000000
42                           management         103  0.679612  0.679612      0.000000
43                        jurisprudence         108  0.583333  0.583333      0.000000
51                    computer_security         100  0.560000  0.560000      0.000000
50                   conceptual_physics         235  0.417021  0.417021      0.000000
35                      human_sexuality         131  0.526718  0.526718      0.000000
46                             virology         166  0.439759  0.439759      0.000000
47                       moral_disputes         346  0.514451  0.514451      0.000000
48                              anatomy         135  0.459259  0.459259      0.000000
36                      college_physics         102  0.215686  0.215686      0.000000
0            high_school_macroeconomics         390  0.420513  0.420513      0.000000
34                  high_school_physics         151  0.311258  0.311258      0.000000
33             college_computer_science         100  0.420000  0.420000      0.000000
2                     international_law         121  0.636364  0.636364      0.000000
3                   college_mathematics         100  0.330000  0.330000      0.000000
4                      college_medicine         173  0.410405  0.410405      0.000000
8                high_school_statistics         216  0.314815  0.314815      0.000000
9                      medical_genetics         100  0.450000  0.450000      0.000000
10                    college_chemistry         100  0.290000  0.290000      0.000000
11              professional_accounting         282  0.411348  0.411348      0.000000
14                            sociology         201  0.601990  0.601990      0.000000
15                professional_medicine         272  0.452206  0.452206      0.000000
16                    logical_fallacies         163  0.521472  0.521472      0.000000
18                      college_biology         144  0.506944  0.506944      0.000000
19         high_school_european_history         165  0.575758  0.575758      0.000000
21                     abstract_algebra         100  0.280000  0.280000      0.000000
23         high_school_computer_science         100  0.430000  0.430000      0.000000
26                     machine_learning         112  0.375000  0.375000      0.000000
27                            astronomy         152  0.500000  0.500000      0.000000
1                          formal_logic         126  0.222222  0.222222      0.000000
29              high_school_mathematics         270  0.259259  0.259259      0.000000
32                      business_ethics         100  0.450000  0.450000      0.000000
28                         global_facts         100  0.380000  0.380000      0.000000
```
|  dataset | subject_num | question_num | reference_acc |NPU acc|
|:---:|:-----------:|:------------:|:-------------:|:---:|
| MMLU |     57      |    14042     |    0.4691     |0.4698|

# LLaMA2-13B

## Training

Here's a hardware summary of pre-training  LLaMA2-13B:

| Hardware |                      Value                      |
| :------: | :---------------------------------------------: |
|   NPU    |               8 x Ascend NPUs                   |

### Script

1. Clone the repository to your local server:

    ```shell
    git clone https://gitee.com/ascend/MindSpeed-LLM.git 
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    git checkout -f bcce6f
    cp -r megatron ../MindSpeed-LLM/
    cd ..
    cd MindSpeed-LLM
    git checkout 1.0
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

    # install torch 和 torch_npu
    pip install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl
    pip install torch_npu-2.1.0*-cp38-cp38m-linux_aarch64.whl
    pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

    # modify ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # install MindSpeed
    git clone https://gitee.com/ascend/MindSpeed.git
    cd MindSpeed
    git checkout 224ae35e8fc96778f957029d1371ddb623452a50
    pip install -r requirements.txt 
    pip3 install -e .
    cd ..

    # install other packages
    pip install -r requirements.txt 
    ```
3. Prepare pretrained weights and tokenizer
    Download the LLaMA2-13B checkpoint from [here](https://huggingface.co/NousResearch/Llama-2-13b-hf/tree/main)

    ```bash
    cd ./model_from_hf
    git lfs install
    git clone https://huggingface.co/NousResearch/Llama-2-13b-hf
    cd ..
    ```
4. Weights convert

   *Note that if you want to use the weight from huggingface, please run the weight conversion script first. The following uses llama-2-13b model weight conversion as an example.*

    ```bash
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # convert weights
    python tools/checkpoint/util.py \
        --model-type GPT \
        --loader llama2_hf \
        --saver megatron \
        --target-tensor-parallel-size 8 \
        --load-dir ./model_from_hf/Llama-2-13b-hf/ \
        --target-pipeline-parallel-size 1 \
        --save-dir ./model_weights/Llama-2-13b-hf-v0.1-tp8-pp1/ \
        --tokenizer-model ./llama2-13b-hf/tokenizer.model
    ```

    Weight conversion is suitable for pre-training, fine-tuning, inference and evaluation. Adjust the parameters `target-tensor-parallel-size` and `target-pipeline-parallel-size` according to different tasks.

    Any Megatron weights with parallel slicing strategy --> Any Megatron weights with parallel slicing strategy

    ```shell
    # Modify the ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python tools/checkpoint/util.py \
        --model-type GPT \
        --loader megatron \
        --saver megatron \
        --save-model-type save_huggingface_llama \
        --load-dir ./model_weights/Llama-2-13b-hf-v0.1-tp8-pp1/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --save-dir ./model_from_hf/Llama-2-13b-hf/  # <-- Fill in the original HF model path here, new weights will be saved in ./model_from_hf/Llama-2-13b-hf/mg2hg/
   ```
5. pre-training

   5.1 Prepare dataset

   Download the LLAMA2-13B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

   ```shell
   # download datasets
   cd ./dataset
   wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
   cd ..

   # process datasets  
   mkdir ./dataset/Llama-2-13b-hf/
   python ./tools/preprocess_data.py \
       --input ./dataset_llama2/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
       --tokenizer-name-or-path ./model_from_hf/Llama-2-13b-hf/ \
       --output-prefix ./dataset/Llama-2-13b-hf/alpaca \
       --workers 4 \
       --log-interval 1000 \
       --tokenizer-type PretrainedFromHF
   ```

5. pre-training

   5.1 Prepare dataset

   Download the LLAMA2-13B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

   ```shell
   # download datasets
   cd ./dataset
   wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
   cd ..

   # process datasets  
   python ./tools/preprocess_data.py \
       --input ./dataset_llama2/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
       --tokenizer-name-or-path ./model_from_hf/Llama-2-13b-hf/ \
       --output-prefix ./dataset/Llama-2-13b-hf_alpaca \
       --workers 4 \
       --log-interval 1000 \
       --tokenizer-type PretrainedFromHF
   ```

   5.2 pre-training using ptd mode
   Config LLAMA2-13B pre-training script: examples/llama2/pretrain_llama2_13B_ptd_8p.sh

   ```shell
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # modify config according to your own actual situation
    LOAD_CHECKPOINT_PATH="./model_weights/Llama-2-13b-hf-v0.1-tp8-pp1/"
    SAVE_CHECKPOINT_PATH="./ckpt/Llama-2-13b-hf/"
    TOKENIZER_MODEL="./model_from_hf/Llama-2-13b-hf/tokenizer.model"  #tokenizer path
    DATA_PATH="./dataset/Llama-2-13b-hf/alpaca_text_document"  #processed dataset
   ```

   Multi-machine training requires the addition of parameter --overlap-grad-reduce

   Launch LLAMA2-13B  pre-training script: examples/llama2/pretrain_llama2_13B_ptd_8p.sh

   ```shell
    bash examples/llama2/pretrain_llama2_13B_ptd_8p.sh
   ```
   **Note**: If using multi machine training, and no data sharing configuration on the mechines, it's necessary to add the parameter `--no-shared-storage`. This parameter will determine whether non master nodes need to load data based on distributed parameters, and check the corresponding cache and generated data.

   ```shell
   # download datasets
   mkdir finetune_dataset
   cd ./finetune_dataset
   wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
   cd ..

   # process datasets  
   mkdir ./finetune_dataset/llama-2-13b-hf/
   python ./tools/preprocess_data.py \
     --input ./dataset_llama2/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
     --tokenizer-name-or-path ./model_from_hf/llama-2-13b-hf \
     --output-prefix ./finetune_dataset/llama-2-13b-hf/alpaca \
     --workers 4 \
     --log-interval 1000 \
     --tokenizer-type PretrainedFromHF \
     --handler-name GeneralInstructionHandler \
     --append-eod
   ```

   6.1 Prepare fine-tuning dataset
   Download the LLAMA2-13B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

   ```shell
   # download datasets
   mkdir finetune_dataset
   cd ./finetune_dataset
   wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
   cd ..

   # process datasets  
   python ./tools/preprocess_data.py \
     --input ./dataset_llama2/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
     --tokenizer-name-or-path ./model_from_hf/llama-2-13b-hf \
     --output-prefix ./finetune_dataset/llama-2-13b-hf/alpaca \
     --workers 4 \
     --log-interval 1000 \
     --tokenizer-type PretrainedFromHF \
     --handler-name GeneralInstructionHandler \
     --append-eod
   ```

   6.2 Full Parameters Fine-Tuning
   The configuration script for full parameters fine-tuning  is basically the same as that for pretrain_llama2_7b_ptd.sh.*The difference is that the dataset and the training parameter is-instruction-dataset are added.*

   Add the fine-tuning parameter `--finetune` and add pretrained-weight load parameter `--load`, so that fine-tuning starts from the first step.

   ```bash
   DATA_PATH="./finetune_dataset/llama-2-13b-hf/alpaca"
   TOKENIZER_PATH="./model_from_hf/llama-2-13b-hf"
   CKPT_PATH="./ckpt"
   --load ${CKPT_PATH} \
   --finetune \
   --is-instruction-dataset \
   --tokenizer-type PretrainedFromHF \
   --tokenizer-name-or-path ${TOKENIZER_PATH} \
   --tokenizer-not-use-fast \
   ```

   6.3 Lora Fine-Tuning
   The Lora fine-tuning script is configured by adding the following lora parameters based on the full-parameter finetune script pretrain_llama2_7b_ptd.sh:

   ```bash
       --lora-target-modules query_key_value dense proj dense_4h_to_h \
       --lora-r 16 \
       --lora-alpha 32 \
   ```

   If the vocabulary is changed, add the following parameters:

   ```bash
     --lora-modules-to-save word_embeddings output_layer \
   ```
   
   
   Launch LLAMA2-13B lora fine tune script: tasks/finetune/tune_llama2_13b_ptd.sh
   
   ```shell
    bash tasks/finetune/tune_llama2_13b_ptd.sh 
   ```

### Performance

#### Machine performance

The performance of LLaMA2-13B in **Ascend NPU** and **Reference**:

|  Device  |   Model   | total Iterations | throughput rate (samples/s/p) | throughput rate (tokens/s/p) | single-step time (s/step) |
| :-------: | :--------: | :--------------: | :---------------------------: | :--------------------------: | :-----------------------: |
|   NPUs   | LLaMA2-13B |       5000       |             3.027             |             1550             |           5.285           |
| Reference | LLaMA2-13B |        --        |              --              |             1750             |            --            |

## Inference

We support Inference for text generation with Llama2 13B.
Inference different from pre-training, such as we need to Load pre-training checkpoint and the length of the output samples:

Config Llama2-13B inference script: tasks/inference/generate_llama2_13b_ptd.sh

```shell
# modify the model weight path and tokenizer path
CHECKPOINT="./model_weights/Llama-2-13b-hf-v0.1-tp8-pp1/"
TOKENIZER_PATH="./model_from_hf/Llama-2-13b-hf/"
```

Config Llama2-13B lora inference script: tasks/inference/generate_llama2_13b_lora_ptd.sh

```bash
# modify lora model directory path
CHECKPOINT_LORA="your lora model directory path"
```

Launch Llama2-13B inference script.

```shell
bash ./tasks/inference/generate_llama2_13b_ptd.sh
```

Launch Llama2-13B lora inference script.

```shell
bash ./tasks/inference/generate_llama2_13b_lora_ptd.sh
```

Some inference samples are as follows:
![llama2-13B-generate.png](../../sources/images/llama2/llama2-13B-generate.png)

## Evaluation

We use boolq benchmark to evaluate our model. Benchmark Download [here](https://huggingface.co/datasets/boolq).

```shell
    # modify the model weight path and tokenizer path
    CHECKPOINT="./model_weights/Llama-2-13b-hf-v0.1-tp8-pp1/"
    TOKENIZER_PATH="./model_from_hf/Llama-2-13b-hf/"
```

```shell
bash tasks/evaluation/evaluate_llama2_13B_ptd.sh
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
      <td>Test</td>
      <th>Llama2 13B</th>
      <td>0.821</td>
      <td><a href="https://paperswithcode.com/sota/question-answering-on-boolq">0.817</a></td>
    </tr>
  </tbody>
</table>

# LLaMA2-34B/70B

## Training-2

Here's a hardware summary of pre-training  LLaMA2-34B/70B:

| Model | Hardware |      Value      |
| :---: | :------: | :--------------: |
|  34B  |   NPU   | 16 x Ascend NPUs |
|  70B  |   NPU   | 64 x Ascend NPUs |

### Script-2

1. Clone the repository to your local server:

```shell
git clone https://gitee.com/ascend/MindSpeed-LLM.git 
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout -f bcce6f
cp -r megatron ../MindSpeed-LLM/
cd ..
cd MindSpeed-LLM
git checkout 1.0
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
git checkout 224ae35e8fc96778f957029d1371ddb623452a50
pip install -r requirements.txt 
pip3 install -e .
cd ..

# install other packages
pip install -r requirements.txt 
```

3. Prepare pretrained weights and tokenizer
   Download the LLaMA2-70B checkpoint from [here](https://huggingface.co/meta-llama/Llama-2-70b-hf)

   ```shell
   mkdir ./model_from_hf/llama2-70b-hf/
   cd ./model_from_hf/llama2-70b-hf/
   wget https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/config.json
   wget https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/generation_config.json
   wget https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/pytorch_model-00001-of-00015.bin
   wget https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/pytorch_model-00002-of-00015.bin
   wget https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/pytorch_model-00003-of-00015.bin
   wget https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/pytorch_model-00004-of-00015.bin
   wget https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/pytorch_model-00005-of-00015.bin
   wget https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/pytorch_model-00006-of-00015.bin
   wget https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/pytorch_model-00007-of-00015.bin   
   wget https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/pytorch_model-00008-of-00015.bin
   wget https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/pytorch_model-00009-of-00015.bin
   wget https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/pytorch_model-00010-of-00015.bin
   wget https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/pytorch_model-00011-of-00015.bin
   wget https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/pytorch_model-00012-of-00015.bin   
   wget https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/pytorch_model-00013-of-00015.bin
   wget https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/pytorch_model-00014-of-00015.bin
   wget https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/pytorch_model-00015-of-00015.bin   
   wget https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/pytorch_model.bin.index.json
   wget https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/special_tokens_map.json
   wget https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/tokenizer.json
   wget https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/tokenizer.model
   wget https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/tokenizer_config.json
   cd ../../
   ```

   For LLaMA2-34B, we use CodeLlama-34b weights and LLaMA2-70B tokenizer.

   CodeLlama-34b weights can be downloaded from [here](https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf/tree/main),

   ```bash
   mkdir ./model_from_hf/codellama-34b-hf/
   cd ./model_from_hf/codellama-34b-hf/
   wget https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf/resolve/main/config.json
   wget https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf/resolve/main/generation_config.json
   wget https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf/resolve/main/pytorch_model-00001-of-00007.bin
   wget https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf/resolve/main/pytorch_model-00002-of-00007.bin
   wget https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf/resolve/main/pytorch_model-00003-of-00007.bin
   wget https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf/resolve/main/pytorch_model-00004-of-00007.bin
   wget https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf/resolve/main/pytorch_model-00005-of-00007.bin
   wget https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf/resolve/main/pytorch_model-00006-of-00007.bin
   wget https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf/resolve/main/pytorch_model-00007-of-00007.bin
   wget https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf/resolve/main/pytorch_model.bin.index.json
   cd ../../
   ```

   LLaMA2-70B tokenizer can be downloaded from [here](https://huggingface.co/meta-llama/Llama-2-70b-hf)

   ```bash
   mkdir ./model_from_hf/llama2-70b-hf/
   cd ./model_from_hf/llama2-70b-hf/
   wget https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/special_tokens_map.json
   wget https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/tokenizer.json
   wget https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/tokenizer.model
   wget https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/tokenizer_config.json
   cd ../../
   ```
4. Weights convert

   *Note that if you want to use the weight from huggingface, please run the weight conversion script first. *
    The following converts llama-2-70b model weight.

    ```bash
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # convert to megatron weights
    python tools/checkpoint/util.py \
    --model-type GPT \
    --loader llama2_hf \
    --saver megatron \
    --target-tensor-parallel-size 8 \
    --target-pipeline-parallel-size 4 \
    --load-dir ./llama2-70b-hf/ \
    --save-dir ./load_ckpt \
    --tokenizer-model ./llama2-70b-hf/tokenizer.model                                                                    
    ```

    The following converts llama-2-34b model weight.

    ```bash
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # convert to megatron weights
    python tools/checkpoint/util.py \
        --model-type GPT \
        --loader llama2_hf \
        --saver megatron \
        --target-tensor-parallel-size 8 \
        --target-pipeline-parallel-size 4 \
        --load-dir ./codellama-34b-hf \
        --save-dir ./load_ckpt \
        --tokenizer-model ./llama2-70b-hf/tokenizer.model \
        --params-dtype bf16                                                              
    ```

    Any Megatron weights with parallel slicing strategy --> Any Megatron weights with parallel slicing strategy.

    * The following converts llama-2-70b model weight.

    ```shell
    # Modify the ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python tools/checkpoint/util.py \
        --model-type GPT \
        --loader megatron \
        --saver megatron \
        --save-model-type save_huggingface_llama \
        --load-dir ./model_weights/llama2-70b-hf-v0.1-tp8-pp4/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --save-dir ./model_from_hf/llama2-70b-hf/  # <-- Fill in the original HF model path here, new weights will be saved in ./model_from_hf/llama2-70b-hf/mg2hg/
    ```

    * The following converts llama-2-34b model weight.

    ```shell
    # Modify the ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python tools/checkpoint/util.py \
        --model-type GPT \
        --loader megatron \
        --saver megatron \
        --save-model-type save_huggingface_llama \
        --load-dir ./model_weights/codellama-34b-hf-v0.1-tp8-pp4/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --save-dir ./model_from_hf/codellama-34b-hf/  # <-- Fill in the original HF model path here, new weights will be saved in ./model_from_hf/codellama-34b-hf/mg2hg/
    ```

    Weight conversion is suitable for pre-training, fine-tuning, inference and evaluation. Adjust the parameters `target-tensor-parallel-size` and `target-pipeline-parallel-size` according to different tasks.
5. pre-training

  5.1 Prepare dataset

    There are two dataset examples: Alpaca and Moss.

    1. Alpaca Dataset

      Download the Alpaca datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

    ```shell
    # download datasets
    cd ./dataset
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..

    # process datasets  
    mkdir ./dataset/llama2-70b-hf/
    python ./tools/preprocess_data.py \
    --input ./dataset_llama2/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./model_from_hf/llama2-70b-hf/ \
    --output-prefix ./dataset/llama2-70b-hf/alpaca \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF
    ```

   2. Moss Dataset

      Download the Moss datasets from [here](https://huggingface.co/datasets/fnlp/moss-003-sft-data/tree/main)

    ```shell
    # download datasets
    cd ./dataset
    wget https://huggingface.co/datasets/fnlp/moss-003-sft-data/resolve/main/moss-003-sft-no-tools.jsonl.zip --no-check-certificate
    unzip moss-003-sft-no-tools.jsonl.zip
    cd ..

    # process datasets  
    python tools/preprocess_data.py \
    --input ./dataset/moss-003-sft-no-tools.jsonl \
    --output-prefix ./dataset/llama2-70b-hf_moss \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ./model_from_hf/llama2-70b-hf/ \
    --tokenizer-not-use-fast \
    --handler-name MOSSInstructionHandler
    ```

  5.2 pre-training using ptd mode

    LLaMA2-34B: examples/llama2/pretrain_llama2_34B_ptd_16p.sh

    ```shell
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # modify script orign dataset path according to your own dataset path
    TOKENIZER_MODEL="./model_from_hf/llama2-70b-hf/tokenizer.model"  #tokenizer path
    DATA_PATH="./dataset/llama2-70b-hf/alpaca_text_document"  #processed dataset
    ```

    LLaMA2-70B: examples/llama2/pretrain_llama2_70b_ptd.sh

    ```shell
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # modify script orign dataset path according to your own dataset path
    TOKENIZER_MODEL="./model_from_hf/llama2-70b-hf/tokenizer.model"  #tokenizer path
    DATA_PATH="./dataset/llama2-70b-hf/alpaca_text_document"  #processed dataset
    ```

    Launch pre-training script

    **Note**: If using multi machine training, and no data sharing configuration on the mechines, it's necessary to add the parameter `--no-shared-storage`. This parameter will determine whether non master nodes need to load data based on distributed parameters, and check the corresponding cache and generated data.
    
    LLaMA2-34B: examples/llama2/pretrain_llama2_34B_ptd_16p.sh

    ```shell
    bash examples/llama2/pretrain_llama2_34B_ptd_16p.sh
    ```

    LLaMA2-70B: examples/llama2/pretrain_llama2_70b_ptd.sh

    ```shell
    bash examples/llama2/pretrain_llama2_70b_ptd.sh
    ```

  6.1 Prepare fine-tuning dataset
  Download the LLAMA2-13B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

  ```shell
  # download datasets
  mkdir finetune_dataset
  cd ./finetune_dataset
  wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
  cd ..

  # process datasets  
  mkdir ./finetune_dataset/llama2-70b-hf/
  python ./tools/preprocess_data.py \
      --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
      --tokenizer-name-or-path ./model_from_hf/llama2-70b-hf/ \
      --output-prefix ./finetune_dataset/llama2-70b-hf/alpaca \
      --workers 4 \
      --log-interval 1000 \
      --tokenizer-type PretrainedFromHF \
      --handler-name GeneralInstructionHandler \
      --append-eod
  ```

  6.1 Prepare fine-tuning dataset
  Download the LLAMA2-13B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

  ```shell
  # download datasets
  mkdir finetune_dataset
  cd ./finetune_dataset
  wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
  cd ..

  # process datasets  
  python ./tools/preprocess_data.py \
      --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
      --tokenizer-name-or-path ./model_from_hf/llama2-70b-hf/ \
      --output-prefix ./finetune_dataset/llama2-70b-hf/alpaca \
      --workers 4 \
      --log-interval 1000 \
      --tokenizer-type PretrainedFromHF \
      --handler-name GeneralInstructionHandler \
      --append-eod
  ```

   6.2 Full Parameters Fine-Tuning
   The configuration script for full parameters fine-tuning  is basically the same as that for pretrain_llama2_7b_ptd.sh.*The difference is that the dataset and the training parameter is-instruction-dataset are added.*

   Add the fine-tuning parameter `--finetune` so that fine-tuning starts from the first step.

   ```bash
   DATA_PATH="./finetune_dataset/llama2-70b-hf/alpaca"
   TOKENIZER_PATH="/model_from_hf/llama2-70b-hf/"
   CKPT_PATH="./ckpt"
   --load ${CKPT_PATH} \
   --finetune \
   --is-instruction-dataset \
   --tokenizer-type PretrainedFromHF \
   --tokenizer-name-or-path ${TOKENIZER_PATH} \
   --tokenizer-not-use-fast \
   ```

   6.3 Lora Fine-Tuning
   The Lora fine-tuning script is configured by adding the following lora parameters to the pretrain_llama2_7b_ptd.sh script:

   ```bash
       --lora-target-modules query_key_value dense proj dense_4h_to_h \
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
   
   Launch LLAMA2-34B lora fine tune script: tasks/finetune/tune_llama2_34b_ptd.sh
   
   ```shell
    bash tasks/finetune/tune_llama2_34b_ptd.sh 
   ```
   
   Launch LLAMA2-70B lora fine tune script: tasks/finetune/tune_llama2_70b_ptd.sh
   
   ```shell
    bash tasks/finetune/tune_llama2_70b_ptd.sh 
   ```

### Performance-2

#### Machine performance-2

The performance of LLaMA2-34B/70B in **Ascend NPU** and **Reference**

|     Device      |     Model     |  throughput (tokens/s/p) |  
|:---------------:|:----------:|:---------------------:|
|      NPUs       | LLaMA2-34B |          690          |
|    Reference    | LLaMA2-34B |          796          |
|      NPUs       | LLaMA2-70B |          350          |
|    Reference    | LLaMA2-70B |          339          |




## Inference-2

Models could generate with 8 NPUs, for example:

Config inference script:

LLaMA2-34B:`tasks/inference/generate_llama2_34B_ptd.sh`.

LLaMA2-70B:`tasks/inference/generate_llama2_70b_ptd.sh`.

```shell
# Modify checkpoint path and vocabfile path.
CHECKPOINT=<checkpoint-path>
TOKENIZER_PATH=<tokenizer-path>
```

Config lora inference script:

```bash
# modify lora model directory path
CHECKPOINT_LORA="your lora model directory path"
```

Launch LLaMA2-34B inference:

```shell
bash ./tasks/inference/generate_llama2_34B_ptd.sh
```

Launch LLaMA2-34B lora inference:

```shell
bash ./tasks/inference/generate_llama2_34b_lora_ptd.sh
```

Launch LLaMA2-70B inference:

```shell
bash ./tasks/llama2/generate_llama2_70b_ptd.sh
```

Launch LLaMA2-70B lora inference:

```shell
bash ./tasks/llama2/generate_llama2_70b_lora_ptd.sh
```

Some inference samples of LLaMA2-34B are as follows:

![llama2-34B-generate](../../sources/images/llama2/llama2-34B-generate.png)

Some inference samples of LLaMA2-70B are as follows:
![llama2-70B_generate.png](../../sources/images/llama2/llama2-70B-generate.png)

## Evaluation-2

We use BoolQ benchmark to evaluate our model. Benchmark [here](https://huggingface.co/datasets/boolq)
Download dev part[here](https://storage.googleapis.com/boolq/dev.jsonl) and put it in a directory named “boolq_dev”.

Config evaluation script:

LLaMA2-34B:`tasks/evaluation/evaluate_llama2_34B_ptd.sh`.

LLaMA2-70B:`tasks/evaluation/evaluate_llama2_70B_ptd.sh`.

```shell
# Modify checkpoint path and vocabfile path.
CHECKPOINT=<checkpoint-path>
TOKENIZER_PATH=<tokenizer-path>
```

Launch LLaMA2-34B evaluation:

```shell
bash tasks/evaluation/evaluate_llama2_34B_ptd.sh
```

Launch LLaMA2-70B evaluation:

```shell
bash tasks/evaluation/evaluate_llama2_70B_ptd.sh
```

Evaluation results with BoolQ dataset:

<table>
  <thead>
    <tr>
      <th>Task</th>
      <th>Subset</th>
      <th>Model</th>
      <th>NPU</th>
      <th>Benchmark</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://huggingface.co/datasets/boolq">BoolQ</a></td>
      <td>dev</td>
      <th>Llama2-70b</th>
      <td>0.859</td>
      <td><a href="https://paperswithcode.com/sota/question-answering-on-boolq">(Llama2-70b test) 0.877</a></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/datasets/boolq">BoolQ</a></td>
      <td>dev</td>
      <th>Llama2-34b</th>
      <td>0.651</td>
      <td><a href="https://hub.opencompass.org.cn/dataset-detail/BoolQ">(AquilaChat2-34B test) 0.571</a></td>
    </tr>
  </tbody>
</table>
