<p align="center"> <img src="sources/images/logo.png" height="110px" width="500px"> </p>

<p align="center">
    <a>
        <img alt="Docs" src="https://readthedocs.org/projects/internlm/badge/?version=latest">
    </a>
    <a href="https://gitee.com/ascend/MindSpeed/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue">
    </a>
    <a href="https://gitee.com/ascend/MindSpeed">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a>
        <img src="https://app.codacy.com/project/badge/Grade/1710faac5e634acaabfc26b0a778cdde">
    </a>
</p>
<p align="center">
        <b><a href="README.md">简体中文</a> </b> |
        <b>English</b>
</p>

MindSpeed-LLM, originally named ModelLink, provides end-to-end solutions for large language models on Ascend chips, including models, algorithms and tasks.

## MindSpeed-LLM Solution Overview

---


## supported features
Current MindSpeed-LLM supported features for large model usage:
* [Dataset Preparation for Pre-training](#jump11)/[Fine-tuning Instruction Dataset Preparation](#jump12)
* [Pre-training](#jump13)/[Full-parameter Fine-tuning](#jump14)/[Low-parameter Fine-tuning](#jump15)
* [Inference: human-machine dialogue](#jump16)
* [Evaluation with numerous benchmarks](#jump17)
* [Utilizing Acceleration Features (Acceleration Algorithms + Fusion Operators)](#jump18)

More novel and useful features are developing for LLMs training on Ascend ...


## Supported Models
Current MindSpeed-LLM supports pre-training and fine-tuning for the following models:

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Parameters</th>
      <th>Fine-tuning</th>
      <th>Inference</th>
      <th>Evaluation</th>
      <th>Dataset Support</th>
      <th>Contributor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="1"> <a href="examples/aquila/README.md">Aquila</a> </td>
      <td> 7B </td>
      <td> -- </td>
      <td> <a href="tasks/inference/generate_aquila_7b_ptd.sh">inference</a> </td>
      <td> <a href="tasks/evaluation/evaluate_aquila_7b_ptd.sh">evaluation</a> </td>
      <td> <a href="https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json">alpaca_data.json</a> </td>
      <td>【Model contributed by Ascend】</td>
    </tr>
    <tr>
      <td rowspan="2"> <a href="examples/baichuan/README.md">Baichuan</a> </td>
      <td> 7B </td>
      <td> -- </td>
      <td> <a href="tasks/inference/generate_baichuan_7b_ptd.sh">inference</a> </td>
      <td> <a href="tasks/evaluation/evaluate_baichuan_7B_ptd.sh">evaluation</a> </td>
      <td> <a href="https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json">alpaca_data.json</a> </td>
      <td>【Model contributed by Ascend】</td>
    </tr>
    <tr>
      <td> 13B </td>
      <td> -- </td>
      <td> <a href="tasks/inference/generate_baichuan_13b_ptd.sh">inference</a> </td>
      <td> <a href="tasks/evaluation/evaluate_baichuan_13B_ptd.sh">evaluation</a> </td>
      <td> <a href="https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json">alpaca_data.json</a> </td>
      <td>【Model contributed by Ascend】</td>
    </tr>
    <tr>
      <td rowspan="2"> <a href="examples/baichuan2/README.md">Baichuan2</a> </td>
      <td> 7B </td>
      <td> -- </td>
      <td> <a href="tasks/inference/generate_baichuan2_7b_ptd.sh">inference</a> </td>
      <td> <a href="tasks/evaluation/evaluate_baichuan2_7B_ptd.sh">evaluation</a> </td>
      <td> <a href="https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json">alpaca_data.json</a> </td>
      <td>【Model contributed by Ascend】</td>
    </tr>
    <tr>
      <td> 13B </td>
      <td> -- </td>
      <td> <a href="tasks/inference/generate_baichuan2_13b_ptd.sh">inference</a> </td>
      <td> <a href="tasks/evaluation/evaluate_baichuan2_13B_ptd.sh">evaluation</a> </td>
      <td> <a href="https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json">alpaca_data.json</a> </td>
      <td>【Model contributed by Ascend】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/bloom/README.md">Bloom</a></td>
      <td> 7B1 </td>
      <td> -- </td>
      <td> <a href="tasks/inference/generate_bloom_7b_ptd.sh">inference</a> </td>
      <td> <a href="tasks/evaluation/evaluate_baichuan_7B_ptd.sh">evaluation</a> </td>
      <td> <a href="https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json">alpaca_data.json </a> </td>
      <td>【Model contributed by Ascend】</td>
    </tr>
    <tr>
      <td> 176B </td>
      <td> -- </td>
      <td> <a href="tasks/inference/generate_bloom_176b_ptd.sh">inference</a> </td>
      <td> <a href="tasks/evaluation/evaluate_bloom_176b_ptd.sh">evaluation</a> </td>
      <td> <a href="https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json">alpaca_data.json</a>  </td>
      <td>【Model contributed by Ascend】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/intern/README.md">InternLM</a></td>
      <td>7B</td>
      <td> -- </td>
      <td> <a href="tasks/inference/generate_lnternlm_7b_ptd.sh">inference</a> </td>
      <td> <a href="tasks/evaluation/evaluate_internlm_7B_ptd.sh">evaluation</a> </td>
      <td> <a href="https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json">alpaca_data.json</a> </td>
      <td>【Model contributed by Ascend】</td>
    </tr>
    <tr>
      <td>65B</td>
      <td> -- </td>
      <td> -- </td>
      <td> -- </td>
      <td> <a href="https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json">alpaca_data.json</a> </td>
      <td>【Model contributed by Ascend】</td>
    </tr>
    <tr>
      <td rowspan="4"><a href="examples/llama/README.md">LLaMA</a></td>
      <td>7B</td>
      <td> <a href="tasks/finetune/tune_llama_7b_ptd.sh">lora</a> </td>
      <td> <a href="tasks/inference/generate_llama_7b_lora_ptd.sh">inference</a> </td>
      <td> <a href="tasks/evaluation/evaluate_llama_7B_ptd.sh">evaluation</a> </td>
      <td> <a href="https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json">alpaca_data.json</a> </td>
      <td>【Model contributed by Ascend】</td>
    </tr>
    <tr>
      <td>13B</td>
      <td> <a href="tasks/finetune/tune_llama_13b_ptd.sh">lora</a> </td>
      <td> <a href="tasks/inference/generate_llama_13b_lora_ptd.sh">inference</a> </td>
      <td> <a href="tasks/evaluation/evaluate_llama_13B_ptd.sh">evaluation</a> </td>
      <td> <a href="https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json">alpaca_data.json</a> </td>
      <td>【Model contributed by Ascend】</td>
    </tr>
    <tr>
      <td>33B</td>
      <td> <a href="tasks/finetune/tune_llama_33b_ptd.sh">lora</a> </td>
      <td> <a href="tasks/inference/generate_llama_33b_ptd.sh">inference</a> </td>
      <td> <a href="tasks/evaluation/evaluate_llama_33B_ptd.sh">evaluation</a> </td>
      <td> <a href="https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json">alpaca_data.json</a> </td>
      <td>【Model contributed by Ascend】</td>
    </tr>
    <tr>
      <td > 65B </td>
      <td > <a href="tasks/finetune/tune_llama_65b_ptd.sh">lora</a> </td>
      <td> <a href="tasks/inference/generate_llama_65b_lora_ptd.sh">inference</a> </td>
      <td> <a href="tasks/evaluation/evaluate_llama_65B_ptd.sh">evaluation</a> </td>
      <td> <a href="https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json">alpaca_data.json</a>  </td>
      <td>【Model contributed by Ascend】</td>
    </tr>
    <tr>
      <td rowspan="4"><a href="examples/llama2/README.md">LLaMA2</a></td>
      <td>7B</td>
      <td> <a href="tasks/finetune/tune_llama2_7b_ptd.sh">lora </a> </td>
      <td> <a href="tasks/inference/generate_llama2_7b_ptd.sh">inference</a> </td>
      <td> <a href="tasks/evaluation/evaluate_llama2_7B_ptd.sh">evaluation</a>  </td>
      <td> <a href="https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json">alpaca_data.json </a> </td>
      <td>【Model contributed by Ascend】</td>
    </tr>
    <tr>
      <td>13B</td>
      <td> <a href="tasks/finetune/tune_llama2_13b_ptd.sh">lora </a> </td>
      <td> <a href="tasks/inference/generate_llama2_13b_ptd.sh">inference</a> </td>
      <td> <a href="tasks/evaluation/evaluate_llama2_13B_ptd.sh">evaluation</a> </td>
      <td> <a href="https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json">alpaca_data.json</a>  </td>
      <td>【Model contributed by Ascend】</td>
    </tr>
    <tr>
      <td> 34B </td>
      <td> <a href="tasks/finetune/tune_llama2_34b_ptd.sh">lora </a> </td>
      <td> <a href="tasks/inference/generate_llama2_34B_ptd.sh">inference</a> </td>
      <td> <a href="tasks/evaluation/evaluate_llama2_34B_ptd.sh">evaluation</a> </td>
      <td> <a href="https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json">alpaca_data.json</a>  </td>
      <td>【Model contributed by Ascend】</td>
    </tr>
    <tr>
      <td > 70B </td>
      <td > <a href="tasks/finetune/tune_llama2_70b_ptd.sh">lora</a> </td>
      <td> <a href="tasks/inference/generate_llama2_70b_lora_ptd.sh">inference</a> </td>
      <td> <a href="tasks/evaluation/evaluate_llama2_70B_ptd.sh">evaluation</a> </td>
      <td>  <a href="https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json">alpaca_data.json</a>  </td>
      <td>【Model contributed by Ascend】</td>
    </tr>
    <tr>
      <td rowspan="3"><a href="examples/qwen/README.md">Qwen</a></td>
      <td>7B</td>
      <td> -- </td>
      <td> <a href="tasks/inference/generate_qwen_7b_ptd.sh">inference</a></td>
      <td> <a href="tasks/evaluation/evaluate_qwen_7b_ptd.sh">evaluation</a></td>
      <td> <a href="https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json">alpaca_data.json </a> </td>
      <td>【Model contributed by Ascend】</td>
    </tr>
    <tr>
      <td>14B</td>
      <td> -- </td>
      <td> <a href="tasks/inference/generate_qwen_14b_ptd.sh">inference</a> </td>
      <td> <a href="tasks/evaluation/evaluate_qwen_14b_ptd.sh">evaluation</a> </td>
      <td> <a href="https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json">alpaca_data.json </a> </td>
      <td>【Model contributed by Ascend】</td>
    </tr>
    <tr>
      <td>72B</td>
      <td> -- </td>
      <td> <a href="tasks/inference/generate_qwen_72b_ptd.sh">inference</a> </td>
      <td> <a href="tasks/evaluation/evaluate_qwen_72b_ptd.sh">evaluation</a> </td>
      <td> <a href="https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json">alpaca_data.json </a> </td>
      <td>【Model contributed by Ascend】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/mixtral/README.md">Mixtral</a></td>
      <td>8x7B</td>
      <td> -- </td>
      <td> <a href="tasks/inference/generate_mixtral_8x7b_ptd.sh">inference</a> </td>
      <td> <a href="tasks/evaluation/evaluate_mixtral_8x7b_ptd.sh">evaluation</a>  </td>
      <td> <a href="https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json">alpaca_data.json </a> </td>
      <td>【Model contributed by Ascend】</td>
    </tr>
  </tbody>
</table>


## Script Naming Rules
|            Script             |    Rule    |
|:-------------------------:|:--------:|
|      pretrain_xxx.sh      |  Pre-training Script   |
|        tune_xxx.sh        |   Fine-tuning Script   |
|      generate_xxx.sh      |   Inference Script   |
|     evaluation_xxx.sh     |   Evaluation Script   |

---

# Model Usage Guide and Version Notes


Model Usage Guide and Version Notes
For the supported models listed above, we provide training scripts and readme instructions in the examples folder, which contain detailed processes for model training, inference, and evaluation.

【Please note the corresponding environment versions for model usage, as follows】

|           软件            | [版本](https://www.hiascend.com/zh/) |
| :-----------------------: |:----------------------------------:|
|          Python           |                3.8                 |
|          driver           |              Ascend HDK 24.1.RC1              |
|         firmware          |              Ascend HDK 24.1.RC1              |
|           CANN            |              CANN 8.0.RC1              |
|           torch           |               2.1.0                |
|         torch_npu         |              release v6.0.RC1               |

The current repository uses Megatron commitId [bcce6f54](https://github.com/NVIDIA/Megatron-LM/tree/bcce6f54e075e3c3374ea67adefe54f3f2da2b07)

【Based on the latest version, the performance statistics from our testing are as follows】

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Parameters</th>
      <th>Cluster Scale</th>
      <th>Precision Mode</th>
      <th>Performance </th>
      <th>Reference Performance </th>
      <th>Scripts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="1"><a href="examples/aquila/README.md">Aquila</a></td>
      <td>7B</td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 2849 </td>
      <td> 2874 </td>
      <td> <a href="examples/aquila/pretrain_aquila_7b_ptd.sh">train</a> </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/baichuan/README.md">Baichuan</a></td>
      <td>7B</td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 2685 </td>
      <td> 2036 </td>
      <td> <a href="examples/baichuan/pretrain_baichuan_ptd_7B.sh">train</a> </td>
    </tr>
    <tr>
      <td>13B</td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 1213 </td>
      <td> 862 </td>
      <td> <a href="examples/baichuan/pretrain_baichuan_ptd_13B.sh">train</a> </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/baichuan2/README.md">Baichuan2</a></td>
      <td>7B</td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 2664 </td>
      <td> 3969 </td>
      <td> <a href="examples/baichuan2/pretrain_baichuan2_ptd_7B.sh">train</a> </td>
    </tr>
    <tr>
      <td>13B</td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 1310 </td>
      <td> 872 </td>
      <td> <a href="examples/baichuan2/pretrain_baichuan2_ptd_13B.sh">train</a> </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/bloom/README.md">Bloom</a></td>
      <td>7B1</td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 2034 </td>
      <td> 2525 </td>
      <td> <a href="examples/bloom/pretrain_bloom_ptd_7B.sh">train</a> </td>
    </tr>
    <tr>
      <td >176B</td>
      <td >12x8</td>
      <td> BF16 </td>
      <td> 100 </td>
      <td> 107 </td>
      <td> <a href="examples/bloom/pretrain_bloom_176b.sh">train</a> </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/intern/README.md">InternLM</a></td>
      <td>7B</td>
      <td>1x8</td>
      <td>BF16</td>
      <td> 2776 </td>
      <td> 2854 </td>
      <td> <a href="examples/intern/pretrain_internlm_7b_ptd.sh">train</a> </td>
    </tr>
    <tr>
      <td >65B</td>
      <td >4x8</td>
      <td> BF16 </td>
      <td> 341 </td>
      <td> 414 </td>
      <td> <a href="examples/intern/pretrain_internlm_65b_ptd.sh">train</a> </td>
    </tr>
    <tr>
      <td rowspan="5"><a href="examples/llama/README.md">LLaMA</a></td>
      <td>7B</td>
      <td>1x8</td>
      <td>FP16</td>
      <td> 3600 </td>
      <td> 3804 </td>
      <td> <a href="examples/llama/pretrain_llama_7b_ptd.sh">train</a> </td>
    </tr>
    <tr>
      <td>13B</td>
      <td>1x8</td>
      <td>FP16</td>
      <td> 1895 </td>
      <td> 2012 </td>
      <td> <a href="examples/llama/pretrain_llama_13b_ptd.sh">train</a> </td>
    </tr>
    <tr>
        <td>33B</td>
        <td>4x8</td>
        <td>FP16</td>
        <td>621</td>
        <td>776</td>
        <td><a href="examples/llama/pretrain_llama_33B_ptd_32p.sh">train</a> </td>
    </tr>
    <tr>
      <td rowspan="2">65B</td>
      <td rowspan="2">4x8</td>
    </tr>
    <tr>
      <td>BF16 </td>
      <td> 348 </td>
      <td> 426 </td>
      <td> <a href="examples/llama/pretrain_llama_65b_ptd.sh">train</a> </td>
    </tr>
    <tr>
      <td rowspan="4"><a href="examples/llama2/README.md">LLaMA2</a></td>
      <td>7B</td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 2884 </td>
      <td> 2884 </td>
      <td> <a href="examples/llama2/pretrain_llama2_7b_ptd.sh">train</a> </td>
    </tr>
    <tr>
      <td>13B</td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 1550 </td>
      <td> 1750 </td>
      <td> <a href="examples/llama2/pretrain_llama2_13B_ptd_8p.sh">train</a> </td>
    </tr>
    <tr>
      <td>34B</td>
      <td>2x8</td>
      <td>BF16 </td>
      <td> 690 </td>
      <td> 796 </td>
      <td> <a href="examples/llama2/pretrain_llama2_34B_ptd_16p.sh">train</a> </td>
    </tr>
    <tr>
      <td>70B</td>
      <td>8x8</td>
      <td>BF16 </td>
      <td> 350 </td>
      <td> 339 </td>
      <td> <a href="examples/llama2/pretrain_llama2_70b_ptd.sh">train</a> </td>
    </tr>
    <tr>
      <td rowspan="3"><a href="examples/qwen/README.md">Qwen</a></td>
      <td>7B</td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 2499 </td>
      <td> 2867 </td>
      <td> <a href="examples/qwen/pretrain_qwen_7b_ptd.sh">train</a> </td>
    </tr>
    <tr>
      <td>14B</td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 1560 </td>
      <td> 1578 </td>
      <td> <a href="examples/qwen/pretrain_qwen_14b_ptd.sh">train</a> </td>
    </tr>
    <tr>
      <td>72B</td>
      <td>16x8</td>
      <td>BF16 </td>
      <td> 285 </td>
      <td> 345 </td>
      <td> <a href="examples/qwen/pretrain_qwen_72b_ptd.sh">train</a> </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/mixtral/README.md">Mixtral</a></td>
      <td>8x7B</td>
      <td>2x8</td>
      <td>BF16 </td>
      <td> 487 </td>
      <td> 610 </td>
      <td> <a href="examples/mixtral/pretrain_mixtral_8x7b_ptd.sh">train</a> </td>
    </tr>
  </tbody>
</table>




# Function Usage Guide

## Instruction/Pretraining dataset support

#### Quick Start
Use the [preprocess_data.py](tools/preprocess_data.py) data preprocessing tool to process raw data into binary format data for training. Below is an example of processing the Alpaca dataset:

```bash
# for llama, download alpaca dataset, like
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet

# download tokenizer configs and (selective) weights from
# https://huggingface.co/yahma/llama-7b-hf/tree/main
# revise "LLaMATokenizer" as "LlamaTokenizer" in tokenizer_config.json (This is a bug of huggingface)
mkdir dataset
python tools/preprocess_data.py --input train-00000-of-00001-a09b74b3ef9c3b56.parquet \
                                --output-prefix dataset/alpaca \
                                --tokenizer-type PretrainedFromHF \
                                --tokenizer-name-or-path llama-7b-hf \
                                --tokenizer-not-use-fast \
                                --handler-name GeneralInstructionHandler
```

The output will be two files, named alpaca_packed_input_ids_document.bin and alpaca_packed_input_ids_document.idx. In subsequent training, specify --data-path with the full path and new filename, but without the file extension. Use --tokenizer-type to specify the data preprocessing method corresponding to the model, --tokenizer-name-or-path to specify the tokenizer model path, usually downloaded along with the pre-trained weights in the open-source project, and --handler-name to specify the data set's instruction data construction method.

#### <span id="jump11"> reprocessing pretraining dataset </span>

##### wikipedia dataset

+ download [wikipedia data](https://huggingface.co/datasets/wikipedia/tree/main) from huggingface to WORKSPACE/wikipedia
+ download [llama tokenizer model and config](https://huggingface.co/yahma/llama-7b-hf/tree/main) from huggingface to WORKSPACE/llama-7b-hf
+ use preprocessing script to preprocess wikipedia data

```shell
# We assume that data and tokenizer has already been downloaded to WORKSPACE.
cd WORKSPACE
mkdir wikipedia_preprocessed

# specify huggingface load_dataset parameters.(--input param will be ignored)
# these params will just be feed into datasets.load_dataset function
hf_config_json="./hf_config_json.json"
cat <<EOT > $hf_config_json
{
    "path": "WORKSPACE/wikipedia",
    "name": "20220301.en",
    "streaming: True,
    "split": "train"
}
EOT

python tools/preprocess_data.py \
    --input "WORKSPACE/wikipedia" \
    --hf-datasets-params ${hf_config_json} \
    --output-prefix WORKSPACE/wikipedia_preprocessed/wikipedia \
    --dataset-impl mmap \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path WORKSPACE/llama-7b-hf \
    --tokenizer-not-use-fast \
    --streaming \
    --workers 8
```

After preprocessing, there will be a `wikipedia_text_document.bin` and a `wikipedia_text_document.idx` in the `WORKSPACE/wikipedia_preprocessed` dictionary.
Then, we can train a model with `--data-path WORKSPACE/wikipedia_preprocessed/wikipedia_text_document` flag.

Note that datasets in huggingface have a format like [this](https://huggingface.co/datasets/wikipedia/viewer/20220301.en/train). The name of the text field of the dataset can be changed by using the `--json-key` flag which default is `text`.
In wikipedia dataset, it has four columns, including `id`, `url`, `title` and `text`, where we can choose a column used for training by `--json-key` flag.

##### alpaca dataset

Besides, we can also use [alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet) dataset for pretraining as below.

```shell
python tools/preprocess_data.py --input WORKSPACE/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
                                --output-prefix WORKSPACE/alpaca_preprocessed/alpaca \
                                --tokenizer-type PretrainedFromHF \
                                --tokenizer-name-or-path WORKSPACE/llama-7b-hf \
                                --tokenizer-not-use-fast \
                                --json-key text
```


#### <span id="jump12"> Preprocessing instruction dataset </span>
##### alpaca dataset
```bash
# for llama, download alpaca dataset, like
# wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet

# download tokenizer configs and (selective) weights from
# https://huggingface.co/yahma/llama-7b-hf/tree/main
# revise "LLaMATokenizer" as "LlamaTokenizer" in tokenizer_config.json (This is a bug of huggingface)

cd WORKSPACE
mkdir alpaca_preprocessed
python tools/preprocess_data.py --input WORKSPACE/alpaca/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
                                --output-prefix WORKSPACE/alpaca_preprocessed/alpaca \
                                --tokenizer-type PretrainedFromHF \
                                --tokenizer-name-or-path WORKSPACE/llama-7b-hf \
                                --tokenizer-not-use-fast \
                                --handler-name GeneralInstructionHandler \
                                --append-eod
```

After preprocessing, there will be three `bin` files and three `idx` files in the `WORKSPACE/alpaca_preprocessed` dictionary. Then, we can train a model with `--data-path WORKSPACE/alpaca_preprocessed/alpaca` and `--is-instruction-dataset` flags.
In addition, we have developed the dynamic padding function based on the instruction dataset, which can be implemented using the `--variable-seq-lengths` flag.

Note that instruction dataset has a `--handler-name GeneralInstructionHandler` flag which will choose `GeneralInstructionHandler` class to create prompt in `modellink/data/data_handler.py`.
If you have an alpaca-style dataset which have `instruction`, `input` and `output` columns, just use `GeneralInstructionHandler`.
In addition, `BelleMultiTurnInstructionHandler` is used to handle [belle dataset](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M),
`MOSSInstructionHandler` is used to handle [MOSS dataset](https://huggingface.co/datasets/fnlp/moss-003-sft-data) and `LeetcodePythonInstructionHandler` is used to handle [Leetcode dataset](https://huggingface.co/datasets/mhhmm/leetcode-solutions-python).
### <span id="jump13"> Pre-training </span>
```shell
 # Configure LLaMA-7B pre-training script: pretrain_llama_7b.sh
 # Configure vocabulary, dataset, and model parameter saving path according to actual conditions
 TOKENIZER_PATH=WORKSPACE/llama-7b-hf/tokenizer.model  # Path to the vocabulary
 DATA_PATH=WORKSPACE/alpaca_preprocessed/alpaca_text_document  # Path to pre-training dataset
```

Launch LLaMA-7B pre-training script: examples/llama/pretrain_llama_7b_ptd.sh
```shell
 bash examples/llama2/pretrain_llama_7b_ptd.sh
```

### <span id="jump14"> Full-parameter Fine-tuning </span>
```shell
 # Based on the pre-training script, provide the pre-training weight path, use instruction dataset path, and enable fine-tuning switch --finetune
 LOAD_CHECKPOINT_PATH="your init model weight load path"
 DATA_PATH=WORKSPACE/alpaca_preprocessed/alpaca_text_document  # Instruction fine-tuning dataset path
 
 torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
     --load ${LOAD_CHECKPOINT_PATH} \
     --finetune \
     ... \
     ...
```


### <span id="jump15"> Low-parameter fine-tuning </span>
#### Lora

Now, we support Lora to fine-tune your models.

First, you need to install version 0.4.0 of the peft library, like this:
```shell
pip install peft==0.4.0
```
When torch==1.11.0, You can also choose to install from [the source package in the GitHub repository](https://github.com/huggingface/peft/archive/refs/tags/v0.4.0.tar.gz), so you can modify the setup.py file to avoid some dependency issues.

Next, you just need to add this argument in your script to open Lora:

```shell
# Llama example
--lora-target-modules query_key_value dense gate_proj dense_h_to_4h dense_4h_to_h \
```

There are other Lora related arguments here, you can find their definitions in the [PEFT](https://github.com/huggingface/peft) library.

```shell
# Llama example
--lora-r 64 \
--lora-alpha 128 \
--lora-modules-to-save word_embeddings output_layer \
--lora-register-forward-hook word_embeddings input_layernorm \
```

Among them, the argument `--lora-register-forward-hook` is used to repair the gradient chain break caused by PP. It only needs to be set to the input layer of each PP stage, and the repair will not increase the trainable parameters. The argument `--lora-modules-to-save` is used for fine-tuning when expanding the vocabulary. If there is no need for this, there is no need to pass in this argument.

Finally, only Lora's parameters are saved after turning on Lora. Similarly, when loading a model, you need to specify the original model weight path and the Lora weight path. Parameters such as the optimizer are subject to those in the Lora weight path.

```shell
--load ${ORIGIN_CHECKPOINT} \
--lora-load ${LORA_CHECKPOINT} \
```

There is an [example](examples/llama/tune_llama_ptd_13b.sh) could be referred.

After using Lora to fine-tune the Llama model, the instruction dialogue effect is as follows:

```shell
You >> Give three tips for staying healthy.

MindSpeed-LLM:

- Start exercising regularly and eat healthy food.
- Get a good eight hours of sleep each night.
- Take medications regularly.
```

### <span id="jump16"> Inference: human-machine dialogue </span>
Currently, we support the following four cases of inference:
- PTD
- Model fine-tuned with lora

【For supported models, we also provide examples. Please refer to the following quick start】

#### Quick Start

***Please Note that:***
1. If you want to use the weight from huggingface, please run the weight conversion script first.
    Take Llama-7B, for example:

      - PTD only
           ```bash
            python tools/checkpoint/util.py --model-type GPT \
                                            --loader llama2_hf \
                                            --saver megatron \
                                            --target-tensor-parallel-size 1 \
                                            --target-pipeline-parallel-size 8 \
                                            --load-dir ./model_from_hf/llama-7b-hf \
                                            --save-dir ./model_weights/llama-7b-tp1-pp8 \
                                            --tokenizer-model ./model_from_hf/llama-7b-hf/tokenizer.model
           ```

5. You need to modify some variables in the shell script such as **model weight path** and **vocab path**.

    - **PTD only:** In this mode, the model is split by pipeline parallel and tensor parallel mode in megatron ways.
        ```bash
        sh examples/llama/generate_llama_7B_tp2_pp2.sh
        ```
    - **If you want to use lora model**, for details, refer to:
        ```bash
        sh tasks/inference/generate_llama_7b_lora_ptd.sh
        ```

#### Usage Guide
***Some examples with [Chinese-LLaMA-Alpaca-13B weights](https://github.com/ymcui/Chinese-LLaMA-Alpaca) is as below***:

##### Initializing the Distributed Environment
```python
initialize_megatron(args_defaults={'no_load_rng': True, 'no_load_optim': True})
```
##### Initializing model and loading weights

```python
from modellink import get_args
from modellink.model import GPTModel
from modellink.arguments import core_transformer_config_from_args


def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    config = core_transformer_config_from_args(get_args())
    init_model = GPTModel(
        config,
        num_tokentypes=0,
        parallel_output=False,
        return_moe_loss=False,
        pre_process=pre_process,
        post_process=post_process
    )
    return init_model


model = GPTModel.from_pretrained(
    model_provider=model_provider,
    pretrained_model_name_or_path="your model weight path"
)

"""
This is an API for initializing model and loading weight.

Parameters:
----------
model_provider(`func`):
    Function used to generate model objects which is similar to the training define.
pretrained_model_name_or_path(`str`, *optional*, defaults to None):
    File path of Model weight in megatron format (TP, PP may be used).
    If it is None, the random initialized weights will be used.
"""
```
##### <span id="case1"> Generate text in HuggingFace-like ways </span>

- Greedy Search
    ```python
    responses = model.generate(
        "Write quick sort code in python",
        max_new_tokens=512
    )
    ```
    <img src="sources/images/greedy.png">

- Do sample with top-k and top-p
    ```python
    responses = model.generate(
        "Write quick sort code in python",
        do_sample=True,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        max_new_tokens=512
    )
    ```
    <img src="sources/images/sampling.png">

- Beam search with top-k and top-p
    ```python
    responses = model.generate(
        "Write quick sort code in python",
        num_beams=4,
        top_k=50,
        top_p=0.95,
        max_new_tokens=512
    )
    ```
    <img src="sources/images/beam_search.png">

- Beam search with top-k and top-p sampling
    ```python
    responses = model.generate(
        "Write quick sort code in python",
        do_sample=True,
        temperature=0.6,
        num_beams=4,
        top_k=50,
        top_p=0.95,
        max_new_tokens=512
    )
    ```
    <img src="sources/images/beam_search_sampling.png">

### <span id="jump17"> Evaluation with Numerous Benchmarks </span>



#### Dataset Evaluation Results

<table>
  <thead>
    <tr>
      <th>Task</th>
      <th>Subset</th>
      <th>Model</th>
      <th>Ascend</th>
      <th>Reference</th>
      <th>Benchmark</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://huggingface.co/datasets/lukaemon/bbh">BBH</a></td>
      <td>test</td>
      <th>Llama7b</th>
      <td>0.334</td>
      <td>0.333</td>
      <td><a href="https://opencompass.org.cn/dataset-detail/BBH">0.335</a></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/datasets/lighteval/agi_eval_en">AGIEval</a></td>
      <td>test</td>
      <th>Llama7b</th>
      <td>0.210</td>
      <td>0.210</td>
      <td><a href="https://opencompass.org.cn/dataset-detail/AGIEval">0.206</a></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/datasets/openai_humaneval">HumanEval</a></td>
      <td>test</td>
      <th>Llama7b</th>
      <td>0.128</td>
      <td>0.128</td>
      <td><a href="https://opencompass.org.cn/dataset-detail/HumanEval">0.128</a></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/datasets/boolq">BoolQ</a></td>
      <td>test</td>
      <th>Llama7b</th>
      <td>0.742</td>
      <td>0.742</td>
      <td><a href="https://opencompass.org.cn/dataset-detail/BoolQ">0.754</a></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/datasets/gsm8k">GSM8K</a></td>
      <td>test</td>
      <th>Llama7b</th>
      <td>0.102</td>
      <td>0.103</td>
      <td><a href="https://opencompass.org.cn/dataset-detail/GSM8K">0.100</a></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/datasets/ceval/ceval-exam">CEval</a></td>
      <td>val</td>
      <th>Llama7b</th>
      <td>0.408</td>
      <td>0.404</td>
      <td>/</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/datasets/cais/mmlu">MMLU</a></td>
      <td>test</td>
      <th>Llama7b</th>
      <td>0.333</td>
      <td>0.324</td>
      <td><a href="https://browse.arxiv.org/pdf/2302.13971v1.pdf">0.351</a></td>
    </tr>
  </tbody>
</table>

#### Quick Start
```bash
# Configure model path and vocab_file path
# Vocab file can be downloaded from https://huggingface.co/yahma/llama-7b-hf
CHECKPOINT=../models/llama-7b-tp2-pp4/
VOCAB_FILE=../models/llama7b-hf/
# configure task and data path
DATA_PATH="dataset/boolq/test"
TASK="boolq"
# configure generation parameters
python -m torch.distributed.launch $DISTRIBUTED_ARGS evaluation_llama.py   \
       --task-data-path $DATA_PATH \
       --task $TASK\
       --seq-length 512 \
       --max-new-tokens 1 \
       --max-position-embeddings 512 \
       --tensor-model-parallel-size 2  \
       --pipeline-model-parallel-size 4  \
       --num-layers 32  \
       --hidden-size 4096  \
       --ffn-hidden-size 11008 \
       --load ${CHECKPOINT[images](sources%2Fimages)}  \
       --num-attention-heads 32  \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path $VOCAB_FILE \
       --tokenizer-not-use-fast \
       --fp16  \
       --micro-batch-size 1  \
       --seed 42 | tee logs/train.log
# start evaluation
bash tasks/evaluation/evaluate_llama_7B_ptd.sh
```

#### Task Introduction
The most important evaluation parameters must be `--max-new-tokens`, which means the output length of model generation. For example, multiple-choice
questions' output length is obviously shorter than coding tasks. Besides, this parameter largely decides the speed of model generation.

```bash
python -m torch.distributed.launch $DISTRIBUTED_ARGS evaluation_llama.py   \
       --task-data-path $DATA_PATH \
       --task $TASK\
       --seq-length 512 \
       --max-new-tokens 1 \
       --evaluation-batch-size 1 \
       --max-position-embeddings 512 \
       --tensor-model-parallel-size 2  \
       --pipeline-model-parallel-size 4  \
       --num-layers 32  \
       --hidden-size 4096  \
       --ffn-hidden-size 11008 \
       --load ${CHECKPOINT}  \
       --num-attention-heads 32  \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path $VOCAB_FILE \
       --tokenizer-not-use-fast \
       --fp16  \
       --micro-batch-size 1  \
       --seed 42 | tee logs/train.log
```
#### Evaluation Script Instructions

#### Baseline Dataset Introduction


##### MMLU
Since MMLU is a multidisciplinary task and 5 shots are performed, the length of each subject question varies greatly. If you want to run 57 subjects at the same time, you need to set `TASK="mmlu"`, `--seq-length=2048`, `--max-position-embeddings=2048`, `--max-new-token=2`. (`--max-new-tokens` can be set to between 2-4).
On many websites, the accuracy of the MMLU is evaluated according to disciplines. The 57 categories of single subjects belong to four main categories. Therefore, the statistics should be summarized according to the major categories of the subjects. The [website](https://github.com/hendrycks/test/blob/master/categories.py) gives the major categories of subjects for 57 categories of subjects.


##### GSM8K
GSM8K is a dataset of 8.5K high quality linguistically diverse grade school math word problems created by human problem writers. The answer of each question is a specific number. Since few shots are performed,  the question length is relatively long in GSM8K, and the output answer contains a chain of thoughts, it is necessary to configure `TASK="gsm8k"`, `--seq-length=2048`, `--max-position-embeddings=2048`, `--max-new-token=128`. (`--max-new-tokens` can be set between 256-512).

##### HumanEval
HumanEval dataset is a handcrafted set of 164 programming problems designed to challenge code generation models. The problems include a function signature, docstring, body, and several unit tests, all handwritten to ensure they're not included in the training set of code generation models.
Since the answer of HumanEval dataset contains long codes, it is necessary to configure `TASK="human_eval"`, `--seq-length=2048`, `--max-position-embeddings=2048`, `--max-new-token=1024`.

##### AGIEval
AGIEval is a human-centric benchmark specifically designed to evaluate the general
abilities of foundation models in tasks pertinent to human cognition and problem-solving. This benchmark is derived from 20 official, public, and high-standard admission and qualification exams intended for general human test-takers, such as general college admission tests (e.g., Chinese College Entrance Exam (Gaokao) and American SAT), law school admission tests, math competitions, lawyer qualification tests, and national civil service exams.Since the length of answers to different type of questions varies, we have to configure `TASK="agieval"`, `--seq-length=2048`, `--max-position-embeddings=2048`, `--max-new-token=1024` to fit the longest answer.

##### Big-Bench-Hard
Big-bench-hard dataset is a subset of big bench, which is a diverse evaluation suite that focuses on a suite of 23 challenging BIG-Bench tasks. These are the task for which prior language model evaluations did not outperform the average human-rater. This dataset covers multiple areas including text understanding, reasoning, logical reasoning, mathematical reasoning, and common sense reasoning.
Except word_sorting, all datasets are multiple-choice questions. So we can set `TASK="bbh"`, `--seq-length=2048`, `--max-position-embeddings=2048`, `--max-new-token=32`. (`--max-new-tokens` can be set between 32-64).

##### CEval
As [C-Eval](https://cevalbenchmark.com/) shows, C-Eval is a comprehensive Chinese evaluation suite for foundation models. It consists of 13948 multi-choice questions spanning 52 diverse disciplines and four difficulty levels, as shown below. You may explore our dataset examples at Explore, or check our paper for more details. The dataset contains validation and test data, however, only validation data has label for auto-evaluation. If
you want to evaluate on test data, you should email your results to [C-Eval](https://cevalbenchmark.com/).

### <span id="jump18">Acceleration Features
MindSpeed-LLM supports various acceleration algorithms such as tensor parallelism, pipeline parallelism, sequence parallelism, recomputation, distributed optimizer, and more. The table below shows the enable switches corresponding to each acceleration feature:

|   Acceleration Feature    |   Enable Parameter   |
|:---------:|:--------:|
|   Tensor Parallel    |  --tensor-model-parallel-size  |
|   Pipeline Parallel   | --pipeline-model-parallel-size |
|   Sequence Parallel    | --sequence-parallel |
|    Recomputation    | --recompute-granularity |
|  Distributed Optimizer   | --use-distributed-optimizer |

```bash
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 72 \
    --use-distributed-optimizer \
    ... \
    ...
```
In model training and downstream tasks, you can use --tensor-model-parallel-size and --pipeline-model-parallel-size to distribute the model across different devices, reducing memory usage per device.
Using --sequence-parallel to split the data enables the model to train on extremely long sequences. The options recompute-granularity or use-distributed-optimizer can also reduce memory usage per device.

MindSpeed-LLM supports multiple fused operators such as attn, rmsnorm, swiglu, rotary-pos-emb, mlp, etc., to improve model training performance:
```bash
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    --use-flash-attn \
    --use-fused-rmsnorm \
    --use-fused-swiglu \
    --use-fused-rotary-pos-emb \
    --mlp-layer-fusion \
    ... \
    ...
```


## Acknowledgments

---
MindSpeed-LLM is jointly contributed by the following departments of Huawei Corporation:
- Ascend Computing Product Unit
- Algorithm Unit of Computing Product Unit
- Research Unit of Computing Product Unit
- Open Computing Kit of Computing Product Unit
- General Development Department
- Global Technical Service Department

We appreciate every PR from community, and welcome to contribute to MindSpeed-LLM.

## Appendix

---
- Safety Statement: [Safety Statement](https://gitee.com/ascend/MindSpeed-LLM/wikis/%E5%AE%89%E5%85%A8%E5%A3%B0%E6%98%8E)
