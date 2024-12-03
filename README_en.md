<p align="center"> <img src="sources/images/logo.png" height="110px" width="500px"> </p>

<p align="center">
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

MindSpeed-LLM, originally named ModelLink, aims to provide end-to-end large language model solutions for Huawei Ascend chips, including models, algorithms, and downstream tasks.

---

## MindSpeed-LLM Solution Overview

Current MindSpeed-LLM supported features for large model usage:
* [Dataset Preparation for Pre-training](#jump11)/[Fine-tuning Instruction Dataset Preparation](#jump12)
* [Pre-training](#jump13)/[Full-parameter Fine-tuning](#jump14)/[Low-parameter Fine-tuning](#jump15)
* [Inference: human-machine dialogue](#jump16)
* [Evaluation with numerous benchmarks](#jump17)
* [Utilizing Acceleration Features (Acceleration Algorithms + Fusion Operators)](#jump18)
* [Profiling data based on Ascend chips](#jump19)
* [Convert ckpt between huggingface and megatron](#jump19)
* [Enbale deterministic computing function for Ascend](#jump21)

More novel and useful features are developing for LLMs training on Ascend ...

---


## MindSpeed-LLM Maintenance Policies

MindSpeed-LLM release has the following five maintenance phases:

| **Status**        | **Duration** | **Description**                                                                                                                |
|-------------------|--------------|--------------------------------------------------------------------------------------------------------------------------------|
| Planning          | 1-3 months   | Plan features.                                                                                                                 |
| Development       | 3 months     | Develop features.                                                                                                              |
| Maintained        | 6-12 months | Allow the incorporation of all resolved issues and release the version, Different versions of MindSpeed-LLM adopt varying support policies. The maintenance periods for Regular Releases and Long-Term Support versions are 6 months and 12 months, respectively. |
| Unmaintained      | 0-3 months   | Allow the incorporation of all resolved issues. No dedicated maintenance personnel are available. No version will be released. |
| End Of Life (EOL) | N/A          | Do not accept any modification to a branch.                                                                                    |


 The Released Version Maintenance Policy for MindSpeed-LLM is as follows:

| **MindSpeed-LLM** |  **Maintenance Policies** | **Status** | **Launch Date** | **Subsequent Status**             | **EOL Date**     |
|---------------|--------------------|--------------|-----------------|-----------------------------------|-----------|
| 1.0           |  Regular Release  | Maintained   | 2024/03/30      | Unmaintained 2024/09/30 estimated |           |
| 1.1           |  Regular Release  | Maintained   | 2024/06/30      | Unmaintained 2024/12/30 estimated |           |

---

## Supporing version and models

【Please note the corresponding environment versions for model usage, as follows】


<table border="0">
  <tr>
    <th>Software</th>
    <th>Version</th>
    <th>Software Installation Guide</th>
    <th>Recommended Hardware Form</th>
  </tr>

  <tr>
    <td>Ascend NPU Driver</td>
    <td rowspan="2">Ascend HDK 24.1.RC2</td>
    <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/softwareinst/instg/instg_0003.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit">Driver And Firm Installation Guide</a>》</td>
    <td rowspan="8">Atlas 900 A2 PODc</td>
  </tr>
  <tr>
    <td>Ascend NPU Firm</td>
  </tr>
  <tr>
    <td>Toolkit</td>
    <td rowspan="2">CANN 8.0.RC2</td>
    <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/softwareinst/instg/instg_0000.html">CANN Installation Guide</a>》</td>
  </tr>
  <tr>
    <td>Kernel</td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td>PyTorch</td>
    <td rowspan="3">release v6.0.RC2</td>
    <td rowspan="3">《<a href="https://www.hiascend.com/document/detail/zh/Pytorch/60RC2/configandinstg/instg/insg_0001.html">Ascend Extension for PyTorch Installation Guide</a>》</td>
  </tr>
  <tr>
    <td>torch_npu</td>
  </tr>
  <tr>
    <td>apex</td>
  </tr>
</table>


【Based on the current version of megatron, the performance statistics from our testing are as follows (Hardware info：Atlas 900 A2 PODc)】

For the following supported models, we provide training scripts and readme instructions in the examples folder, which contain detailed processes for model training, inference, and evaluation.

The following is the measured performance after turning on the mc2 acceleration feature, which is only supported in the version above 24RC2, the warehouse code level is closed by default, if you want to use, please refer to the `Acceleration Features` chapter on the home page

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Parameters</th>
      <th>Sequence length</th>
      <th>Cluster Scale</th>
      <th>Precision Mode</th>
      <th>Performance </th>
      <th>Reference Performance </th>
      <th>Contributor </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="1"><a href="examples/aquila/README.md">Aquila</a></td>
      <td>7B</td>
      <td> 2K </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 2849 </td>
      <td> 2874 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/aquila2/README_en.md">Aquila2</a></td>
      <td>7B</td>
      <td> 2K </td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 3323 </td>
      <td> 2673 </td>
      <td>【Community】</td>
    </tr>
    <tr>
      <td>34B</td>
      <td> 4K </td>
      <td> 2x8</td>
      <td> BF16 </td>
      <td> 854 </td>
      <td> 732 </td>
      <td>【Community】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/baichuan/README.md">Baichuan</a></td>
      <td>7B</td>
      <td> 4K </td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 2685 </td>
      <td> 2036 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td>13B</td>
      <td> 4K </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 1213 </td>
      <td> 862 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/baichuan2/README.md">Baichuan2</a></td>
      <td>7B</td>
      <td> 4K </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 2664 </td>
      <td> 3969 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td>13B</td>
      <td> 4K </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 1668 </td>
      <td> 2062 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/bloom/README.md">Bloom</a></td>
      <td>7B1</td>
      <td> 2K </td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 2034 </td>
      <td> 2525 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td >176B</td>
      <td> 2K </td>
      <td >12x8</td>
      <td> BF16 </td>
      <td> 100 </td>
      <td> 107 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/chatglm3/README.md">ChatGLM3</a></td>
      <td>6B</td>
      <td> 8K </td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 4297 </td>
      <td> 4267 </td>
      <td>【Community】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/codellama/README.md">CodeLlama</a></td>
      <td>34B</td>
      <td> 4K </td>
      <td> 2x8</td>
      <td> BF16 </td>
      <td> 837 </td>
      <td> 762 </td>
      <td>【Community】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/intern/README.md">InternLM</a></td>
      <td>7B</td>
      <td> 2K </td>
      <td>1x8</td>
      <td>BF16</td>
      <td> 2776 </td>
      <td> 2854 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td >65B</td>
      <td> 2K </td>
      <td >4x8</td>
      <td> BF16 </td>
      <td> 341 </td>
      <td> 414 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td rowspan="4"><a href="examples/llama/README.md">LLaMA</a></td>
      <td>7B</td>
      <td> 2K </td>
      <td>1x8</td>
      <td>FP16</td>
      <td> 3600 </td>
      <td> 3804 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td>13B</td>
      <td> 2K </td>
      <td>1x8</td>
      <td>FP16</td>
      <td> 1895 </td>
      <td> 2012 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
        <td>33B</td>
        <td> 2K </td>
        <td>4x8</td>
        <td>FP16</td>
        <td>621</td>
        <td>776</td>
        <td>【Ascend】</td>
    </tr>
    <tr>
      <td>65B</td>
      <td> 2K </td>
      <td>4x8</td>
      <td>BF16 </td>
      <td> 348 </td>
      <td> 426 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td rowspan="4"><a href="examples/llama2/README.md">LLaMA2</a></td>
      <td>7B</td>
      <td> 4K </td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 4200 </td>
      <td> 3850 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td>13B</td>
      <td> 4K </td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 1990 </td>
      <td> 1920 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td>34B</td>
      <td> 4K </td>
      <td>2x8</td>
      <td>BF16 </td>
      <td> 749 </td>
      <td> 796 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td>70B</td>
      <td> 4K </td>
      <td>4x8</td>
      <td>BF16 </td>
      <td> 420 </td>
      <td> 430 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/llama3/README.md">LLaMA3</a></td>
      <td>8B</td>
      <td> 8K </td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 2483 </td>
      <td> 2674 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td>70B</td>
      <td> 8K </td>
      <td>8x8</td>
      <td>BF16 </td>
      <td> 283 </td>
      <td> 355 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td rowspan="3"><a href="examples/qwen/README.md">Qwen</a></td>
      <td>7B</td>
      <td> 8K </td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 2499 </td>
      <td> 2867 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td>14B</td>
      <td> 2K </td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 1560 </td>
      <td> 1578 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td>72B</td>
      <td> 8K </td>
      <td>16x8</td>
      <td>BF16 </td>
      <td> 285 </td>
      <td> 345 </td>
      <td>【Ascend】</td>
    </tr>
   <tr>
      <td rowspan="7"><a href="examples/qwen15/README.md">Qwen1.5</a></td>
      <td> 0.5B </td>
      <td> 8K </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 22834 </td>
      <td> 25306 </td>
      <td>【Community】</td>
      <tr>
      <td> 1.8B </td>
      <td> 8K </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 13029 </td>
      <td> 12181 </td>
      <td>【Community】</td>
      <tr>
      <td> 4B </td>
      <td> 8K </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  5033 </td>
      <td> 5328 </td>
      <td>【Community】</td>
      </tr>
      <td> 7B </td>
      <td> 8K </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  2862 </td>
      <td> 2621 </td>
      <td>【Community】</td>
      <tr>
      <td> 14B </td>
      <td> 8K </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 1717 </td>
      <td> 1702 </td>
      <td>【Community】</td>
      <tr>
      <td> 32B </td>
      <td> 8K </td>
      <td> 4x8 </td>
      <td> BF16 </td>
      <td> 751 </td>
      <td> 708 </td>
      <td>【Community】</td>
      <tr>
      <td> 72B </td>
      <td> 8K </td>
      <td> 8x8 </td>
      <td> BF16 </td>
      <td> 301 </td>
      <td> 317 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/yi/README_en.md">Yi</a></td>
      <td>34B</td>
      <td> 4K </td>
      <td>2x8</td>
      <td>BF16 </td>
      <td> 809 </td>
      <td> 730 </td>
      <td>【Community】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/mixtral/README.md">Mixtral</a></td>
      <td>8x7B</td>
      <td> 32K </td>
      <td>2x8</td>
      <td>BF16 </td>
      <td> 487 </td>
      <td> 610 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/mistral/README.md">Mistral</a></td>
      <td>7B</td>
      <td> 32K </td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 2806 </td>
      <td> 2734 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/gemma/README.md">Gemma</a></td>
      <td>2B</td>
      <td> 8K </td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 6821 </td>
      <td> 7602 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td>7B</td>
      <td> 8K </td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 2938 </td>
      <td> 2607 </td>
      <td>【Ascend】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/gpt3/README.md">GPT3</a></td>
      <td>175B</td>
      <td> 2K </td>
      <td> 16x8 </td>
      <td> FP16 </td>
      <td> 153 </td>
      <td> -- </td>
      <td>【Community】</td>
    </tr>
  </tbody>
</table>

---


## Acceleration Features
MindSpeed-LLM supports various acceleration algorithms such as tensor parallelism, pipeline parallelism, sequence parallelism, recomputation, distributed optimizer, and more. The table below shows the enable switches corresponding to each acceleration feature:

|         Acceleration Feature         |        Enable Parameter        |
|:------------------------------------:|:------------------------------:|
|           Tensor Parallel            |  --tensor-model-parallel-size  |
|          Pipeline Parallel           | --pipeline-model-parallel-size |
|       Dynamic division for PP        |        --num-layer-list        |
|          Sequence Parallel           |      --sequence-parallel       |
|            Recomputation             |    --recompute-granularity     |
|        Distributed Optimizer         |  --use-distributed-optimizer   |
|        overlap DDP allreduce         |     --overlap-grad-reduce      |
|           Flash attention            |        --use-flash-attn        |
|            Fused rmsnorm             |      --use-fused-rmsnorm       |
|             Fused swiglu             |       --use-fused-swiglu       |
|                 mc2                  |     --use-mc2                  |
| Fused rotary <br/>position embedding |   --use-fused-rotary-pos-emb   |
|       Sliding window attention       |        --sliding-window        |



```bash
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layer-list 1,2,2,2,1 \
    --sequence-parallel \
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 72 \
    --use-distributed-optimizer \
    --use-flash-attn \
    --use-fused-rmsnorm \
    --use-fused-swiglu \
    --overlap-grad-reduce \
    --use-fused-rotary-pos-emb \
    --use-mc2 \
    --sliding-window 4096 \
    ... \
    ...
```

```bash
Note:
To enable mc2, ensure the following:
1. The environment version matches the description on the repository homepage;
2. Comment out line 283 in the validate_args_decorator function within modellink\arguments.py:
   #args.use_mc2 = False
```

---


## Analyze profiling data based on Ascend chips 
MindSpeed-LLM supports analyze profiling data based on Ascend chips, which is useful for modelling:

```bash
--profile                        # enable profiling
--profile-step-start  5          # the start step
--profile-step-end 6             # the end step
--profile-ranks 0 1 2 3 4        # ranks for profiling
--profile-level level2           # level0, 1, 2 for data profiling
--profile-with-cpu               # profiling cpu information
--profile-with-stack             # profile stack information
--profile-with-memory            # profile memory information
--profile-record-shapes          # profile shape information
--profile-save-path ./profile_dir    # path to save data
```

---

## Enable deterministic computing based on Ascend chips 

- add choice in script
```shell
--use-deter-comp
```
- add environment variable
```shell
export HCCL_DETERMINISTIC=True
```

---

## Acknowledgments

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
