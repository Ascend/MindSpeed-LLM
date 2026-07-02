<h1 align="center"> <img src="docs/en/pytorch/figures/readme/logo.png" height="110px" width="500px"> </h1>

<p align="center">
    <a href="https://gitcode.com/ascend/MindSpeed-LLM/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue">
    </a>
    <a href="https://gitcode.com/ascend/MindSpeed-LLM">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a>
        <img src="https://app.codacy.com/project/badge/Grade/1710faac5e634acaabfc26b0a778cdde">
    </a>
</p>

# Introduction

---

MindSpeed LLM is a distributed large language model training toolkit built on the Ascend ecosystem. It aims to provide ecosystem partners in the Huawei [Ascend chips](https://www.hiascend.com/) ecosystem with an end-to-end large language model training solution, including distributed pretraining, distributed instruction fine-tuning, and the corresponding development toolchain, such as data preprocessing, weight conversion, online inference, and baseline evaluation.

**<small>Note: The original repository name, ModelLink, has been changed to MindSpeed LLM, and the original package name, modellink, has been changed to mindspeed_llm. </small>**

# Roadmap

---

The latest roadmap updates are published in [MindSpeed LLM RoadMap](https://gitcode.com/Ascend/MindSpeed-LLM/issues/982). Visit there for the latest LLM plans and updates.

# Community Meetings

---
Please see the [Ascend Meeting Center](https://meeting.ascend.osinfra.cn/) for the schedule of MindSpeed LLM TC and SIG meetings.

# Join Us

---

To share development experience, exchange usage tips, and stay up to date on project releases, we created the MindSpeed LLM community group. Whether you already use this project or have new ideas, you are welcome to join.

Ways to join:

1. Scan the QR code to join the WeChat group directly. The QR code is valid for 7 days and is updated regularly.
2. Add the Ascend open-source assistant to get the group link and join the MindSpeed LLM community group.

<div style="display: flex; justify-content: flex-start; gap: 30px; align-items: flex-start; padding-left: 60px;">
  <div style="text-align: center;">
    <div>MindSpeed LLM community group</div>
    <img src="docs/en/pytorch/figures/wechat/llm_group.jpg" width="150" alt="MindSpeed LLM WeChat group">
  </div>
  <div style="text-align: center;">
    <div>Ascend open-source assistant</div>
    <img src="docs/en/pytorch/figures/wechat/ascend_assistant.jpg" width="150" alt="Ascend assistant WeChat">
  </div>
</div>

# Latest News

---

- [Apr. 25, 2026]: 🚀 [**DeepSeekV4-Flash** fixed-length data pretraining support](./examples/mcore/deepseek4_flash/README.md) 【Prototype】

- [Apr. 16, 2026]: 🚀 [**MiniMax_M27** model support](./examples/fsdp2/minimax_m27/) 【Prototype】

- [Mar. 28, 2026]: 🚀 [**Mamba3-block** demo model support](./examples/fsdp2/mamba3/) 【Prototype】
- [Mar. 27, 2026]: 🌴 MindSpeed LLM released the [v26.0.0 branch](https://gitcode.com/Ascend/MindSpeed-LLM/tree/26.0.0), which supports core_v0.12.1.
- [Mar. 10, 2026]: 🚀 MindSpeed LLM model sunset plan phase two ([link](https://gitcode.com/Ascend/MindSpeed-LLM/issues/1224)) started. Thank you for every contribution over the years.
- [Feb. 12, 2026]: 🚀 [**GLM-5** model support](./examples/mcore/glm5) 【Prototype】
- [Feb. 11, 2026]: 🚀 [**Step-3.5-Flash** model support](./examples/fsdp2/step35) 【Prototype】
- [Feb. 10, 2026]: 🚀 [FSDP2 training backend is now available, supporting the **Qwen3-Next** model](./examples/fsdp2/qwen3_next) 【Prototype】
- [Feb. 04, 2026]: 🚀 [**Qwen3-Coder-Next** model support for the mcore backend](./examples/mcore/qwen3_coder_next) 【Prototype】
- [Jan. 28, 2026]: 🌴 [Community image package 2.3.0 branch is now available](https://gitcode.com/Ascend/MindSpeed-LLM/blob/2.3.0/docs/pytorch/install_guide.md) 【Prototype】
- [Jan. 23, 2026]: 🌴 [Community image package 2.2.0 branch is now available](https://gitcode.com/Ascend/MindSpeed-LLM/blob/2.2.0/docs/pytorch/install_guide.md) 【Prototype】
- [Jan. 16, 2026]: 🌴 MindSpeed LLM released the [v2.3.0 branch](https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.3.0), which supports core_v0.12.1.
- [Dec. 24, 2025]: 🚀 **gpt-oss** model support
- [Dec. 11, 2025]: 🚀 **Qwen3-Next** model training supports Triton fusion to accelerate GDN module computation 【Prototype】
- [Nov. 25, 2025]: 🚀 [Online data and weight loading for training](./docs/en/pytorch/training/pretrain/mcore/train_from_hf.md)
- [Nov. 14, 2025]: 🚀 **magistral** model support 【Prototype】
- [Oct. 30, 2025]: 🚀 MindSpeed LLM model sunset plan ([link](https://gitcode.com/Ascend/MindSpeed-LLM/issues/943)) started. Thank you for every contribution over the years.
- [Oct. 28, 2025]: 🌴 MindSpeed LLM released the [v2.2.0 branch](https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0), which supports core_v0.12.1.
- [Oct. 16, 2025]: 🚀 **Qwen3-30B** supports DPO training.
- [Oct. 14, 2025]: 🚀 **DeepSeek-V3** pretraining now supports running based on the **[MindSpore AI framework](./docs/en/mindspore/readme.md)**.
- [Sep. 16, 2025]: 🚀 **Qwen3-Next** model support
- [Aug. 23, 2025]: 🚀 An optimized version of large-parameter model [weight conversion v2](./docs/en/pytorch/tools/checkpoint_convert_hf_mcore_large_params.md) is now available.
- [Jul. 28, 2025]: 🚀 Simultaneous first-release support for the **GLM-4.5-Air** model series
- [Jul. 25, 2025]: 🌴 MindSpeed LLM released the [v2.1.0 branch](https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.1.0), which supports core_r0.8.0.
- [Jul. 10, 2025]: 🚀 Features in the **[DeepSeek-R1](https://gitcode.com/Ascend/MindSpeed-RL/blob/master/docs/zh/solutions/r1_zero_deepseek_671b.md)** series are being rolled out gradually.
- [May. 19, 2025]: 🚀 Simultaneous first-release support for the **Qwen3** model series
- [Mar. 27, 2025]: 🚀 **[DeepSeek-R1-ZERO Qwen-7B](https://gitcode.com/ascend/MindSpeed-RL/blob/master/docs/zh/solutions/r1_zero_qwen25_7b.md)** **[DeepSeek-R1-ZERO Qwen-32B](https://gitcode.com/ascend/MindSpeed-RL/blob/master/docs/zh/solutions/r1_zero_qwen25_32b.md)**
- [Mar. 26, 2025]: 🚀 **[DeepSeek-V3-671B model suite](./examples/mcore/deepseek3)** is now available.

Note: 【Prototype】 indicates that the feature has not been fully validated. If you encounter any issues while using it, please report them in the [issue tracker](https://gitcode.com/Ascend/MindSpeed-LLM/issues).

- [MindSpeed LLM takes the lead in supporting MiniMax M2.7 training reproduction and accelerates model iteration to complete complex tasks](https://mp.weixin.qq.com/s/FWcQLu8InQvLh6YBd5Sq2w)
- [Say goodbye to tedious preprocessing. MindSpeed LLM launches Train_from_HF to enable load-and-train](https://mp.weixin.qq.com/s/kMUVWyCYLGKgceHzYXjigg)
- [Fast response. MindSpeed LLM seamlessly adapts to Step-3.5-Flash, unlocking new possibilities for large-scale MoE deployment](https://mp.weixin.qq.com/s/g7f_mpDgnvsc22P6XGxbmg)
- [MindSpeed LLM receives a major upgrade. It now supports the FSDP training backend and day-level adaptation for the Qwen3-Next-Coder model](https://mp.weixin.qq.com/s/Ihfc54P66bcO0r2j_mMX8A)
- [Get started quickly with the Qwen3-Coder-Next model on Ascend. A step-by-step guide is here.](https://mp.weixin.qq.com/s/yo0RlfU9gIY20NKyYQp4QA)

# Directory Structure

---

The MindSpeed LLM project code is organized according to modular design principles. For details, see the [Project Guide](./docs/en/project_guide.md).

``` shell
MindSpeed-LLM/
 ├── ci                        # CI watchdog.
 ├── configs                   # Configuration files.
 ├── docs                      # Project documentation.
 ├── examples                  # Model example scripts.
 ├── mindspeed_llm             # Core code.
 ├── tests                     # Test cases.
 ├── convert_ckpt.py           # Weight conversion tool.
 ├── convert_ckpt_v2.py        # Weight conversion tool v2.
 ├── preprocess_data.py        # Data preprocessing tool.
 ├── pretrain_gpt.py           # Pretraining workflow.
 ├── pretrain_mamba.py         # Pretraining workflow for Mamba models.
 ├── posttrain_gpt.py          # Post-training workflow.
 ├── preprocess_prompt.py      # Prompt preprocessing tool.
 ├── rlhf_gpt.py               # RLHF training workflow.
 ├── train_fsdp2.py            # FSDP2 training workflow.
 ├── inference.py              # Model inference tool.
 ├── evaluation.py             # Model evaluation tool.
 ├── setup.py                  # Installation configuration file.
 ├── README.md                 # Project overview document.
```

# Documentation Navigation

---

The [Documentation Guide](./docs/en/docs_guide.md) provides the complete usage guide for MindSpeed LLM and covers the following core topics:

- **Environment setup guide**: installation and configuration instructions for MindSpeed LLM.
- **Quick start**: beginner guidance from environment setup to launching training.
- **Model list**: models supported by the PyTorch and MindSpore frameworks.
- **Feature list**: performance optimization and memory optimization features.
- **Training solutions**: complete solutions for pretraining, fine-tuning, inference, and evaluation.
- **Toolchain**: usage instructions for weight conversion, dataset processing, performance collection and analysis, deterministic computation, and other tools.

# Release Notes

---

See the [release notes](docs/en/release_notes_llm.md).

# Installation

---

- For detailed installation steps and environment configuration, see the [MindSpeed LLM installation guide for PyTorch](./docs/en/pytorch/training/install_guide.md).
- For detailed installation steps and environment configuration, see the [MindSpeed LLM installation guide for MindSpore](./docs/en/mindspore/install_guide.md).

# Quick Start

---

This section helps you quickly launch large language model pretraining and fine-tuning tasks. See the following guides for details:

- [Quick start for the PyTorch framework](./docs/en/pytorch/training/quick_start.md)
- [Quick start for the MindSpore framework](./docs/en/mindspore/quick_start.md)

# Supported Models

---

MindSpeed LLM now includes built-in support for pretraining and fine-tuning more than 100 widely used large language models. See the supported model lists:

- [PyTorch framework supported model list](./docs/en/pytorch/models/supported_models.md)
- [MindSpore framework supported model list](./docs/en/mindspore/models/supported_models.md)

# Training Solutions and Features

---

MindSpeed LLM includes training solutions such as distributed pretraining and distributed fine-tuning. For details, see [Training Solutions and Features](./docs/en/pytorch/features/README.md).

# Online Inference

---

<table>
  <thead>
    <tr>
      <th>Feature</th>
      <th>MCore</th>
      <th>Released</th>
      <th>Contributor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="docs/en/pytorch/training/inference/inference.md">Streaming inference</a></td>
      <td>✅</td>
      <td>✅</td>
      <td>【NAIE】</td>
    </tr>
    <tr>
      <td><a href="docs/en/pytorch/training/inference/chat.md">Chat conversation</a></td>
      <td>✅</td>
      <td>✅</td>
      <td>【NAIE】</td>
    </tr>
    <tr>
      <td><a href="docs/en/pytorch/features/mcore/yarn.md">YARN context extension</a></td>
      <td>✅</td>
      <td>❌</td>
      <td>【Ascend】</td>
    </tr>
  </tbody>
</table>

# Open Dataset Evaluation

---

See the [open dataset evaluation baselines](docs/en/pytorch/training/evaluation/models_evaluation.md) for model baselines in the repository.
<table>
  <thead>
    <tr>
      <th>Scenario</th>
      <th>Dataset</th>
      <th>MCore</th>
      <th>Released</th>
      <th>Contributor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="8"><a href="docs/en/pytorch/training/evaluation/evaluation_guide.md">Evaluation</a></td>
      <td><a href="https://people.eecs.berkeley.edu/~hendrycks/data.tar">MMLU</a></td>
      <td>✅</td>
      <td>❌</td>
      <td>【NAIE】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/datasets/ceval/ceval-exam/tree/main">CEval</a></td>
      <td>✅</td>
      <td>❌</td>
      <td>【NAIE】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/google-research-datasets/boolean-questions">BoolQ</a></td>
      <td>✅</td>
      <td>❌</td>
      <td>【NAIE】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/suzgunmirac/BIG-Bench-Hard/tree/main/bbh">BBH</a></td>
      <td>✅</td>
      <td>❌</td>
      <td>【NAIE】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/ruixiangcui/AGIEval/tree/main">AGIEval</a></td>
      <td>✅</td>
      <td>❌</td>
      <td>【NAIE】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/openai/human-eval/tree/master/data">HumanEval</a></td>
      <td>✅</td>
      <td>❌</td>
      <td>【NAIE】</td>
    </tr>
  </tbody>
</table>

# Developer Toolchain

---

## Weight Conversion

MindSpeed LLM supports two-way weight conversion between Hugging Face and Megatron-core formats, and it supports merging LoRA weights. For parameters and usage instructions for the weight conversion feature, see [Weight Conversion](docs/en/pytorch/tools/checkpoint_convert_hf_mcore.md).

<table>
  <thead>
    <tr>
      <th>Source format</th>
      <th>Target format</th>
      <th>Sharding features</th>
      <th>LoRA</th>
      <th>Contributor</th>
      <th>Released</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Hugging Face</td>
      <td>Megatron-core</td>
      <td>tp, pp, dpp, vpp, cp, ep, loop layer</td>
      <td>❌</td>
      <td rowspan="3">【Ascend】</td>
      <td rowspan="3">❌</td>
    </tr>
    <tr>
      <td rowspan="2">Megatron-core</td>
      <td>Hugging Face</td>
      <td></td>
      <td>✅</td>
    </tr>
    <tr>
      <td>Megatron-core</td>
      <td>tp, pp, dpp, vpp, cp, ep, loop layer</td>
      <td>✅</td>
    </tr>
  </tbody>
</table>

## Data Preprocessing

MindSpeed LLM supports data preprocessing for pretraining, instruction fine-tuning, and other tasks.

<table>
  <thead>
    <tr>
      <th>Task scenario</th>
      <th>Dataset</th>
      <th>MCore</th>
      <th>Released</th>
      <th>Contributor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Pretraining</td>
      <td><a href="docs/en/pytorch/tools/data_process_pretrain.md">Pretraining data processing</a></td>
      <td>✅</td>
      <td>✅</td>
      <td rowspan="3">【Ascend】</td>
    </tr>
    <tr>
      <td rowspan="2">Fine-tuning</td>
      <td><a href="docs/en/pytorch/tools/data_process_sft_alpaca_style.md">Alpaca style</a></td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td><a href="docs/en/pytorch/tools/data_process_sft_sharegpt_style.md">ShareGPT style</a></td>
      <td>✅</td>
      <td>✅</td>
    </tr>
  </tbody>
</table>

## Performance Collection

<table>
  <thead>
    <tr>
      <th>Scenario</th>
      <th>Feature</th>
      <th>MCore</th>
      <th>Released</th>
      <th>Contributor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="1">Performance collection</td>
      <td><a href="docs/en/pytorch/tools/profiling.md">Collect profiling data on Ascend chips</a></td>
      <td>✅</td>
      <td>❌</td>
      <td>【Ascend】</td>
    </tr>
  </tbody>
</table>

## High Availability

<table>
  <thead>
    <tr>
      <th>Scenario</th>
      <th>Feature</th>
      <th>MCore</th>
      <th>Released</th>
      <th>Contributor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">High availability</td>
      <td><a href="docs/en/pytorch/tools/deterministic_computation.md">Enable deterministic computation on Ascend chips</a></td>
      <td>✅</td>
      <td>❌</td>
      <td rowspan="2">【Ascend】</td>
    </tr>
  </tbody>
</table>

# Version Maintenance Policy

---

MindSpeed LLM versions go through the following five maintenance stages:

| **Status** | **Time** | **Description** |
| ------------------- | --------- | ------------------------------------------------------------ |
| Planning | 1 to 3 months | Planned features. |
| Development | 3 months | Features under development. |
| Maintenance | 6 to 12 months | Merge all resolved issues and release versions. Different MindSpeed LLM versions follow different maintenance strategies. The maintenance period is six months for regular releases and twelve months for long-term support releases. |
| No maintenance | 0 to 3 months | Merge all resolved issues. No dedicated maintenance staff. No version release. |
| End of life (EOL) | N/A | The branch no longer accepts any changes. |

MindSpeed LLM released version maintenance policy:

| **MindSpeed LLM version** | **Corresponding tag** | **Maintenance policy** | **Current status** | **Release date** | **Next status** | **EOL date** |
| --------------------- | ------------| ------------ | ------------ | ------------ | ---------------------- | ----------- |
| 2.3.0 | v2.3.0 | Regular release | Maintenance | 2025/12/30 | Expected to enter no-maintenance status starting 2026/6/30 | |
| 2.2.0 | v2.2.0 | Regular release | Maintenance | 2025/9/30 | Expected to enter no-maintenance status starting 2026/3/30 | |
| 2.1.0 | v2.1.0 | Regular release | EOL | 2025/6/30 | End of life | 2025/12/30 |
| 2.0.0 | v2.0.0 | Regular release | EOL | 2025/3/30 | End of life | 2025/9/30 |
| 1.0.0 | v1.0.0 | Regular release | EOL | 2024/12/30 | End of life | 2025/6/30 |
| 1.0.RC3 | v1.0.RC3.0 | Regular release | EOL | 2024/09/30 | End of life | 2025/3/30 |
| 1.0.RC2 | v1.0.RC2.0 | Regular release | EOL | 2024/06/30 | End of life | 2024/12/30 |
| 1.0.RC1 | v1.0.RC1.0 | Regular release | EOL | 2024/03/30 | End of life | 2024/9/30 |
| bk_origin_23 | \ | Demo | EOL | 2023 | End of life | 2024/6/30 |

# Security Statement

---

[MindSpeed LLM Security Statement](https://gitcode.com/Ascend/MindSpeed-LLM/wiki/%E5%AE%89%E5%85%A8%E7%9B%B8%E5%85%B3%2F%E5%AE%89%E5%85%A8%E5%A3%B0%E6%98%8E.md)

# Disclaimer

---

## To MindSpeed LLM Users

1. The models provided by MindSpeed LLM are for non-commercial use only.
2. Third-party open source software that MindSpeed LLM depends on, such as Megatron, is provided and maintained by third-party communities. Fixes for problems caused by third-party open source software depend on contributions from and feedback to the relevant communities. You should understand that the MindSpeed LLM repository does not guarantee fixes for problems in the third-party open source software itself, and it also does not guarantee that it tests or corrects all vulnerabilities and errors in third-party open source software.
3. For each model, the MindSpeed LLM platform only suggests datasets that you can use for training. Huawei does not provide any datasets. If you use these datasets for training, you must especially ensure that you comply with the license of the corresponding dataset. If you become involved in an infringement dispute because of your use of a dataset, Huawei bears no responsibility.
4. If you find any issues while using MindSpeed LLM models, including but not limited to functional issues and compliance issues, please submit an issue on GitCode. We will review and address it in a timely manner.

## To Dataset Owners

If you do not want your dataset to be mentioned in MindSpeed LLM model descriptions, or if you want to update the description of your dataset in MindSpeed LLM, please submit an issue on GitCode. We will delete or update the dataset description according to your request. We sincerely thank you for your understanding and contribution to MindSpeed LLM.

# License Statement

- The license for the MindSpeed LLM product is described in [LICENSE](LICENSE).
- The documents in the `docs` directory of the MindSpeed LLM tool are subject to the CC-BY 4.0 license. See [LICENSE](./docs/zh/LICENSE) for details.

# Contribution Statement

---

If you want to report issues or contribute code to MindSpeed LLM, see the [Contribution Guide](./CONTRIBUTING.md).

# FAQ

---

For common questions about basic MindSpeed LLM usage, see the [MindSpeed LLM FAQ](./docs/en/FAQ.md). If the FAQ does not cover your question, you can search for similar issues in the repository [issues list](https://gitcode.com/Ascend/MindSpeed-LLM/issues), or submit a new issue.

# Acknowledgments

---

MindSpeed LLM is jointly contributed by the following Huawei departments and Ascend ecosystem partners:

Huawei:

- Computing Product Line: Ascend
- Public Development Department: NAIE
- Global Technical Service Department: GTS
- Huawei Cloud Computing: Cloud

Ecosystem partners:

- Mobile Cloud (China Mobile Cloud): Dayun Zhenze Intelligent Computing Platform
- Big Data and Artificial Intelligence Lab of the Software Development Center of Industrial and Commercial Bank of China

Thank you for every PR from the community. Contributions to MindSpeed LLM are welcome.
