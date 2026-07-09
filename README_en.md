# <p align="center"> <img src="docs/zh/pytorch/figures/readme/logo.png" height="110px" width="500px"> </p>

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

- [Mar. 10, 2026]: 🚀 MindSpeed LLM model sunset plan phase two ([link](https://gitcode.com/Ascend/MindSpeed-LLM/issues/1224)) started. Thank you for every contribution over the years.
- [Feb. 12, 2026]: 🚀 [**GLM-5** model support](./examples/mcore/glm5) 【Prototype】
- [Feb. 11, 2026]: 🚀 [**Step-3.5-Flash** model support](./examples/fsdp2/step35) 【Prototype】
- [Feb. 10, 2026]: 🚀 [FSDP2 training backend is now available, supporting the **Qwen3-Next** model](./examples/fsdp2/qwen3_next) 【Prototype】
- [Feb. 04, 2026]: 🚀 [**Qwen3-Coder-Next** model support for the MCore backend](examples/mcore/qwen3_coder_next) 【Prototype】
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
- [Mar. 27, 2025]: 🚀 **[DeepSeek-R1-ZERO Qwen-7B](https://gitcode.com/ascend/MindSpeed-RL/blob/master/docs/zh/solutions/r1_zero_qwen25_7b.md)** **[DeepSeek-R1-ZERO Qwen-32B](https://gitcode.com/ascend/MindSpeed-RL/blob/master/docs/zh/solutions/r1_zero_qwen25_32b.md)**.
- [Mar. 26, 2025]: 🚀 **[DeepSeek-V3-671B model suite](./examples/mcore/deepseek3)** is now available.

Note: 【Prototype】 indicates that the feature has not been fully validated. If you encounter any issues while using it, please report them in the [issue tracker](https://gitcode.com/Ascend/MindSpeed-LLM/issues).

# Directory Structure

---

The MindSpeed LLM project code is organized according to modular design principles. For details, see the [Project Guide](./docs/en/project_guide.md).

```bash
MindSpeed-LLM/
 ├── ci                        # CI watchdog
 ├── configs                   # Configuration files
 ├── docs                      # Project documentation
 ├── examples                  # Model example scripts
 ├── mindspeed_llm             # Core code
 ├── tests                     # Test cases
 ├── convert_ckpt.py           # Weight conversion tool
 ├── convert_ckpt_v2.py        # Weight conversion tool v2
 ├── preprocess_data.py        # Data preprocessing tool
 ├── pretrain_gpt.py           # Pretraining workflow
 ├── pretrain_mamba.py         # Pretraining workflow for Mamba models
 ├── posttrain_gpt.py          # Post-training workflow
 ├── preprocess_prompt.py      # Prompt preprocessing tool
 ├── rlhf_gpt.py               # RLHF training workflow
 ├── train_fsdp2.py            # FSDP2 training workflow
 ├── inference.py              # Model inference tool
 ├── evaluation.py             # Model evaluation tool
 ├── setup.py                  # Installation configuration file
 ├── README.md                 # Project overview document
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

MindSpeed LLM includes training solutions such as distributed pretraining and distributed fine-tuning.

## Distributed Pretraining

The measured pretraining performance based on MindSpeed LLM is as follows.

<table>
  <thead>
    <tr>
      <th>Model Series</th>
      <th>Experimental Model</th>
      <th>Hardware</th>
      <th>Cluster Size</th>
      <th>MFU</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">Llama-2</td>
      <td><a href="./examples/mcore/llama2/pretrain_llama2_7b_pack_ptd.sh">Llama-2-7B</a></td>
      <td>Atlas 900 A2 PODc</td>
      <td>1x8</td>
      <td>69.0%</td>
    </tr>
    <tr>
      <td><a href="./examples/mcore/llama2/pretrain_llama2_13b_pack_ptd.sh">Llama-2-13B</a></td>
      <td>Atlas 900 A2 PODc</td>
      <td>1x8</td>
      <td>64.7%</td>
    </tr>
    <tr>
      <td><a href="./examples/mcore/llama2/pretrain_llama2_70b_pack_ptd.sh">Llama-2-70B</a></td>
      <td>Atlas 900 A2 PODc</td>
      <td>4x8</td>
      <td>44.1%</td>
    </tr>
    <tr>
      <td>Mixtral</td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/blob/2.2.0/examples/mcore/mixtral/pretrain_mixtral_8x7b_ptd.sh">Mixtral-8x7B</a></td>
      <td>Atlas 900 A2 PODc</td>
      <td>8x8</td>
      <td>31.7%</td>
    </tr>
  </tbody>
</table>

### Pretraining Solutions

<table>
  <thead>
    <tr>
      <th>Solution Category</th>
      <th>MCore</th>
      <th>Released</th>
      <th>Contributor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="docs/en/pytorch/training/pretrain/mcore/pretrain.md">Multi-dataset pretraining</a></td>
      <td>✅</td>
      <td>✅</td>
      <td rowspan="2">[Ascend]</td>
    </tr>
    <tr>
      <td><a href="docs/en/pytorch/training/pretrain/mcore/pretrain_eod.md">Multi-sample pack-mode pretraining</a></td>
      <td>✅</td>
      <td>❌</td>
</tr>
  </tbody>
</table>

### Acceleration Features

<table><thead>
  <tr>
    <th>Scenario</th>
    <th>Feature</th>
    <th>MCore</th>
    <th>Released</th>
    <th>Contributor</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="5">SPTD parallelism</td>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/tensor-parallel.md">Tensor parallelism</a></td>
    <td>✅</td>
    <td>✅</td>
    <td rowspan="29">[Ascend]</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/pipeline-parallel.md">Pipeline parallelism</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="docs/en/pytorch/features/mcore/virtual_pipeline_parallel.md">Virtual pipeline parallelism</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/sequence-parallel.md">Sequence parallelism</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/noop-layers.md">noop layers</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td rowspan="3">Long-sequence parallelism</td>
    <td><a href="docs/en/pytorch/features/mcore/ring-attention-context-parallel.md">Ascend Ring Attention Long-sequence parallelism</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/ulysses-context-parallel.md">Ulysses Long-sequence parallelism</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/hybrid-context-parallel.md">Hybrid long-sequence parallelism</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td rowspan="2">MoE</td>
    <td><a href="https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md">MoE parallelism</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/megatron_moe/megatron-moe-allgather-dispatcher.md">MoE rerouting communication optimization</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td rowspan="6">Memory optimization</td>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/reuse-fp32-param.md">Parameter copy reuse</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
    <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/distributed-optimizer.md">Distributed optimizer</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/swap_attention.md">Swap Attention</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="docs/en/pytorch/features/mcore/recompute_relative.md">Recomputation</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/norm-recompute.md">Norm recomputation</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="docs/en/pytorch/features/mcore/o2.md">O2 BF16 Optimizer</a></td>
    <td>✅</td>
    <td>❌</td>
  </tr>
  <tr>
    <td rowspan="7">Fusion operators</td>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/flash-attention.md">Flash attention</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="docs/en/pytorch/features/mcore/variable_length_flash_attention.md">Flash attention variable length</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/rms_norm.md">Fused rmsnorm</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/swiglu.md">Fused swiglu</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/rotary-embedding.md">Fused rotary position embedding</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/megatron_moe/megatron-moe-gmm.md">GMM</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/npu_matmul_add.md">Matmul Add</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td rowspan="6">Communication optimization</td>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/async-ddp-param-gather.md">Gradient reduce communication-compute overlap</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/recompute_independent_pipelining.md">Recompute in advance</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/async-ddp-param-gather.md">Weight all-gather communication-compute overlap</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="docs/en/pytorch/features/mcore/mc2.md">MC2</a></td>
    <td>✅</td>
    <td>❌</td>
  </tr>
  <tr>
    <td><a href="docs/en/pytorch/features/mcore/communication-over-computation.md">CoC</a></td>
    <td>✅</td>
    <td>❌</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/hccl-replace-gloo.md">Ascend Gloo archive flushing optimization</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
</tbody></table>

## Distributed Fine-Tuning

The measured instruction fine-tuning performance based on MindSpeed LLM is as follows.

<table>
  <tr>
    <th>Model</th>
    <th>Hardware</th>
    <th>Cluster</th>
    <th>Solution</th>
    <th>Sequence</th>
    <th>Performance</th>
    <th>MFU</th>
  </tr>
  <tr>
    <td rowspan="3">Llama-2-7B</td>
    <td rowspan="3">Atlas 900 A2 PODc</td>
    <td rowspan="3">1x8</td>
    <td>Full-parameter</td>
    <td><a href="./examples/mcore/llama2/tune_llama2_7b_full_ptd.sh">dynamic</a></td>
    <td>15.87 samples/s</td>
    <td>-</td>
  </tr>
  <tr>
    <td>Full-parameter</td>
    <td><a href="./examples/mcore/llama2/tune_llama2_7b_full_pack_16k.sh">16K</a></td>
    <td>1.14 samples/s</td>
    <td>37.4%</td>
  </tr>
  <tr>
    <td>Full-parameter</td>
    <td><a href="./examples/mcore/llama2/tune_llama2_7b_full_pack_32k.sh">32K</a></td>
    <td>0.51 samples/s</td>
    <td>48.4%</td>
  </tr>
  <tr>
    <td rowspan="1">Llama-2-13B</td>
    <td rowspan="1">Atlas 900 A2 PODc</td>
    <td rowspan="1">1x8</td>
    <td>Full-parameter</td>
    <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/blob/2.0.0/examples/legacy/llama2/tune_llama2_13b_full_ptd.sh">dynamic</a></td>
    <td>50.4 samples/s</td>
    <td>-</td>
  </tr>
  <tr>
    <td>Llama-2-70B</td>
    <td>Atlas 900 A2 PODc</td>
    <td>1x8</td>
    <td>LoRA</td>
    <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/blob/2.0.0/examples/legacy/llama2/tune_llama2_70b_lora_ptd.sh">dynamic</a></td>
    <td>15.2 samples/s</td>
    <td>-</td>
  </tr>
</table>

### Fine-Tuning Solution

<table><thead>
  <tr>
    <th>Solution</th>
    <th>MCore</th>
    <th><a href="docs/en/pytorch/training/finetune/mcore/lora_finetune.md">LoRA</a></th>
    <th><a href="docs/en/pytorch/training/finetune/mcore/qlora_finetune.md">QLoRA</a></th>
    <th>Released</th>
    <th>Contributor</th>
  </tr></thead>
<tbody>
  <tr>
    <td><a href="docs/en/pytorch/training/finetune/mcore/instruction_finetune.md">Single-sample fine-tuning</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>[Ascend]</td>
  </tr>
  <tr>
    <td><a href="docs/en/pytorch/training/finetune/mcore/multi_sample_pack_finetune.md">Multi-sample pack fine-tuning</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>❌</td>
    <td>❌</td>
    <td>[NAIE]</td>
  </tr>
    <tr>
    <td><a href="docs/en/pytorch/training/finetune/mcore/multi_turn_conversation.md">Multi-turn conversation fine-tuning</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>❌</td>
    <td>❌</td>
    <td>[Ascend]</td>
  </tr>
</tbody></table>

### Acceleration Features

<table><thead>
  <tr>
    <th>Scenario</th>
    <th>Feature</th>
    <th>MCore</th>
    <th>Released</th>
    <th>Contributor</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="1"><a href="docs/en/pytorch/training/finetune/mcore/lora_finetune.md">LoRA fine-tuning</a></td>
    <td><a href="docs/en/pytorch/features/mcore/cc_lora.md">CCLoRA</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>[Ascend]</td>
  </tr>
  <tr>
      <td rowspan="1"><a href="docs/en/pytorch/training/finetune/mcore/qlora_finetune.md">QLoRA fine-tuning</a></td>
    <td><a href="docs/en/pytorch/features/mcore/cc_lora.md">CCLoRA</a></td>
    <td>❌</td>
    <td>❌</td>
    <td>[NAIE]</td>
  </tr>
  <tr>
    <td>Long-sequence fine-tuning</td>
    <td><a href="docs/en/pytorch/features/mcore/fine-tuning-with-context-parallel.md">Long-sequence CP</a></td>
    <td>✅</td>
    <td>❌</td>
    <td>[Ascend]</td>
  </tr>
</tbody></table>

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
      <td>[NAIE]</td>
    </tr>
    <tr>
      <td><a href="docs/en/pytorch/training/inference/chat.md">Chat conversation</a></td>
      <td>✅</td>
      <td>✅</td>
      <td>[NAIE]</td>
    </tr>
    <tr>
      <td><a href="docs/en/pytorch/features/mcore/yarn.md">YARN context extension</a></td>
      <td>✅</td>
      <td>❌</td>
      <td>[Ascend]</td>
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
      <td>[NAIE]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/datasets/ceval/ceval-exam/tree/main">CEval</a></td>
      <td>✅</td>
      <td>❌</td>
      <td>[NAIE]</td>
    </tr>
    <tr>
      <td><a href="https://github.com/google-research-datasets/boolean-questions">BoolQ</a></td>
      <td>✅</td>
      <td>❌</td>
      <td>[NAIE]</td>
    </tr>
    <tr>
      <td><a href="https://github.com/suzgunmirac/BIG-Bench-Hard/tree/main/bbh">BBH</a></td>
      <td>✅</td>
      <td>❌</td>
      <td>[NAIE]</td>
    </tr>
    <tr>
      <td><a href="https://github.com/ruixiangcui/AGIEval/tree/main">AGIEval</a></td>
      <td>✅</td>
      <td>❌</td>
      <td>[NAIE]</td>
    </tr>
    <tr>
      <td><a href="https://github.com/openai/human-eval/tree/master/data">HumanEval</a></td>
      <td>✅</td>
      <td>❌</td>
      <td>[NAIE]</td>
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
      <th>Source Format</th>
      <th>Target Format</th>
      <th>Sharding Features</th>
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
      <td rowspan="3">[Ascend]</td>
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
      <th>Task Scenario</th>
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
      <td rowspan="3">[Ascend]</td>
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
      <td>[Ascend]</td>
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
      <td rowspan="1">High availability</td>
      <td><a href="docs/en/pytorch/tools/deterministic_computation.md">Enable deterministic computation on Ascend chips</a></td>
      <td>✅</td>
      <td>❌</td>
      <td rowspan="1">[Ascend]</td>
    </tr>
  </tbody>
</table>

# Version Maintenance Policy

---

MindSpeed LLM versions go through the following five maintenance stages:

| Status | Time | Description |
| ------------------- | --------- | ------------------------------------------------------------ |
| Planning | 1 to 3 months | Planned features. |
| Development | 3 months | Features under development. |
| Maintenance | 6 to 12 months | Merge all resolved issues and release versions. Different MindSpeed LLM versions follow different maintenance strategies. The maintenance period is six months for regular releases and twelve months for long-term support releases. |
| No maintenance | 0 to 3 months | Merge all resolved issues. No dedicated maintenance staff. No version release. |
| End of life (EOL) | N/A | The branch no longer accepts any changes. |

MindSpeed LLM released version maintenance policy:

| MindSpeed LLM Version | Corresponding Tag | Maintenance Policy | Current Status | Release Date | Next Status | EOL Date |
|---------------------|------------| ------------ | ------------ |------------|-----------------| ----------- |
| 26.0.0              | v26.0.0    | Regular release     | Maintenance         | Mar 30, 2026  | Expected to enter no-maintenance status starting Sep 30, 2026 |             |
| 2.3.0               | v2.3.0     | Regular release     | Maintenance         | Dec 30, 2025 | Expected to enter no-maintenance status starting Jun 30, 2026 |             |
| 2.2.0               | v2.2.0     | Regular release     | Maintenance         | Sep 30, 2025  | Expected to enter no-maintenance status starting Mar 30, 2026 |             |
| 2.1.0               | v2.1.0     | Regular release     | EOL         | Jun 30, 2025  | End of life          |     Dec 30, 2025        |
| 2.0.0               | v2.0.0     | Regular release     | EOL          | Mar 30, 2025  | End of life          | Sep 30, 2025    |
| 1.0.0               | v1.0.0     | Regular release     | EOL          | Dec 30, 2024 | End of life          | Jun 30, 2025    |
| 1.0.RC3             | v1.0.RC3.0 | Regular release     | EOL          | Sep 30, 2024 | End of life          | Mar 30, 2025    |
| 1.0.RC2             | v1.0.RC2.0 | Regular release     | EOL          | Jun 30, 2024 | End of life          | Dec 30, 2024   |
| 1.0.RC1             | v1.0.RC1.0 | Regular release     | EOL          | Mar 30, 2024 | End of life          | Sep 30, 2024    |
| bk_origin_23        | \          | Demo        | EOL          | 2023       | End of life          | Jun 30, 2024     |

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
- The documents in the `docs` directory of the MindSpeed LLM tool are subject to the CC-BY 4.0 license. See [LICENSE](./docs/LICENSE) for details.

# Contribution Statement

**1. Reporting Issues**

- If you find any issues, first check the repository [issues list](https://gitcode.com/Ascend/MindSpeed-LLM/issues) and try to find similar issues or solutions.

- If the existing [issues list](https://gitcode.com/Ascend/MindSpeed-LLM/issues) does not contain your issue, you can [submit a new issue](https://gitcode.com/Ascend/MindSpeed-LLM/issues/create/choose) and provide a clear problem description, reproduction steps, and environment information where possible.

**2. Performance Optimization and New Features**

- For performance optimization proposals, use the `Performance` label when submitting the issue and describe the performance optimization feature and usage scenario.

- For new feature suggestions or discussions, use the `Feature` label when submitting the issue and describe the background, expectations, and proposal.

**3. Code Contribution Process**

To submit code changes, follow these brief steps:

- Develop and commit on your personal branch, then open a Pull Request (PR) to this repository.

- Register for a PR review slot at our [SIG regular meetings](https://gitcode.com/Ascend/MindSpeed-LLM/issues/1108) by filling in the specified format, and attend the corresponding review meeting on time.

- Modify the PR according to review comments and update it.

- After the PR passes review, enter `compile` in the comment area to trigger the gated CI pipeline.

- After the PR CI passes and the PR obtains enough labels, the repository Committer will perform the final review and merge it into the development branch.

Thank you for your participation and contribution. We look forward to advancing the project together with you.

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
