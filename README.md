  <p align="center"> <img src="sources/images/readme/logo.png" height="110px" width="500px"> </p>

<p align="center">
    <a href="https://gitcode.com/ascend/MindSpeed-LLM/blob/master/LICENSE">
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


# 简介

---

MindSpeed LLM：基于昇腾生态的大语言模型分布式训练框架，旨在为华为 [昇腾芯片](https://www.hiascend.com/) 生态合作伙伴提供端到端的大语言模型训练方案，包含分布式预训练、分布式指令微调以及对应的开发工具链，如：数据预处理、权重转换、在线推理、基线评估等。

**<small>注意 : 原仓名ModelLink更改为MindSpeed LLM，原包名modellink更改为mindspeed_llm </small>**


# 未来规划

---

未来规划会刷新在[MindSpeed LLM RoadMap](https://gitcode.com/Ascend/MindSpeed-LLM/issues/982)中，欢迎访问LLM最新规划动态。


# 加入我们

---

为了交流开发经验、分享使用心得、及时获取项目更新，我们创建了MindSpeed LLM社区交流群。无论你是正在使用这个项目，还是有奇思妙想，都欢迎加入。

加入方式：
1. 直接扫码加入微信交流群（二维码7天有效，定期更新）
2. 添加昇腾开源小助手，获取群链接，进入MindSpeed LLM社区交流群

<div style="display: flex; justify-content: flex-start; gap: 30px; align-items: flex-start; padding-left: 60px;">
  <div style="text-align: center;">
    <div>MindSpeed LLM社区交流群</div>
    <img src="sources/images/wechat/llm_group.jpg" width="150" alt="MindSpeed LLM 微信群">
  </div>
  <div style="text-align: center;">
    <div>昇腾开源小助手</div>
    <img src="sources/images/wechat/ascend_assistant.jpg" width="150" alt="昇腾小助手 微信">
  </div>
</div>

# 最新消息

---
- [Dec. 25, 2025]: 🚀 权重转换、数据处理和训练功能合一【Prototype】
- [Dec. 14, 2025]: 🚀 **magistral** 模型支持 【Prototype】
- [Dec. 03, 2025]: 🚀 MindSpeed LLM 模型下架**[夕阳计划](https://gitcode.com/Ascend/MindSpeed-LLM/issues/943)** 启动，感谢每一份曾经的贡献。
- [Oct. 28, 2025]: 🌴 MindSpeed LLM发布v2.2.0分支，支持core_v0.12.1版本
- [Oct. 16, 2025]: 🚀 Qwen3-30B支持DPO训练
- [Oct. 14, 2025]: 🚀 **DeepSeek-V3**预训练已支持基于 **[MindSpore AI框架](./docs/mindspore/readme.md)** 运行
- [Oct. 22, 2025]: 🚀 **Ling mini v2** 模型支持
- [Sep. 16, 2025]: 🚀 **Qwen3-Next** 模型支持 【Prototype】
- [Aug. 23, 2025]: 🚀 大参数模型权重转换v2优化版本上线
- [Jul. 28, 2025]: 🚀 **glm4.5-moe** 系列模型同步首发支持
- [Jul. 25, 2025]: 🌴 MindSpeed LLM发布v2.1.0分支，支持core_r0.8.0版本
- [Jul. 10, 2025]: 🚀 **[DeepSeek-R1](https://gitcode.com/Ascend/MindSpeed-RL/blob/master/docs/solutions/r1_zero_deepseek_671b.md)**  系列功能逐步上线
- [May. 19, 2025]: 🚀 **Qwen3** 系列模型同步首发支持
- [Mar. 27, 2025]: 🚀 **[DeepSeek-R1-ZERO Qwen-7B](https://gitcode.com/ascend/MindSpeed-RL/blob/master/docs/solutions/r1_zero_qwen25_7b.md)**  **[DeepSeek-R1-ZERO Qwen-32B](https://gitcode.com/ascend/MindSpeed-RL/blob/master/docs/solutions/r1_zero_qwen25_32b.md)**

- [Mar. 26, 2025]: 🚀  **[DeepSeek-V3-671B模型全家桶](./examples/mcore/deepseek3/)** 上线


# 版本说明

---

MindSpeed LLM支持Atlas 800T A2等昇腾训练硬件形态，软件版本配套表如下：

| MindSpeed LLM版本 | MindSpeed版本      | Megatron版本 | PyTorch版本  | torch_npu版本 | CANN版本 | Python版本            |
| ---------------- | ------------------ | ------------ | -----------  | ------------- | -------- | --------------------- |
| master（主线）   | master             | core_v0.12.1  | 2.7.1       | 在研版本       | 在研版本  | Python3.10            |
| 2.2.0（商用）    | 2.2.0_core_r0.12.1 | core_v0.12.1  | 2.7.1       | 7.2.0         | 8.3.RC1  | Python3.10            |
| 2.1.0（商用）    | 2.1.0_core_r0.8.0  | core_r0.8.0   | 2.1.0, 2.6.0 | 7.1.0         | 8.2.RC1  | Python3.8, Python3.10 |


# 安装

---



MindSpeed LLM具体的安装步骤请参考[安装指导](./docs/pytorch/install_guide.md)。

# 快速上手

---



MindSpeed LLM将以Qwen2.5-7B模型为例，指导开发者完成Qwen2.5-7B大语言模型的预训练任务，具体的操作请参考[快速入门](./docs/quick_start.md)。

# 预置模型

---



MindSpeed LLM目前已内置支持百余个业界常用LLM大模型的预训练与微调，预置模型清单详见下表。

<table><thead>
  <tr>
    <th>模型类别</th>
    <th>模型列表</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="1">稠密模型</td>
    <td><a href="docs/pytorch/models/dense_model.md">Dense</a></td>
  </tr>
  <tr>
    <td rowspan="1">MOE模型</td>
    <td><a href="docs/pytorch/models/moe_model.md">MOE</a></td>
  </tr>
  <tr>
    <td rowspan="1">SSM模型</td>
    <td><a href="./docs/pytorch/models/ssm_model.md">SSM</a></td>
  </tr>  
</tbody></table>

# 训练方案与特性

---



MindSpeed LLM包含分布式预训练、分布式微调等训练方案。

## 分布式预训练

基于MindSpeed LLM的实测预训练性能如下：

<table>
  <thead>
    <tr>
      <th>模型系列</th>
      <th>实验模型</th>
      <th>硬件信息</th>
      <th>集群规模</th>
      <th>MFU</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">LLAMA2</td>
      <td><a href="./examples/mcore/llama2/pretrain_llama2_7b_pack_ptd.sh">LLAMA2-7B</a></td>
      <td>Atlas 900 A2 PODc</td>
      <td>1x8</td>
      <td>69.0%</td>
    </tr>
    <tr>
      <td><a href="./examples/mcore/llama2/pretrain_llama2_13b_pack_ptd.sh">LLAMA2-13B</a></td>
      <td>Atlas 900 A2 PODc</td>
      <td>1x8</td>
      <td>64.7%</td>
    </tr>
    <tr>
      <td><a href="./examples/mcore/llama2/pretrain_llama2_70b_pack_ptd.sh">LLAMA2-70B</a></td>
      <td>Atlas 900 A2 PODc</td>
      <td>4x8</td>
      <td>44.1%</td>
    </tr>
    <tr>
      <td>Mixtral</td>
      <td><a href="./examples/mcore/mixtral/pretrain_mixtral_8x7b_ptd.sh">Mixtral-8x7B</a></td>
      <td>Atlas 900 A2 PODc</td>
      <td>8x8</td>
      <td>31.7%</td>
    </tr>
  </tbody>
</table>

### 预训练方案

<table>
  <thead>
    <tr>
      <th>方案类别</th>
      <th>Mcore</th>
      <th>Released</th>
      <th>贡献方</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="docs/pytorch/solutions/pretrain/pretrain.md">多样本集预训练</a></td>
      <td>✅</td>
      <td>✅</td>
      <td rowspan="2">【Ascend】</td>
    </tr>
    <tr>
      <td><a href="docs/pytorch/solutions/pretrain/pretrain_eod.md">多样本pack模式预训练</a></td>
      <td>✅</td>
      <td>❌</td>
</tr>
  </tbody>
</table>

### 加速特性

<table><thead>
  <tr>
    <th>场景</th>
    <th>特性名称</th>
    <th>Mcore</th>
    <th>Released</th>
    <th>贡献方</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="5">SPTD并行</td>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/tensor-parallel.md">张量并行</a></td>
    <td>✅</td>
    <td>✅</td>
    <td rowspan="29">【Ascend】</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/pipeline-parallel.md">流水线并行</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="docs/pytorch/features/virtual_pipeline_parallel.md">虚拟流水并行</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/sequence-parallel.md">序列并行</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/noop-layers.md">noop layers</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td rowspan="3">长序列并行</td>
    <td><a href="docs/pytorch/features/ring-attention-context-parallel.md">Ascend Ring Attention 长序列并行</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/ulysses-context-parallel.md">Ulysses 长序列并行</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/hybrid-context-parallel.md">混合长序列并行</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td rowspan="2">MOE</td>
    <td><a href="https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md">MOE 专家并行</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/megatron_moe/megatron-moe-allgather-dispatcher.md">MOE 重排通信优化</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td rowspan="6">显存优化</td>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/reuse-fp32-param.md">参数副本复用</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
    <tr>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/distributed-optimizer.md">分布式优化器</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/swap_attention.md">Swap Attention</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="docs/pytorch/features/recompute_relative.md">重计算</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/norm-recompute.md">Norm重计算</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="docs/pytorch/features/o2.md">O2 BF16 Optimizer</a></td>
    <td>✅</td>
    <td>❌</td>
  </tr>
  <tr>
    <td rowspan="7">融合算子</td>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/flash-attention.md">Flash attention</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="docs/pytorch/features/variable_length_flash_attention.md">Flash attention variable length</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/rms_norm.md">Fused rmsnorm</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/swiglu.md">Fused swiglu</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/rotary-embedding.md">Fused rotary position embedding</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/megatron_moe/megatron-moe-gmm.md">GMM</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/npu_matmul_add.md">Matmul Add</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td rowspan="6">通信优化</td>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/async-ddp-param-gather.md">梯度reduce通算掩盖</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/recompute_independent_pipelining.md">Recompute in advance</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/async-ddp-param-gather.md">权重all-gather通算掩盖</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="docs/pytorch/features/mc2.md">MC2</a></td>
    <td>✅</td>
    <td>❌</td>
  </tr>
  <tr>
    <td><a href="docs/pytorch/features/communication-over-computation.md">CoC</a></td>
    <td>✅</td>
    <td>❌</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/hccl-replace-gloo.md">Ascend Gloo 存档落盘优化</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
</tbody></table>

## 分布式微调

基于MindSpeed LLM的实测指令微调性能如下：

<table>
  <tr>
    <th>模型</th>
    <th>硬件</th>
    <th>集群</th>
    <th>方案</th>
    <th>序列</th>
    <th>性能</th>
    <th>MFU</th>
  </tr>
  <tr>
    <td rowspan="3">Llama2-7B</td>
    <td rowspan="3">Atlas 900 A2 PODc</td>
    <td rowspan="3">1x8</td>
    <td>全参</td>
    <td><a href="./examples/mcore/llama2/tune_llama2_7b_full_ptd.sh">dynamic</a></td>
    <td>15.87 samples/s</td>
    <td>-</td>
  </tr>
  <tr>
    <td>全参</td>
    <td><a href="./examples/mcore/llama2/tune_llama2_7b_full_pack_16k.sh">16K</a></td>
    <td>1.14 samples/s</td>
    <td>37.4%</td>
  </tr>
  <tr>
    <td>全参</td>
    <td><a href="./examples/mcore/llama2/tune_llama2_7b_full_pack_32k.sh">32K</a></td>
    <td>0.51 samples/s</td>
    <td>48.4%</td>
  </tr>
  <tr>
    <td rowspan="1">Llama2-13B</td>
    <td rowspan="1">Atlas 900 A2 PODc</td>
    <td rowspan="1">1x8</td>
    <td>全参</td>
    <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/blob/2.0.0/examples/legacy/llama2/tune_llama2_13b_full_ptd.sh">dynamic</a></td>
    <td>50.4 samples/s</td>
    <td>-</td>
  </tr>
  <tr>
    <td>Llama2-70B</td>
    <td>Atlas 900 A2 PODc</td>
    <td>1x8</td>
    <td>LoRA</td>
    <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/blob/2.0.0/examples/legacy/llama2/tune_llama2_70b_lora_ptd.sh">dynamic</a></td>
    <td>15.2 samples/s</td>
    <td>-</td>
  </tr>
</table>

### 微调方案

<table><thead>
  <tr>
    <th>方案名称</th>
    <th>Mcore</th>
    <th><a href="docs/pytorch/solutions/finetune/lora_finetune.md">LoRA</a></th>
    <th><a href="docs/pytorch/solutions/finetune/qlora_finetune.md">QLoRA</a></th>
    <th>Released</th>
    <th>贡献方</th>
  </tr></thead>
<tbody>
  <tr>
    <td><a href="docs/pytorch/solutions/finetune/instruction_finetune.md">单样本微调</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>【Ascend】</td>
  </tr>
  <tr>
    <td><a href="docs/pytorch/solutions/finetune/multi_sample_pack_finetune.md">多样本pack微调</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>❌</td>
    <td>❌</td>
    <td>【NAIE】</td>
  </tr>
    <tr>
    <td><a href="docs/pytorch/solutions/finetune/multi-turn_conversation.md">多轮对话微调</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>❌</td>
    <td>❌</td>
    <td>【Ascend】</td>
  </tr>  
</tbody></table>

### 加速特性

<table><thead>
  <tr>
    <th>场景</th>
    <th>特性</th>
    <th>Mcore</th>
    <th>Released</th>
    <th>贡献方</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="2"><a href="docs/pytorch/solutions/finetune/lora_finetune.md">LoRA微调</a></td>
    <td><a href="docs/pytorch/features/cc_lora.md">CCLoRA</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>【Ascend】</td>
  </tr>
  <tr>
    <td><a href="docs/pytorch/features/fused_mlp.md">Fused_MLP</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>【Ascend】</td>
  </tr>
  <tr>
    <td rowspan="2"><a href="docs/pytorch/solutions/finetune/qlora_finetune.md">QLoRA微调</a></td>
    <td><a href="docs/pytorch/features/cc_lora.md">CCLoRA</a></td>
    <td>❌</td>
    <td>❌</td>
    <td>【NAIE】</td>
  </tr>
  <tr>
    <td><a href="docs/pytorch/features/fused_mlp.md">Fused_MLP</a></td>
    <td>❌</td>
    <td>❌</td>
    <td>【NAIE】</td>
  </tr>
  <tr>
    <td>长序列微调</td>
    <td><a href="docs/pytorch/features/fine-tuning-with-context-parallel.md">长序列CP</a></td>
    <td>✅</td>
    <td>❌</td>
    <td>【Ascend】</td>
  </tr>
</tbody></table>

# 在线推理

---



<table>
  <thead>
    <tr>
      <th>特性</th>
      <th>Mcore</th>
      <th>Released</th>
      <th>贡献方</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="docs/pytorch/solutions/inference/inference.md">流式推理 </a></td>
      <td>✅</td>
      <td>✅</td>
      <td>【NAIE】</td>
    </tr>
    <tr>
      <td><a href="docs/pytorch/solutions/inference/chat.md"> Chat对话</a></td>
      <td>✅</td>
      <td>✅</td>
      <td>【NAIE】</td>
    </tr>
    <tr>
      <td><a href="docs/pytorch/features/yarn.md"> yarn上下文扩展 </a></td>
      <td>✅</td>
      <td>❌</td>
      <td>【Ascend】</td>
    </tr>
  </tbody>
</table>

# 开源数据集评测

---



仓库模型基线见[开源数据集评测基线](docs/pytorch/models/models_evaluation.md)
<table>
  <thead>
    <tr>
      <th>场景</th>
      <th>数据集</th>
      <th>Mcore</th>
      <th>Released</th>
      <th>贡献方</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="8"><a href="docs/pytorch/solutions/evaluation/evaluation_guide.md">评测</a></td>
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


# 开发工具链

---



## 权重转换

MindSpeed LLM支持Huggingface、Megatron-core两种格式的权重互转，支持LoRA权重合并。权重转换特性参数和使用说明参考[权重转换](docs/pytorch/solutions/checkpoint/checkpoint_convert.md)。

<table>
  <thead>
    <tr>
      <th>源格式</th>
      <th>目标格式</th>
      <th>切分特性</th>
      <th>LoRA</th>
      <th>贡献方</th>
      <th>Released</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Huggingface</td>
      <td>Megatron-core</td>
      <td>tp、pp、dpp、vpp、cp、ep、loop layer</td>
      <td>❌</td>
      <td rowspan="3">【Ascend】</td>
      <td rowspan="3">❌</td>
    </tr>
    <tr>
      <td rowspan="2">Megatron-core</td>
      <td>Huggingface</td>
      <td></td>
      <td>✅</td>
    </tr>
    <tr>
      <td>Megatron-core</td>
      <td>tp、pp、dpp、vpp、cp、ep、loop layer</td>
      <td>✅</td>
    </tr>
  </tbody>
</table>

## 数据预处理

MindSpeed LLM支持预训练、指令微调等多种任务的数据预处理。

<table>
  <thead>
    <tr>
      <th>任务场景</th>
      <th>数据集</th>
      <th>Mcore</th>
      <th>Released</th>
      <th>贡献方</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>预训练</td>
      <td><a href="docs/pytorch/solutions/pretrain/pretrain_dataset.md">预训练数据处理</a></td>
      <td>✅</td>
      <td>✅</td>
      <td rowspan="3">【Ascend】</td>
    </tr>
    <tr>
      <td rowspan="2">微调</td>
      <td><a href="docs/pytorch/solutions/finetune/datasets/alpaca_dataset.md">Alpaca风格</a></td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td><a href="docs/pytorch/solutions/finetune/datasets/sharegpt_dataset.md">ShareGPT风格</a></td>
      <td>✅</td>
      <td>✅</td>
    </tr>
  </tbody>
</table>


## 性能采集

<table>
  <thead>
    <tr>
      <th>场景</th>
      <th>特性</th>
      <th>Mcore</th>
      <th>Released</th>
      <th>贡献方</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="1">性能采集</td>
      <td><a href="docs/pytorch/features/profiling.md">基于昇腾芯片采集 profiling 数据</a></td>
      <td>✅</td>
      <td>❌</td>
      <td>【Ascend】</td>
    </tr>
  </tbody>
</table>


## 高可用性

<table>
  <thead>
    <tr>
      <th>场景</th>
      <th>特性</th>
      <th>Mcore</th>
      <th>Released</th>
      <th>贡献方</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">高可用性</td>
      <td><a href="docs/pytorch/features/deterministic_computation.md">基于昇腾芯片开启确定性计算</a></td>
      <td>✅</td>
      <td>❌</td>
      <td rowspan="2">【Ascend】</td>
    </tr>
  </tbody>
</table>


# 版本维护策略

---



MindSpeed LLM版本有以下五个维护阶段：

| **状态**            | **时间**  | **说明**                                                     |
| ------------------- | --------- | ------------------------------------------------------------ |
| 计划                | 1—3 个月  | 计划特性                                                     |
| 开发                | 3 个月    | 开发特性                                                     |
| 维护                | 6-12 个月 | 合入所有已解决的问题并发布版本，针对不同的MindSpeed LLM版本采取不同的维护策略，常规版本和长期支持版本维护周期分别为6个月和12个月 |
| 无维护              | 0—3 个月  | 合入所有已解决的问题，无专职维护人员，无版本发布             |
| 生命周期终止（EOL） | N/A       | 分支不再接受任何修改                                         |


MindSpeed LLM已发布版本维护策略：

| **MindSpeed LLM版本** | **对应标签** | **维护策略** | **当前状态** | **发布时间** | **后续状态**              | **EOL日期** |
| --------------------- | ------------| ------------ | ------------ | ------------ | ---------------------- | ----------- |
| 2.3.0                 | v2.3.0       | 常规版本     | 维护         | 2025/12/30   | 预计2026/6/30起无维护   |             |
| 2.2.0                 | v2.2.0       | 常规版本     | 维护         | 2025/9/30    | 预计2026/3/30起无维护   |             |
| 2.1.0                 | v2.1.0       | 常规版本     | 维护         | 2025/6/30    | 预计2025/12/30起无维护  |             |
| 2.0.0                 | v2.0.0       | 常规版本     | EOL          | 2025/3/30    | 生命周期终止           | 2025/9/30    |
| 1.0.0                 | v1.0.0       | 常规版本     | EOL          | 2024/12/30   | 生命周期终止           | 2025/6/30    |
| 1.0.RC3               | v1.0.RC3.0   | 常规版本     | EOL          | 2024/09/30   | 生命周期终止           | 2025/3/30    |
| 1.0.RC2               | v1.0.RC2.0   | 常规版本     | EOL          | 2024/06/30   | 生命周期终止           | 2024/12/30   |
| 1.0.RC1               | v1.0.RC1.0   | 常规版本     | EOL          | 2024/03/30   | 生命周期终止           | 2024/9/30    |
| bk_origin_23          | \            | Demo        | EOL          | 2023         | 生命周期终止           | 2024/6/30     |

# 致谢

---



MindSpeed LLM由华为公司的下列部门以及昇腾生态合作伙伴联合贡献 ：

华为公司：

- 计算产品线：Ascend
- 公共开发部：NAIE
- 全球技术服务部：GTS
- 华为云计算：Cloud

生态合作伙伴：

- 移动云（China Mobile Cloud）：大云震泽智算平台
- 工商银行软件开发中心大数据人工智能实验室

感谢来自社区的每一个PR，欢迎贡献 MindSpeed LLM。

# 安全声明

---



[MindSpeed LLM安全声明](https://gitcode.com/Ascend/MindSpeed-LLM/wiki/%E5%AE%89%E5%85%A8%E7%9B%B8%E5%85%B3%2F%E5%AE%89%E5%85%A8%E5%A3%B0%E6%98%8E.md)

# 免责声明

---



## 致MindSpeed LLM使用者

1. MindSpeed LLM提供的模型仅供您用于非商业目的。
2. MindSpeed LLM功能依赖的Megatron等第三方开源软件，均由第三方社区提供和维护，因第三方开源软件导致的问题修复依赖相关社区的贡献和反馈。您应理解，MindSpeed LLM仓库不保证对第三方开源软件本身的问题进行修复，也不保证会测试、纠正所有第三方开源软件的漏洞和错误。
3. 对于各模型，MindSpeed LLM平台仅提示性地向您建议可用于训练的数据集，华为不提供任何数据集，如您使用这些数据集进行训练，请您特别注意应遵守对应数据集的License，如您因使用数据集而产生侵权纠纷，华为不承担任何责任。
4. 如您在使用MindSpeed LLM模型过程中，发现任何问题（包括但不限于功能问题、合规问题），请在Gitcode提交issue，我们将及时审视并解决。

## 致数据集所有者

如果您不希望您的数据集在MindSpeed LLM中的模型被提及，或希望更新MindSpeed LLM中的模型关于您的数据集的描述，请在Gitcode提交issue，我们将根据您的issue要求删除或更新您的数据集描述。衷心感谢您对MindSpeed LLM的理解和贡献。

## License声明

Ascend MindSpeed LLM提供的模型，如模型目录下存在License的，以该License为准。如模型目录下不存在License的，以Apache 2.0许可证许可，对应许可证文本可查阅Ascend MindSpeed LLM根目录。
