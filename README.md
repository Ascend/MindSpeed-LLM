  <p align="center"> <img src="sources/images/readme/logo.png" height="110px" width="500px"> </p>

<p align="center">
    <a href="https://gitee.com/ascend/MindSpeed-LLM/blob/master/LICENSE">
    <a href="https://gitee.com/ascend/MindSpeed-LLM/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue">
    </a>
    <a href="https://gitee.com/ascend/MindSpeed-LLM">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a>
        <img src="https://app.codacy.com/project/badge/Grade/1710faac5e634acaabfc26b0a778cdde">
    </a>
</p>

MindSpeed LLM是基于昇腾生态的大语言模型分布式训练框架，旨在为华为 [昇腾芯片](https://www.hiascend.com/) 生态合作伙伴提供端到端的大语言模型训练方案，包含分布式预训练、分布式指令微调以及对应的开发工具链，如：数据预处理、权重转换、在线推理、基线评估。

***<small>注 : 原仓名ModelLink更改为MindSpeed LLM，原包名modellink更改为mindspeed_llm </small>***

---

## NEWS !!! 📣📣📣
🚀🚀🚀**DeepSeek-V3**预训练已支持基于 **[MindSpore AI框架](./docs/mindspore/readme.md)** 运行！！！🚀🚀🚀

🚀🚀🚀**Qwen3** 系列模型同步首发支持！！！🚀🚀🚀

**[Qwen3系列模型](https://gitee.com/ascend/MindSpeed-LLM/tree/master/tests/0day/qwen3)** 😊

🚀🚀🚀**DeepSeek-R1** 系列功能逐步上线！！🚀🚀🚀

😊 **[DeepSeek-R1-ZERO Qwen-7B](https://gitee.com/ascend/MindSpeed-RL/blob/master/docs/solutions/r1_zero_qwen25_7b.md)**  **[DeepSeek-R1-ZERO Qwen-32B](https://gitee.com/ascend/MindSpeed-RL/blob/master/docs/solutions/r1_zero_qwen25_32b.md)**

🚀🚀🚀 **[DeepSeek-V3-671B模型全家桶](./examples/mcore/deepseek3/)** 已上线！！！🚀🚀🚀

😊**支持数据集处理、权重转换、预训练、全参微调、lora微调、qlora微调** 

🚀🚀🚀**DeepSeek-R1-Distill** 系列模型已上线！！🚀🚀🚀

😊 **[DeepSeek-R1-Distill-Qwen](./examples/mcore/deepseek_r1_distill_qwen/)**  **[DeepSeek-R1-Distill-LLaMA](./examples/mcore/deepseek_r1_distill_llama/)**

> 注：当前qwen3系列模型功能完善、验证中，非商用版本。


## 版本配套表

MindSpeed LLM的依赖配套如下表，安装步骤参考[安装指导](./docs/features/install_guide.md)。

<table>
  <tr>
    <th>依赖软件</th>
    <th>版本</th>
  </tr>
  <tr>
    <td>昇腾NPU驱动</td>
    <td rowspan="2">在研版本</td>
  <tr>
    <td>昇腾NPU固件</td>
  </tr>
  <tr>
    <td>Toolkit（开发套件）</td>
      <td rowspan="3">在研版本</td>
  </tr>
  <tr>
    <td>Kernel（算子包）</td>
  </tr>
  <tr>
    <td>NNAL（Ascend Transformer Boost加速库）</td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td>Python</td>
    <td>>=3.10</td>
  </tr>
  <tr>
    <td>PyTorch</td>
    <td>2.6</td>
  </tr>
  <tr>
    <td>torch_npu插件</td>
    <td rowspan="2">在研版本</td>
  </tr>
  <tr>
    <td>apex</td>
  </tr>
</table>


注：由于master分支使用在研版本的驱动以及CANN包，因此master上的一些新特性老版本配套可能有不支持情况，要使用稳定版本，请切换到商发分支并下载对应依赖版本进行安装。

## 预置模型

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

## 训练方案与特性

MindSpeed LLM包含分布式预训练、分布式微调等训练方案。

### 分布式预训练

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

#### 预训练方案

<table>
  <thead>
    <tr>
      <th>方案类别</th>
      <th>Legacy</th>
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
      <td>✅</td>
      <td rowspan="2">【Ascend】</td>
    </tr>
    <tr>
      <td><a href="docs/pytorch/solutions/pretrain/pretrain_eod.md">多样本pack模式预训练</a></td>
      <td>✅</td>
      <td>✅</td>
      <td>❌</td>
</tr>
  </tbody>
</table>


#### 加速特性

<table><thead>
  <tr>
    <th>场景</th>
    <th>特性名称</th>
    <th>Mcore</th>
    <th>Legacy</th>
    <th>Released</th>
    <th>贡献方</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="5">SPTD并行</td>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/tensor-parallel.md">张量并行</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td rowspan="29">【Ascend】</td>
  </tr>
  <tr>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/pipeline-parallel.md">流水线并行</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="docs/pytorch/features/virtual_pipeline_parallel.md">虚拟流水并行</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/sequence-parallel.md">序列并行</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/noop-layers.md">noop layers</a></td>
    <td>✅</td>
    <td>❌</td>
    <td>✅</td>
  </tr>
  <tr>
    <td rowspan="3">长序列并行</td>
    <td><a href="docs/pytorch/features/ring-attention-context-parallel.md">Ascend Ring Attention 长序列并行</a></td>
    <td>✅</td>
    <td>❌</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/ulysses-context-parallel.md">Ulysses 长序列并行</a></td>
    <td>✅</td>
    <td>❌</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/hybrid-context-parallel.md">混合长序列并行</a></td>
    <td>✅</td>
    <td>❌</td>
    <td>✅</td>
  </tr>
  <tr>
    <td rowspan="2">MOE</td>
    <td><a href="https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md">MOE 专家并行</a></td>
    <td>✅</td>
    <td>❌</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/megatron_moe/megatron-moe-allgather-dispatcher.md">MOE 重排通信优化</a></td>
    <td>✅</td>
    <td>❌</td>
    <td>✅</td>
  </tr>
  <tr>
    <td rowspan="6">显存优化</td>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/reuse-fp32-param.md">参数副本复用</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
    <tr>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/distributed-optimizer.md">分布式优化器</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/swap_attention.md">Swap Attention</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="docs/pytorch/features/recompute_relative.md">重计算</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/norm-recompute.md">Norm重计算</a></td>
    <td>✅</td>
    <td>❌</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="docs/pytorch/features/o2.md">O2 BF16 Optimizer</a></td>
    <td>✅</td>
    <td>❌</td>
    <td>❌</td>
  </tr>
  <tr>
    <td rowspan="7">融合算子</td>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/flash-attention.md">Flash attention</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="docs/pytorch/features/variable_length_flash_attention.md">Flash attention variable length</a></td>
    <td>✅</td>
    <td>❌</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/rms_norm.md">Fused rmsnorm</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/swiglu.md">Fused swiglu</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/rotary-embedding.md">Fused rotary position embedding</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/megatron_moe/megatron-moe-gmm.md">GMM</a></td>
    <td>✅</td>
    <td>❌</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/npu_matmul_add.md">Matmul Add</a></td>
    <td>✅</td>
    <td>❌</td>
    <td>✅</td>
  </tr>
  <tr>
    <td rowspan="6">通信优化</td>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/async-ddp-param-gather.md">梯度reduce通算掩盖</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/recompute_independent_pipelining.md">Recompute in advance</a></td>
    <td>✅</td>
    <td>❌</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/async-ddp-param-gather.md">权重all-gather通算掩盖</a></td>
    <td>✅</td>
    <td>❌</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="docs/pytorch/features/mc2.md">MC2</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>❌</td>
  </tr>
  <tr>
    <td><a href="docs/pytorch/features/communication-over-computation.md">CoC</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>❌</td>
  </tr>
  <tr>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/hccl-replace-gloo.md">Ascend Gloo 存档落盘优化</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>❌</td>
  </tr>
</tbody></table>

### 分布式微调

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
    <td rowspan="3">llama2-7B</td>
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
    <td rowspan="1">llama2-13B</td>
    <td rowspan="1">Atlas 900 A2 PODc</td>
    <td rowspan="1">1x8</td>
    <td>全参</td>
    <td><a href="./examples/legacy/llama2/tune_llama2_13b_full_ptd.sh">dynamic</a></td>
    <td>50.4 samples/s</td>
    <td>-</td>
  </tr>
  <tr>
    <td>llama2-70B</td>
    <td>Atlas 900 A2 PODc</td>
    <td>1x8</td>
    <td>LoRA</td>
    <td><a href="./examples/legacy/llama2/tune_llama2_70b_lora_ptd.sh">dynamic</a></td>
    <td>15.2 samples/s</td>
    <td>-</td>
  </tr>
</table>

#### 微调方案

<table><thead>
  <tr>
    <th>方案名称</th>
    <th>Mcore</th>
    <th>Legacy</th>
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
    <td>✅</td>
    <td>【Ascend】</td>
  </tr>
  <tr>
    <td><a href="docs/pytorch/solutions/finetune/multi_sample_pack_finetune.md">多样本pack微调</a></td>
    <td>✅</td>
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
    <td>✅</td>
    <td>❌</td>
    <td>❌</td>
    <td>【Ascend】</td>
  </tr>  
</tbody></table>


#### 加速特性

<table><thead>
  <tr>
    <th>场景</th>
    <th>特性</th>
    <th>Mcore</th>
    <th>Legacy</th>
    <th>Released</th>
    <th>贡献方</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="2"><a href="docs/pytorch/solutions/finetune/lora_finetune.md">LoRA微调</a></td>
    <td><a href="docs/pytorch/features/cc_lora.md">CCLoRA</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>【Ascend】</td>
  </tr>
  <tr>
    <td><a href="docs/pytorch/features/fused_mlp.md">Fused_MLP</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>【Ascend】</td>
  </tr>
  <tr>
    <td rowspan="2"><a href="docs/pytorch/solutions/finetune/qlora_finetune.md">QLoRA微调</a></td>
    <td><a href="docs/pytorch/features/cc_lora.md">CCLoRA</a></td>
    <td>❌</td>
    <td>❌</td>
    <td>❌</td>
    <td>【NAIE】</td>
  </tr>
  <tr>
    <td><a href="docs/pytorch/features/fused_mlp.md">Fused_MLP</a></td>
    <td>❌</td>
    <td>❌</td>
    <td>❌</td>
    <td>【NAIE】</td>
  </tr>
  <tr>
    <td>长序列微调</td>
    <td><a href="docs/pytorch/features/fine-tuning-with-context-parallel.md">长序列CP</a></td>
    <td>✅</td>
    <td>❌</td>
    <td>❌</td>
    <td>【Ascend】</td>
  </tr>
</tbody></table>

## 在线推理

<table>
  <thead>
    <tr>
      <th>特性</th>
      <th>Mcore</th>
      <th>Legacy</th>
      <th>Released</th>
      <th>贡献方</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="docs/pytorch/solutions/inference/inference.md">流式推理 </a></td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td>【NAIE】</td>
    </tr>
    <tr>
      <td><a href="docs/pytorch/solutions/inference/chat.md"> Chat对话</a></td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td>【NAIE】</td>
    </tr>
    <tr>
      <td><a href="docs/pytorch/features/yarn.md"> yarn上下文扩展 </a></td>
      <td>✅</td>
      <td>❌</td>
      <td>❌</td>
      <td>【Ascend】</td>
    </tr>
  </tbody>
</table>

## 开源数据集评测
仓库模型基线见[开源数据集评测基线](docs/pytorch/models/models_evaluation.md)
<table>
  <thead>
    <tr>
      <th>场景</th>
      <th>数据集</th>
      <th>Mcore</th>
      <th>Legacy</th>
      <th>Released</th>
      <th>贡献方</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="8"><a href="docs/pytorch/solutions/evaluation/evaluation_guide.md">评测</a></td>
      <td><a href="https://people.eecs.berkeley.edu/~hendrycks/data.tar">MMLU</a></td>
      <td>✅</td>
      <td>✅</td>
      <td>❌</td>
      <td>【NAIE】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/datasets/ceval/ceval-exam/tree/main">CEval</a></td>
      <td>✅</td>
      <td>✅</td>
      <td>❌</td>
      <td>【NAIE】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/google-research-datasets/boolean-questions">BoolQ</a></td>
      <td>✅</td>
      <td>✅</td>
      <td>❌</td>
      <td>【NAIE】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/suzgunmirac/BIG-Bench-Hard/tree/main/bbh">BBH</a></td>
      <td>✅</td>
      <td>✅</td>
      <td>❌</td>
      <td>【NAIE】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/ruixiangcui/AGIEval/tree/main">AGIEval</a></td>
      <td>✅</td>
      <td>✅</td>
      <td>❌</td>
      <td>【NAIE】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/openai/human-eval/tree/master/data">HumanEval</a></td>
      <td>✅</td>
      <td>✅</td>
      <td>❌</td>
      <td>【NAIE】</td>
    </tr>
  </tbody>
</table>


## 开发工具链

### 权重转换

MindSpeed LLM支持huggingface、megatron-core、megatron-legacy三种格式的权重互转，支持Lora权重合并。权重转换特性参数和使用说明参考[权重转换](docs/pytorch/solutions/checkpoint_convert.md)。

<table>
  <thead>
    <tr>
      <th>源格式</th>
      <th>目标格式</th>
      <th>切分特性</th>
      <th>lora</th>
      <th>贡献方</th>
      <th>Released</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">huggingface</td>
      <td>megatron-core</td>
      <td rowspan="2">tp、pp、dpp、vpp、cp、ep、loop layer</td>
      <td>❌</td>
      <td rowspan="8">【Ascend】</td>
      <td rowspan="8">❌</td>
    </tr>
    <tr>
      <td>megatron-legacy</td>
      <td>❌</td>
    </tr>
    <tr>
      <td rowspan="3">megatron-core</td>
      <td>huggingface</td>
      <td></td>
      <td>✅</td>
    </tr>
    <tr>
      <td>megatron-legacy</td>
      <td rowspan="2">tp、pp、dpp、vpp、cp、ep、loop layer</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>megatron-core</td>
      <td>✅</td>
    </tr>
    <tr>
      <td rowspan="3">megatron-legacy</td>
      <td>huggingface</td>
      <td></td>
      <td>✅</td>
    </tr>
    <tr>
      <td>megatron-core</td>
      <td rowspan="2">tp、pp、dpp、vpp、cp、ep、loop layer</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>megatron-legacy</td>
      <td>✅</td>
    </tr>
  </tbody>
</table>


### 数据预处理

MindSpeed LLM支持预训练、指令微调等多种任务的数据预处理。

<table>
  <thead>
    <tr>
      <th>任务场景</th>
      <th>数据集</th>
      <th>Mcore</th>
      <th>Legacy</th>
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
      <td>✅</td>
      <td rowspan="3">【Ascend】</td>
    </tr>
    <tr>
      <td rowspan="2">微调</td>
      <td><a href="docs/pytorch/solutions/finetune/datasets/alpaca_dataset.md">Alpaca风格</a></td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td><a href="docs/pytorch/solutions/finetune/datasets/sharegpt_dataset.md">ShareGPT风格</a></td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
  </tbody>
</table>


### 性能采集

<table>
  <thead>
    <tr>
      <th>场景</th>
      <th>特性</th>
      <th>Mcore</th>
      <th>Legacy</th>
      <th>Released</th>
      <th>贡献方</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="1">性能采集</td>
      <td><a href="docs/pytorch/features/profiling.md">基于昇腾芯片采集 profiling 数据</a></td>
      <td>✅</td>
      <td>✅</td>
      <td>❌</td>
      <td>【Ascend】</td>
    </tr>
  </tbody>
</table>


### 高可用性

<table>
  <thead>
    <tr>
      <th>场景</th>
      <th>特性</th>
      <th>Mcore</th>
      <th>Legacy</th>
      <th>Released</th>
      <th>贡献方</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">高可用性</td>
      <td><a href="docs/pytorch/features/deterministic_computation.md">基于昇腾芯片开启确定性计算</a></td>
      <td>✅</td>
      <td>✅</td>
      <td>❌</td>
      <td rowspan="2">【Ascend】</td>
    </tr>
  </tbody>
</table>


## 版本维护策略

MindSpeed LLM版本有以下五个维护阶段：

| **状态**            | **时间**  | **说明**                                                     |
| ------------------- | --------- | ------------------------------------------------------------ |
| 计划                | 1—3 个月  | 计划特性                                                     |
| 开发                | 3 个月    | 开发特性                                                     |
| 维护                | 6-12 个月 | 合入所有已解决的问题并发布版本，针对不同的MindSpeed LLM版本采取不同的维护策略，常规版本和长期支持版本维护周期分别为6个月和12个月 |
| 无维护              | 0—3 个月  | 合入所有已解决的问题，无专职维护人员，无版本发布             |
| 生命周期终止（EOL） | N/A       | 分支不再接受任何修改                                         |


MindSpeed LLM已发布版本维护策略：

| **MindSpeed LLM版本** | **对应标签** | **维护策略** | **当前状态** | **发布时间** | **后续状态**           | **EOL日期** |
| --------------------- | ------------ | ------------ | ------------ | ------------ | ---------------------- | ----------- |
| 2.0.0                 | \            | 常规版本     | 维护         | 2025/3/30   | 预计2025/09/30起无维护 |             |
| 1.0.0                 | v1.0.0       | 常规版本     | 维护         | 2024/12/30   | 预计2025/06/30起无维护 |             |
| 1.0.RC3               | v1.0.RC3.0   | 常规版本     | EOL          | 2024/09/30   | 生命周期终止           | 2025/3/30  |
| 1.0.RC2               | v1.0.RC2.0   | 常规版本     | EOL          | 2024/06/30   | 生命周期终止           | 2024/12/30  |
| 1.0.RC1               | v1.0.RC1.0   | 常规版本     | EOL          | 2024/03/30   | 生命周期终止           | 2024/9/30   |
| bk_origin_23          | \            | Demo         | EOL          | 2023         | 生命周期终止           | 2024/6/30   |

## 致谢

MindSpeed LLM由华为公司的下列部门以及昇腾生态合作伙伴联合贡献 ：

华为公司：

- 计算产品线：Ascend
- 公共开发部：NAIE
- 全球技术服务部：GTS
- 华为云计算：Cloud

生态合作伙伴：

- 移动云（China Mobile Cloud）：大云震泽智算平台

感谢来自社区的每一个PR，欢迎贡献 MindSpeed LLM。

## 安全声明

[MindSpeed LLM安全声明](https://gitee.com/ascend/ModelLink/wikis/%E5%AE%89%E5%85%A8%E7%9B%B8%E5%85%B3/%E5%AE%89%E5%85%A8%E5%A3%B0%E6%98%8E)

# 免责声明

## 致MindSpeed LLM使用者
1. MindSpeed LLM提供的模型仅供您用于非商业目的。
2. 对于各模型，MindSpeed LLM平台仅提示性地向您建议可用于训练的数据集，华为不提供任何数据集，如您使用这些数据集进行训练，请您特别注意应遵守对应数据集的License，如您因使用数据集而产生侵权纠纷，华为不承担任何责任。
3. 如您在使用MindSpeed LLM模型过程中，发现任何问题（包括但不限于功能问题、合规问题），请在Gitee提交issue，我们将及时审视并解决。

## 致数据集所有者
如果您不希望您的数据集在MindSpeed LLM中的模型被提及，或希望更新MindSpeed LLM中的模型关于您的数据集的描述，请在Gitee提交issue，我们将根据您的issue要求删除或更新您的数据集描述。衷心感谢您对MindSpeed LLM的理解和贡献。

## License声明
Ascend MindSpeed LLM提供的模型，如模型目录下存在License的，以该License为准。如模型目录下不存在License的，以Apache 2.0许可证许可，对应许可证文本可查阅Ascend MindSpeed LLM根目录。