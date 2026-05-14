# 训练方案与特性说明

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
      <td><a href="../../../../examples/mcore/llama2/pretrain_llama2_7b_pack_ptd.sh">LLAMA2-7B</a></td>
      <td>Atlas 900 A2 PODc</td>
      <td>1x8</td>
      <td>69.0%</td>
    </tr>
    <tr>
      <td><a href="../../../../examples/mcore/llama2/pretrain_llama2_13b_pack_ptd.sh">LLAMA2-13B</a></td>
      <td>Atlas 900 A2 PODc</td>
      <td>1x8</td>
      <td>64.7%</td>
    </tr>
    <tr>
      <td><a href="../../../../examples/mcore/llama2/pretrain_llama2_70b_pack_ptd.sh">LLAMA2-70B</a></td>
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
      <td><a href="../training/pretrain/mcore/pretrain.md">多样本集预训练</a></td>
      <td>✅</td>
      <td>✅</td>
      <td rowspan="2">【Ascend】</td>
    </tr>
    <tr>
      <td><a href="../training/pretrain/mcore/pretrain_eod.md">多样本pack模式预训练</a></td>
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
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/tensor-parallel.md">张量并行</a></td>
    <td>✅</td>
    <td>✅</td>
    <td rowspan="29">【Ascend】</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/pipeline-parallel.md">流水线并行</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="mcore/virtual_pipeline_parallel.md">虚拟流水并行</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/sequence-parallel.md">序列并行</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/noop-layers.md">noop layers</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td rowspan="3">长序列并行</td>
    <td><a href="mcore/ring-attention-context-parallel.md">Ascend Ring Attention 长序列并行</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/ulysses-context-parallel.md">Ulysses 长序列并行</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/hybrid-context-parallel.md">混合长序列并行</a></td>
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
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/megatron_moe/megatron-moe-allgather-dispatcher.md">MOE 重排通信优化</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td rowspan="6">显存优化</td>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/reuse-fp32-param.md">参数副本复用</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
    <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/distributed-optimizer.md">分布式优化器</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/swap_attention.md">Swap Attention</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="mcore/recompute_relative.md">重计算</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/norm-recompute.md">Norm重计算</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="mcore/o2.md">O2 BF16 Optimizer</a></td>
    <td>✅</td>
    <td>❌</td>
  </tr>
  <tr>
    <td rowspan="7">融合算子</td>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/flash-attention.md">Flash attention</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="mcore/variable_length_flash_attention.md">Flash attention variable length</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/rms_norm.md">Fused rmsnorm</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/swiglu.md">Fused swiglu</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/rotary-embedding.md">Fused rotary position embedding</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/megatron_moe/megatron-moe-gmm.md">GMM</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/npu_matmul_add.md">Matmul Add</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td rowspan="6">通信优化</td>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/async-ddp-param-gather.md">梯度reduce通算掩盖</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/recompute_independent_pipelining.md">Recompute in advance</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/async-ddp-param-gather.md">权重all-gather通算掩盖</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="mcore/mc2.md">MC2</a></td>
    <td>✅</td>
    <td>❌</td>
  </tr>
  <tr>
    <td><a href="mcore/communication-over-computation.md">CoC</a></td>
    <td>✅</td>
    <td>❌</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/hccl-replace-gloo.md">Ascend Gloo 存档落盘优化</a></td>
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
    <td><a href="../../../../examples/mcore/llama2/tune_llama2_7b_full_ptd.sh">dynamic</a></td>
    <td>15.87 samples/s</td>
    <td>-</td>
  </tr>
  <tr>
    <td>全参</td>
    <td><a href="../../../../examples/mcore/llama2/tune_llama2_7b_full_pack_16k.sh">16K</a></td>
    <td>1.14 samples/s</td>
    <td>37.4%</td>
  </tr>
  <tr>
    <td>全参</td>
    <td><a href="../../../../examples/mcore/llama2/tune_llama2_7b_full_pack_32k.sh">32K</a></td>
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
    <th><a href="../training/finetune/mcore/lora_finetune.md">LoRA</a></th>
    <th><a href="../training/finetune/mcore/qlora_finetune.md">QLoRA</a></th>
    <th>Released</th>
    <th>贡献方</th>
  </tr></thead>
<tbody>
  <tr>
    <td><a href="../training/finetune/mcore/instruction_finetune.md">单样本微调</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>【Ascend】</td>
  </tr>
  <tr>
    <td><a href="../training/finetune/mcore/multi_sample_pack_finetune.md">多样本pack微调</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>❌</td>
    <td>❌</td>
    <td>【NAIE】</td>
  </tr>
    <tr>
    <td><a href="../training/finetune/mcore/multi_turn_conversation.md">多轮对话微调</a></td>
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
    <td rowspan="1"><a href="../training/finetune/mcore/lora_finetune.md">LoRA微调</a></td>
    <td><a href="mcore/cc_lora.md">CCLoRA</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>【Ascend】</td>
  </tr>
  <tr>
      <td rowspan="1"><a href="../training/finetune/mcore/qlora_finetune.md">QLoRA微调</a></td>
      <td><a href="mcore/cc_lora.md">CCLoRA</a></td>
    <td>❌</td>
    <td>❌</td>
    <td>【NAIE】</td>
  </tr>
  <tr>
    <td>长序列微调</td>
    <td><a href="mcore/fine-tuning-with-context-parallel.md">长序列CP</a></td>
    <td>✅</td>
    <td>❌</td>
    <td>【Ascend】</td>
  </tr>
</tbody></table>
