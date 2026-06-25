# 训练方案与特性说明

---

MindSpeed LLM包含分布式预训练、分布式微调等训练方案。

> [!NOTE]
>
> 文档表格中的“Released”代表商用版本已发布，“✅”代表支持，“❌”代表不支持。

## 分布式预训练

基于MindSpeed LLM的实测预训练性能如下：

<table>
  <thead>
    <tr>
      <th>模型系列</th>
      <th>实验模型</th>
      <th>硬件信息</th>
      <th>集群规模</th>
      <th>吞吐（tokens/s）</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">Qwen3</td>
      <td><a href="../../../../examples/mcore/qwen3/pretrain_qwen3_8b_4K_ptd_A3.sh">8B</a></td>
      <td>Atlas 900 A3 SuperPoD</td>
      <td>1x16</td>
      <td>7617.002</td>
    </tr>
    <tr>
      <td><a href="../../../../examples/mcore/qwen3_moe/pretrain_qwen3_30b_a3b_4K_ptd.sh">30B</a></td>
      <td>Atlas 900 A2 PODc</td>
      <td>2x8</td>
      <td>2318.373</td>
    </tr>
    <tr>
      <td>DeepSeek-V3</td>
      <td><a href="../../../../examples/mcore/deepseek3/pretrain_deepseek3_671b_4k_A3_ptd.sh">671B</a></td>
      <td>Atlas 900 A3 SuperPoD</td>
      <td>32x16</td>
      <td>914.97</td>
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
    <td rowspan="30">【Ascend】</td>
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
  <tr>
    <td rowspan="1">优化器</td>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/muon-optimizer.md">Muon优化器</a></td>
    <td>✅</td>
    <td>❌</td>
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
    <th>吞吐（tokens/s）</th>
  </tr>
  <tr>
    <td rowspan="1">Qwen3-30B</td>
    <td rowspan="1">Atlas 900 A3 SuperPoD</td>
    <td rowspan="1">8x16</td>
    <td>全参</td>
    <td><a href="../../../../examples/mcore/qwen3_moe/tune_qwen3_30b_a3b_256K_full_pack_A3_ptd.sh">256K</a></td>
    <td>3774.914</td>
  </tr>
  <tr>
    <td rowspan="1">Qwen3-32B</td>
    <td rowspan="1">Atlas 900 A3 SuperPoD</td>
    <td rowspan="1">8x16</td>
    <td>全参</td>
    <td><a href="../../../../examples/mcore/qwen3/tune_qwen3_32b_256K_full_pack_A3_ptd.sh">256K</a></td>
    <td>1435.603</td>
  </tr>
  <tr>
    <td rowspan="1">DeepSeek-V3-671B</td>
    <td rowspan="1">Atlas 900 A2 PODc</td>
    <td rowspan="1">8x8</td>
    <td>LoRA</td>
    <td><a href="../../../../examples/mcore/deepseek3/tune_deepseek3_671b_4k_lora_ptd.sh">4K</a></td>
    <td>978.914</td>
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
    <td>【Ascend】</td>
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
    <td>【Ascend】</td>
  </tr>
  <tr>
    <td>长序列微调</td>
    <td><a href="mcore/fine-tuning-with-context-parallel.md">长序列CP</a></td>
    <td>✅</td>
    <td>❌</td>
    <td>【Ascend】</td>
  </tr>
</tbody></table>
