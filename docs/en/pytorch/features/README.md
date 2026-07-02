# Training Schemes and Features

---

MindSpeed LLM includes distributed pretraining and distributed fine-tuning schemes.

## Distributed Pretraining

The measured pretraining performance of MindSpeed LLM is as follows.

<table>
  <thead>
    <tr>
      <th>Model Family</th>
      <th>Experimental Model</th>
      <th>Hardware</th>
      <th>Cluster Size</th>
      <th>Throughput (Tokens/s)</th>
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

### Pretraining Schemes

<table>
  <thead>
    <tr>
      <th>Scheme Type</th>
      <th>MCore</th>
      <th>Released</th>
      <th>Contributor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="../training/pretrain/mcore/pretrain.md">Multi-Sample Dataset Pretraining</a></td>
      <td>✅</td>
      <td>✅</td>
      <td rowspan="2">【Ascend】</td>
    </tr>
    <tr>
      <td><a href="../training/pretrain/mcore/pretrain_eod.md">Multi-Sample Pack-Mode Pretraining</a></td>
      <td>✅</td>
      <td>❌</td>
</tr>
  </tbody>
</table>

### Acceleration Features

<table><thead>
  <tr>
    <th>Scenario</th>
    <th>Feature Name</th>
    <th>MCore</th>
    <th>Released</th>
    <th>Contributor</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="5">SPTD Parallelism</td>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/tensor-parallel.md">Tensor Parallelism</a></td>
    <td>✅</td>
    <td>✅</td>
    <td rowspan="29">【Ascend】</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/pipeline-parallel.md">Pipeline Parallelism</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="mcore/virtual_pipeline_parallel.md">Virtual Pipeline Parallelism</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/sequence-parallel.md">Sequence Parallelism</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/noop-layers.md">noop layers</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td rowspan="3">Long-Sequence Parallelism</td>
    <td><a href="mcore/ring-attention-context-parallel.md">Ascend Ring Attention Long-Sequence Parallelism</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/ulysses-context-parallel.md">Ulysses Long-Sequence Parallelism</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/hybrid-context-parallel.md">Hybrid Long-Sequence Parallelism</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td rowspan="2">MoE</td>
    <td><a href="https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md">MoE Expert Parallelism</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/megatron_moe/megatron-moe-allgather-dispatcher.md">MoE Reordering Communication Optimization</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td rowspan="6">Memory Optimization</td>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/reuse-fp32-param.md">Parameter Replica Reuse</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
    <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/distributed-optimizer.md">Distributed Optimizer</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/swap_attention.md">Swap Attention</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="mcore/recompute_relative.md">Recompute</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/norm-recompute.md">Norm Recompute</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="mcore/o2.md">O2 BF16 Optimizer</a></td>
    <td>✅</td>
    <td>❌</td>
  </tr>
  <tr>
    <td rowspan="7">Fused Operators</td>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/flash-attention.md">Flash Attention</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="mcore/variable_length_flash_attention.md">Flash Attention Variable Length</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/rms_norm.md">Fused RMSNorm</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/swiglu.md">Fused SwiGLU</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/rotary-embedding.md">Fused Rotary Position Embedding</a></td>
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
    <td rowspan="6">Communication Optimization</td>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/async-ddp-param-gather.md">Gradient Reduce Communication Overlap</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/recompute_independent_pipelining.md">Recompute in Advance</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/async-ddp-param-gather.md">Weight All-Gather Communication Overlap</a></td>
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
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/hccl-replace-gloo.md">Ascend Gloo Archive-to-Drive Optimization</a></td>
    <td>✅</td>
    <td>✅</td>
  </tr>
</tbody></table>

## Distributed Fine-Tuning

The measured instruction fine-tuning performance of MindSpeed LLM is as follows.

<table>
  <tr>
    <th>Model</th>
    <th>Hardware</th>
    <th>Cluster</th>
    <th>Scheme</th>
    <th>Sequence</th>
    <th>Throughput (Tokens/s)</th>
  </tr>
  <tr>
    <td rowspan="1">Qwen3-30B</td>
    <td rowspan="1">Atlas 900 A3 SuperPoD</td>
    <td rowspan="1">8x16</td>
    <td>Full Parameters</td>
    <td><a href="../../../../examples/mcore/qwen3_moe/tune_qwen3_30b_a3b_256K_full_pack_A3_ptd.sh">256K</a></td>
    <td>3774.914</td>
  </tr>
  <tr>
    <td rowspan="1">Qwen3-32B</td>
    <td rowspan="1">Atlas 900 A3 SuperPoD</td>
    <td rowspan="1">8x16</td>
    <td>Full Parameters</td>
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

### Fine-Tuning Schemes

<table><thead>
  <tr>
    <th>Scheme Name</th>
    <th>MCore</th>
    <th><a href="../training/finetune/mcore/lora_finetune.md">LoRA</a></th>
    <th>Released</th>
    <th>Contributor</th>
  </tr></thead>
<tbody>
  <tr>
    <td><a href="../training/finetune/mcore/instruction_finetune.md">Single-Sample Fine-Tuning</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>【Ascend】</td>
  </tr>
  <tr>
    <td><a href="../training/finetune/mcore/multi_sample_pack_finetune.md">Multi-Sample Pack Fine-Tuning</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>❌</td>
    <td>【Ascend】</td>
  </tr>
    <tr>
    <td><a href="../training/finetune/mcore/multi_turn_conversation.md">Multi-Turn Conversation Fine-Tuning</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>❌</td>
    <td>【Ascend】</td>
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
    <td rowspan="1"><a href="../training/finetune/mcore/lora_finetune.md">LoRA Fine-Tuning</a></td>
    <td><a href="mcore/cc_lora.md">CCLoRA</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>【Ascend】</td>
  </tr>
  <tr>
    <td>Long-Sequence Fine-Tuning</td>
    <td><a href="mcore/fine-tuning-with-context-parallel.md">Long-Sequence CP</a></td>
    <td>✅</td>
    <td>❌</td>
    <td>【Ascend】</td>
  </tr>
</tbody></table>
