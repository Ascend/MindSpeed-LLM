## MindSpore后端支持说明


## 介绍

MindSpeed-LLM已支持接入华为自研AI框架MindSpore，旨在提供华为全栈易用的端到端的大语言模型训练解决方案，以此获得更极致的性能体验。MindSpore后端提供了一套对标PyTorch的API，用户无需进行额外代码适配即可无缝切换。

## 版本配套表

MindSpeed-LLM + MindSpore后端的依赖配套如下表，安装步骤参考[基础安装指导](./features/install_guide.md)。

| 依赖软件        |                                                                                                                                    |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| 昇腾NPU驱动固件 | [在研版本](https://www.hiascend.com/hardware/firmware-drivers/community?product=1&model=30&cann=8.0.RC3.alpha002&driver=1.0.26.alpha) |
| 昇腾 CANN       | [在研版本](https://www.hiascend.com/zh/developer/download/community/result?module=cann)                                               |
| MindSpore       | [2.6.0-rc1](https://www.mindspore.cn/install/)                                                                                        |
| Python          | >=3.9                                                                                                                              |

## 模型支持

MindSpore后端现已支持DeepSeek-V3模型预训练，模型指南详见[DeepSeek3预训练](../../examples/mindspore/deepseek3/README.md)。
更多模型支持即将上线，敬请期待！

## 特性支持

MindSpore后端对MindSpeed的重要加速特性的支持情况如下表所示，部分不支持的特性将在后续迭代中逐步支持，敬请期待。

<table><thead>
  <tr>
    <th>场景</th>
    <th>特性名称</th>
    <th>支持情况</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="5">SPTD并行</td>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/tensor-parallel.md">张量并行</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/pipeline-parallel.md">流水线并行</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="./docs/features/virtual_pipeline_parallel.md">虚拟流水并行</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/sequence-parallel.md">序列并行</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/noop-layers.md">Noop Layers</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td rowspan="3">长序列并行</td>
    <td><a href="../docs/features/ring-attention-context-parallel.md">Ascend Ring Attention 长序列并行</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/ulysses-context-parallel.md">Ulysses 长序列并行</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/hybrid-context-parallel.md">混合长序列并行</a></td>
    <td>❌</td>
  </tr>
  <tr>
    <td rowspan="2">MOE</td>
    <td><a href="https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md">MOE 专家并行</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/megatron_moe/megatron-moe-allgather-dispatcher.md">MOE 重排通信优化</a></td>
    <td>仅支持alltoall</td>
  </tr>
  <tr>
    <td rowspan="6">显存优化</td>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/reuse-fp32-param.md">参数副本复用</a></td>
    <td>✅</td>
  </tr>
    <tr>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/distributed-optimizer.md">分布式优化器</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/swap_attention.md">Swap Attention</a></td>
    <td>❌</td>
  </tr>
  <tr>
    <td><a href="../docs/features/recompute_relative.md">重计算</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/norm-recompute.md">Norm重计算</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="../docs/features/o2.md">O2 BF16 Optimizer</a></td>
    <td>❌</td>
  </tr>
  <tr>
    <td rowspan="7">融合算子</td>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/flash-attention.md">Flash attention</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="../docs/features/variable_length_flash_attention.md">Flash attention variable length</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/rms_norm.md">Fused rmsnorm</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/swiglu.md">Fused swiglu</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/rotary-embedding.md">Fused rotary position embedding</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/megatron_moe/megatron-moe-gmm.md">GMM</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/npu_matmul_add.md">Matmul Add</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td rowspan="6">通信优化</td>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/async-ddp-param-gather.md">梯度reduce通算掩盖</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/recompute_independent_pipelining.md">Recompute in advance</a></td>
    <td>❌</td>
  </tr>
  <tr>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/async-ddp-param-gather.md">权重all-gather通算掩盖</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="../docs/features/mc2.md">MC2</a></td>
    <td>❌</td>
  </tr>
  <tr>
    <td><a href="../docs/features/communication-over-computation.md">CoC</a></td>
    <td>❌</td>
  </tr>
  <tr>
    <td><a href="https://gitee.com/ascend/MindSpeed/blob/master/docs/features/hccl-replace-gloo.md">Ascend Gloo 存档落盘优化</a></td>
    <td>❌</td>
  </tr>
</tbody></table>

## 开发工具链


### 数据预处理

MindSpore后端已完全支持MindSpeed-LLM的预训练、指令微调、RLHF等多种任务的数据预处理。

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
      <td><a href="../pytorch/solutions/pretrain/pretrain_dataset.md">预训练数据处理</a></td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td rowspan="3">【Ascend】</td>
    </tr>
    <tr>
      <td rowspan="2">微调</td>
      <td><a href="../pytorch/solutions/finetune/datasets/alpaca_dataset.md">Alpaca风格</a></td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td><a href="../pytorch/solutions/finetune/datasets/sharegpt_dataset.md">ShareGPT风格</a></td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>DPO</td>
      <td rowspan="3"><a href="../pytorch/solutions/finetune/datasets/pairwise_dataset.md">Pairwise数据集处理</a></td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td rowspan="3">【NAIE】</td>
    </tr>
    <tr>
      <td>SimPO</td>
      <td>✅</td>
      <td>✅</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>ORM</td>
      <td>✅</td>
      <td>✅</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>PRM</td>
      <td rowspan="1"><a href="../pytorch/solutions/preference-alignment/process_reward_dataset.md">PRM数据集处理</a></td>
      <td>✅</td>
      <td>✅</td>
      <td>❌</td>
      <td rowspan="1">【Ascend】</td>
    </tr>
  </tbody>
</table>

### 权重转换

即将上线，敬请期待！

### 在线推理

即将上线，敬请期待！

### 开源数据集评测

即将上线，敬请期待！

