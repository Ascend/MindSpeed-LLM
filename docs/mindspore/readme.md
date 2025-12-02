## MindSpore后端支持说明

MindSpeed-LLM已支持接入华为自研AI框架MindSpore，旨在提供华为全栈易用的端到端的大语言模型训练解决方案，以此获得更极致的性能体验。MindSpore后端提供了一套对标PyTorch的API，用户无需进行额外代码适配即可无缝切换。

---

## NEWS !!! 📣📣📣

🚀🚀🚀MindSpore后端已支持 **[Deepseek-V3](../../examples/mindspore/deepseek3/README.md)/[QWEN3](../../examples/mindspore/qwen3/README.md)/[GLM-4.5](../../examples/mindspore/glm45-moe/README.md)** ！！！🚀🚀🚀


## 版本配套表

MindSpeed-LLM + MindSpore后端的依赖配套如下表，安装步骤参考[基础安装指导](./install_guide.md)。

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
    <td>3.10</td>
  </tr>
</table>
注：由于master分支使用在研版本的驱动以及CANN包，因此master上的一些新特性老版本配套可能有不支持情况，要使用稳定版本，请切换到商发分支并下载对应依赖版本进行安装。                                                                                                                           |

## 模型支持

MindSpore后端仅支持以 mcore 方式实现的模型，当前模型支持详情见下表，更多模型支持将逐步上线，敬请期待！

<table><thead>
  <tr>
    <th>模型类别</th>
    <th>模型列表</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="1">稠密模型</td>
    <td><a href="models/dense_model.md">Dense</a></td>
  </tr>
  <tr>
    <td rowspan="1">MOE模型</td>
    <td><a href="models/moe_model.md">MOE</a></td>
  </tr>
</tbody></table>

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
    <td rowspan="7">SPTD并行</td>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/tensor-parallel.md">张量并行</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/pipeline-parallel.md">流水线并行</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="../pytorch/features/virtual_pipeline_parallel.md">虚拟流水并行</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/sequence-parallel.md">序列并行</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/noop-layers.md">Noop Layers</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="../mindspore/features/seq1f1b.md">Seq1F1B流水线并行</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/custom_fsdp.md">全分片并行</a></td>
    <td>暂不支持开启pp及--reuse-fp32-param参数配置</td>
  </tr>
  <tr>
    <td rowspan="2">长序列并行</td>
    <td><a href="../pytorch/features/ring-attention-context-parallel.md">Ascend Ring Attention 长序列并行</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/ulysses-context-parallel.md">Ulysses 长序列并行</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td rowspan="2">MOE</td>
    <td><a href="https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md">MOE 专家并行</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/megatron_moe/megatron-moe-allgather-dispatcher.md">MOE 重排通信优化</a></td>
    <td>仅支持alltoall</td>
  </tr>
  <tr>
    <td rowspan="5">显存优化</td>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/reuse-fp32-param.md">参数副本复用</a></td>
    <td>须和分布式优化器特性一起使用</td>
  </tr>
    <tr>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/distributed-optimizer.md">分布式优化器</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="../pytorch/features/recompute_relative.md">重计算</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/norm-recompute.md">Norm重计算</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/features/virtual-optimizer.md">Virtual Optimizer</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td rowspan="7">融合算子</td>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/flash-attention.md">Flash attention</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="../pytorch/features/variable_length_flash_attention.md">Flash attention variable length</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/rms_norm.md">Fused rmsnorm</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/swiglu.md">Fused swiglu</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/rotary-embedding.md">Fused rotary position embedding</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/megatron_moe/megatron-moe-gmm.md">GMM</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/npu_matmul_add.md">Matmul Add</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td rowspan="4">通信优化</td>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/async-ddp-param-gather.md">梯度reduce通算掩盖</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/async-ddp-param-gather.md">权重all-gather通算掩盖</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="../pytorch/features/communication-over-computation.md">CoC</a></td>
    <td>✅</td>
  </tr>
    <tr>
    <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/blob/master/docs/mindspore/features/alltoallvc.md">AllToAllVC 通信算子</a></td>
    <td>✅</td>
  </tr>
</tbody></table>

### 在线推理

<table>
  <thead>
    <tr>
      <th>特性</th>
      <th>是否支持</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="../pytorch/solutions/inference/inference.md">流式推理 </a></td>
      <td>✅</td>
    </tr>
    <tr>
      <td><a href="../pytorch/solutions/inference/chat.md"> Chat对话</a></td>
      <td>✅</td>
    </tr>
    <tr>
      <td><a href="../pytorch/features/yarn.md"> yarn上下文扩展 </a></td>
      <td>✅</td>
    </tr>
  </tbody>
</table>

### 开源数据集评测

即将上线，敬请期待！


## 开发工具链

### 数据预处理

MindSpore后端已完全支持MindSpeed-LLM的预训练、指令微调、RLHF等多种任务的数据预处理。

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
      <td><a href="../pytorch/solutions/pretrain/pretrain_dataset.md">预训练数据处理</a></td>
      <td>✅</td>
      <td>✅</td>
      <td rowspan="3">【Ascend】</td>
    </tr>
    <tr>
      <td rowspan="2">微调</td>
      <td><a href="../pytorch/solutions/finetune/datasets/alpaca_dataset.md">Alpaca风格</a></td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td><a href="../pytorch/solutions/finetune/datasets/sharegpt_dataset.md">ShareGPT风格</a></td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>DPO</td>
      <td rowspan="3"><a href="../pytorch/solutions/finetune/datasets/pairwise_dataset.md">Pairwise数据集处理</a></td>
      <td>✅</td>
      <td>✅</td>
      <td rowspan="3">【NAIE】</td>
    </tr>
  </tbody>
</table>

### 权重转换

MindSpeed MindSpore后端的权重转换与PyTorch后端保持了一致，当前支持huggingface、megatron-core两种格式的权重互转。权重转换特性参数和使用说明参考[权重转换](../pytorch/solutions/checkpoint/checkpoint_convert.md)。

<table>
  <thead>
    <tr>
      <th>源格式</th>
      <th>目标格式</th>
      <th>切分特性</th>
      <th>lora</th>
      <th>贡献方</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>huggingface</td>
      <td>megatron-core</td>
      <td>tp、pp、dpp、vpp、cp、ep、loop layer</td>
      <td>❌</td>
      <td rowspan="3">【Ascend】</td>
    </tr>
    <tr>
      <td rowspan="2">megatron-core</td>
      <td>huggingface</td>
      <td></td>
      <td>✅</td>
    </tr>
    <tr>
      <td>megatron-core</td>
      <td>tp、pp、dpp、vpp、cp、ep、loop layer</td>
      <td>❌</td>
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
      <th>贡献方</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="1">性能采集</td>
      <td><a href="../pytorch/features/profiling.md">基于昇腾芯片采集 profiling 数据</a></td>
      <td>✅</td>
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
      <th>贡献方</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">高可用性</td>
      <td><a href="../pytorch/features/deterministic_computation.md">基于昇腾芯片开启确定性计算</a></td>
      <td>✅</td>
      <td rowspan="2">【Ascend】</td>
    </tr>
  </tbody>
</table>