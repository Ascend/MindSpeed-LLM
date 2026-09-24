# 版本说明

## 关键特性

MindSpeed LLM 是面向昇腾的大语言模型分布式训练套件，支持Qwen3/DeepSeek/Mamba2等100+主流LLM及Dense/MoE/SSM架构，提供开箱即用的昇腾高性能训练脚本；支持分布式预训练、数据预处理方案和TP/PP/DP/CP/EP多维并行；支持全参/LoRA分布式指令微调；同时支持权重转换与在线推理评估。

## 版本配套说明

### 产品版本信息

<table>
  <tbody>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>产品名称</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>MindSpeed</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>产品版本</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>26.2.0</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>版本类型</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>Release版本</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>组件名称</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>MindSpeed LLM</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>发布时间</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>2026年10月</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>维护周期</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>6个月</p></td>
    </tr>
  </tbody>
</table>

> [!NOTE]
>
> 有关MindSpeed LLM的版本维护，具体请参见[版本维护策略](https://gitcode.com/Ascend/MindSpeed-LLM/tree/master#%E7%89%88%E6%9C%AC%E7%BB%B4%E6%8A%A4%E7%AD%96%E7%95%A5)。

### 相关产品版本配套说明

**表 1**  MindSpeed LLM软件版本配套表

| MindSpeed LLM版本 | MindSpeed Core代码分支名称 | Megatron-LM版本 | PyTorch版本  | TorchNPU版本 | CANN版本 | Triton-Ascend版本 | FSDPTurbo版本 | Python版本 |
| ---------------- | ------------------------- | ------------ | -----------  | ------------ | ------- | ----------------- | ------------ | ---------- |
| master（在研版本）| master（在研版本）        | core_v0.12.1  | 2.10.0       | 在研版本      | 在研版本 | 在研版本          | 在研版本      | Python3.12 |
| 26.2.0           | 26.2.0_core_r0.12.1      | core_v0.12.1  | 2.10.0       | 26.2.0       | 9.2.0   | 3.2.2             | 0.1.0        | Python3.12 |
| 26.1.0           | 26.1.0_core_r0.12.1      | core_v0.12.1  | 2.7.1        | 26.1.0       | 9.1.0   | 3.2.2             | /            | Python3.10 |

>[!NOTE]
>
> - 用户可根据需要选择MindSpeed LLM代码分支下载源码并进行安装。
> - Triton-Ascend版本与CANN版本强绑定，Triton-Ascend的使用应该与CANN版本一一对应，详见[Triton-Ascend兼容性](https://triton-ascend.readthedocs.io/zh-cn/latest/release_note.html#id13)。
> - 26.1.0版本未使用FSDPTurbo组件，表中“/”表示不配套。

## 版本兼容性说明

> [!NOTE]
>
> 本节表格中“/”表示不可配套，“Y”表示可配套。

**表 2**  MindSpeed LLM与TorchNPU版本兼容

<table style="table-layout: fixed; width: 750px; text-align:center">
  <colgroup>
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
  </colgroup>
  <thead>
    <tr>
      <th rowspan="2">MindSpeed LLM</th>
      <th colspan="4">TorchNPU版本</th>
    </tr>
    <tr>
      <th>7.3.0</th>
      <th>26.0.0</th>
      <th>26.1.0</th>
      <th>26.2.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>26.1.0</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>/</td>
    </tr>
    <tr>
      <td>26.2.0</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
  </tbody>
</table>

**表 3**  MindSpeed LLM与CANN版本兼容

<table style="table-layout: fixed; width: 750px; text-align:center">
  <colgroup>
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
  </colgroup>
  <thead>
    <tr>
      <th rowspan="2">MindSpeed LLM</th>
      <th colspan="4">CANN版本</th>
    </tr>
    <tr>
      <th>8.5.X</th>
      <th>9.0.X</th>
      <th>9.1.X</th>
      <th>9.2.X</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>26.1.0</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>/</td>
    </tr>
    <tr>
      <td>26.2.0</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
  </tbody>
</table>

## 更新说明

### 新增特性

|组件|描述|目的|
|--|--|--|
|MindSpeed LLM|新增模型支持|支持GLM5.2模型训练|
|MindSpeed LLM|模型训练场景扩充|新增DSeepSeekV4长序列支持|
|MindSpeed LLM|易用性提升|权重/数据合一功能覆盖全量模型用例|

### 删除特性

|组件|描述|目的|
|--|--|--|
|MindSpeed LLM|模型下架|模型下架清单：<br>LLaMA2-13B<br>Qwen2.5-72B<br>Qwen3-0.6B/1.7B/4B/14B/32B<br>Gemma2-9B/27B<br>Phi3.5-mini‑instruct<br>Magistral-24B<br>PLM-1.8B<br>Qwen3‑Next<br>Qwen3‑Coder‑Next<br>DeepSeek‑V2‑Lite<br>Ling‑mini‑2.0<br>Ring-1T<br>Phi3.5<br>GLM4.5‑Air<br>Step3.5‑Flash<br>GPT‑OSS<br>MiniMax‑M2.7<br>Mamba3|
|MindSpeed LLM|特性下架|下架DPO特性以及相关脚本|
|MindSpeed LLM|架构下架|下架mindspore架构相关脚本、代码、文档|

### 接口变更说明

无

### 已解决问题

无

### 遗留问题

无

## 升级影响

### 升级过程中对现行系统的影响

- 对业务的影响

    软件版本升级过程中会导致业务中断。

- 对网络通信的影响

    对通信无影响。

### 升级后对现行系统的影响

无

## 版本配套文档

|文档名称|内容简介|更新说明|
|--|--|--|
|《[MindSpeed LLM软件安装](./pytorch/training/install_guide.md)》|指导用户如何在NPU上完成MindSpeed LLM的安装，内容涵盖硬件与操作系统兼容性说明、驱动固件及CANN基础软件安装，以及基于PyTorch框架的完整安装流程，帮助用户快速搭建大语言模型分布式训练环境。|安装操作适配版本配套分支，新增安装FSDPTurbo加速库；新增源码安装（Submodule统一安装）。|
|《[MindSpeed LLM快速入门（基于Megatron训练后端）](./pytorch/training/quick_start.md)》|以Qwen3-8B为例，指导初次接触MindSpeed LLM的开发者完成NPU上基于Megatron训练后端的预训练和微调任务，帮助用户快速上手大模型分布式训练。|无|
|《[MindSpeed LLM快速入门（基于FSDP2训练后端）](./pytorch/training/fsdp2_quick_start.md)》|以Qwen3-8B为例，指导初次接触MindSpeed LLM的开发者完成NPU上基于FSDP2训练后端的预训练和微调任务，帮助用户快速上手大模型分布式训练。|无|
|《[FSDP2 后端微调使用指南](./pytorch/training/finetune/fsdp2/finetune_fsdp2.md)》|介绍如何基于HuggingFace格式的预训练模型，使用MindSpeed LLM FSDP2训练后端完成全参数微调。|新增模型使用文档。|
|《[精度数据采集](./pytorch/tools/msprobe.md)》|基于MindSpeed LLM FSDP2与Megatron训练后端，指导用户如何在训练过程中调用msProbe进行数据采集。|新增开发工具文档。|
|《[模型输入输出采集](./pytorch/tools/model_io_trace.md)》|适用于快速查看指定训练step的Module前反向输入输出和batch字段等简单定位场景。|新增开发工具文档。|

## 病毒扫描结果

|防病毒软件名称|防病毒软件版本|病毒库版本|扫描时间|扫描结果|
|---|---|---|---|---|
|QiAnXin|8.0.5.5260|2026-07-05 08:00:00.0|2026-07-06|无病毒，无恶意|
|Kaspersky|12.0.0.6672|2026-07-06 10:03:00|2026-07-06|无病毒，无恶意|
|Bitdefender|7.5.1.200224|7.101158|2026-07-06|无病毒，无恶意|

## 漏洞修补列表

无
