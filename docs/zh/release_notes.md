# 版本说明

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
      <td class="cellrowborder" valign="top" width="73.75%"><p>26.0.0</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>版本类型</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>正式版本</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>组件名称</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>MindSpeed LLM</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>发布时间</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>2026年4月</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>维护周期</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>6个月</p></td>
    </tr>
  </tbody>
</table>

> [!NOTE]  
> 有关MindSpeed LLM的版本维护，具体请参见[版本维护策略](https://gitcode.com/Ascend/MindSpeed-LLM/tree/master#%E7%89%88%E6%9C%AC%E7%BB%B4%E6%8A%A4%E7%AD%96%E7%95%A5)。

### 相关产品版本配套说明

**表 1**  MindSpeed LLM软件版本配套表

| MindSpeed LLM版本 | MindSpeed版本      | Megatron版本 | PyTorch版本  | torch_npu版本 | CANN版本 | Python版本            | MindSpeed-Core-MS版本 |
| ---------------- | ------------------ | ------------ | -----------  | ------------- | -------- | --------------------- | ------------------- |
| master（在研版本）| master（在研版本）  | core_v0.12.1  | 2.7.1       | 在研版本       | 在研版本  | Python3.10            | master              |
| 26.0.0（商用）   | 26.0.0             | core_v0.12.1  | 2.7.1       | 26.0.0        | 9.0.0  | Python3.10            | r0.5.0                 |
| 2.3.0（商用）    | 2.3.0_core_r0.12.1 | core_v0.12.1  | 2.7.1       | 7.3.0         | 8.5.0  | Python3.10            | r0.5.0                 |
| 2.2.0（商用）    | 2.2.0_core_r0.12.1 | core_v0.12.1  | 2.7.1       | 7.2.0         | 8.3.RC1  | Python3.10            | r0.4.0               |

>[!NOTE]  
>用户可根据需要选择MindSpeed LLM代码分支下载源码并进行安装。

## 版本兼容性说明

|MindSpeed LLM版本|CANN版本|Ascend Extension for PyTorch版本|MindSpeed-Core-MS版本|
|--|--|--|--|
|2.3.0|CANN 8.5.0<br>CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1<br>CANN 8.0.0<br>|7.3.0|r0.5.0|
|2.2.0|CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1<br>CANN 8.0.0<br>CANN 8.0.RC3<br>CANN 8.0.RC2|7.2.0|r0.4.0|

## 版本使用注意事项

无

## 更新说明

### 新增特性

|组件|描述|目的|
|--|--|--|
|MindSpeed LLM|新增FSDP2训练支持|支持Qwen3-30B/32B/235B、Qwen3-Next模型训练|
|MindSpeed LLM|128K训练支持|支持GPT-OSS、DeepSeekV3.2模型超长序列训练|
|MindSpeed LLM|工具效率提升|支持权重转换与训练合一、数据预处理与训练合一|
|MindSpeed LLM|安全加固|支持文本大模型微调PMCC保护|

### 删除特性

|组件|描述|目的|
|--|--|--|
|MindSpeed LLM|模型下架|Dense模型下架清单：<br>LLaMA2-34B<br>LLaMA3-8B/70B<br>LLaMA3.1-8B/50B/70B/200B<br>LLaMA3.2-1B/3B<br>LLaMA3.3-70B-Instruct<br>ChatGLM3-6B<br>GLM4-9B<br>Baichuan2-7B/13B<br>InternLM2.5-1.8B/7B/20B<br>Qwen2.5-0.5B/1.5B/3B/7B/14B/32B<br>Qwen3-8B(Megatron FSDP2)<br><br>MoE模型下架清单：<br>Qwen3-30B(Megatron FSDP2)<br>GPT4-MoE-175B<br>Hunyuan-389B|

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

## 配套文档

|文档名称|内容简介|更新说明|
|--|--|--|
|《[MindSpeed LLM安装指导](./install_guide.md)》|指导如何在NPU上基于PyTorch和MindSpore框架完成MindSpeed LLM的安装，内容涵盖硬件与操作系统兼容性说明、驱动固件及CANN基础软件安装，以及两种框架下的完整安装流程，帮助用户快速搭建大语言模型分布式训练环境。|-|
|《[MindSpeed LLM快速入门](./quick_start.md)》|以Qwen2.5-7B为例，指导初次接触MindSpeed LLM的开发者完成NPU上的预训练任务，帮助用户快速上手大模型分布式训练。|-|

## 病毒扫描及漏洞修补列表

### 病毒扫描结果

|防病毒软件名称|防病毒软件版本|病毒库版本|扫描时间|扫描结果|
|---|---|---|---|---|
|QiAnXin|8.0.5.5260|2026-01-03 08:00:00.0|2025-12-06|无病毒，无恶意|
|Kaspersky|12.0.0.6672|2026-01-05 10:03:00.0|2025-12-06|无病毒，无恶意|
|Bitdefender|7.5.1.200224|7.100087|2025-12-06|无病毒，无恶意|

### 漏洞修补列表

无
