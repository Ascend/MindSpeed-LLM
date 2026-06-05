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
      <td class="cellrowborder" valign="top" width="73.75%"><p>26.1.0</p></td>
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
      <td class="cellrowborder" valign="top" width="73.75%"><p>2026年6月</p></td>
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

| MindSpeed LLM版本 | MindSpeed Core代码分支名称 | Megatron版本 | PyTorch版本  | Ascend Extension for PyTorch版本 | CANN版本 | Triton-Ascend版本 | Python版本     |
| ---------------- | ------------------ | ------------ | -----------  | ------------- |--------------------- |-----------------| ------------------- |
| master（在研版本）| master（在研版本）  | core_v0.12.1  | 2.7.1       | 在研版本       | 在研版本  | 在研版本            | Python3.10      |
| 26.1.0（商用）   | 26.1.0_core_r0.12.1 | core_v0.12.1  | 2.7.1       | 26.1.0        | 9.1.0  | 3.2.2           | Python3.10      |
| 26.0.0（商用）   | 26.0.0_core_r0.12.1 | core_v0.12.1  | 2.7.1       | 26.0.0        | 9.0.0  | 3.2.1           | Python3.10      |

>[!NOTE]
>
>用户可根据需要选择MindSpeed LLM代码分支下载源码并进行安装。

## 版本兼容性说明

|MindSpeed LLM版本|CANN版本|Ascend Extension for PyTorch版本|
|--|--|--|
|26.1.0|CANN 9.1.0<br>CANN 9.0.0<br>CANN 8.5.0<br>CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1|26.1.0|
|26.0.0|CANN 9.0.0<br>CANN 8.5.0<br>CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1|26.0.0|

|CANN版本| Triton-Ascend版本 |
|--|-----------------|
|CANN 9.1.0| 3.2.2           |
|CANN 9.0.0| 3.2.1           |
|CANN 8.5.0| 3.2.0           |

## 版本使用注意事项

- Triton-Ascend版本与CANN版本强绑定，Triton-Ascend的使用应该与CANN版本一一对应。

## 更新说明

### 新增特性

|组件|描述|目的|
|--|--|--|
|MindSpeed LLM|新增Mcore训练支持|支持Seed-OSS、GLM5模型训练|
|MindSpeed LLM|工具效率提升|支持异步保存权重|

### 删除特性

无

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
|《[MindSpeed LLM安装指导](./pytorch/training/install_guide.md)》|指导用户如何在NPU上完成MindSpeed LLM的安装，内容涵盖硬件与操作系统兼容性说明、驱动固件及CANN基础软件安装，以及基于PyTorch框架的完整安装流程，帮助用户快速搭建大语言模型分布式训练环境。|-|
|《[MindSpeed LLM快速入门](./pytorch/training/quick_start.md)》|以Qwen3-8B为例，指导初次接触MindSpeed LLM的开发者完成NPU上的预训练和微调任务，帮助用户快速上手大模型分布式训练。|-|

## 病毒扫描及漏洞修补列表

### 病毒扫描结果

|防病毒软件名称|防病毒软件版本|病毒库版本|扫描时间|扫描结果|
|---|---|---|---|---|
|QiAnXin|8.0.5.5260|2026-04-01 08:00:00.0|2026-04-02|无病毒，无恶意|
|Kaspersky|12.0.0.6672|2026-04-02 10:05:00.0|2026-04-02|无病毒，无恶意|
|Bitdefender|7.5.1.200224|7.100588|2026-04-02|无病毒，无恶意|

### 漏洞修补列表

无
