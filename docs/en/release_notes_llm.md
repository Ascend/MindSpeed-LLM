# Release Notes

## Key Features

MindSpeed LLM now supports GLM-5.2 model training and long-sequence training for DeepSeek-V4, and the online data and weight loading feature now covers all models, significantly improving model training capabilities.

## Version Mapping

### Product Version Information

<table>
  <tbody>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>Product</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>MindSpeed</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>Product Version</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>26.2.0</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>Version Type</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>Release version</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>Component Name</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>MindSpeed LLM</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>Release Date</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>October 2026</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>Maintenance</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>6 months</p></td>
    </tr>
  </tbody>
</table>

> [!NOTE]
>
> For version maintenance of MindSpeed LLM, see [Version Maintenance Policy](https://gitcode.com/Ascend/MindSpeed-LLM/tree/master#%E7%89%88%E6%9C%AC%E7%BB%B4%E6%8A%A4%E7%AD%96%E7%95%A5).

### Related Product Version Mapping

**Table 1** MindSpeed LLM software version compatibility matrix

| MindSpeed LLM version | MindSpeed Core code branch name | Megatron version | PyTorch version | TorchNPU version | CANN version | Triton-Ascend version | FSDPTurbo version | Python version |
| -------------------- | ------------------------------ | ---------------- | --------------- | ----------------------------------- | ------------ | --------------------- | -------------- | -------------- |
| 26.2.0               | 26.2.0_core_r0.12.1            | core_v0.12.1     | 2.10.0           | 26.2.0                             | 9.2.0        | 3.6.0                 | 26.2.0         | Python 3.12    |
| 26.1.0               | 26.1.0_core_r0.12.1            | core_v0.12.1     | 2.7.1           | 26.1.0                             | 9.1.0        | 3.2.2                 | /              | Python 3.10    |
| 26.0.0               | 26.0.0_core_r0.12.1            | core_v0.12.1     | 2.7.1           | 26.0.0                             | 9.0.0        | 3.2.1                 | /              | Python 3.10    |

> [!NOTE]
>
> - You can choose the MindSpeed LLM code branch as needed to download the source code and install it.
> - The Triton-Ascend version is strongly bound to the CANN version. The Triton-Ascend version must correspond to the CANN version one-to-one. For details, see [Triton-Ascend Compatibility](https://triton-ascend.readthedocs.io/en/latest/release_note.html#id13).
> - Versions 26.1.0 and 26.0.0 do not use the FSDPTurbo component. In the table, "/" indicates incompatibility.

## Version Compatibility Information

> [!NOTE]
>
> In the tables in this section, "/" indicates incompatibility and "Y" indicates compatibility.

**Table 2** MindSpeed LLM and TorchNPU version compatibility

<table style="table-layout: fixed; width: 750px; text-align:center">
  <colgroup>
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
  </colgroup>
  <thead>
    <tr>
      <th rowspan="2">MindSpeed LLM</th>
      <th colspan="5">TorchNPU Version</th>
    </tr>
    <tr>
      <th>7.2.0</th>
      <th>7.3.0</th>
      <th>26.0.X</th>
      <th>26.1.X</th>
      <th>26.2.X</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>26.0.X</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
    <tr>
      <td>26.1.X</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
    <tr>
      <td>26.2.X</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
  </tbody>
</table>

**Table 3** MindSpeed LLM and CANN version compatibility

<table style="table-layout: fixed; width: 750px; text-align:center">
  <colgroup>
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
  </colgroup>
  <thead>
    <tr>
      <th rowspan="2">MindSpeed LLM</th>
      <th colspan="5">CANN Version</th>
    </tr>
    <tr>
      <th>8.3.RCX</th>
      <th>8.5.0</th>
      <th>9.0.X</th>
      <th>9.1.X</th>
      <th>9.2.X</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>26.0.X</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
    <tr>
      <td>26.1.X</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
    <tr>
      <td>26.2.X</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
  </tbody>
</table>

## Update Notes

### New Features

| Component | Description | Purpose |
| -- | -- | -- |
| MindSpeed LLM | Added model support | Supports GLM-5.2 model training |
| MindSpeed LLM | Expanded model training scenarios | Adds long sequence support for DeepSeek-V4 |
| MindSpeed LLM | Improved ease of use | Extends online data and weight loading to all model use cases |

### Removed Features

| Component | Description | Purpose |
| -- | -- | -- |
| MindSpeed LLM | Model retirement | Model retirement list:<br>Llama-2-13B<br>Qwen2.5-72B<br>Qwen3-0.6B/1.7B/4B/14B/32B<br>Gemma2-9B/27B<br>Phi3.5-mini-instruct<br>Magistral-24B<br>PLM-1.8B<br>Qwen3-Next<br>Qwen3-Coder-Next<br>DeepSeek-V2-Lite<br>Ling-mini-2.0<br>Ring-1T<br>Phi3.5<br>GLM-4.5-Air<br>Step3.5-Flash<br>GPT-OSS<br>MiniMax-M2.7<br>Mamba3 |
| MindSpeed LLM | Feature retirement | Retired the DPO feature and related scripts |
| MindSpeed LLM | Architecture retirement | Retired the MindSpore architecture scripts, code, and documentation |

### API Changes

None.

### Resolved Issues

None.

### Known Issues

None.

## Upgrade Impact

### Impact on the Current System During Upgrade

- Service impact.

    Upgrading the software version interrupts service.

- Network communication impact.

    It has no impact on communication.

### Impact on the Current System After Upgrade

None.

## Documentation

| Document | Summary | Update Notes |
| -- | -- | -- |
| [MindSpeed LLM Installation](./pytorch/training/install_guide.md) | This guide helps you install MindSpeed LLM on an NPU. It covers hardware and operating system compatibility, driver firmware and CANN base software installation, and the complete installation process based on the PyTorch framework. It helps you quickly build a distributed LLM training environment. | The installation operations have been adapted for the version-compatible branch. Installation of the FSDPTurbo acceleration library and source code installation (unified Submodule installation) are now supported. |
| [MindSpeed LLM Quick Start (Megatron Training Backend)](./pytorch/training/quick_start.md) | Using Qwen3-8B as an example, this guide helps developers who are new to MindSpeed LLM complete pretraining and fine-tuning tasks on the NPU based on the Megatron training backend. It helps you quickly get started with distributed LLM training. | None. |
| [MindSpeed LLM Quick Start (FSDP2 Training Backend)](./pytorch/training/fsdp2_quick_start.md) | Using Qwen3-8B as an example, this guide helps developers who are new to MindSpeed LLM complete pretraining and fine-tuning tasks on the NPU based on the FSDP2 training backend. It helps you quickly get started with distributed LLM training. | None. |
| [FSDP2 Backend Fine-Tuning Guide](./pytorch/training/finetune/fsdp2/finetune_fsdp2.md) | Describes how to use the MindSpeed LLM FSDP2 training backend to complete full-parameter fine-tuning based on a pretrained model in the Hugging Face format. | New model usage document. |
| [Precision Data Collection](./pytorch/tools/msprobe.md) | This guide is based on the MindSpeed LLM FSDP2 and Megatron training backends and describes how you can call msProbe for data collection during training. | New development tool document. |
| [Model Input and Output Collection](./pytorch/tools/model_io_trace.md) | This guide applies to simple troubleshooting scenarios such as quickly viewing the forward and backward inputs and outputs of a module and the batch fields at a specified training step. | New development tool document. |

## Virus Scan Results

| Antivirus Software | Antivirus Software Version | Virus Database Version | Scan Time | Scan Result |
| --- | --- | --- | --- | --- |
| QiAnXin | 8.0.5.5260 | 2026-09-27 08:00:00.0 | 2026-09-28 | No viruses or malware |
| Kaspersky | 12.0.0.6672 | 2026-09-28 10:09:00 | 2026-09-28 | No viruses or malware |
| Bitdefender | 7.5.1.200224 | 7.101487 | 2026-09-28 | No viruses or malware |

## Fixed Vulnerabilities

None

## Revision History

| Document Version | Release Date | Change Description |
| --- | --- | --- |
| 01 | 2026-10-15 | First official release. |
