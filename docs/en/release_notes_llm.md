# Release Notes

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
      <td class="cellrowborder" valign="top" width="73.75%"><p>26.1.0</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>Version Type</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>Official release</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>Component Name</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>MindSpeed LLM</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>Release Date</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>July 2026</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>Maintenance</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>6 months</p></td>
    </tr>
  </tbody>
</table>

> [!NOTE]
>
> For version maintenance of MindSpeed LLM, see [Version Maintenance Policy](../../README_en.md#version-maintenance-policy).

### Related Product Version Mapping

**Table 1** MindSpeed LLM software version compatibility matrix

| MindSpeed LLM version | MindSpeed Core code branch name | Megatron version | PyTorch version | TorchNPU version | CANN version | Triton-Ascend version | Python version |
| -------------------- | ------------------------------ | ---------------- | --------------- | ----------------------------------- | ------------ | --------------------- | -------------- |
| 26.1.0               | 26.1.0_core_r0.12.1            | core_v0.12.1     | 2.7.1           | 26.1.0                             | 9.1.0        | 3.2.2                 | Python 3.10    |
| 26.0.0               | 26.0.0_core_r0.12.1            | core_v0.12.1     | 2.7.1           | 26.0.0                             | 9.0.0        | 3.2.1                 | Python 3.10    |

> [!NOTE]
>
> - You can choose the MindSpeed LLM code branch as needed to download the source code and install it.
> - The Triton-Ascend version is strongly bound to the CANN version. The Triton-Ascend version must correspond to the CANN version one-to-one. For details, see [Triton-Ascend Compatibility](https://triton-ascend.readthedocs.io/en/latest/release_note.html#version-compatibility-matrix).

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
  </colgroup>
  <thead>
    <tr>
      <th rowspan="2">MindSpeed LLM</th>
      <th colspan="4">TorchNPU version</th>
    </tr>
    <tr>
      <th>7.2.0</th>
      <th>7.3.0</th>
      <th>26.0.0</th>
      <th>26.1.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>26.0.0</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>/</td>
    </tr>
    <tr>
      <td>26.1.0</td>
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
  </colgroup>
  <thead>
    <tr>
      <th rowspan="2">MindSpeed LLM</th>
      <th colspan="4">CANN version</th>
    </tr>
    <tr>
      <th>8.3.RCX</th>
      <th>8.5.X</th>
      <th>9.0.X</th>
      <th>9.1.X</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>26.0.0</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>/</td>
    </tr>
    <tr>
      <td>26.1.0</td>
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
| MindSpeed LLM | Added model support to the Megatron training backend | Supports Seed-OSS and GLM5 model training |
| MindSpeed LLM | Improved tool efficiency | Supports asynchronous weight saving |
| MindSpeed LLM | Added hardware support | Supports <term>Ascend 950 products</term> |

### Removed Features

| Component | Description | Purpose |
| -- | -- | -- |
| MindSpeed LLM | Model retirement | Model retirement list:<br>InternLM3-8B<br>Llama-2-7B/70B<br>Llama-3.1-405B<br>Mamba2-2.7B/8B<br>Mamba2-Hybrid-8B |
| MindSpeed LLM | Feature retirement | Retired QLoRA and related scripts |

### API Changes

None.

### Resolved Issues

None.

### Known Issues

None.

## Upgrade Impact

### Impact on the Current System during Upgrade

- Service impact.

    Upgrading the software version interrupts service.

- Network communication impact.

    It has no impact on communication.

### Impact on the Current System after Upgrade

None.

## Related Documents

| Document | Summary | Update Notes |
| -- | -- | -- |
| [MindSpeed LLM Installation](./pytorch/training/install_guide.md) | This guide helps you install MindSpeed LLM on an NPU. It covers hardware and operating system compatibility, driver firmware and CANN base software installation, and the complete installation process based on the PyTorch framework. It helps you quickly build a distributed LLM training environment. | The installation operations have been adapted for the version-compatible branch, and Triton-Ascend installation has been added. |
| [MindSpeed LLM Quick Start (Megatron Training Backend)](./pytorch/training/quick_start.md) | Using Qwen3-8B as an example, this guide helps developers who are new to MindSpeed LLM complete pretraining and fine-tuning tasks on the NPU based on the Megatron training backend. It helps you quickly get started with distributed LLM training. | Data and weights can be loaded online for training on Qwen3 series models. Training operations have been optimized accordingly. |
| [MindSpeed LLM Quick Start (FSDP2 Training Backend)](./pytorch/training/fsdp2_quick_start.md) | Using Qwen3-8B as an example, this guide helps developers who are new to MindSpeed LLM complete pretraining and fine-tuning tasks on the NPU based on the FSDP2 training backend. It helps you quickly get started with distributed LLM training. | New document. Describes model pretraining and fine-tuning using the FSDP2 backend in MindSpeed LLM. |

## Virus Scan Results and Vulnerability Patch List

### Virus Scan Results

| Antivirus Software | Antivirus Software Version | Virus Database Version | Scan Time | Scan Result |
| --- | --- | --- | --- | --- |
| QiAnXin | 8.0.5.5260 | 2026-07-05 08:00:00.0 | 2026-07-06 | No viruses or malware |
| Kaspersky | 12.0.0.6672 | 2026-07-06 10:03:00 | 2026-07-06 | No viruses or malware |
| Bitdefender | 7.5.1.200224 | 7.101158 | 2026-07-06 | No viruses or malware |

### Fixed Vulnerabilities

None
