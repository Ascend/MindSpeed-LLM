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
      <td class="cellrowborder" valign="top" width="73.75%"><p>26.0.0</p></td>
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
      <td class="cellrowborder" valign="top" width="73.75%"><p>April 2026</p></td>
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

| MindSpeed LLM version | MindSpeed Core code branch name | Megatron version | PyTorch version | TorchNPU version | CANN version | Python version |
| -------------------- | ------------------------------ | ---------------- | --------------- | ----------------------------------- | ------------ | -------------- |
| master (under development) | master (under development) | core_v0.12.1 | 2.7.1 | In development | In development | Python 3.10 |
| 26.0.0 (commercial) | 26.0.0_core_r0.12.1 | core_v0.12.1 | 2.7.1 | 26.0.0 | 9.0.0 | Python 3.10 |
| 2.3.0 (commercial) | 2.3.0_core_r0.12.1 | core_v0.12.1 | 2.7.1 | 7.3.0 | 8.5.0 | Python 3.10 |
| 2.2.0 (commercial) | 2.2.0_core_r0.12.1 | core_v0.12.1 | 2.7.1 | 7.2.0 | 8.3.RC1 | Python 3.10 |

> [!NOTE]
>
> You can choose the MindSpeed LLM code branch as needed to download the source code and install it.

## Version Compatibility Information

| MindSpeed LLM version | CANN version | TorchNPU version |
| -- | -- | -- |
| 26.0.0 | CANN 9.0.0<br>CANN 8.5.0<br>CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1 | 26.0.0 |
| 2.3.0 | CANN 8.5.0<br>CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1<br>CANN 8.0.0<br> | 7.3.0 |
| 2.2.0 | CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1<br>CANN 8.0.0<br>CANN 8.0.RC3<br>CANN 8.0.RC2 | 7.2.0 |

## Version Usage Notes

None.

## Update Notes

### New Features

| Component | Description | Purpose |
| -- | -- | -- |
| MindSpeed LLM | Added FSDP2 training support | Supports Qwen3-30B, Qwen3-32B, Qwen3-235B, and Qwen3-Next model training |
| MindSpeed LLM | Added 128K training support | Supports ultra-long sequence training for gpt-oss and DeepSeekV3.2 models |
| MindSpeed LLM | Improved tool efficiency | Supports combined weight conversion and training, and combined data preprocessing and training |
| MindSpeed LLM | Security hardening | Supports PMCC protection for LLM fine-tuning |

### Removed Features

| Component | Description | Purpose |
| -- | -- | -- |
| MindSpeed LLM | Model retirement | Dense model retirement list:<br>Llama-2-34B<br>Llama-3-8B/70B<br>Llama-3.1-8B/50B/70B/200B<br>Llama-3.2-1B/3B<br>Llama-3.3-70B-Instruct<br>ChatGLM3-6B<br>GLM4-9B<br>Baichuan2-7B/13B<br>InternLM2.5-1.8B/7B/20B<br>Qwen2.5-0.5B/1.5B/3B/7B/14B/32B<br>Qwen3-8B (Megatron FSDP2)<br><br>MoE model retirement list:<br>Qwen3-30B (Megatron FSDP2)<br>GPT4-MoE-175B<br>Hunyuan-389B |

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
| [MindSpeed LLM Installation Guide](./pytorch/training/install_guide.md) | This guide helps you install MindSpeed LLM on an NPU. It covers hardware and operating system compatibility, driver firmware and CANN base software installation, and the complete installation process based on the PyTorch framework. It helps you quickly build a distributed LLM training environment. | - |
| [Quick Start: Qwen3-8B Model Pretraining and Fine-Tuning](./pytorch/training/quick_start.md) | Using Qwen3-8B as an example, this guide helps developers who are new to MindSpeed LLM complete pretraining and fine-tuning tasks on the NPU. It helps you quickly get started with distributed LLM training. | - |

## Virus Scan and Vulnerability Fix List

### Virus Scan Results

| Antivirus Software Name | Antivirus Software Version | Virus Database Version | Scan Time | Scan Result |
| --- | --- | --- | --- | --- |
| QiAnXin | 8.0.5.5260 | 2026-04-01 08:00:00.0 | 2026-04-02 | No viruses, no malware |
| Kaspersky | 12.0.0.6672 | 2026-04-02 10:05:00.0 | 2026-04-02 | No viruses, no malware |
| Bitdefender | 7.5.1.200224 | 7.100588 | 2026-04-02 | No viruses, no malware |

### Vulnerability Fix List

None.
