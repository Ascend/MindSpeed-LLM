# MindSpeed LLM Installation Guide

This document explains how to quickly install MindSpeed LLM, the distributed training toolkit for large language models, on the MindSpore framework.

## Hardware and Supported OSs

**Table 1** Product hardware support list

| Product | Supported |
|--|:-:|
| <term>Atlas A3 training products</term> | √ |
| <term>Atlas A3 inference products</term> | x |
| <term>Atlas A2 training products</term> | √ |
| <term>Atlas A2 inference products</term> | x |
| <term>Atlas 200I/500 A2 inference products</term> | x |
| <term>Atlas inference products</term> | x |
| <term>Atlas training products</term> | x |

> [!NOTE]
>
> The "√" in the table indicates support, and "x" indicates no support.

- For the OSs supported by each hardware product in physical machine deployment scenarios, see the [Compatibility Query Assistant](https://www.hiascend.com/hardware/compatibility).
- For the OSs supported by each hardware product in VM and container deployment scenarios, see the "OS Compatibility" section in CANN Software Installation, either the commercial edition or the community edition.

## Preparation before Installation

See the "Related Product Version Support" section in the Release Notes to download and install the corresponding software version.

### Installing Driver Firmware

Download the [Firmware and Drivers](https://www.hiascend.com/hardware/firmware-drivers/community). Choose the community or commercial firmware and driver package according to the OS and hardware model.

Use the following commands for installation:

```shell
chmod +x Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run
chmod +x Ascend-hdk-<chip_type>-npu-firmware_<version>.run
./Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run --full --force
./Ascend-hdk-<chip_type>-npu-firmware_<version>.run --full
```

### Installing CANN

Refer to [CANN Quick Installation](https://www.hiascend.com/cann/download) to install the CANN software, including the Toolkit, ops, and NNAL packages, and configure the environment variables.

```shell
# Set environment variables.
source /usr/local/Ascend/cann/set_env.sh                 # Change this to the actual Toolkit installation path.
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0 # Change this to the actual NNAL installation path.
```

> [!NOTICE]
>
> You are advised to install and run the software as a non-root user. You are also advised to control permissions for the installer directory and files. Set directory permissions to `750` and file permissions to `640`. You can control the permissions after installation by setting `umask`, for example `umask 0027`.
> For more security-related information, see the explanation of file permission control for each component in [Security Statement](../SECURITYNOTE.md).

### Installing MindSpore

Refer to the [official MindSpore installation guide](https://www.mindspore.cn/install). Choose the installation command for MindSpore 2.9.0 based on the OS type, CANN version, and Python version. Ensure that network access is available before installation.

## Installing MindSpeed LLM

Follow these steps to install MindSpeed LLM and its dependencies.

1. Enable the environment variables.

    ```shell
    source /usr/local/Ascend/cann/set_env.sh                  # Change this to the actual Toolkit installation path.
    source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0  # Change this to the actual NNAL installation path.
    ```

2. Install the MindSpeed-Core-MS conversion tool.

    ```shell
    git clone https://gitcode.com/ascend/MindSpeed-Core-MS.git -b master
    ```

3. Set up the environment with the internal script provided by MindSpeed-Core-MS.

    ```shell
    cd MindSpeed-Core-MS
    pip3 install -r requirements.txt  # Install third-party dependencies.
    source auto_convert.sh llm        # Pull the component libraries required for training.
    source tests/scripts/set_path.sh  # Set environment variables.
    ```
