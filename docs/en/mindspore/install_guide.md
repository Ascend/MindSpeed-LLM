# MindSpeed LLM Installation

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
> The "√" in the table indicates support, and "x" indicates no support.

- For the OSs supported by each hardware product in physical machine deployment scenarios, see the [Compatibility Query Assistant](https://www.hiascend.com/hardware/compatibility).
- For the OSs supported by each hardware product in VM and container deployment scenarios, see [OS Compatibility](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/softwareinst/instg/instg_0101.html?OS=openEuler&InstallType=netyum) in the community edition of CANN Software Installation.

## Preparation before Installation

See the "[Related Product Version Mapping](../release_notes_llm.md#related-product-version-mapping)" section in *Release Notes* to download and install the corresponding software version.

### Installing Driver Firmware

Click [Firmware and Drivers](https://www.hiascend.com/hardware/firmware-drivers) and follow the guidance to install the firmware and drivers.

### Installing CANN

Install the matching NPU driver, firmware, and CANN software (Toolkit, ops, and NNAL), and configure the CANN environment variables. For details, see [CANN Software Installation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/softwareinst/instg/instg_0000.html).

CANN provides a process-level environment variable setting script. Before executing business code using the NPU in training or inference scenarios, you must invoke this script; otherwise, the business code cannot be executed.

```shell
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0
```

The preceding commands use the default paths after installation as the root user as an example. Replace the paths with the actual `set_env.sh` paths.

> [!NOTICE]
>
> You are advised to install and run the program as a non-root user. You are also advised to control permissions for the installer directory and files. Set directory permissions to `750` and file permissions to `640`. You can control the permissions after installation by setting `umask`, for example `umask 0027`.
> For more security-related information, see the explanation of file permission control for each component in [Security Statement](../SECURITYNOTE.md).

### Installing MindSpore

Refer to the [official MindSpore installation guide](https://www.mindspore.cn/install). Choose the installation command for MindSpore 2.9.0 based on the OS type, CANN version, and Python version. Ensure that network access is available before installation.

## Installing MindSpeed LLM

Follow these steps to install MindSpeed LLM and its dependencies.

1. Enable the environment variables.

    ```shell
    source /usr/local/Ascend/cann/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0
    ```

    The preceding commands use the default paths after installation as the root user as an example. Replace the paths with the actual `set_env.sh` paths.

2. Install the MindSpeed-Core-MS conversion tool.

    ```shell
    git clone https://gitcode.com/ascend/MindSpeed-Core-MS.git -b master
    ```

3. Set up the environment with the internal script provided by MindSpeed-Core-MS.

    ```shell
    cd MindSpeed-Core-MS
    pip3 install -r requirements.txt  # Install third-party dependencies
    source auto_convert.sh llm        # Pull the component libraries required for training
    source tests/scripts/set_path.sh  # Set environment variables
    ```
