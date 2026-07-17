# MindSpeed LLM Installation Guide

This document explains how to quickly install MindSpeed LLM, the PyTorch-based distributed training toolkit for large language models.

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
- For the OSs supported by each hardware product in VM and container deployment scenarios, see [OS Compatibility](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0101.html?OS=openEuler&InstallType=netyum) in CANN Software Installation for the community edition.

## Preparation before Installation

See [Related Product Version Compatibility](../../release_notes_llm.md#related-product-version-mapping) in the Release Notes to download and install the corresponding software version.

> [!NOTICE]
>
> You are advised to install and run the software as a non-root user. You are also advised to control permissions for the installer directory and files. Set directory permissions to `750` and file permissions to `640`. You can control the permissions after installation by setting `umask`, for example `umask 0027`.
> For more security-related information, see the explanation of file permission control for each component in [Security Note](../../SECURITYNOTE.md).

Download the [Firmware and Drivers](https://www.hiascend.com/hardware/firmware-drivers/community). Choose the community or commercial firmware and driver package according to the OS and hardware model.

Use the following commands for installation:

```shell
chmod +x Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run
chmod +x Ascend-hdk-<chip_type>-npu-firmware_<version>.run
./Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run --full --force
./Ascend-hdk-<chip_type>-npu-firmware_<version>.run --full
```

## Installing MindSpeed LLM

### Method 1: Image Installation

> [!NOTE]
>
> - Before using the image, confirm the machine model. The latest image supports only the AArch64 architecture. Run `uname -a` to verify.
> - The image is pre-installed with CANN 9.0.0 and Ascend Extension for PyTorch 26.0.0. You can use it as needed.
> - If your environment is incompatible with the provided image, choose [Method 2: Installation from Source](#method-2-installation-from-source).

1. Pull the image.

   The latest images correspond to the [MindSpeed LLM 26.0.0 branch](https://gitcode.com/Ascend/MindSpeed-LLM/tree/26.0.0). [Pull the image](https://www.hiascend.com/developer/ascendhub/detail/e26da9266559438b93354792f25b2f4a) as needed.

   - <term>Atlas A2 training products</term>: `26.0.0-910b-openeuler24.03-py3.11-aarch64`
   - <term>Atlas A3 training products</term>: `26.0.0-a3-openeuler24.03-py3.11-aarch64`

   ```bash
   # Verify that the image was pulled successfully
   docker image list
   ```

2. Create a container.

   ```bash
   # Mount the image
   docker run -dit --ipc=host --network host --name '<container_name>' --privileged \
       -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
       -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
       -v /usr/local/sbin/:/usr/local/sbin/ \
       -v /home/:/home/ \
       -v /data/:/data \
       <image_name>:<tag> /bin/bash
   ```

   By default, the driver and firmware are installed in `/usr/local/Ascend`. If the paths differ, modify the command accordingly.

   The container initializes the NPU driver and CANN environment by default. To install new ones, replace or source them manually. See `~/.bashrc` in the container for details.

   - Example 1: Basic run

      ```bash
      docker run -it --rm \
          mindspeed-llm:26.0.0-a3-openeuler24.03-py3.11-aarch64 bash
      ```

   - Example 2: Using an NPU device (for example, `/dev/davinci1`)

      ```bash
      # Modify the ascend-toolkit path based on the actual situation
      # Assume the NPU device is installed at /dev/davinci1 and the NPU driver is installed in /usr/local/Ascend
      docker run -it --rm \
         --device=/dev/davinci1 \
         --device=/dev/davinci_manager \
         --device=/dev/devmm_svm \
         --device=/dev/hisi_hdc \
         -v /usr/local/dcmi:/usr/local/dcmi \
         -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
         -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
         -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
         -v /etc/ascend_install.info:/etc/ascend_install.info \
         mindspeed-llm:26.0.0-a3-openeuler24.03-py3.11-aarch64 bash
      ```

   - Example 3: Mounting a data directory (for example, `/dev/davinci1`)

      ```bash
      # Modify the ascend-toolkit path based on the actual situation
      docker run -it --rm \
         --device=/dev/davinci1 \
         --device=/dev/davinci_manager \
         --device=/dev/devmm_svm \
         --device=/dev/hisi_hdc \
         -v /usr/local/dcmi:/usr/local/dcmi \
         -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
         -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
         -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
         -v /etc/ascend_install.info:/etc/ascend_install.info \
         -v /data:/data \
         -v /weights:/weights \
         mindspeed-llm:26.0.0-a3-openeuler24.03-py3.11-aarch64 bash
      ```

3. Load the container and verify the environment.

   ```bash
   # Load the container
   docker exec -it <container_name> bash
   # Verify that the NPU is working
   npu-smi info
   ```

### Method 2: Installation from Source

1. Install CANN.

   Install the matching versions of the NPU driver and firmware, and install the CANN software, including the Toolkit, ops, and NNAL packages, and configure the CANN environment variables. For details, see [CANN Software Installation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0000.html) for the community edition.

   CANN software provides a script for setting process-level environment variables. Before you run application code with NPU acceleration in training or inference scenarios, you must call this script. Otherwise, the application code cannot run.

   ```shell
   source /usr/local/Ascend/cann/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh
   ```

   The preceding commands use the default installation paths after installation for the root user as an example. Replace them with the actual path to `set_env.sh`.

2. Install PyTorch and `torch_npu`.

   Refer to the "[Install PyTorch](https://www.hiascend.com/document/detail/zh/Pytorch/2600/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md)" section in the Ascend Extension for PyTorch Installation Guide to obtain matching versions of the PyTorch and `torch_npu` packages.

   You can use the following installation commands:

   ```shell
   # Refer to https://gitcode.com/ascend/pytorch/releases for torch and torch_npu build instructions
   pip3 install torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl
   pip3 install torch_npu-2.7.1post4-cp310-cp310-manylinux_2_28_aarch64.whl
   ```

3. Install the MindSpeed acceleration library.

   ```shell
   git clone https://gitcode.com/ascend/MindSpeed.git
   cd MindSpeed
   git checkout 26.0.0_core_r0.12.1  # Switch to the 26.0.0_core_r0.12.1 branch of MindSpeed
   pip3 install -r requirements.txt
   pip3 install -e .
   cd ..
   ```

4. Prepare the MindSpeed LLM and Megatron-LM source code.

   ```shell
   git clone https://gitcode.com/ascend/MindSpeed-LLM.git
   git clone https://github.com/NVIDIA/Megatron-LM.git  # Download Megatron-LM from GitHub. Ensure that the network is accessible
   cd Megatron-LM
   git checkout core_v0.12.1
   cp -r megatron ../MindSpeed-LLM/
   cd ../MindSpeed-LLM
   git checkout 26.0.0
   mkdir logs

   pip3 install -r requirements.txt  # Install the remaining dependency packages
   ```
