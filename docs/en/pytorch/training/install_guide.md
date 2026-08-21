# MindSpeed LLM Installation

This document explains how to quickly install MindSpeed LLM, the PyTorch-based distributed training toolkit for large language models.

## Hardware and Supported OSs

**Table 1** Product hardware support list

| Product | Supported |
|--|:-:|
| <term>Ascend 950 products</term> | √ |
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
- For the OSs supported by each hardware product in VM and container deployment scenarios, see the ["OS Compatibility"](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/softwareinst/instg/instg_0101.html?OS=openEuler&InstallType=netyum) section in *CANN Software Installation*.

## Preparation Before Installation

See [Related Product Version Mapping](../../release_notes_llm.md#related-product-version-mapping) in the *Release Notes* to download and install the corresponding software version.

Click [Firmware and Drivers](https://www.hiascend.com/hardware/firmware-drivers) and follow the guidance to install the firmware and drivers.

> [!NOTICE]
>
> You are advised to install and run the software as a non-root user. You are also advised to control permissions for the installer directory and files. Set directory permissions to 750 and file permissions to 640. You can control the permissions after installation by setting `umask`, for example `umask 0027`.
> For more security-related information, see the explanation of file permission control for each component in [Security Statement](../../SECURITYNOTE.md).

## Installing MindSpeed LLM

### Method 1: Image Installation

> [!NOTE]
>
> - Before using the image, confirm the machine model. The latest image supports only the AArch64 architecture. Run `uname -a` to verify.
> - The image is pre-installed with CANN 9.1.0 and TorchNPU 26.1.0. You can use it as needed.
> - If your environment is incompatible with the provided image, choose [Method 2: Installation from Source](#method-2-installation-from-source).
> - The 26.1.0 branch will be updated with new images. For custom image building, see [Image Overview](../../../../docker/OVERVIEW.md).

1. Obtain the image.

   The latest images correspond to the [MindSpeed LLM 26.1.0 branch](https://gitcode.com/Ascend/MindSpeed-LLM/tree/26.1.0). The image is coming soon. Currently, you can use the image corresponding to the MindSpeed LLM 26.0.0 branch. Click [Obtain the image](https://www.hiascend.com/developer/ascendhub/detail/e26da9266559438b93354792f25b2f4a).

   - <term>Atlas A2 training products</term>: `26.0.0-910b-openeuler24.03-py3.11-aarch64`

   - <term>Atlas A3 training products</term>: `26.0.0-a3-openeuler24.03-py3.11-aarch64`

   ```bash
   # Verify that the image was obtained successfully
   docker image list
   ```

2. Start a container.

   Run the following command to start a container. This command is for reference only. Modify it as needed. For parameters, see [Table 2](#table1).

   ```bash
   docker run -it -d \
      --ipc=host \
      --network=host \
      --pid=host \
      --name mindspeed_llm \
      --privileged \
      --shm-size=512g \
      --device=/dev/davinci0 \
      --device=/dev/davinci_manager \
      --device=/dev/devmm_svm \
      --device=/dev/hisi_hdc \
      -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
      -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
      -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
      -v /usr/local/dcmi:/usr/local/dcmi \
      -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
      -v /etc/ascend_install.info:/etc/ascend_install.info \
      -v /data:/data \
      -v /weights:/weights \
      mindspeed-llm:26.0.0-a3-openeuler24.03-py3.11-aarch64 \
      /bin/bash
   ```

   > [!NOTE]
   >
   > - By default, the driver and firmware are installed in `/usr/local/Ascend`. If the paths differ, modify the command accordingly.
   > - Before copying the startup command, replace the `/data` and `/weights` paths in the `-v` parameters with the actual local directories on the host. Otherwise, the container fails to start.
   > - The container initializes the NPU driver and CANN environment by default. To install new ones, replace or source them manually. See `~/.bashrc` in the container for details.
   > - `mindspeed-llm:26.0.0-a3-openeuler24.03-py3.11-aarch64` is the image name and tag. Modify it based on your requirements. Run `docker images` on the host to view existing images.

   **Table 2** Parameters <a id="table1"></a>

   | Parameter | Description |
   |----|----|
   |-it|Starts an interactive terminal (-i) and connects it to the standard input/output (-t) of the container, allowing interaction with the container, such as running CLI operations.|
   |-d|Runs the container in the background (detached mode). This parameter does not block the current terminal, allowing you to continue other operations after starting the container.|
   |--ipc|Uses the host inter-process communication (IPC) namespace.|
   |--network|Uses the host network stack.|
   |--pid|Uses the host PID namespace. With this parameter, processes inside the container can view all process IDs on the host.|
   |--name|Specifies a name for the container. `mindspeed_llm` is the container identifier, which can be customized and must be unique on the current system. If not set, Docker automatically assigns a random name.|
   |--privileged|Removes the default permission restrictions of the container and grants the container almost host-level permissions, ensuring that Ascend driver invocation, `npu-smi`, and other tools can interact with hardware devices properly.|
   |--shm-size|Specifies the size of the shared memory (`/dev/shm`) of the container. You can set it as needed. `512g` is an example value.<br>This value must not exceed the remaining physical memory on the host. Run `free -h` to check.|
   |--device|Maps host devices to the container. Each `--device` parameter shares a host device (for example, a hardware acceleration card or other hardware) with the container for direct access.<ul><li>`/dev/davinci_manager`: Davinci-related management device.</li><li>`/dev/hisi_hdc`: HDC-related management device.</li><li>`/dev/devmm_svm`: Memory management-related device.</li><li>`/dev/davinci*X*`: NPU device, where *X* is the ID, for example, davinci0.</li></ul>Run `ll /dev/ \| grep davinci` to query the device count and names, and bind devices as needed by modifying `--device=****` in the command.|
   |-v|Maps a physical machine folder to the corresponding directory in the container. Modify the following parameters based on the actual paths.<ul><li>`/usr/local/Ascend/driver`: Contains hardware driver files. The driver is installed on the host and must be mapped to the container for use inside the container.</li><li>`/usr/local/Ascend/firmware`: Contains hardware firmware files. The firmware is installed on the host and must be mapped to the container for use inside the container.</li><li>`/usr/local/bin/npu-smi`: Contains the `npu-smi` and other NPU status query commands. Modify based on the actual path.</li><li>`/usr/local/dcmi`: Mount point for the dcmi tool.</li><li>`/usr/local/Ascend/driver/version.info`: Contains the driver version information file.</li><li>`/etc/ascend_install.info`: Contains the installation version information file.</li><li>`/data`: Mount path for datasets, pointing to the dataset directory for the container to access datasets.</li><li>`/weights`: Mount path for weights, pointing to the weight directory for the container to access weights.</li></ul>|

3. Load the container and verify the environment.

   ```bash
   # Query the ID/name of running containers locally
   docker ps -a
   # Load the container
   docker exec -it <container_ID/name> bash
   # Verify that the NPU is working
   npu-smi info
   ```

### Method 2: Installation from Source

Follow these steps to obtain the corresponding source code, install the required dependencies, and complete the installation of MindSpeed LLM.

1. (Optional) Create a virtual environment.

   Python 3.10 is recommended. For details, see the [Release Notes](../../release_notes_llm.md). To avoid affecting the global Python environment, you can use virtual environment management tools such as venv, conda, or uv to create an isolated virtual environment.

   Using conda as an example, run the following commands:

   ```bash
   conda create -n test python=3.10
   conda activate test
   ```

2. Install CANN.

   Install the matching versions of the NPU driver and firmware, and install the CANN software, including the Toolkit, ops, and NNAL packages, and configure the CANN environment variables. For details, see [CANN Software Installation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/softwareinst/instg/instg_0000.html).

   CANN software provides a script for setting process-level environment variables. Before you run application code with NPU acceleration in training or inference scenarios, you must call this script. Otherwise, the application code cannot run.

   ```shell
   # Default installation path for non-root users
   source ${HOME}/Ascend/cann/set_env.sh
   source ${HOME}/Ascend/nnal/atb/set_env.sh
   ```

   ```shell
   # Default installation path for the root user
   source /usr/local/Ascend/cann/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh
   ```

   The preceding commands use the default installation paths when no installation path is specified during offline CANN installation as an example, showing the default installation paths and corresponding environment variable configuration commands for different users.

   If you specify an installation path or use another installation method, see [CANN Software Installation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/softwareinst/instg/instg_0000.html) for instructions on configuring environment variables.

3. Install PyTorch and TorchNPU.

   Refer to [Install TorchNPU](https://www.hiascend.com/document/detail/zh/Pytorch/2610/installguide/swinstall/docs/zh/installation_guide/installation_via_binary_package.md) in the *TorchNPU Software Installation Guide* to obtain matching versions of the PyTorch and TorchNPU packages.

   You can use the following installation commands:

   ```shell
   pip3 install torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl
   pip3 install torch_npu-2.7.1post8-cp310-cp310-manylinux_2_28_aarch64.whl
   ```

   > [!NOTE]
   >
   > For more TorchNPU versions, click [Link](https://gitcode.com/ascend/pytorch/releases).

4. Install Triton-Ascend.

   Install the matching version of Triton-Ascend. For details, see [Quick Installation](https://triton-ascend.readthedocs.io/en/latest/installation_guide.html#quick-installation) in Triton-Ascend to obtain the installation command.

   You can use the following installation command:

   ```shell
   pip install triton-ascend==3.2.2 --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
   ```

   > [!NOTE]
   >
   > For Triton-Ascend 3.2.0 and earlier, Triton-Ascend and Triton cannot coexist. You need to uninstall the community Triton before installing Triton-Ascend.

5. Install the MindSpeed Core acceleration library.

   ```shell
   git clone https://gitcode.com/ascend/MindSpeed.git
   cd MindSpeed
   git checkout 26.1.0_core_r0.12.1  # Switch to the 26.1.0_core_r0.12.1 branch of MindSpeed Core
   pip3 install -r requirements.txt
   pip3 install -e .
   cd ..
   ```

6. Prepare the MindSpeed LLM and Megatron-LM source code.

   ```shell
   git clone https://gitcode.com/ascend/MindSpeed-LLM.git
   git clone https://github.com/NVIDIA/Megatron-LM.git  # Download Megatron-LM from GitHub. Ensure that the network is accessible
   cd Megatron-LM
   git checkout core_v0.12.1
   cp -r megatron ../MindSpeed-LLM/
   cd ../MindSpeed-LLM
   git checkout 26.1.0
   mkdir logs

   pip3 install -r requirements.txt  # Install the remaining dependency packages
   ```
