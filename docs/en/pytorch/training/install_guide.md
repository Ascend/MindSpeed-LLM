# MindSpeed LLM Installation

This document explains how to quickly install MindSpeed LLM, the PyTorch-based distributed training toolkit for large language models.

## Hardware and Supported OSs

**Table 1** Product hardware support list

| Product | Supported |
|--|:-:|
| <term>Ascend 950PR&950DT products</term> | √ |
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
- For the OSs supported by each hardware product in VM and container deployment scenarios, see [OS Compatibility](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920/softwareinst/instg/instg_0101.html?OS=openEuler&InstallType=netyum) in *CANN Software Installation*.

## Preparation Before Installation

See [Related Product Version Mapping](../../release_notes_llm.md#related-product-version-mapping) in the *Release Notes* to download and install the corresponding software version.

Click [Firmware and Drivers](https://www.hiascend.com/hardware/firmware-drivers) and follow the guidance to install the firmware and drivers.

> [!NOTICE]
>
> You are advised to install and run the software as a non-root user. You are also advised to control permissions for the installer directory and files. Set directory permissions to 750 and file permissions to 640. You can control the permissions after installation by setting `umask`, for example `umask 0027`.
>
> For more security-related information, see the explanation of file permission control for each component in [Security Statement](../../SECURITYNOTE.md).

## Installing MindSpeed LLM

MindSpeed LLM supports image installation and installation from source.

The usage scenarios of each installation method are as follows. Choose the appropriate installation method based on your actual situation.

Image installation: This method is relatively simple. You can download a pre-built image directly from the Ascend community. The image already contains the necessary dependencies and software such as CANN, TorchNPU, and MindSpeed LLM. You only need to pull the image and start a container.

Installation from source: To try the latest features or modify and enhance the source code, you can download the MindSpeed LLM code of the corresponding branch and install it yourself.

Installation from source (unified installation using submodules): This optional method suits scenarios where you need to manage the versions of third-party dependencies such as MindSpeed in a unified manner.

### Method 1: Image Installation

> [!NOTE]
>
> - Before using the image, confirm the machine model. The latest image supports the AArch64 and X86_64 architectures. Run `uname -a` to verify that your environment meets the requirements.
> - The image will be pre-installed with CANN 9.2.0 and TorchNPU 26.2.0. You can use it as needed.
> - If your environment is incompatible with the provided image, choose [Method 2: Installation from Source](#method-2-installation-from-source).
> - The 26.2.0 branch will be updated with new images. For custom image building, see [Image Overview](../../../../docker/OVERVIEW.md).

1. Obtain the image.

   The latest images correspond to the [MindSpeed LLM 26.2.0 branch](https://gitcode.com/Ascend/MindSpeed-LLM/tree/26.2.0). This image will be released soon. For now, you can use the image corresponding to the MindSpeed LLM 26.1.0 branch. Click [Obtain the image](https://www.hiascend.com/developer/ascendhub/detail/e26da9266559438b93354792f25b2f4a) as needed.

   <!-- npu="950" id1 -->
   - <term>Ascend 950PR&950DT products</term>: `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-950-openeuler24.03-py3.12`

   - <term>Ascend 950PR&950DT products</term>: `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-950-ubuntu22.04-py3.12`
   <!-- end id1 -->
   <!-- npu="A3" id2 -->
   - <term>Atlas A3 training products</term>: `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.12`

   - <term>Atlas A3 training products</term>: `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-a3-ubuntu22.04-py3.12`
   <!-- end id2 -->
   <!-- npu="910b" id3 -->
   - <term>Atlas A2 training products</term>: `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-910b-openeuler24.03-py3.12`

   - <term>Atlas A2 training products</term>: `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-910b-ubuntu22.04-py3.12`
   <!-- end id3 -->

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
      mindspeed-llm:v26.1.0-cann9.1.0-torch_npu2.7.1.post8-910b-openeuler24.03-py3.12 \
      /bin/bash
   ```

   > [!NOTE]
   >
   > - By default, the driver and firmware are installed in `/usr/local/Ascend`. If the paths differ, modify the command accordingly.
   > - Before copying the startup command, replace the `/data` and `/weights` paths in the `-v` parameters with the actual local directories on the host. Otherwise, the container fails to start.
   > - The container initializes the NPU driver and CANN environment by default. To install new ones, replace or source them manually. See `~/.bashrc` in the container for details.
   > - `mindspeed-llm:v26.1.0-cann9.1.0-torch_npu2.7.1.post8-910b-openeuler24.03-py3.12` is the image name and tag. Modify it based on your requirements. Run `docker images` on the host to view existing images.

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
   |--shm-size|Specifies the size of the shared memory (`/dev/shm`) of the container. You can set it as needed. 512g is an example value.<br>This value must not exceed the remaining physical memory on the host. Run `free -h` to check.|
   |--device|Maps host devices to the container. Each `--device` parameter shares a host device (for example, a hardware acceleration device or other hardware) with the container for direct access.<ul><li>`/dev/davinci_manager`: Davinci-related management device.</li><li>`/dev/hisi_hdc`: HDC-related management device.</li><li>`/dev/devmm_svm`: Memory management-related device.</li><li>`/dev/davinci*X*`: NPU device, where `X` is the ID, for example, davinci0.</li></ul>Run `ll /dev/ \| grep davinci` to query the device count and names, and bind devices as needed by modifying `--device=****` in the command.|
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

   Python 3.12 is recommended. For details, see the [Release Notes](../../release_notes_llm.md). To avoid affecting the global Python environment, you can use virtual environment management tools such as venv, conda, or uv to create an isolated virtual environment.

   Using conda as an example, run the following commands:

   ```bash
   conda create -n test python=3.12
   conda activate test
   ```

2. Install CANN.

   Install the matching versions of the NPU driver and firmware, and install the CANN software, including the Toolkit, ops, and NNAL packages, and configure the CANN environment variables. For details, see [CANN Quick Installation](https://www.hiascend.com/en/cann/download?versionId=791&ids=d806%2Ch0501%2Ch0601%2Ch0703).

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

   If you specify an installation path or use another installation method, see [CANN Software Installation](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/920/softwareinst/instg/instg_0000.html) for instructions on configuring environment variables.

3. Install PyTorch and TorchNPU.

   Install the matching versions of PyTorch 2.10.0 and TorchNPU 26.2.0 by following the guidance. For details, see [TorchNPU Quick Installation](https://www.hiascend.com/en/developer/software/ai-frameworks/pytorch/download?versionId=175&ids=89dda9ba9de741349efa03687a487678%2C98%2C107%2C1%2C6%2C177%2C).

   > [!NOTE]
   >
   > - For more TorchNPU versions, click [Link](https://gitcode.com/ascend/pytorch/releases).
   > - For TorchNPU documentation, see the [TorchNPU Getting Started Guide](https://www.hiascend.com/document/detail/en/Pytorch/2620/index/index.html).

4. Install Triton-Ascend.

   Install the matching version of Triton-Ascend. For details, see the [Triton-Ascend Installation Guide](https://triton-ascend.readthedocs.io/en/latest/installation_guide.html#installation-guide) to obtain the installation command.

   You can use the following installation command:

   ```shell
   pip install triton-ascend==3.6.0 --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
   ```

   > [!NOTE]
   >
   > For Triton-Ascend 3.2.0 and earlier, Triton-Ascend and Triton cannot coexist. You need to uninstall the community Triton before installing Triton-Ascend.

5. Install the MindSpeed Core acceleration library.

   ```shell
   git clone https://gitcode.com/ascend/MindSpeed.git
   cd MindSpeed
   git checkout 26.2.0_core_r0.12.1  # Switch to the 26.2.0_core_r0.12.1 branch of MindSpeed Core
   pip3 install -r requirements.txt
   pip3 install -e .
   cd ..
   ```

6. Install the FSDPTurbo acceleration library.

   ```shell
   git clone https://gitcode.com/Ascend/FSDPTurbo.git
   cd FSDPTurbo
   git checkout 26.2.0   # Switch to the 26.2.0 branch of FSDPTurbo
   pip3 install -e .
   cd ..
   ```

7. Prepare the MindSpeed LLM and Megatron-LM source code.

   ```shell
   git clone https://gitcode.com/ascend/MindSpeed-LLM.git
   git clone https://github.com/NVIDIA/Megatron-LM.git  # Download Megatron-LM from GitHub. Ensure that the network is accessible
   cd Megatron-LM
   git checkout core_v0.12.1
   cp -r megatron ../MindSpeed-LLM/
   cd ../MindSpeed-LLM
   git checkout 26.2.0
   mkdir logs

   pip3 install -r requirements.txt  # Install the remaining dependency packages
   ```

### Method 3: Installation from Source (Unified Installation Using Submodules)

> [!NOTE]
>
> - With this method, you can complete the steps of pulling each dependency repository and switching versions in [Method 2: Installation from Source](#method-2-installation-from-source) with a single command.
> - If you have manually cloned the related repositories using Method 2, delete the corresponding directories before using this method to avoid directory conflicts.

MindSpeed LLM supports unified management of third-party dependency repositories through Git submodules, which facilitates version tracing and collaborative development. The submodules are placed in the `3rdparty/` directory:

- **MindSpeed**: Pulls the latest code from the remote 26.2.0_core_r0.12.1 branch by default.
- **FSDPTurbo**: Pulls the latest code from the remote 26.2.0 branch by default.
- **Megatron-LM**: Is pinned to the tag `core_v0.12.1`.

1. Prepare the basic environment.

   Prepare the basic environment, including the virtual environment, CANN, TorchNPU, and Triton-Ascend. For details, see steps 1 to 4 in [Method 2: Installation from Source](#method-2-installation-from-source).

2. Download the source code.

   After you clone the MindSpeed LLM source code, initialize and pull all submodules:

   ```shell
   git clone -b 26.2.0 https://gitcode.com/ascend/MindSpeed-LLM.git
   cd MindSpeed-LLM
   git submodule update --init --recursive
   git submodule update --init --remote 3rdparty/MindSpeed 3rdparty/FSDPTurbo
   ```

   > [!NOTE]
   >
   > To update the submodules, perform the following operations:
   > - MindSpeed and FSDPTurbo track the latest remote versions by default. They are refreshed during initialization. To pull the latest commits later, run `git submodule update --init --remote 3rdparty/MindSpeed 3rdparty/FSDPTurbo`. (Commits in the `3rdparty` directory can be ignored.)
   > - Megatron-LM is pinned to the corresponding version. Running `git submodule update --init` always checks out this version and does not update it automatically. To switch versions, see the following section [(Optional) Switch the submodule version](#checkout_submodules).

   The source code structure is as follows:

   ```shell
   MindSpeed-LLM/
   ├── 3rdparty/
   │   ├── MindSpeed/          # Acceleration library (corresponds to step 5 of Method 2)
   │   ├── FSDPTurbo/          # FSDP acceleration library (corresponds to step 6 of Method 2)
   │   └── Megatron-LM/        # Megatron-LM core code (corresponds to step 7 of Method 2)
   └── ...
   ```

3. Install the dependency submodules.

   After pulling the submodules, install the dependencies of each component in sequence:

   ```shell
   # Install MindSpeed dependencies
   pip3 install -r 3rdparty/MindSpeed/requirements.txt

   # Create symbolic links to the dependency repository source code in the MindSpeed LLM root directory
   ln -s 3rdparty/MindSpeed/mindspeed mindspeed
   ln -s 3rdparty/FSDPTurbo/fsdp_turbo fsdp_turbo
   ln -s 3rdparty/Megatron-LM/megatron megatron

   # Install the remaining MindSpeed LLM dependencies
   pip3 install -r requirements.txt
   ```

   > [!NOTE]
   >
   > - MindSpeed and FSDPTurbo are installed through symbolic links. Therefore, `pip3 install -e .` is not required. After the submodule code is updated, the symbolic links take effect automatically, and no reinstallation is required.
   > - The Megatron-LM submodule contains the complete repository. Link the `megatron/` source directory to the MindSpeed LLM root directory before use.
   > - To install MindSpeed or FSDPTurbo with `pip3 install -e .`, go to the corresponding submodule directory and run the command.

4. Create the log directory.

   Go to the MindSpeed LLM code root directory and create a log directory.

   ```shell
   cd ../MindSpeed-LLM
   mkdir logs
   ```

5. (Optional) Switch the submodule version.<div id="checkout_submodules"/>

   To switch a submodule to a specified branch or tag, for example, to switch Megatron-LM to another version:

   ```shell
   cd 3rdparty/Megatron-LM
   git checkout <tag or branch name>
   cd ../..
   ```

   To update MindSpeed or FSDPTurbo to the latest remote version:

   ```shell
   cd 3rdparty/MindSpeed
   git pull origin master
   cd ../..
   ```

   To commit version switching changes to the repository:

   ```shell
   git add 3rdparty/<name of the changed repository>
   git commit -m "chore: pin <name of the changed repository> to <version>"
   ```
