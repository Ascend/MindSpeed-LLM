# MindSpeed LLM安装指导

本文主要向用户介绍如何快速基于PyTorch框架完成MindSpeed LLM（大语言模型分布式训练套件）的安装。

## 硬件配套和支持的操作系统

**表 1**  产品硬件支持列表

|产品|是否支持|
|--|:-:|
|<term>Ascend 950 系列产品</term>|√|
|<term>Atlas A3 训练系列产品</term>|√|
|<term>Atlas A3 推理系列产品</term>|x|
|<term>Atlas A2 训练系列产品</term>|√|
|<term>Atlas A2 推理系列产品</term>|x|
|<term>Atlas 200I/500 A2 推理产品</term>|x|
|<term>Atlas 推理系列产品</term>|x|
|<term>Atlas 训练系列产品</term>|x|

> [!NOTE]
>
> 本节表格中“√”代表支持，“x”代表不支持。

- 各硬件产品对应物理机部署场景支持的操作系统请参考[兼容性查询助手](https://www.hiascend.com/hardware/compatibility)。
- 各硬件产品对应虚拟机及容器部署场景支持的操作系统请参考《CANN 软件安装》的“[操作系统兼容性说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0101.html?OS=openEuler&InstallType=netyum)”章节。

## 安装前准备

请参见《版本说明》中的“[相关产品版本配套说明](../../release_notes_llm.md#相关产品版本配套说明)”章节，下载安装对应的软件版本。

> [!NOTICE]
>
> 安装运行程序建议使用非root用户，且建议对安装程序的目录文件做好权限管控：文件夹权限设置为750，文件权限设置为640。可以通过设置umask控制安装后文件的权限，如设置umask为0027。
> 更多安全相关内容请参见《[安全声明](../../SECURITYNOTE.md)》中各组件关于“文件权限控制”的说明。

下载[固件与驱动](https://hiascend.com/hardware/firmware-drivers/community)，请根据系统和硬件产品型号选择对应版本的社区版本或商用版本的固件与驱动。

参考如下命令安装：

```shell
chmod +x Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run
chmod +x Ascend-hdk-<chip_type>-npu-firmware_<version>.run
./Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run --full --force
./Ascend-hdk-<chip_type>-npu-firmware_<version>.run --full
```

## 安装MindSpeed LLM

### 方式一：镜像安装

> [!NOTE]
>
> - 使用镜像前，请先确认机器型号。最新镜像仅支持aarch64架构，可通过uname -a命令确认当前环境是否符合要求。
> - 配套镜像已预装配套的CANN 9.0.0软件及Ascend Extension for PyTorch 26.0.0插件，您可根据需要选用。
> - 若您当前环境与提供的镜像不兼容，请选择[方式二：源码安装](#方式二源码安装)。
> - master分支后续会更新新的镜像，如果需要自定义构建镜像请参见[镜像概述](../../../../docker/OVERVIEW.zh.md)。

1. 获取镜像

   最新镜像均配套[MindSpeed LLM的26.0.0分支](https://gitcode.com/Ascend/MindSpeed-LLM/tree/26.0.0)，请按需[获取镜像](https://www.hiascend.com/developer/ascendhub/detail/e26da9266559438b93354792f25b2f4a)。

   - <term>Atlas A2 训练系列产品</term>：26.0.0-910b-openeuler24.03-py3.11-aarch64

   - <term>Atlas A3 训练系列产品</term>：26.0.0-a3-openeuler24.03-py3.11-aarch64

   ```bash
   # 确认是否成功获取镜像
   docker image list
   ```

2. 启动容器

   执行以下命令启动容器，该命令仅供参考，可根据需求自行修改，命令参数介绍如[表2](#table1)所示。

   ```bash
   docker run -it -d \
      --ipc=host \
      --network=host \
      --pid=host \
      --name mindspeed_llm \
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
   > - 当前默认配置驱动和固件安装在/usr/local/Ascend，如有差异请修改指令路径。
   > - 复制启动命令前，请将-v参数内的/data、/weights两处路径，替换为宿主机本地真实目录，否则容器启动失败。
   > - 当前容器默认初始化NPU驱动和CANN环境信息，如需要安装新的，请自行替换或手动source，详见容器的~/.bashrc。
   > - “_mindspeed-llm:26.0.0-a3-openeuler24.03-py3.11-aarch64_”为镜像名称和标签，可根据实际情况修改。可在宿主机执行`docker images`命令查看当前机器上已有的镜像。

   **表 2**  参数说明 <a id="table1"></a>

    |参数|说明|
    |----|----|
    |-it|表示启动一个交互式终端（-i）并将其连接到容器的标准输入输出 （-t），能够与容器内部进行交互，如运行命令行操作。|
    |-d|表示容器将以后台模式运行，即容器在后台启动。使用该参数后不会阻塞当前终端的操作，可以在启动容器后继续进行其他操作。|
    |--ipc|表示使用宿主机的IPC（进程间通信）命名空间。|
    |--network|表示使用宿主机的网络栈。|
    |--pid|表示使用宿主机的PID命名空间。使用该参数后容器内的进程可以查看宿主机上的所有进程ID。|
    |--name|表示给容器指定一个名称。mindspeed_llm是容器的标识符，可以自行设置，且在当前系统中具有唯一性。如果不设置，Docker会自动分配一个随机名称。|
    |--shm-size|表示指定容器的共享内存（/dev/shm）大小，用户可自行设置，512g为示例值。<br>该值不能超过宿主机剩余的物理内存总量，可使用`free -h`命令查看。|
    |--device|表示将宿主机的设备映射到容器内。每个--device参数将宿主机设备（例如硬件加速卡或其他硬件设备）共享给容器，以便容器可以直接访问。<br>/dev/davinci_manager：davinci相关的管理设备。<br>/dev/hisi_hdc：hdc相关管理设备。<br>/dev/devmm_svm：内存管理相关设备。<br>/dev/davinci*X*：NPU设备，*X*是ID号，如：davinci0。<br>可根据`ll /dev/ \| grep davinci`命令查询device个数及名称，根据需要绑定设备，修改上面命令中的"--device=****"。|
    |-v|表示将物理机的文件夹映射到容器内的相应目录，以下参数请根据实际路径修改。<br>/usr/local/Ascend/driver：该路径包含硬件驱动程序文件，驱动在宿主机上安装，将其映射到容器中，方可在容器中使用。<br>/usr/local/Ascend/firmware：该路径包含硬件固件程序文件，固件在宿主机上安装，将其映射到容器中，方可在容器中使用。<br>/usr/local/bin/npu-smi：该路径包含npu-smi等NPU状态查看命令，请根据实际路径修改。<br>/usr/local/dcmi：该路径用于挂载dcmi工具。<br>/usr/local/Ascend/driver/version.info：该路径包含驱动版本信息文件。<br>/etc/ascend_install.info：该路径包含安装版本信息文件。<br>/data：该路径为设定数据集挂载的路径，指向保存数据集的目录，使容器能访问数据集。<br>/weights：该路径为设定权重挂载的路径，指向保存权重的目录，使容器能访问权重。|

3. 加载容器并确认环境状态

   ```bash
   # 查询本地运行中的容器ID/名称
   docker ps -a
   # 加载容器
   docker exec -it 容器ID/名称 bash
   # 确认NPU是否可以正常使用
   npu-smi info
   ```

### 方式二：源码安装

请参考如下操作获取对应源码及安装相关依赖，完成MindSpeed LLM的安装。

1. （可选）创建虚拟环境

   建议使用Python 3.10版本，具体可参见[版本说明](../../release_notes_llm.md)。若不希望影响全局Python环境，可使用venv、conda、uv等常用虚拟环境管理工具创建独立的虚拟环境。

   以conda为例，可参考如下命令：

   ```bash
   conda create -n test python=3.10
   conda activate test
   ```

2. 安装CANN

   安装配套版本的NPU驱动固件、CANN软件（Toolkit、ops和NNAL）并配置CANN环境变量，具体请参考《[CANN 软件安装](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0000.html)》。

   CANN软件提供进程级环境变量设置脚本，训练或推理场景下使用NPU执行业务代码前需要调用该脚本，否则业务代码将无法执行。

   ```shell
   # 非root用户的默认安装路径
   source ${HOME}/Ascend/cann/set_env.sh
   source ${HOME}/Ascend/nnal/atb/set_env.sh
   ```

   ```shell
   # root用户默认安装路径
   source /usr/local/Ascend/cann/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh
   ```

   以上命令以CANN离线安装场景未指定安装路径为例，给出了不同用户的默认安装路径及对应的配置环境变量的命令。

   若用户指定安装路径或使用其他安装方式，请参考《[CANN 软件安装](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0000.html)》查阅对应的配置环境变量的命令。

3. 安装PyTorch以及torch_npu

   请参考《Ascend Extension for PyTorch 软件安装指南》中的“[安装PyTorch](https://www.hiascend.com/document/detail/zh/Pytorch/2600/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md)”章节，获取配套版本的PyTorch以及torch_npu软件包。

   可参考如下安装命令：

   ```shell
   # 安装torch和torch_npu构建参考 https://gitcode.com/ascend/pytorch/releases
   pip3 install torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl
   pip3 install torch_npu-2.7.1rc1-cp310-cp310-manylinux_2_28_aarch64.whl
   ```

4. 安装Triton-Ascend

   安装配套版本的Triton-Ascend，请参考《Triton-Ascend》中的"[通过pip安装Triton-Ascend](https://triton-ascend.readthedocs.io/zh-cn/latest/installation_guide.html#piptriton-ascend)"章节，获取配套版本的Triton-Ascend安装指令。

   可参考如下安装命令：

   ```shell
   pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
   ```

   > [!NOTE]
   >
   > Triton-Ascend 3.2.0及以下版本，Triton-Ascend和Triton不能同时存在。需先卸载社区Triton，再安装Triton-Ascend。

5. 安装MindSpeed加速库

    ```shell
    git clone https://gitcode.com/ascend/MindSpeed.git
    cd MindSpeed
    git checkout master  # 切换至MindSpeed的master分支
    pip3 install -r requirements.txt
    pip3 install -e .
    cd ..
    ```

6. 准备MindSpeed LLM及Megatron-LM源码

    ```shell
    git clone https://gitcode.com/ascend/MindSpeed-LLM.git
    git clone https://github.com/NVIDIA/Megatron-LM.git  # 从github下载Megatron-LM，请确保网络能访问
    cd Megatron-LM
    git checkout core_v0.12.1
    cp -r megatron ../MindSpeed-LLM/
    cd ../MindSpeed-LLM
    git checkout master
    mkdir logs

    pip3 install -r requirements.txt  # 安装其余依赖库
    ```
