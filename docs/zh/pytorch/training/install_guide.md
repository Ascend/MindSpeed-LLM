# MindSpeed LLM安装指导

本文主要向用户介绍如何快速基于PyTorch框架完成MindSpeed LLM（大语言模型分布式训练套件）的安装。

## 硬件配套和支持的操作系统

**表 1**  产品硬件支持列表

|产品|是否支持|
|--|:-:|
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
- 各硬件产品对应虚拟机及容器部署场景支持的操作系统请参考《CANN 软件安装》的“[操作系统兼容性说明](https://www.hiascend.com/document/detail/zh/canncommercial/900/softwareinst/instg/instg_0101.html?OS=openEuler&InstallType=netyum)”章节（商用版）或“[操作系统兼容性说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0101.html?OS=openEuler&InstallType=netyum)”章节（社区版）。

## 安装前准备

请参见《版本说明》中的“[相关产品版本配套说明](../../release_notes_llm.md#相关产品版本配套说明)”章节，下载安装对应的软件版本。

> [!NOTICE]
>
> 安装运行程序建议使用非root用户，且建议对安装程序的目录文件做好权限管控：文件夹权限设置为750，文件权限设置为640。可以通过设置umask控制安装后文件的权限，如设置umask为0027。更多安全相关内容请参见《[安全声明](../../SECURITYNOTE.md)》中各组件关于“文件权限控制”的说明。

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

1. 拉取镜像

   最新镜像均配套[MindSpeed LLM的26.0.0分支](https://gitcode.com/Ascend/MindSpeed-LLM/tree/26.0.0)，请按需[拉取镜像](https://www.hiascend.com/developer/ascendhub/detail/e26da9266559438b93354792f25b2f4a)。

   - <term>Atlas A2 训练系列产品</term>：26.0.0-910b-openeuler24.03-py3.11-aarch64

   - <term>Atlas A3 训练系列产品</term>：26.0.0-a3-openeuler24.03-py3.11-aarch64

   ```bash
   # 确认是否成功拉取镜像
   docker image list
   ```

2. 创建容器

   ```bash
   # 挂载镜像
   docker run -dit --ipc=host --network host --name '容器名' --privileged -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware -v /usr/local/sbin/:/usr/local/sbin/ -v /home/:/home/ -v /data/:/data 镜像名:标签 /bin/bash
   ```

   当前默认配置驱动和固件安装在/usr/local/Ascend，如有差异请修改指令路径。

   当前容器默认初始化NPU驱动和CANN环境信息，如需要安装新的，请自行替换或手动source，详见容器的~/.bashrc。

   - 示例一：基本运行

      ```bash
      docker run -it --rm \
          mindspeed-llm:26.0.0-a3-openeuler24.03-py3.11-aarch64 bash
      ```

   - 示例二：使用 NPU 设备运行（示例：设备 /dev/davinci1）

      ```bash
      # 根据实际情况修改 ascend-toolkit 路径
      # 假设 NPU 设备安装在 /dev/davinci1 上，并且 NPU 驱动程序安装在 /usr/local/Ascend 上
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

   - 示例三：挂载数据目录运行（示例：设备 /dev/davinci1）

      ```bash
      # 根据实际情况修改 ascend-toolkit 路径
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
         -v /path/to/data:/data \
         -v /path/to/weights:/weights \
         mindspeed-llm:26.0.0-a3-openeuler24.03-py3.11-aarch64 bash
      ```

3. 加载容器并确认环境状态

   ```bash
   # 加载容器
   docker exec -it 容器名 bash
   # 确认NPU是否可以正常使用
   npu-smi info
   ```

### 方式二：源码安装

1. 安装CANN

   安装配套版本的NPU驱动固件、CANN软件（Toolkit、ops和NNAL）并配置CANN环境变量，具体请参考《[CANN 软件安装](https://www.hiascend.com/document/detail/zh/canncommercial/900/softwareinst/instg/instg_0000.html)》（商用版）或《[CANN 软件安装](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0000.html)》（社区版）。

   CANN软件提供进程级环境变量设置脚本，训练或推理场景下使用NPU执行业务代码前需要调用该脚本，否则业务代码将无法执行。

   ```shell
   source /usr/local/Ascend/cann/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh
   ```

   以上命令以root用户安装后的默认路径为例，请用户根据set_env.sh的实际路径进行替换。

2. 安装PyTorch以及torch_npu

   请参考《Ascend Extension for PyTorch 软件安装指南》中的“[安装PyTorch框架](https://www.hiascend.com/document/detail/zh/Pytorch/2600/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md)”章节，获取配套版本的PyTorch以及torch_npu软件包。
   可参考如下安装命令：

   ```shell
   # 安装torch和torch_npu构建参考 https://gitcode.com/ascend/pytorch/releases
   pip3 install torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl
   pip3 install torch_npu-2.7.1post4-cp310-cp310-manylinux_2_28_aarch64.whl
   ```

3. 安装MindSpeed加速库

    ```shell
    git clone https://gitcode.com/ascend/MindSpeed.git
    cd MindSpeed
    git checkout 26.0.0_core_r0.12.1  # 切换分支至MindSpeed 26.0.0_core_r0.12.1
    pip3 install -r requirements.txt
    pip3 install -e .
    cd ..
    ```

4. 准备MindSpeed LLM及Megatron-LM源码

    ```shell
    git clone https://gitcode.com/ascend/MindSpeed-LLM.git
    git clone https://github.com/NVIDIA/Megatron-LM.git  # 从github下载Megatron-LM，请确保网络能访问
    cd Megatron-LM
    git checkout core_v0.12.1
    cp -r megatron ../MindSpeed-LLM/
    cd ../MindSpeed-LLM
    git checkout 26.0.0
    mkdir logs

    pip3 install -r requirements.txt  # 安装其余依赖库
    ```
