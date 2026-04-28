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
> 本节表格中“√”代表支持，“x”代表不支持。

- 各硬件产品对应物理机部署场景支持的操作系统请参考[兼容性查询助手](https://www.hiascend.com/hardware/compatibility)。
- 各硬件产品对应虚拟机及容器部署场景支持的操作系统请参考《CANN 软件安装》的“[操作系统兼容性说明](https://www.hiascend.com/document/detail/zh/canncommercial/900/softwareinst/instg/instg_0101.html?OS=openEuler&InstallType=netyum)”章节（商用版）或“[操作系统兼容性说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0101.html?OS=openEuler&InstallType=netyum)”章节（社区版）。

## 安装前准备

请参见《版本说明》中的“[相关产品版本配套说明](../../release_notes_llm.md#相关产品版本配套说明)”章节，下载安装对应的软件版本。

### 安装驱动固件

下载[固件与驱动](https://hiascend.com/hardware/firmware-drivers/community)，请根据系统和硬件产品型号选择对应版本的社区版本或商用版本的固件与驱动。
参考如下命令安装：

```shell
chmod +x Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run
chmod +x Ascend-hdk-<chip_type>-npu-firmware_<version>.run
./Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run --full --force
./Ascend-hdk-<chip_type>-npu-firmware_<version>.run --full
```

### 安装CANN

请参考《[CANN 快速安装](https://www.hiascend.com/cann/download)》安装CANN软件（包含Toolkit、ops和NNAL包），并配置环境变量。

```shell
# 设置环境变量
source /usr/local/Ascend/cann/set_env.sh     # 修改为实际安装的Toolkit包路径
source /usr/local/Ascend/nnal/atb/set_env.sh # 修改为实际安装的nnal包路径
```

> [!NOTICE]  
> 建议使用非root用户安装运行torch\_npu，且建议对安装程序的目录文件做好权限管控：文件夹权限设置为750，文件权限设置为640。可以通过设置umask控制安装后文件的权限，如设置umask为0027。
> 更多安全相关内容请参见《[安全声明](../../SECURITYNOTE.md)》中各组件关于“文件权限控制”的说明。

## 安装PyTorch以及torch_npu

请参考《Ascend Extension for PyTorch 软件安装指南》中的“[安装PyTorch框架](https://www.hiascend.com/document/detail/zh/Pytorch/2600/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md)”章节，获取配套版本的PyTorch以及torch_npu软件包。
可参考如下安装命令：

```shell
# 安装torch和torch_npu构建参考 https://gitcode.com/ascend/pytorch/releases
pip3 install torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl 
pip3 install torch_npu-2.7.1post4-cp310-cp310-manylinux_2_28_aarch64.whl
```

## 安装MindSpeed LLM

请参考如下操作获取对应源码及安装相关依赖，完成MindSpeed LLM的安装。

1. 使能环境变量。

    ```shell
    # 请根据实际路径进行替换
    source /usr/local/Ascend/cann/set_env.sh 
    source /usr/local/Ascend/nnal/atb/set_env.sh 
    ```

2. 安装MindSpeed加速库。

    ```shell
    git clone https://gitcode.com/ascend/MindSpeed.git
    cd MindSpeed
    git checkout 26.0.0_core_r0.12.1  # 切换分支至MindSpeed 26.0.0_core_r0.12.1
    pip3 install -r requirements.txt 
    pip3 install -e .
    cd ..
    ```

3. 准备MindSpeed LLM及Megatron-LM源码。

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
