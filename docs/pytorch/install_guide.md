# 安装指导

# 1.依赖配套总览

MindSpeed LLM的依赖配套如下表

<table>
  <tr>
    <th>依赖软件</th>
    <th>版本</th>
  </tr>
  <tr>
    <td>昇腾NPU驱动</td>
    <td rowspan="2">Ascend HDK 25.3.0</td>
  <tr>
    <td>昇腾NPU固件</td>
  </tr>
  <tr>
    <td>Toolkit（开发套件）</td>
      <td rowspan="3">CANN 8.3.RC1</td>
  </tr>
  <tr>
    <td>Kernel（算子包）</td>
  </tr>
  <tr>
    <td>NNAL（Ascend Transformer Boost加速库）</td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td>Python</td>
    <td>3.10</td>
  </tr>
  <tr>
    <td>PyTorch</td>
    <td>2.7.1</td>
  </tr>
  <tr>
    <td>torch_npu插件</td>
    <td rowspan="2">7.2.0</td>
  </tr>
</table>

# 2.依赖安装指导

## 2.1驱动固件安装

下载[驱动固件](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=26&cann=8.0.RC3.beta1&driver=1.0.27.alpha)，请根据系统和硬件产品型号选择对应版本的`driver`和`firmware`。参考[安装NPU驱动固件](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/softwareinst/instg/instg_0005.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)或执行以下命令安装：

```shell
chmod +x Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run
chmod +x Ascend-hdk-<chip_type>-npu-firmware_<version>.run
./Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run --full --force
./Ascend-hdk-<chip_type>-npu-firmware_<version>.run --full
```

## 2.2 安装方式选择

### 2.2.1 使用配套镜像安装

使用镜像前请先确定机器型号，最新镜像只支持aarch64系统，先通过uname -a确认自身系统是aarch64，Atlas 800T A2 训练服务器请选择openeuler22.03-mindspeed-llm-2.2.0-a2-arm 镜像，Atlas 900 A3 SuperPoD服务器请选择openeuler22.03-mindspeed-llm-2.2.0-a3-arm 镜像。
本仓库在昇腾社区提供aarch64系统的配套镜像，且镜像已安装CANN 8.3.RC1和配套torch_npu插件，您可以按需使用。
如果您的环境与提供的镜像不兼容，也可以[使用自定义环境安装](#222-使用自定义环境安装)。

**1.拉取镜像**

- 最新镜像区分A2、A3机型，均配套MindSpeed-LLM的2.2.0分支。
  - openeuler22.03-mindspeed-llm-2.2.0-a2-arm 镜像版本匹配 [MindSpeed-LLM的2.2.0分支](https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0)
  
  - openeuler22.03-mindspeed-llm-2.2.0-a3-arm 镜像版本匹配 [MindSpeed-LLM的2.2.0分支](https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0)

- 按需[拉取镜像](https://www.hiascend.com/developer/ascendhub/detail/e26da9266559438b93354792f25b2f4a)。

```bash
# 确认是否成功拉取镜像
docker image list
```

**2.创建容器**

注意当前默认配置驱动和固件安装在/usr/local/Ascend，如有差异请修改指令路径。
当前容器默认初始化npu驱动和CANN环境信息，如需要安装新的，请自行替换或手动source，详见容器的~/.bashrc

```bash
# 挂载镜像
docker run -dit --ipc=host --network host --name 'llm_test' --privileged -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware -v /usr/local/sbin/:/usr/local/sbin/ -v /home/:/home/ -v /data/:/data 镜像名:标签 /bin/bash
```

**3.进入容器并确认环境状态**

```bash
# 进入容器
docker exec -it llm_test bash
# 确认npu是否可以正常使用，否则返回3.检查配置
npu-smi info
```

**4.单机以及多机模型的预训练任务运行**

进入容器后会进入conda环境llm_test，并且工作目录切换到/MindSpeed-LLM/MindSpeed-LLM,该目录下为MindSpeed-LLM的2.2.0分支代码仓，用户可直接参考[基于PyTorch后端的预训练](https://gitcode.com/Ascend/MindSpeed-LLM/blob/2.2.0/docs/quick_start.md#3-%E5%9F%BA%E4%BA%8Epytorch%E5%90%8E%E7%AB%AF%E7%9A%84%E9%A2%84%E8%AE%AD%E7%BB%83)进行训练。

### 2.2.2 使用自定义环境安装

**1.安装CANN**

下载[CANN](https://www.hiascend.com/developer/download/community/result?module=cann)，请根据系统选择`aarch64`或`x86_64`对应版本的`cann-toolkit`、`cann-kernel`和`cann-nnal`。参考[CANN安装](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)或执行以下命令安装：

```shell
# 因为版本迭代，包名存在出入，根据实际修改
chmod +x Ascend-cann-toolkit_<version>_linux-<arch>.run
./Ascend-cann-toolkit_<version>_linux-<arch>.run --install
chmod +x Ascend-cann-kernels-<chip_type>_<version>_linux.run
./Ascend-cann-kernels-<chip_type>_<version>_linux.run --install
source /usr/local/Ascend/ascend-toolkit/set_env.sh # 安装nnal包需要source环境变量
chmod +x Ascend-cann-nnal-<chip_type>_<version>_linux.run
./Ascend-cann-nnal-<chip_type>_<version>_linux.run --install
# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

**2.安装PyTorch框架**

准备[torch_npu](https://www.hiascend.com/developer/download/community/result?module=pt)，参考[Ascend Extension for PyTorch 安装](https://www.hiascend.com/document/detail/zh/Pytorch/710/configandinstg/instg/insg_0001.html)或执行以下命令安装：

```shell
# 安装torch和torch_npu 构建参考 https://gitcode.com/ascend/pytorch/releases
pip3 install torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl 
pip3 install torch_npu-2.7.1rc1-cp310-cp310-manylinux_2_28_aarch64.whl
```

**3.安装MindSpeed-LLM及相关依赖**

```shell
# 使能环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 安装MindSpeed加速库
git clone https://gitcode.com/ascend/MindSpeed.git
cd MindSpeed
git checkout 2.2.0_core_r0.12.1 
pip3 install -r requirements.txt 
pip3 install -e .
cd ..

# 准备MindSpeed-LLM及Megatron-LM源码
git clone https://gitcode.com/ascend/MindSpeed-LLM.git 
git clone https://github.com/NVIDIA/Megatron-LM.git  # megatron从github下载，请确保网络能访问
cd Megatron-LM
git checkout core_v0.12.1
cp -r megatron ../MindSpeed-LLM/
cd ../MindSpeed-LLM
git checkout 2.2.0

pip3 install -r requirements.txt  # 安装其余依赖库
```

**4.单机以及多机模型的预训练任务运行**

参考[基于PyTorch后端的预训练](https://gitcode.com/Ascend/MindSpeed-LLM/blob/2.2.0/docs/quick_start.md#3-%E5%9F%BA%E4%BA%8Epytorch%E5%90%8E%E7%AB%AF%E7%9A%84%E9%A2%84%E8%AE%AD%E7%BB%83)进行训练。