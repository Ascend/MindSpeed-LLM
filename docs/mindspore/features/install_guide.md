## 安装指导

### 驱动固件安装

下载[驱动固件](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=26&cann=8.0.RC3.beta1&driver=1.0.27.alpha)，请根据系统和硬件产品型号选择对应版本的`driver`和`firmware`。参考[安装NPU驱动固件](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/softwareinst/instg/instg_0005.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)或执行以下命令安装：

```shell
bash Ascend-hdk-910b-npu-firmware_7.5.0.1.129.run --full
bash Ascend-hdk-910b-npu-driver_24.1.rc3_linux-aarch64.run --full --force
```

### CANN安装

下载[CANN](https://www.hiascend.com/developer/download/community/result?module=cann)，请根据根据系统选择`aarch64`或`x86_64`对应版本的`cann-toolkit`、`cann-kernel`和`cann-nnal`。参考[CANN安装](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)或执行以下命令安装：

```shell
# 因为版本迭代，包名存在出入，根据实际修改
bash Ascend-cann-toolkit_8.1.RC1_linux-aarch64.run --full
bash Ascend-cann-kernels-910b_8.1.RC1_linux-aarch64.run --install
source /usr/local/Ascend/ascend-toolkit/set_env.sh # 安装nnal包需要source环境变量
bash Ascend-cann-nnal_8.1.RC1_linux-aarch64.run --install
# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxxabi=0
```

### MindSpore安装

参考[mindspore](https://gitee.com/mindspore/mindspore#%E5%AE%89%E8%A3%85)完成mindspore的安装


### MindSpeed-LLM及相关依赖安装

```shell
# 使用环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxxabi=0

# 安装MindSpeed-Core-MS转换工具
git clone https://gitee.com/ascend/MindSpeed-Core-MS.git -b master

# 使用MindSpeed-Core-MS内部脚本提供配置环境
cd MindSpeed-Core-MS

source auto_convert_llm.sh
```

