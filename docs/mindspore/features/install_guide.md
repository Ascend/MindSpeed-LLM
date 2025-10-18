## 安装指导

### 驱动固件安装

下载[驱动固件](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=26&cann=8.2.RC2&driver=Ascend+HDK+25.2.0)，请根据系统和硬件产品型号选择对应版本的`driver`和`firmware`。参考[安装NPU驱动固件](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/softwareinst/instg/instg_0005.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit)或执行以下命令安装：

```shell
chmod +x Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run
chmod +x Ascend-hdk-<chip_type>-npu-firmware_<version>.run
./Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run --full --force
./Ascend-hdk-<chip_type>-npu-firmware_<version>.run --full
```

### CANN安装

下载[CANN](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC2)，请根据根据系统选择`aarch64`或`x86_64`对应版本的`cann-toolkit`、`cann-kernel`和`cann-nnal`。参考[CANN安装](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=local&OS=Debian&Software=cannToolKit)或执行以下命令安装：

```shell
# 因为版本迭代，包名存在出入，根据实际修改
chmod +x Ascend-cann-toolkit_<version>_linux-<arch>.run
./Ascend-cann-toolkit_<version>_linux-<arch>.run --install
chmod +x Ascend-cann-kernels-<chip_type>_<version>_linux-<arch>.run
./Ascend-cann-kernels-<chip_type>_<version>_linux-<arch>.run --install
source /usr/local/Ascend/ascend-toolkit/set_env.sh # 安装nnal包需要source环境变量
chmod +x Ascend-cann-nnal-<chip_type>_<version>_linux-<arch>.run
./Ascend-cann-nnal-<chip_type>_<version>_linux-<arch>.run --install
# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0
```

### MindSpore安装

参考[mindspore](https://gitee.com/mindspore/mindspore#%E5%AE%89%E8%A3%85)完成mindspore的安装


### MindSpeed-LLM及相关依赖安装

```shell
# 使用环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0

# 安装MindSpeed-Core-MS转换工具
git clone https://gitcode.com/ascend/MindSpeed-Core-MS.git -b r0.3.0

# 使用MindSpeed-Core-MS内部脚本提供配置环境
cd MindSpeed-Core-MS
pip install -r requirements.txt
source auto_convert_llm.sh
```

