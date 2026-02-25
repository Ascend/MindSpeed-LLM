## 安装指导

### 驱动固件安装

下载[驱动固件](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/softwareinst/instg/instg_0003.html?Mode=PmIns&OS=Debian&Software=cannToolKit)，请根据系统和硬件产品型号选择对应版本的`driver`和`firmware`。参考[安装NPU驱动固件](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/softwareinst/instg/instg_0005.html?Mode=PmIns&OS=Debian&Software=cannToolKit)或执行以下命令安装：

```shell
chmod +x Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run
chmod +x Ascend-hdk-<chip_type>-npu-firmware_<version>.run
./Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run --full --force
./Ascend-hdk-<chip_type>-npu-firmware_<version>.run --full
```

### CANN安装

下载[CANN](https://www.hiascend.com/developer/download/community/result?module=cann)，请根据系统选择`aarch64`或`x86_64`对应版本的`cann-toolkit`、`cann-ops`和`cann-nnal`。参考[CANN安装](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/softwareinst/instg/instg_0008.html?OS=Debian&Software=cannToolKit)或执行以下命令安装：

```shell
# 因为版本迭代，包名存在出入，根据实际修改
chmod +x Ascend-cann-toolkit_<version>_linux-<arch>.run
./Ascend-cann-toolkit_<version>_linux-<arch>.run --install
chmod +x Ascend-cann-<chip_type>-ops_<version>_linux-<arch>.run
./Ascend-cann-<chip_type>-ops_<version>_linux-<arch>.run --install
source /usr/local/Ascend/cann/set_env.sh # 修改为实际安装的Toolkit包路径
chmod +x Ascend-cann-nnal_<version>_linux-<arch>.run
./Ascend-cann-nnal_<version>_linux-<arch>.run --install
# 设置环境变量
source /usr/local/Ascend/cann/set_env.sh # 修改为实际安装的Toolkit包路径
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0 # 修改为实际安装的nnal包路径
```

### MindSpore安装

参考[MindSpore](https://gitcode.com/mindspore/mindspore#%E5%AE%89%E8%A3%85)完成MindSpore的安装。


### MindSpeed-LLM及相关依赖安装

```shell
# 使用环境变量
source /usr/local/Ascend/cann/set_env.sh # 修改为实际安装的Toolkit包路径
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0 # 修改为实际安装的nnal包路径

# 安装MindSpeed-Core-MS转换工具
git clone https://gitcode.com/ascend/MindSpeed-Core-MS.git -b master

# 使用MindSpeed-Core-MS内部脚本提供配置环境
cd MindSpeed-Core-MS
pip3 install -r requirements.txt # 安装第三方依赖库
source auto_convert.sh llm # 拉取训练所需组件库
source tests/scripts/set_path.sh # 设置环境变量
```

