## 安装指导

请参考首页[版本说明](./gitcode.com/ascend/MindSpeed-LLM/tree/master/#%E7%89%88%E6%9C%AC%E8%AF%B4%E6%98%8E)选择下载对应依赖版本。

>注意：<br>
> 1.torch2.6及以上版本不支持python3.8，请优先使用python3.10。<br>
> 2.qwen3，glm45-moe系列模型要求高版本transformers，因此需要使用python3.10及以上版本。<br>


### 驱动固件安装

下载[驱动固件](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=26&cann=8.0.RC3.beta1&driver=1.0.27.alpha)，请根据系统和硬件产品型号选择对应版本的`driver`和`firmware`。参考[安装NPU驱动固件](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/softwareinst/instg/instg_0005.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)或执行以下命令安装：

```shell
chmod +x Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run
chmod +x Ascend-hdk-<chip_type>-npu-firmware_<version>.run
./Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run --full --force
./Ascend-hdk-<chip_type>-npu-firmware_<version>.run --full
```

### CANN安装

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

### PTA安装

准备[torch_npu](https://www.hiascend.com/developer/download/community/result?module=pt)，执行以下命令安装或参考[Ascend Extension for PyTorch 配置与安装](https://www.hiascend.com/document/detail/zh/Pytorch/710/configandinstg/instg/insg_0001.html)：

```shell
# 安装torch和torch_npu 构建参考 https://gitcode.com/ascend/pytorch/releases
pip3 install torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl 
pip3 install torch_npu-2.7.1rc1-cp310-cp310-manylinux_2_28_aarch64.whl
```

### MindSpeed-LLM及相关依赖安装

```shell
# 使能环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 安装MindSpeed加速库
git clone https://gitcode.com/ascend/MindSpeed.git
cd MindSpeed
git checkout master  # checkout commit from MindSpeed master
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
git checkout master

pip3 install -r requirements.txt  # 安装其余依赖库
```


> 注意:<br>
> 1.qwen3,llama3.3系列模型依赖transformers 4.51.0, 需要在环境配置完成后手动执行pip install transformers==4.51.0;<br>
> 2.glm4.5-moe系列模型依赖transformers 4.54.0, 需要在环境配置完成后手动执行pip install transformers==4.54.0;<br>
