#  StarCoder模型-双芯推理指导

- [概述](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/foundation_models/LLaMA-1/13b#概述)
- [输入输出数据](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/foundation_models/LLaMA-1/13b#输入输出数据)
- [推理环境准备](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/foundation_models/LLaMA-1/13b#推理环境准备)
- [快速上手](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/foundation_models/LLaMA-1/13b#快速上手)
  - [获取源码及依赖](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/foundation_models/LLaMA-1/13b#获取源码及依赖)
  - [模型推理](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/foundation_models/LLaMA-1/13b#模型推理)
- [模型推理性能](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/foundation_models/LLaMA-1/13b#模型推理性能)

# 概述

StarCoder模型是在The Stack (v1.2)的80+种编程语言上训练的15.5B参数模型，不包括选择退出请求。该模型使用多查询注意力，一个包含8192个令牌的上下文窗口，并在1万亿个令牌上使用填充中间目标进行训练。

- 参考实现：

  ```
  https://huggingface.co/bigcode/starcoder
  ```

# 输入输出数据

- 输入数据

  | 输入数据       | 大小                               | 数据类型 | 数据排布格式 | 是否必选 |
  | -------------- | ---------------------------------- | -------- | ------------ | -------- |
  | input_ids      | BATCH_SIZE x SEQ_LEN               | INT64    | ND           | 是       |
  | attention_mask | BATCH_SIZE x 1 x SEQ_LEN x SEQ_LEN | FLOAT32  | ND           | 否       |

- 输出数据

  | 输出数据   | 大小                        | 数据类型 | 数据排布格式 |
  | ---------- | --------------------------- | -------- | ------------ |
  | output_ids | BATCH_SIZE x OUTPUT_SEQ_LEN | INT64    | ND           |

# 推理环境准备

该模型需要以下插件与驱动

**表 1** 版本配套表

| 配套           | 版本   | 下载链接 |
| -------------- | ------ | -------- |
| 固件与驱动     |        | -        |
| CANN           |        | -        |
| Python         | 3.8.16 | -        |
| PytorchAdapter | 2.0.1  | -        |
| 推理引擎       | -      | -        |

**表 2** 推理引擎依赖

| 软件  | 版本要求 |
| ----- | -------- |
| glibc | >= 2.27  |
| gcc   | >= 7.5.0 |

**表 3** 硬件形态

| CPU     | Device   |
| ------- | -------- |
| aarch64 | 300I DUO |
| x86     | 300I DUO |
| aarch64 | 800I A2  |
| x86     | 300I A2  |
# 快速上手

## 获取源码及依赖

1. 环境部署

- 1.1. 安装HDK

> 先安装firmwire，再安装driver

  1.1.1. 安装firmwire

  安装方法:

| 包名                                             |
|------------------------------------------------|
| Ascend-hdk-xxxx-npu-firmware_7.0.0.5.242.run |

  ```bash
  # 安装firmwire
  chmod +x Ascend-hdk-xxxx-npu-firmware_7.0.0.5.242.run
  ./Ascend-hdk-xxxx-npu-firmware_7.0.0.5.242.run --full
  ```

  1.1.2. 安装driver

  安装方法：

| cpu     | 包名                                                         |
|---------|------------------------------------------------------------|
| aarch64 | Ascend-hdk-xxxx-npu-driver_23.0.rc3.b082_linux-aarch64.run |
| x86     | Ascend-hdk-xxxx-npu-driver_23.0.rc3.b082_linux-x86-64.run |

  ```bash
  # 根据CPU架构安装对应的 driver
  chmod +x Ascend-hdk-xxxx-npu-driver_23.0.rc3.b082_*.run
  ./Ascend-hdk-xxxx-npu-driver_23.0.rc3.b082_*.run --full
  ```

- 1.2. 安装CANN

> 先安装toolkit 再安装kernel

  1.2.1. 安装toolkit

  安装方法：

| cpu     | 包名                                                         |
|---------|------------------------------------------------------------|
| aarch64 | Ascend-cann-toolkit_7.0.RC1_linux-aarch64.run |
| x86     | Ascend-cann-toolkit_7.0.RC1_linux-x86_64.run |

  ```bash
  # 安装toolkit
  chmod +x Ascend-cann-toolkit_7.0.RC1_linux-*.run
  ./Ascend-cann-toolkit_7.0.RC1_linux-*.run --install
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ```
  1.2.2. 安装kernel

  安装方法：

| 包名                                         |
|--------------------------------------------|
| Ascend-cann-kernels-{型号}_7.0.RC1_linux.run |

  ```bash
  # 安装 kernel
  chmod +x Ascend-cann-kernels-{型号}_7.0.RC1_linux.run
  ./Ascend-cann-kernels-{型号}_7.0.RC1_linux.run --install
  ```

- 1.3. 安装PytorchAdapter

> 安装apex、torch、torch_npu

  1.3.1 安装torch

  安装方法：

| cpu     | 包名                                       |
| ------- | ------------------------------------------ |
| aarch64 | torch-2.0.1-cp38-cp38-linux_aarch64.whl    |
| x86     | torch-2.0.1+cpu-cp38-cp38-linux_x86_64.whl |

  根据所使用的环境中的python版本，选择torch-1.11.0相应的安装包。

  ```bash
  # 安装torch 2.0.1 的python 3.8 的arm版本为例
  pip install torch-2.0.1-cp38-cp38-linux_aarch64.whl
  ```

  1.3.2 安装torch_npu

  安装方法：

| 包名                       |
| -------------------------- |
| pytorch_v2.0.1_py38.tar.gz |

> 安装选择与torch版本 以及 python版本 一致的torch_npu版本

  ```bash
  # 安装 torch_npu 以torch 1.11.0 的python 3.9的arm版本为例
  tar -zxvf pytorch_v2.0.1_py38.tar.gz
  pip install torch*_aarch64.whl
  ```

## 推理环境准备

> 安装配套软件。安装python依赖。

  ```
  pip3 install -r requirements.txt
  ```

1. 下载starcoder模型权重，放置到自定义`input_dir`

   ```
   https://huggingface.co/bigcode/starcoder
   ```
   
   
   
3. 根据版本发布链接，安装加速库 

   | 加速库包名                                            |
   | ----------------------------------------------------- |
   | Ascend-mindie-atb_{version}_cxx11abi0_linux-aarch64.run |
   | Ascend-mindie-atb_{version}_cxx11abi1_linux-aarch64.run |
   | Ascend-mindie-atb_{version}_cxx11abi1_linux-x86_64.run  |
   | Ascend-mindie-atb_{version}_cxx11abi0_linux-x86_64.run  |
   
   具体使用cxx11abi0 还是cxx11abi1 可通过python命令查询
   
   ```python
   import torch
   
   torch.compiled_with_cxx11_abi()
   ```
   
   若返回True 则使用 cxx11abi1，否则相反。
   
   ```bash
   # 安装
   chmod +x Ascend-cann-atb_7.0.T10_*.run
   ./Ascend-cann-atb_7.0.T10_*.run --install
   source /usr/local/Ascend/atb/set_env.sh
   ```
   
3. 根据版本发布链接，解压大模型文件

   | 大模型包名                                                   |
   | ------------------------------------------------------------ |
   | Ascend-mindie-atb-models_{version_id}_linux-x86_64_torch{pta_version}-abi0.tar.gz |
   | Ascend-mindie-atb-models_{version_id}_linux-x86_64_torch{pta_version}-abi1.tar.gz |
   | Ascend-mindie-atb-models_{version_id}_linux-aarch64_torch{pta_version}-abi0.tar.gz |
   | Ascend-mindie-atb-models_{version_id}_linux-aarch64_torch{pta_version}-abi1.tar.gz |

    具体使用cxx11abi0 还是cxx11abi1 方法同安装atb

   ```bash
   # 安装
   # cd {llm_path}
   tar -xzvf Ascend-cann-llm_*.tar.gz
   source set_env.sh
   ```

   > 注： 每次运行前都需要 source CANN， 加速库，大模型

## 模型推理

1. 切分模型权重 **首次跑模型时**，需要先对模型权重进行**切分**，切分方法如下 (在启动脚本run.sh中，world size为切分数，Atalas推理系列产品中每芯记为1，例：单卡双芯时WORLD SIZE为2)

- 修改代码

  1. 修改`run.sh`中`input_dir`为真实`input_dir`
  
  2. 修改`run.sh`中`output_dir`为自定义路径，用于存放切分后的模型权重

- 执行切分

  ```
  bash run.sh --run --parallel
  # 切分好的模型权重会存放在自定义的output_dir
  
  # run.sh中第44行
  cp $SCRIPT_DIR/modeling_gpt_bigcode_simple.py $transformers....
  modeling_gpt_bigcode_simple.p用于初始切分
  后续需使用modeling_gpt_bigcode_parallel_model_{型号}.py进行模型库的Atalas推理系列产品双芯推理
  patch/model路径下有各类modeling脚本，根据需要替换
  ```

2. **执行模型推理** 模型切分完成后，run_parallel.sh会加载`output_idr`下切分好的模型权重（`output_dir/part_model/0`和`output_dir/part_model/1`）进行推理

- 配置可选参数：最大输入输出长度
  默认值为2048，可以根据用户需要, 在脚本中手动**配置最大输入输出长度**，把`modeling_gpt_bigcode_parallel_model_{型号}.py`脚本中的变量**MAX_SEQ_LENGTH**改为：**期望的最大输入长度 + 最大输出长度**

- 修改配置参数&执行推理
  当前支持单case推理和多case推理。
  multicase=0时，单case；
  multicase=1时，多case；当前多case推理支持用例排列组合，set_case_pair=1时生效。

  ```
  # 双芯模型权重路径
  output_dir="./starcoder_parallel"
  # 指定芯片，默认为0,1
  device_id=0
  multi_batch_size=[1,4,8,16,32]
  
  # 单case生效
  seqlen_in=128
  seqlen_out=128
  
  # 多case生效
  # 单case推理(0) or 多case(1)
  multicase=1
  # 多case推理配置参数，默认执行[1,4,8,16,32]的推理
  set_case_pair=0
  # 以下两个变量set_case_pair=0生效，推理默认case，即输入输出分别为[32,64,128,256,512,1024]组合的36组case;
  # 默认输入长度从2^5到2^10
  seqlen_in_range=[5,11]
  # 默认输出长度从2^5到2^10
  seqlen_out_range=[5,11]
  # 以下两个变量set_case_pair=1生效，推理特定case，默认推理(输入长度，输出长度)分别为(256,64),(256,256),(512,512),(1024,1024)4组case;
  seqlen_in_pair=[256,256,512,1024]
  seqlen_out_pair=[64,256,512,1024]
  # starcoder, 为输出文件名字的后缀
  model_name="starcoder"
  ```

> 单case: 推理[batch_size,seqlen_in,seqlen_out]这个用例；
> 多case: 默认测试batch=1/4/8/16/32，输入32~1024，输出32~1024这些case的性能；当set_case_pair=1时，测试seqlen_in_pair/seqlen_out_pair中的用例排列组合；
> 多case会将性能数据保存在./multibatch_performance_{model_name}_{device_id}.csv，包括case配置、首token、非首token处理时延;
  ```
  bash run.sh --run --parallel
  ```

  该命令会运行一次简单的推理实例warm up，并启动后续的推理

- 自定义运行可参考`run_parallel.py`

3. **优化选项** 
- 开启多stream性能优化，修改run.sh中以下环境变量为1，并执行推理（注意：Atlas 800T A2不涉及，不要打开）
  ```
  export ATB_USE_TILING_COPY_STREAM=1
  ```
- 开启reuse内存优化，修改run.sh中以下环境变量为1，并执行推理
  ```
  export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
  ```

# 量化

1. 先将浮点权重量化

   ```
   python quant.py --model_path <浮点权重路径> --output_path <量化权重路径>
   ```

2. 将浮点权重，量化权重进行切分

   ```
   transformers_package_path=$(python3 -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')
   cp modeling_gpt_bigcode_simple.py $transformers_package_path/models/gpt_bigcode/modeling_gpt_bigcode.py
   python cut_model_util.py --input_path  <浮点权重路径>   --output_path <浮点多卡权重路径> 
   python cut_quant_model_util.py  --input_path <量化权重路径> --output_path <量化多卡权重路径> 
   ```

3.  运行量化modeling

   将 `run.sh` 中的 `cp $SCRIPT_DIR/modeling_gpt_bigcode_simple.py $transformers_package_path/models/gpt_bigcode/modeling_gpt_bigcode.py` 
   改为 ``cp  $SCRIPT_DIR/patch/model/modeling_gpt_bigcode_model_quant_{型号}.py $transformers_package_path/models/gpt_bigcode/modeling_gpt_bigcode.py``

# 模型推理性能

| 硬件形态 | 模型 | Batch | 首token(ms)     |非首token(ms)      |
| ------------------ | ------- | --------- | ----- | --------------- |
| Duo双芯(x86) |      |       |             |               |
| Duo双芯(x86)  |      |       |             |               |
> Batch=1, 输入长度和输出长度取[32,64,128,256,512,1024], 共36组case取均值
# 模型推理精度

[基于human-eval数据集]

|      |      |      |      |      |      |      |
| ------------------ | ------- | --------- | ----- | --------------- | ---------- | ------ |
|      |      |      |      |      |      |      |
|      |      |      |      |      |      |      |
|      |      |      |      |      |      |      |
|      |      |      |      |      |      |      |