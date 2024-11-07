#  LLaMA2-7B/13B模型-推理指导（300I-DUO/800I A2）

- [概述](#概述)
- [输入输出数据](#输入输出数据)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码及依赖](#获取源码及依赖)
  - [模型推理](#模型推理)
- [模型推理性能](#模型推理性能)
- [精度验证指南](#精度验证指南)
- [模型推理精度](#模型推理精度)

# 概述

LLaMA（Large Language Model Meta AI），由 Meta AI 发布的一个开放且高效的大型基础语，可以通过自然语言交互的方式提供知识、文本生成、语言翻译、语言理解、代码编写和解释等任务。

- 参考实现：

  ```
  https://github.com/facebookresearch/llama
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

| 配套           | 版本          | 下载链接 |
| -------------- | ------------- | -------- |
| 固件与驱动     | 23.0.RC3.B082 | -        |
| CANN           | 7.0.RC1.B082  | -        |
| Python         | 3.9.11        | -        |
| PytorchAdapter | 1.11.0        | -        |
| 推理引擎       | -             | -        |

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

# 快速上手

## 获取源码及依赖

1. 环境部署

- 1.1. 安装HDK

> 先安装firmwire，再安装driver

  1.1.1. 安装firmwire

  安装方法:

| 包名                                             |
|------------------------------------------------|
| Ascend-hdk-{型号}-npu-firmware_7.0.0.5.242.run |

  ```bash
  # 安装firmwire
  chmod +x Ascend-hdk-{型号}-npu-firmware_7.0.0.5.242.run
  ./Ascend-hdk-{型号}-npu-firmware_7.0.0.5.242.run --full
  ```

  1.1.2. 安装driver

  安装方法：

| cpu     | 包名                                                         |
|---------|------------------------------------------------------------|
| aarch64 | Ascend-hdk-{型号}-npu-driver_23.0.rc3.b082_linux-aarch64.run |
| x86     | Ascend-hdk-{型号}-npu-driver_23.0.rc3.b082_linux-x86-64.run |

  ```bash
  # 根据CPU架构安装对应的 driver
  chmod +x Ascend-hdk-{型号}-npu-driver_23.0.rc3.b082_*.run
  ./Ascend-hdk-{型号}-npu-driver_23.0.rc3.b082_*.run --full
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

| cpu     | 包名                                                         |
|---------|------------------------------------------------------------|
| aarch64 | torch-1.11.0-cp39-cp39-linux_aarch64.whl |
| x86     | torch-1.11.0+cpu-cp39-cp39-linux_x86_64.whl |

  根据所使用的环境中的python版本，选择torch-1.11.0相应的安装包。

  ```bash
  # 安装torch 1.11.0 的python 3.9 的arm版本为例
  pip install torch-1.11.0-cp39-cp39-linux_aarch64.whl
  ```

  1.3.2 安装torch_npu

  安装方法：

| 包名                          |
|-----------------------------|
| pytorch_v1.11.0_py39.tar.gz |

> 安装选择与torch版本 以及 python版本 一致的torch_npu版本

  ```bash
  # 安装 torch_npu 以torch 1.11.0 的python 3.9的arm版本为例
  tar -zxvf pytorch_v1.11.0_py39.tar.gz
  pip install torch*_aarch64.whl
  ```

## 推理环境准备

> 安装配套软件。安装python依赖。

  ```
  pip3 install -r requirements.txt
  ```

1. 下载LLaMA2-7B/LLaMA2-13B模型权重，放置到自定义`input_dir`

   ```
   https://huggingface.co/NousResearch/Llama-2-13b-hf
   https://huggingface.co/NousResearch/Llama-2-7b-hf
   ```

2. 根据版本发布链接，安装加速库 

   | 加速库包名                                            |
   | ----------------------------------------------------- |
   | Ascend-cann-atb_{version}_cxx11abi0_linux-aarch64.run |
   | Ascend-cann-atb_{version}_cxx11abi1_linux-aarch64.run |
| Ascend-cann-atb_{version}_cxx11abi1_linux-x86_64.run  |
   | Ascend-cann-atb_{version}_cxx11abi0_linux-x86_64.run  |
   
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
   | Ascend-cann-llm_{version_id}_linux-x86_64_torch{pta_version}-abi0.tar.gz |
   | Ascend-cann-llm_{version_id}_linux-x86_64_torch{pta_version}-abi1.tar.gz |
   | Ascend-cann-llm_{version_id}_linux-aarch64_torch{pta_version}-abi0.tar.gz |
   | Ascend-cann-llm_{version_id}_linux-aarch64_torch{pta_version}-abi1.tar.gz |

    具体使用cxx11abi0 还是cxx11abi1 方法同安装atb

   ```bash
   # 安装
   mkdir {llm_path}
   tar -xzvf Ascend-cann-llm_*.tar.gz -C {llm_path} --no-same-owner
   source set_env.sh
   ```

   > 注： 每次运行前都需要 source CANN， 加速库，大模型

## 模型推理
1. 如果跑多卡多芯推理，需要先切分模型权重，切分方法如下：

- 修改代码

  1. 修改`cut_weight.sh`中`input_dir`为实际存放模型权重的路径
  
  2. 修改`cut_weight.sh`中`output_dir`为自定义路径，用于存放切分后的模型权重

- 执行切分

  ```
  # 切分模型权重2份，切分好的权重会存放在自定义的output_dir
  bash cut_weight.sh --float 2
  # 切分模型权重4份
  bash cut_weight.sh --float 4
  ```

2. **执行模型推理**
- 开启CPU Performance模式以提高模型推理性能（首次开启时，根据提示安装依赖）
  ```
  cpupower frequency-set -g performance
  ```
  
- 进入./pytorch/examples/atb_speed_sdk/，安装sdk依赖
  ```
  cd ../atb_speed_sdk/
  pip install .
  ```

- 在800I A2执行推理时，可以通过**绑核**以达到最佳性能
  ```
  # 进入run.sh，设置环境变量BIND_CPU为1（默认为0，不绑核）
  export BIND_CPU=1
  ```

- 配置必选参数：最大输入输出长度
  修改run.sh中环境变量**MAX_SEQ_LENGTH**为：**期望的最大输入长度 + 最大输出长度**，默认值为2048

- 修改配置参数
当前支持单case推理和多case推理。
multicase=0时，单case推理；
multicase=1时，多case推理；支持用例排列组合，set_case_pair=1时生效。

  ```
  # 双芯模型权重路径
  input_dir="./llama2-7b_parallel"
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
  # LLAMA2-7B or LLAMA2-13B, 为输出文件名字的后缀
  model_name="LLAMA2-7B"
  ```
> 单case: 推理用例为[batch_size, seqlen_in, seqlen_out]；
> 多case: 默认测试batch=1/4/8/16/32，输入32-1024，输出32-1024多case的性能；当set_case_pair=1时，测试seqlen_in_pair/seqlen_out_pair中的用例排列组合；
> 推理完成后性能数据保存在./multibatch_performance_{model_name}_{device_id}.csv，包括用例配置、首token、非首token处理时延等;

- 执行推理
  指令：bash run.sh --[RUN_OPTION] [WORLD_SIZE] [DEVICE_TYPE]  
  ```
  # 800I A2环境执行单卡推理
  bash run.sh --performance 1 d9
  # 800I A2环境执行双卡推理
  bash run.sh --performance 2 d9
  # 300I DUO环境执行单卡双芯推理
  bash run.sh --performance 2 d3
  # 300I DUO环境执行双卡四芯推理
  bash run.sh --performance 4 d3
  ```
  > WORLD_SIZE: 指定芯片数量，实现单卡和多卡推理（默认1）
  > DEVICE_TYPE: d9/d3, 分别适配800I A2和300I DUO芯片型号 (默认d3，支持300I DUO推理)

  该命令会运行一次简单的推理实例warm up，并启动后续的推理；自定义运行可参考`main.py`

## 量化推理
> 使用Anti-Outlier（离群值抑制）量化算法，int8模型权重包含anti-outlier**浮点权重**和**量化权重**各一份
1. 获取量化权重
1.1 直接获取量化权重
  - LLaMA2-7B int8权重链接：https://model-weight.obs.cn-north-4.myhuaweicloud.com/llama7b.tar.gz
  - LLaMA2-13B int8权重链接：https://model-weight.obs.cn-north-4.myhuaweicloud.com/llama13b/llama13b.tar.gz
  解压从链接获取的.tar.gz权重压缩包，权重目录示例如下：
    ```
    --llama[7/13]b_quant_weight
        #包含anti-outlier量化权重
        --deq_scale.npy
        --fp_bias.npy
        --quant_weight.npy
        --input_offset.npy
        --input_scale.npy
        --anti_weight
            #包含anti-outlier浮点权重
            --config.json
            --generation_config.json
            --pytorch_model.bin.index.json
            --pytorch_model-00001-of-00002.bin
            --pytorch_model-00002-of-00002.bin

    ```
1.2 cpu生成量化权重
  - 下载ceval量化数据集到./quant_script/下
    - ceval数据集 https://huggingface.co/datasets/ceval/ceval-exam/tree/main
    解压从链接获取的.zip权重压缩包，权重解压在quan_script文件夹下，目录示例如下
  - 生成anti-outlier浮点和量化权重
    ```
    # 1. 修改./quant_script/quantize_weight.sh中input_dir为实际存放**模型原始浮点权重**的路径，例如：
    input_dir="./llama2-7b"
    # 2. 修改./quant_script/quantize_weight.sh中output_dir为自定义路径，用于存放anti-outlier浮点和量化权重，例如：
    output_dir="./llama7b_quant_weight"

    # 如果跑llama2-7b，量化回退层推荐设置为L6
    disable_level="L6"
    # 如果跑llama2-13b，量化回退层推荐设置为L8
    disable_level="L8"
    # 执行量化权重生成
    bash ./quant_script/quantize_weight.sh --anti_quant
    ```

2. 量化权重切分
  - 切分anti-outlier量化权重
    ```
    # 1. 修改cut_weight.sh中input_dir为实际存放anti-outlier量化权重的路径，例如：
    input_dir="./llama7b_quant_weight"
    # 2. 修改cut_weight.sh中output_dir为自定义路径，用于存放切分后的量化权重，例如：
    output_dir="./llama7b_anti_quant_parallel"
    # 3. 执行切分，切分量化权重2份
    bash cut_weight.sh --quant 2
    ```

3. 适配量化推理代码
  - 进入modeling_llama_ascend.py，适配量化权重路径和回退层
    ```
    # 开启量化推理开关，配置anti-outlier算法
    RUN_QUANT_MODEL = True
    RUN_ANTI_QUANT_MODEL = True

    # 修改全局变量QUANT_WEIGHT_PATH、ANTI_QUANT_WEIGHT_PATH、和FLOAT_LAYERS
    QUANT_WEIGHT_PATH = "./llama7b_anti_quant_parallel" # anti-outlier量化权重
    ANTI_QUANT_WEIGHT_PATH = "./llama7b_quant_weight/anti_weight" # anti-outlier浮点权重
    
    # 如果跑llama2-7b，量化回退层设置为0, 1, 2, 8, 30, 31
    FLOAT_LAYERS = [0, 1, 2, 8, 30, 31] 
    # 如果跑llama2-13b，量化回退层设置为0, 1, 3, 7, 9, 27, 38, 39
    FLOAT_LAYERS = [0, 1, 3, 7, 9, 27, 38, 39]
    ```

4. 执行量化模型推理
  ```
  # 800I A2环境执行单卡推理
  bash run.sh --performance 1 d9
  # 300I DUO环境执行单卡双芯推理
  bash run.sh --performance 2 d3
  ```

## 稀疏量化推理
1. 生成稀疏量化权重
  - 进入./quant_script/quantize_llama_sparse_weight.py，修改用于做校准的ceval数据集路径以及生成的稀疏量化权重保存路径
  ```python
  # line 14
  WORKDIR = "./ceval"

  # line 247
  dest_dir = "./llama7b_sparsequant_weight"
  ```

  - 运行./quant_script/quantize_llama_sparse_weight.py脚本
  ```bash
  python ./quant_script/quantize_llama_sparse_weight.py
  ```

2. 稀疏权重切分
    ```
    # 1. 修改cut_weight.sh中input_dir为实际存放稀疏量化权重的路径，例如：
    input_dir="./llama7b_sparsequant_weight"
    # 2. 修改cut_weight.sh中output_dir为自定义路径，用于存放切分后的量化权重，例如：
    output_dir="./llama7b_sparsequant_parallel"
    # 3. 执行切分，切分量化权重2份
    bash cut_weight.sh --quant 2
    ```

3. 压缩切分后的稀疏权重
  - 进入./quant_script/compress_llama_sparse_weight.py，修改切分后的稀疏量化权重路径以及压缩后保存的路径。
  
  【注意】请分别修改切分后的两份权重路径以及压缩后保存的路径，并运行两次./quant_script/compress_llama_sparse_weight.py脚本。
  
  ```python
  # line 14
  weight_path = "./llama7b_sparsequant_parallel/0/quant_weight.npy"  # ./llama7b_sparsequant_parallel/1/quant_weight.npy

  # line 16
  save_path = "./compress_0"  # ./compress_1
  ```

  - 建议将压缩后的权重放在与压缩前权重的相同路径下，目录示例如下：
    ```
    --llama7b_sparsequant_parallel
        #压缩前
        --0
        --1
        #压缩后
        --compress_0
        --compress_1
    ```

    - 运行./quant_script/compress_llama_sparse_weight.py脚本
    ```bash
    python ./quant_script/compress_llama_sparse_weight.py
    ```

    【FAQ】若运行脚本时出现"FileNotFoundError: [Errno2] No such file or directory: '/xxx/modelslim/pytorch/weight_compression/compress_graph/build/compress_executor': ..."，请按照以下步骤进行手动编译之后再执行脚本：

    1. 进入到报错路径下的/xxx/modelslim/pytorch/weight_compression/compress_graph路径下
    2. 执行编译命令，必选参数为当前环境下的CANN路径(依据实际情况而定)：
    ```bash
    bash build.sh /usr/local/Ascend/ascend-toolkit/latest
    ```

4. 适配量化推理代码
  - 进入modeling_llama_ascend.py，适配稀疏量化权重路径和回退层
    ```
    # 修改RUN_SPARSE_MODEL为True (注：RUN_QUANT_MODEL为默认的False即可)
    RUN_SPARSE_MODEL = True

    # 修改全局变量FLOAT_MODEL_PATH、QUANT_WEIGHT_PATH、和FLOAT_LAYERS
    QUANT_WEIGHT_PATH = "./llama7b_sparsequant_parallel" # 压缩前的稀疏量化权重路径
    SPARSE_WEIGHT_PATH = "./llama7b_sparsequant_parallel" # 压缩后的稀疏量化权重路径
    
    # 例如，如果llama2-7b生成权重的浮点回退层为0, 1, 2, 4, 30，那么将对应的层号加入FLOAT_LAYERS列表里
    FLOAT_LAYERS = [0, 1, 2, 4, 30] 
    ```

5. 执行稀疏量化模型推理
  ```
  # 800I A2环境执行单卡推理
  bash run.sh --performance 1 d9
  # 300I DUO环境执行单卡双芯推理
  bash run.sh --performance 2 d3
  ```


# 模型推理性能

| 硬件形态 | 模型 | Batch | 首token(ms)     |非首token(ms)      |
| ------------------ | ------- | --------- | ----- | --------------- |
| Duo双芯(x86) | LLaMA2-7B| 1 | 188 |  65.32  |
| Duo双芯(x86)  | LLaMA2-13B| 1 | 326 | 110.19 |
> Batch=1, 输入长度和输出长度取[32,64,128,256,512,1024], 共36组case取均值


# 精度验证指南
> 模型精度验证基于MMLU数据集，采用5-shot的方式验证模型推理精度。 

## 1.下载MMLU数据集
```
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
tar -xvf data.tar
```

## 2. 安装atb-speed插件
```
cd ../atb_speed_sdk/
pip install .
```

## 3.配置精度测试参数
1. 在当前目录新建工作文件夹${mmlu_test_dir}
2. 将下载的测试数据集进行解压后的数据放置在${mmlu_test_dir}
3. 修改sdk_config.ini文件中精度测试的相关配置，设置模型路径、工作目录、device id(默认0、1卡)、和batch size(默认1)
    * model_path=./llama2-7b_parallel
    * work_dir=./mmlu_test
    * device=0,1
    * batch=1

目录结构示例${mmlu_test_dir}:
--mmlu_test
    --test_result 跑完之后生成  
    --data (包含：数据文件夹dev、test、val三者)

## 4. 运行并查看结果

**4.1 开始精度数据集推理**
```
# 执行单芯推理
python sdk_test.py --task precision

# 执行多卡多芯推理
torchrun --nproc_per_node 2 --master_port 25641 sdk_test.py --task precision
```

**4.2 查看结果**
| test_result目录                        | 用途                   | 
|---------------------------|----------------------| 
| cache.csv                | 结果详情，C列为预期答案，D列为测试答案 |
| summary_classes_acc.json | 测试数据下按不同维度统计准确率      |
| summary_subject_acc.json | 测试数据下按不同学科统计准确率      |

> 注：开始下一次数据集精度推理前，请重命名之前保存的结果文件./mmlu_test/test_result


# 模型推理精度
| llama Model 5-shot | GPU LLaMA2-7B | NPU LLaMA2-7B | GPU LLaMA2-13B | NPU LLaMA2-13B |
| ------------------ | ------------- | ------------- | -------------- | -------------- |
| Average(%) | 45.97 | 45.99	|	55.72 | 55.64	|