[TOC]

# Alpaca2-7B模型-推理指导

# 概述

Alpaca-2是基于Meta发布的可商用大模型Llama-2开发的指令精调大模型，该模型在原版Llama-2的基础上扩充并优化了中文词表，
使用了大规模中文数据进行增量预训练，进一步提升了中文基础语义和指令理解能力，相比一代相关模型获得了显著性能提升。
该模型支持FlashAttention-2训练。标准版模型支持4K上下文长度，长上下文版模型支持16K上下文长度，并可通过NTK方法最高扩展至24K+上下文长度。

- 参考实现：

  ```
  https://huggingface.co/hfl/chinese-alpaca-2-7b
  ```

# 输入输出数据

- 输入数据

| 输入数据           | 大小                                 | 数据类型    | 数据排布格式 | 是否必选 |
|----------------|------------------------------------|---------|--------|------|
| input_ids      | BATCH_SIZE x SEQ_LEN               | INT64   | ND     | 是    |
| attention_mask | BATCH_SIZE x 1 x SEQ_LEN x SEQ_LEN | FLOAT32 | ND     | 否    |

- 输出数据

| 输出数据       | 大小                          | 数据类型  | 数据排布格式 |
|------------|-----------------------------|-------|--------|
| output_ids | BATCH_SIZE x OUTPUT_SEQ_LEN | INT64 | ND     |

# 推理环境准备

- 该模型需要以下插件与驱动

  **表 1** 版本配套表

| 配套             | 版本            | 下载链接 |
|----------------|---------------|------|
| 固件与驱动          | 23.0.RC3.B060 | -    |
| CANN           | 7.0.RC1.B060  | -    |
| python         | 3.8.17        | -    |
| PytorchAdapter | 1.11.0        | -    |
| 推理引擎           | -             | -    |

**表 2** 推理引擎依赖

| 软件    | 版本要求     |
|-------|----------|
| glibc | >= 2.27  |
| gcc   | >= 7.5.0 |

**表 3** 硬件形态

| CPU     | Device |
|---------|--------|
| aarch64 | Atlas 800 A2  |
| aarch64 | Atalas推理系列产品  |

# 快速上手

## 获取源码及依赖

### 1. 环境部署

#### 1.1 安装HDK

先安装firmwire，再安装driver

##### 1.1.1 安装firmwire

安装方法:

| 包名                                             |
|------------------------------------------------|
| Ascend-hdk-xxx-npu-firmware_7.0.t9.0.b221.run |

根据芯片型号选择相应的安装包安装

```bash
# 安装firmwire
chmod +x Ascend-hdk-xxx-npu-firmware_7.0.t9.0.b221.run
./Ascend-hdk-xxx-npu-firmware_7.0.t9.0.b221.run --full
```

##### 1.1.2 安装driver

安装方法：

| cpu     | 包名                                                         |
|---------|------------------------------------------------------------|
| aarch64 | Ascend-hdk-xxxx-npu-driver_23.0.rc3.b060_linux-aarch64.run |
| x86     | Ascend-hdk-xxxx-npu-driver_23.0.rc3.b060_linux-x86_64.run  |

```bash
# 根据CPU架构 以及npu型号 安装对应的 driver
chmod +x Ascend-hdk-xxxx-npu-driver_23.0.rc3.b060_*.run
./Ascend-hdk-xxxx-npu-driver_23.0.rc3.b060_*.run --full
```

#### 1.2 安装CANN

先安装toolkit 再安装kernel

##### 1.2.1 安装toolkit

安装方法：

| cpu     | 包名                                            |
|---------|-----------------------------------------------|
| aarch64 | Ascend-cann-toolkit_7.0.T10_linux-aarch64.run |
| x86     | Ascend-cann-toolkit_7.0.T10_linux-x86_64.run  |

```bash
# 安装toolkit  以arm为例
chmod +x Ascend-cann-toolkit_7.0.T10_linux-aarch64.run
./Ascend-cann-toolkit_7.0.T10_linux-aarch64.run --install
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

##### 1.2.2 安装kernel

安装方法：

| 包名                                         |
|--------------------------------------------|
| Ascend-cann-kernels-xxxx_7.0.T10_linux.run |

```bash
# 安装 kernel
chmod +x Ascend-cann-kernels-xxxx_7.0.T10_linux.run
./Ascend-cann-kernels-xxxx_7.0.T10_linux.run --install
```

#### 1.3 安装PytorchAdapter

先安装torch 再安装torch_npu

##### 1.3.1 安装torch

安装方法：

| 包名                                           |
|----------------------------------------------|
| torch-2.0.1+cpu-cp38-cp38-linux_x86_64.whl   |
| torch-2.0.1+cpu-cp39-cp39-linux_x86_64.whl   |
| torch-2.0.1+cpu-cp310-cp310-linux_x86_64.whl |
| torch-2.0.1-cp310-cp310-linux_aarch64.whl    |
| torch-2.0.1-cp38-cp38-linux_aarch64.whl      |
| torch-2.0.1-cp39-cp39-linux_aarch64.whl      |
| ...                                          |

根据所使用的环境中的python版本以及cpu类型，选择torch-2.0.1相应的安装包。

```bash
# 安装torch 2.0.1 的python 3.8 的arm版本为例
pip install torch-2.0.1-cp38-cp38-linux_aarch64.whl
```

##### 1.3.2 安装torch_npu

安装方法：

| 包名                          |
|-----------------------------|
| pytorch_v2.0.1_py38.tar.gz  |
| pytorch_v2.0.1_py39.tar.gz  |
| pytorch_v2.0.1_py310.tar.gz |
| ...                         |

- 安装选择与torch版本 以及 python版本 一致的npu_torch版本

```bash
# 安装 torch_npu 以torch 2.0.1 的python 3.8的版本为例
tar -zxvf pytorch_v2.0.1_py38.tar.gz
pip install torch*_aarch64.whl
```

### 2. 安装依赖

#### 2.1 推理环境准备

1. 下载alpaca2-7b模型权重，放置到自定义目录下，假定模型存放目录是：`{model_path}`

   ```
   https://huggingface.co/hfl/chinese-alpaca-2-7b
   ```

2. 根据版本发布链接，安装加速库

| 加速库包名                                                 |
|-------------------------------------------------------|
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
# 安装atb 
chmod +x Ascend-cann-atb_*.run
./Ascend-cann-atb_*.run --install
source /usr/local/Ascend/atb/set_env.sh
```

3. 根据版本发布链接，解压大模型文件

| 大模型包名                                                                     |
|---------------------------------------------------------------------------|
| Ascend-cann-llm_{version_id}_linux-x86_64_torch{pta_version}-abi0.tar.gz  |
| Ascend-cann-llm_{version_id}_linux-x86_64_torch{pta_version}-abi1.tar.gz  |
| Ascend-cann-llm_{version_id}_linux-aarch64_torch{pta_version}-abi0.tar.gz |
| Ascend-cann-llm_{version_id}_linux-aarch64_torch{pta_version}-abi1.tar.gz |

具体使用cxx11abi0 还是cxx11abi1 方法同安装atb

 ```bash
 # 安装大模型加速库, 下载文件放在自定义目录下， 假定自定义目录是`{alpaca2_path}`
 cd {alpaca2_path}
 tar -xzvf Ascend-cann-llm_*.tar.gz
 source set_env.sh
 ```

4. 下载CEval数据集

若需执行精度测试，请参考附录中的精度测试指南 进行下载

5. 设置环境变量

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/atb/set_env.sh
source ${alpaca2_path}/set_env.sh
```

> 注： 每次运行前都需要 source CANN， 加速库，大模型

## 模型推理

### 拷贝文件

进入工作目录 `{alpaca2_path}/pytorch/examples/alpaca2/7b`,   
拷贝开源模型文件夹`{model_path}`中的 `config.json` 、`generation_config.json` 、`pytorch_model.bin.index.json` 、
`special_tokens_map.json`、`tokenizer.model` 、`tokenizer_config.json` 文件到工作目录下
示例：

```shell
cd {alpaca2_path}/pytorch/examples/alpaca2/7b
cp {model_path}/config.json ./
cp {model_path}/generation_config.json ./
cp {model_path}/pytorch_model.bin.index.json ./
cp {model_path}/special_tokens_map.json ./
cp {model_path}/tokenizer.model ./
cp {model_path}/tokenizer_config.json ./
```

- 修改`config.json` , 在第5行（"bos_token_id": 1,）前增加如下内容

```shell
"auto_map": {
  "AutoModelForCausalLM": "modeling_alpaca.LlamaForCausalLM"
},
```

### 软链接模型权重文件

```shell
ln -s {model_path}/pytorch_model-00001-of-00002.bin  pytorch_model-00001-of-00002.bin
ln -s {model_path}/pytorch_model-00002-of-00002.bin  pytorch_model-00002-of-00002.bin
```

### 安装 atb_speed_sdk

```shell
cd ${alpaca2_path}/pytorch/examples/atb_speed_sdk
pip install .
```

### 配置 config.ini

- 在工作目录下创建config.ini文件
  ```shell
  cd {alpaca2_path}/pytorch/examples/alpaca2/7b
  vi config.ini
  ```

  [参考atb_speed_sdk下的README.md](../../atb_speed_sdk/README.md)

- 复制readme中”配置文件样例“章节的内容，以config.ini文件名保存

- 配置config.ini中属性值：
  ```
  model_path={alpaca2_path}/pytorch/examples/alpaca2/7b
  work_dir={alpaca2_path}/pytorch/examples/alpaca2/7b/script
  perf_mode=detail
  ```

### 执行推理

```shell
python main.py --task inference
```

该命令会运行一次简单的推理实例warm up，并启动后续的1个问答

# 竞品对比

## 精度

| 精度             | Atlas 800T A2（313T）         | A100                | 对比         |
|----------------|---------------------|---------------------|------------|
| STEM           | 0.38372093023255816 | 0.38372093023255816 | 1.00000000 |
| Social Science | 0.48                | 0.48                | 1.00000000 |
| Humanities     | 0.45136186770428016 | 0.45136186770428016 | 1.00000000 |
| Other          | 0.4036458333333333  | 0.4036458333333333  | 1.00000000 |
| Avg acc        | 0.42199108469539376 | 0.42199108469539376 | 1.00000000 |

## 性能

| 芯片型号        | batch_size | 首token推理速度(token/s) | 增量推理速度(token/s)    |
|-------------|------------|---------------------|--------------------|
| A100        | 1          | 44.09995588         | 11.25243064        |
| Atlas 800T A2（313T） | 1          | 59.44401735729642   | 14.25532105233934  |
| 对比          | 1          | 0.7418737467713726  | 0.7893495066639305 |

# 附录：

# 精度测试指南

## 配置说明

参考 [SDK精度测试指南CEVAL章节](../../atb_speed_sdk/README.md)

## 运行脚本

- 单芯

```shell
cd ${script_path}
python main.py --task precision
```

结束后在{ceval_work_dir}/test_result目录下查看测试结果。

| 文件                        | 用途                   |
|---------------------------|----------------------|
| device0.log               | 运行过程日志               |
| cache0.csv                | 结果详情，C列为预期答案，D列为测试答案 |
| result_0_classes_acc.json | 测试数据下按不同维度统计准确率      |
| result_0_subject_acc.json | 测试数据下按不同学科统计准确率      |

**注意：后续重新运行， 需要删除当前目录下生成的test_result文件夹，否则只会读取当前的目录下的测试结果**

# 性能测试指南

在功能运行正常的基础下，执行以下步骤进行性能测试

## 按照推理指导,下载模型及配置路径，并安装atb_speed_sdk

## 1. 准备

参考 [SDK性能测试指南精确打点法章节](../../atb_speed_sdk/README.md) 进行准备

## 2. 修改配置文件

- 配置config.ini中[performance]属性， 如下：
  ```
  model_name=alpaca2_7b
  perf_mode=detail
  ```

## 3. 执行测试脚本

```shell
RETURN_PERF_DETAIL=1 python main.py --task performance
```

为了不影响正常使用，将`RETURN_PERF_DETAIL`设置成1来返回具体的性能测试的值，默认是0

### 性能测试结果

得到性能测试结果csv `performance_test_npu_${model_name}_xxx.csv`

### 结果分析

| 列名                            | 含义         |
|-------------------------------|------------|
| batch_size                    | batch大小    |
| input_seq_len(Encoding)       | 输入长度       |
| output_seq_len(Decoding)      | 输出长度       |
| ResponseTime(s)               | 总响应时间      |
| forward_first_token_time(ms)  | 首token推理时长 |
| forward_next_token_time(ms)   | 增量推理时长     |
| pre_next_token_time(ms)       | 前处理时长      |
| post_next_token_time_post(ms) | 后处理时长      |
