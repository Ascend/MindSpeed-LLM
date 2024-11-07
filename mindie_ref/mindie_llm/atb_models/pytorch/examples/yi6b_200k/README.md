#  Yi-6B-200K模型-推理指导（800I A2）

- [概述](#概述)
- [输入输出数据](#输入输出数据)
- [推理环境准备](#推理环境准备)
- [安装配套版本](#安装配套版本)
- [快速上手](#快速上手)
- [精度验证指南](#精度验证指南)

# 概述

[Yi](https://huggingface.co/01-ai/Yi-6B-200K) 系列模型是由 01.AI 从头开始训练的新一代开源大型语言模型。[Yi](https://huggingface.co/01-ai/Yi-6B-200K) 模型以双语语言模型为目标，在 3T 多语种语料库上进行训练，已成为全球最强大的 LLM 之一，在语言理解、常识推理、阅读理解等方面展示出良好的前景。

# 输入输出数据

- 输入数据

  | 输入数据       | 大小                 | 数据类型 | 数据排布格式 | 是否必选 |
  | -------------- | -------------------- | -------- | ------------ | -------- |
  | input_ids      | BATCH_SIZE x SEQ_LEN | INT64    | ND           | 是       |
  | attention_mask | BATCH_SIZE x SEQ_LEN | BFLOAT16  | ND           | 否       |

- 输出数据

  | 输出数据   | 大小                        | 数据类型 | 数据排布格式 |
  | ---------- | --------------------------- | -------- | ------------ |
  | output_ids | BATCH_SIZE x OUTPUT_SEQ_LEN | INT64    | ND           |

# 推理环境准备

### 1 安装HDK

详细信息可参见[昇腾社区驱动与固件](https://www.hiascend.com/document/detail/zh/canncommercial/63RC2/envdeployment/instg/instg_000018.html)，先安装firmwire，再安装driver

#### 1.1 安装firmwire

安装方法: `{version}`代表具体版本

| 包名                                   |
|--------------------------------------|
| Ascend-hdk-{型号}-npu-firmware_{version}.run |
| Ascend-hdk-{型号}-npu-firmware_{version}.run |

根据芯片型号选择相应的安装包安装

```bash
# 安装firmwire
chmod +x Ascend-hdk-{型号}-npu-firmware_{version}.run
./Ascend-hdk-{型号}-npu-firmware_{version}.run --full
```

#### 1.2 安装driver

安装方法：

| cpu     | 包名                                               | 
|---------|--------------------------------------------------|
| aarch64 | Ascend-hdk-{型号}-npu-driver_{version}_linux-aarch64.run |
| x86     | Ascend-hdk-{型号}-npu-driver_{version}_linux-x86_64.run  |

```bash
# 根据CPU架构 以及npu型号 安装对应的 driver
chmod +x Ascend-hdk-{型号}-npu-driver_{version}_*.run
./Ascend-hdk-{型号}-npu-driver_{version}_*.run --full
```

### 2 安装CANN

详细信息可参见[昇腾社区CANN软件](https://www.hiascend.com/software/cann)，先安装toolkit 再安装kernel

#### 2.1 安装toolkit

安装方法：`{version}`代表具体版本

| cpu     | 包名                                            |
|---------|-----------------------------------------------|
| aarch64 | Ascend-cann-toolkit_{version}_linux-aarch64.run |
| x86     | Ascend-cann-toolkit_{version}_linux-x86_64.run  |

```bash
# 安装toolkit  以arm为例
chmod +x Ascend-cann-toolkit_{version}_linux-aarch64.run
./Ascend-cann-toolkit_{version}_linux-aarch64.run --install
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

#### 2.2 安装kernel

安装方法：

| 包名                                         |
|--------------------------------------------|
| Ascend-cann-kernels-{型号}_{version}_linux.run |
| Ascend-cann-kernels-{型号}_{version}_linux.run |

```bash
# 安装 kernel
chmod +x Ascend-cann-kernels-{型号}_{version}_linux.run
./Ascend-cann-kernels-{型号}_{version}_linux.run --install
```

### 3 安装PytorchAdapter

先安装torch 再安装torch_npu

#### 3.1 安装torch

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

根据所使用的环境中的python版本以及cpu类型，选择对应版本的torch安装包。

```bash
# 安装torch 2.0.1 的python 3.9 的arm版本为例
pip install torch-2.0.1-cp39-cp39-linux_aarch64.whl
```

#### 3.2 安装torch_npu

[下载PyTorch Adapter](https://www.hiascend.com/software/ai-frameworks/commercial)，安装方法：

| 包名                          |
|-----------------------------|
| pytorch_v2.0.1_py38.tar.gz  |
| pytorch_v2.0.1_py39.tar.gz  |
| pytorch_v2.0.1_py310.tar.gz |
| ...                         |

- 安装选择与torch版本 以及 python版本 一致的npu_torch版本

```bash
# 安装 torch_npu 以torch 2.0.1 的python 3.9的版本为例
tar -zxvf pytorch_v2.0.1_py39.tar.gz
pip install torch*_aarch64.whl
```
# 安装配套版本

#### 1. 下载Yi-6B-200K模型权重，放置到自定义`input_dir`

   ```
   https://huggingface.co/01-ai/Yi-6B-200K
   ```

#### 2. 根据版本发布链接，安装MindIE-ATB 

   | MindIE-ATB包名                                            |
   | ----------------------------------------------------- |
   | Ascend-mindie-atb_{version}_linux-aarch64_abi0.run |
   | Ascend-mindie-atb_{version}_linux-aarch64_abi1.run |
   | Ascend-mindie-atb_{version}_linux-x86_abi0.run     |
   | Ascend-mindie-atb_{version}_linux-x86_abi1.run     |
   
   具体使用abi0 还是abi1 可通过python命令查询
   
   ```python3
   python3 -c "import torch; print(torch.compiled_with_cxx11_abi())"
   ```
   
   若返回True 则使用 abi1，否则相反。
   
   ```bash
   # 安装
   chmod +x Ascend-mindie-atb_{version}_linux-*.run
   ./Ascend-mindie-atb_{version}_linux-*.run --install
   source /usr/local/Ascend/atb/set_env.sh
   ```
   
#### 3. 根据版本发布链接，解压大模型ATB-Models


   | 大模型包名                                                   |
   | ------------------------------------------------------------ |
   | Ascend-mindie-atb-models_1.0.RC1_linux-aarch64_torchxx-abi0.tar.gz |
   | Ascend-mindie-atb-models_1.0.RC1_linux-aarch64_torchxx-abi1.tar.gz |
   | Ascend-mindie-atb-models_1.0.RC1_linux-x86_torchxx-abi0.tar.gz |
   | Ascend-mindie-atb-models_1.0.RC1_linux-x86_torchxx-abi1.tar.gz |

    具体使用abi0 还是abi1 方法同安装atb

   ```bash
   # 安装
   mkdir {llm_path}
   tar -xzvf Ascend-mindie-atb-models_*.tar.gz -C {llm_path} --no-same-owner
   source set_env.sh
   ```

   > 注： 每次运行前都需要 source CANN， MindIE-ATB ，ATB-Models

# 快速上手

> 工作目录和llama一致,  `cd ./pytorch/examples/llama`

#### 1. 跑多卡推理，需要先切分模型权重，切分方法如下：

- 修改代码

  1. 修改`cut_weight.sh`中`input_dir`为实际存放模型权重的路径
  
  2. 修改`cut_weight.sh`中`output_dir`为自定义路径，用于存放切分后的模型权重
  3. 修改`cut_weight.sh`中`yi6b=1`

- 执行切分

  ```bash
  # 切分模型权重8份，切分好的权重会存放在自定义的`output_dir`
  bash cut_weight.sh --float 8
  ```

#### 2. **执行模型推理**
- 开启CPU Performance模式以提高模型推理性能（首次开启时，根据提示安装依赖）
  ```
  cpupower frequency-set -g performance
  ```

- 在800I A2执行推理时，可以通过**绑核**以达到最佳性能
  ```
  # 进入./pytorch/examples/atb_speed_sdk/，安装sdk依赖
  cd ../atb_speed_sdk/
  pip install .

  # 进入sdk_config.ini，设置环境变量bind_cpu为1（默认为1，绑核）
  bind_cpu=1
  ```
- 配置数据类型
  - 修改sdk_test.py中`torch.float16`为`torch.bfloat16`

- 配置必选参数
  - 修改run_sdk_test.sh中环境变量**MAX_SEQ_LENGTH**为：**期望的最大输入长度 + 最大输出长度**，默认值为2048
  - 修改sdk_config.ini中`model.model_path=output_dir`
  - 修改sdk_config.ini中`model.device_ids=0,1,2,3,4,5,6,7`

    > 注意：正常推理时，**MAX_SEQ_LENGTH**必须大于或者等于**输入长度 + 输出长度**, 否则会出现精度问题; 最大吞吐测试时，**MAX_SEQ_LENGTH**必须等于**输入长度 + 输出长度**，否则会影响内存申请

- 配置输入输出长度
  - 修改sdk_config.ini中`performance.model_name=Yi-6B-200K`
  - 修改sdk_config.ini中`performance.batch_size=300`
  - 修改sdk_config.ini中`performance.case_pair=[[256,256]]`

- 执行推理，此时执行batch_size=300，输入256, 输出256的性能测试
  指令：bash run_sdk_test.sh [WORLD_SIZE] [DEVICE_TYPE] [TASK]
  
  ```
  # 800I A2环境执行单机8卡推理
  bash run_sdk_test.sh 8 d9 performance
  ```
  > WORLD_SIZE: 指定芯片数量，实现单卡和多卡推理（默认1）
  > DEVICE_TYPE: d9, 对应800I A2
  > TASK: 可选'run', 'performance', 'precision'

#### 3. **长序列推理**
- 配置必选参数
  - 修改run_sdk_test.sh中`MAX_SEQ_LENGTH=200128`
  - 修改run_sdk_test.sh中`LOG_SEQ_ENABLE=1`
- 配置输入输出长度
  - 修改sdk_config.ini中`performance.model_name=Yi-6B-200K`
  - 修改sdk_config.ini中`performance.batch_size=1`
  - 修改sdk_config.ini中`performance.case_pair=[[200000,128]]`
- 执行推理，此时执行batch_size=1，输入200000, 输出128的性能测试
  - bash run_sdk_test.sh 8 d9 performance

# 精度验证指南

### C-EVAL/MMLU
> 支持C-EVAL、MMLU两种数据集，采用5-shot的方式验证模型推理精度。 

####  1.下载数据集
``` bash
# C-EVAL
wget https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip
unzip ceval-exam.zip -d data
# MMLU
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
tar -xvf data.tar
```
> 先测试C-EVAL, 修改`pytorch/examples/atb_speed_sdk/atb_speed/common/precision/base.py` 132行为test数据集，重安装atb-speed，再测试MMLU

```python3
val_df = pd.read_csv(os.path.join(self.data_dir, "test", task_name + "_test.csv"), header=None)
```
####  2. 安装atb-speed插件
```
cd ../atb_speed_sdk/
pip install .
```

####  3.配置精度测试参数(以MMLU为例)
1. 在当前目录新建工作文件夹${mmlu_test_dir}
2. 将下载的测试数据集进行解压后的数据放置在${mmlu_test_dir}
3. 修改sdk_config.ini文件中精度测试的相关配置，设置模型路径、工作目录、device id(默认0卡)、和batch size(默认1)
    * model_path=./yi_6b_200k
    * work_dir=./mmlu_test_dir
    * device=0,1,2,3,4,5,6,7
    * batch=1

目录结构示例${mmlu_test_dir}:
--mmlu_test_dir
    --test_result 跑完之后生成  
    --data (包含：数据文件夹dev、test、val三者)

####  4. 运行并查看结果

**4.1 开始精度数据集推理**
```
# 执行多卡多芯推理
bash run_sdk_test.sh 8 d9 precision
```

**4.2 查看结果**
| test_result目录                        | 用途                   | 
|---------------------------|----------------------| 
| cache.csv                | 结果详情，C列为预期答案，D列为测试答案 |
| summary_classes_acc.json | 测试数据下按不同维度统计准确率      |
| summary_subject_acc.json | 测试数据下按不同学科统计准确率      |

> CEVAL test数据集golden未公开，因此使用dev作为few-shot来源，val作为测试数据
> MMLU 使用dev作为few-shot来源，test作为测试数据

### LONGBENCH
> [LongBench](https://github.com/THUDM/LongBench)是第一个多任务、中英双语、针对大语言模型长文本理解能力的评测基准，包含14个英文任务、5个中文任务和2个代码任务，多数任务的平均长度在5k-15k之间，共包含4750条测试数据，以此来对大模型在长文本下的多语言能力进行更全面的评估。

####  1.下载数据集和相关代码
1.1 下载数据集, 存放在${dataset_dir}
```bash
wget https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip
unzip data.zip
```

1.2 下载数据集映射信息和评测代码, 存放在./pytorch/examples/llama
https://github.com/THUDM/LongBench/blob/main/config/dataset2maxlen.json
https://github.com/THUDM/LongBench/blob/main/config/dataset2prompt.json
https://github.com/THUDM/LongBench/blob/main/eval.py
https://github.com/THUDM/LongBench/blob/main/metrics.py

####  2. 运行并查看结果
```bash
cd ./pytorch/examples/llama
transformers_package_path=$(python3 -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')
cp ./modeling_llama_ascend.py $transformers_package_path/models/llama/modeling_llama.py
MAX_SEQ_LENGTH=16000 torchrun --nproc_per_node 8 --master_port 25641 longbench.py --model_path ${output_dir} --dataset_path ${dataset_dir} --e
```
运行结束后，可以在`pred_e/对应模型名称`的文件夹下得到模型在LongBench-E所有数据集下的输出，此后运行eval.py的评测代码：
```bash
python eval.py --model yi_6b_200k --e
export RESULT_DIR=./pred_e/yi_6b_200k
python3 -c "from longbench import get_result_scores; get_result_scores()"
```
可以在存储模型输出文件夹下的result.json和result.csv中得到模型在LongBench-E各数据集上的评测结果。