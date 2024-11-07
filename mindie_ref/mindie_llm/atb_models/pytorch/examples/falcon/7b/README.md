#  Falcon7B 模型推理指导

# 概述

Falcon7B 是一个7B参数量的decoder-only模型，使用1,500B tokens训练而成。

- 参考实现：

  ```shell
  https://huggingface.co/tiiuae/falcon-7b
  ```

# 推理环境准备(参考，待完善)

该模型需要以下插件与驱动

**表 1** 版本配套表

| 配套           | 版本                | 下载链接                                                     |
| -------------- | ------------------- | ------------------------------------------------------------ |
| 固件与驱动     | Ascend HDK 23.0.RC3 | https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743/software/260867092? |
| CANN           | CANN 7.0.RC1        | https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/260809541?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C251168373 |
| Python         | 3.9.18              | -                                                            |
| PytorchAdapter | 1.11.0              | https://gitee.com/ascend/pytorch/releases                    |
| 模型加速包     | CANN 7.0.T22        | https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/261305468?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C251168373 |

**表 2** 推理引擎依赖

| 软件  | 版本要求 |
| :---: | :------: |
| glibc | >= 2.27  |
|  gcc  | >= 7.5.0 |

**表 3** 硬件形态

|   CPU   |  Device  |
| :-----: | :------: |
| aarch64 |   Atlas 800T A2  |
|   x86   |   Atlas 800T A2  |

# 快速上手

## 获取源码及依赖

1. 环境部署

   - 安装HDK

   - 安装python环境

     ```shell
     conda create -n falcon_llm python==3.9.18
     conda activate falcon_llm
     # 安装指定torch&torch_npu
     # 安装以下依赖
     transformers==4.30.2
     te==0.4.0
     pandas
     sympy
     ```

   - 安装CANN

     使用测试的python运行`python -c "import torch;print(torch.compiled_with_cxx11_abi())"`，若返回True，则flag=1；若返回False则flag=0；Linux中执行`arch`获取平台架构arch。

     - Ascend-cann-toolkit_7.0.RC1_linux-{arch}.run

     - Ascend-cann-kernels-xxx_7.0.RC1_linux.run

     - Ascend-cann-atb_7.0.RC1_cxx11abi{flag}_linux-{arch}.run

   - 解压模型加速包

     - Ascend-cann-llm\_{version}\_linux-{arch}\_torch{pta_version}-abi{flag}.tar.gz

     ```shell
     mkdir {llm_path}
     tar -xzvf Ascend-cann-llm_*.tar.gz -C {llm_path}
     ```

   - 设置环境变量

     ```shell
     source /usr/local/Ascend/ascend-toolkit/set_env.sh
     source /usr/local/Ascend/atb/set_env.sh
     source {llm_path}/set_env.sh
     ```

## 模型推理

打开falcon推理路径`{llm_path}/pytorch/examples/falcon` 设定当前路径为`{WORKSPACE}`

### 权重准备

1. 下载原始权重
   
   `mkdir {WORKSPACE}/bloom`


   下载falcon-7b模型权重，放置到上述路径

   ```shell
   https://huggingface.co/tiiuae/falcon-7b
   ```

### 执行推理

```shell
# 执行一下指令
export ATB_OPERATION_EXECUTE_ASYNC=1
export TASK_QUEUE_ENABLE=1
export ATB_USE_TILING_COPY_STREAM=1
export ATB_CONTEXT_WORKSPACE_RING=1

export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
export INF_NAN_MODE_ENABLE=0
```

  1. 测试单batch推理性能
      ```shell
      python3 main.py --model_path /weight/to/falcon7b --device device_number
      ```

  2. 测试推理精度mmlu
      ```shell
      wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
      tar -xvf data.tar
      python3 main.py --model_path /weight/to/falcon7b --device device_number --mode precision --dataset-path ./
      ```
  
  3、测试其它case性能
      ```shell
      python3 main.py --model_path /weight/to/falcon7b --device 0 --set_case_pair --seqlen_in_pair 256 512 1024 --seqlen_out_pair 64 128 256 --batch 1 --performance_output_file performance_bs1.csv
      ```
      


## 模型精度与性能

### 模型精度

|    模型     | mmlu  |
| :---------: | :----: |
|     GPU     | 0.2736 |
| NPU(W16A16) | 0.2730 |

### 模型性能(参考)

测试结果为2^5~2^10数据平均值

| 硬件形态 | 模型 | Batch | 首token(ms) | 非首token(ms) |
| :----:| :---: | :----: |:----: |:----: |
| Atlas 800T A2 | Falcon-7B | 1 | 59.18 | 17.24 |
|  A100 | Falcon-7B | 1 | 31.23  | 10.73 |
