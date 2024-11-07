#  Bloom7B 模型推理指导

# 概述

Bloom7B 是由BigScience训练的开源语言模型，BLOOM 通过 46 种自然语言和 13 种编程语言训练生成。

- 参考实现：

  ```shell
  https://huggingface.co/bigscience/bloom-7b1
  ```

+ 支持能力
  + FP16
  + INT8
  + Tensor Parallel

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
| aarch64 | Atalas推理系列产品 Duo |
|   x86   | Atalas推理系列产品 Duo |

# 快速上手

## 获取源码及依赖

1. 环境部署

   - 安装HDK

   - 安装python环境

     ```shell
     conda create -n bloom_llm python==3.9.18
     conda activate bloom_llm
     # 安装指定torch&torch_npu
     依次安装如下依赖
     transformers==4.30.2
     te==0.4.0
     pandas
     sympy
     accelerate
     ```

   - 安装CANN

     使用测试的python运行`python -c "import torch;print(torch.compiled_with_cxx11_abi())"`，若返回True，则flag=1；若返回False则flag=0；Linux中执行`arch`获取平台架构arch。version为当前使用版本，hardware根据实际执行硬件选择Atalas推理系列产品或者Atlas 900 A2 PODc，pta_version为使用的python环境中的torch版本。

     - Ascend-cann-toolkit\_{version}\_linux-{arch}.run

     - Ascend-cann-kernels-{hardware}\_{version}\_linux.run

     - Ascend-cann-atb\_{version}\_cxx11abi{flag}_linux-{arch}.run

   - 解压模型加速包

     - Ascend-cann-llm\_{version}\_linux-{arch}\_torch{pta_version}-abi{flag}.tar.gz

     ```shell
     mkdir {llm_path}
     tar -xzvf Ascend-cann-llm_{version}_linux-{arch}_torch{pta_version}-abi{flag}.tar.gz -C {llm_path}
     ```

   - 设置环境变量

     ```shell
     source /usr/local/Ascend/ascend-toolkit/set_env.sh
     source /usr/local/Ascend/atb/set_env.sh
     source {llm_path}/set_env.sh
     ```

## 模型推理

打开bloom推理路径`{llm_path}/pytorch/examples/bloom7b` 设定当前路径为`{WORKSPACE}`

### 权重准备

1. 下载原始权重
   
   `mkdir {WORKSPACE}/bloom`


   下载bloom-7b模型权重，放置到上述路径

   ```shell
   https://huggingface.co/bigscience/bloom-7b1
   ```

   下载完成后检查`{WORKSPACE}/bloom`下的文件列表如下:

   - config.json
   - flax_model-00002-of-00002.msgpack
   - pytorch_model-00001-of-00002.bin
   - pytorch_model.bin.index.json
   - tokenizer_config.json
   - flax_model-00001-of-00002.msgpack
   - flax_model.msgpack.index.json
   - pytorch_model-00002-of-00002.bin
   - special_tokens_map.json
   - tokenizer.json

2. 浮点权重切分

    执行权重切分命令
    ```Shell
    python3 handle_weights.py --input-path {WORKSPACE}/bloom --output-path {WORKSPACE}/bloom_cut --handle-type cut_float
    ```

    切分权重会模型保存在`{WORKSPACE}/bloom_cut`

3. 权重量化

    ```shell    
    # 导入量化工具路径
    # 请使用torch2.x进行量化，否则会有精度偏差
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/tools/modelslim/pytorch/llm_ptq:/usr/local/Ascend/ascend-toolkit/latest/tools:$PYTHONPATH
    
    # 执行权重量化
    python3 handle_weights.py --input-path {WORKSPACE}/bloom --output-path {WORKSPACE}/bloom_quant --handle-type quant
    ```

    量化权重会保存在`{WORKSPACE}/bloom_quant`

4. 量化权重切分

    ```shell
    # 执行量化权重切分
    python3 handle_weights.py --input-path {WORKSPACE}/bloom_quant --output-path {WORKSPACE}/bloom_quant_cut --handle-type cut_quant
    ```
    量化切分权重会保存在`{WORKSPACE}/bloom_quant_cut`

 注：1、以上指令可以通过修改`--input-path`和`--output-path`自定义输入输出路径; 2、不同设备如310和910的量化权重(包含切分和非切分权重)不通用，请重新生成

### 执行推理

请开启以下变量后进行测试：

```shell
export HCCL_BUFFSIZE=110
export PYTHONWARNINGS="ignore"
export ATB_OPERATION_EXECUTE_ASYNC=1
export TASK_QUEUE_ENABLE=1
export ATB_USE_TILING_COPY_STREAM=1
export ATB_CONTEXT_WORKSPACE_RING=1
export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
```

开启CPU Performance模式以提高模型推理性能（首次开启时，根据提示安装依赖）
```
cpupower frequency-set -g performance
```

主要参数说明：

+ model_path 模型路径
+ device npu卡号，支持单卡或双卡
+ data_dtype 数据格式，支持int8和fp16， 默认int8
+ hardware 支持310和910，默认310
+ mode 支持performance性能测试和precision精度测试，默认performance
+ dataset_path 测试精度的数据集路径

更多参数通过`python3 main.py --help`查看

执行示例：

  1. 测试推理性能

      ```shell
      # 浮点推理(双芯)
      torchrun --nproc_per_node 2 --master_port 39682 main.py --model_path {WORKSPACE}/bloom_cut --device 0 1 --data_dtype fp16 --hardware 310
      # 量化推理(双芯)
      torchrun --nproc_per_node 2 --master_port 39682 main.py --model_path {WORKSPACE}/bloom_quant_cut --device 0 1 --data_dtype int8 --hardware 310
      # 浮点推理(单芯)
      python3 main.py --model_path {WORKSPACE}/bloom --device 0 --data_dtype fp16 --hardware 310
      # 量化推理(单芯)
      python3 main.py --model_path {WORKSPACE}/bloom_quant --device 0 --data_dtype int8 --hardware 310

      # 注：
      # 可通过开启 --set_case_pair 指定输入输出序列长度，例如在不同推理脚本后加上
      --set_case_pair --seqlen_in_pair 256 512 1024 --seqlen_out_pair 64 128 256 --batch 1 --performance_output_file performance_bs1.csv

      # 或使用序列range组合，如下为输入序列长度[2^5, 2^6], 输出序列长度[2^5, 2^6], 左右闭区间
      --seqlen_in_range 5 6 --seqlen_out_range 5 6 --batch 1 --performance_output_file performance_bs1.csv

      ```

  2. 测试推理精度ceval

      ```shell
      # 安装sdk
      cd ../atb_speed_sdk
      pip install .
      cd -
      # 准备ceval数据集
      wget https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip
      unzip ceval-exam.zip -d data
      # 下载映射类别
      wget https://raw.githubusercontent.com/hkust-nlp/ceval/main/subject_mapping.json
      # 执行精度测试
      # 在1中的执行指令后添加 --mode precision --dataset_path ./
      # 可以指定dataset_path为其它路径，需下载上述文件到指定路径，完成后检查dataset_path下存在ceval-exam文件夹和subject_mapping.json文件
      # 如测试浮点双芯的精度如下：
      torchrun --nproc_per_node 2 --master_port 39682 main.py --model_path {WORKSPACE}/bloom_cut --device 0 1 --data_dtype fp16 --hardware 310 --mode precision --dataset_path ./
      # 测试量化双芯精度如下：
      torchrun --nproc_per_node 2 --master_port 39682 main.py --model_path {WORKSPACE}/bloom_quant_cut --device 0 1 --data_dtype int8 --hardware 310 --mode precision --dataset_path ./
      # 多batch请添加如：--batch 8

      # 精度测试结束后结果文件位于dataset_path下的test_result文件夹
      # 精度文件为result_classes_acc.json，记录了四个科目以及平均精度
      ```


## 模型精度与性能

### 模型精度(参考)

|    模型     | ceval  |
| :---------: | :----: |
|     GPU     | 0.2415 |
|  NPU(W8A8)  | 0.2511 |
| NPU(W16A16) | 0.2422 |

### 模型性能(参考)

测试结果为2^5~2^10数据平均值

| 硬件形态 | 数据类型 | Batch | 首token(ms) | 非首token(ms) |
| :----:| :---: | :----: |:----: |:----: |
|  Atalas推理系列产品 | W8A8 | 1 | 184.68 | 45.52 |
| Atalas推理系列产品 | W16A16 | 1 | 165.08 | 61.79 |

