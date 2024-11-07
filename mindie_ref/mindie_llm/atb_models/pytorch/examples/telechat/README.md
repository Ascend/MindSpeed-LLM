# telechat模型-推理指导  


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [基础环境搭建](#section4622531142816)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

  星辰语义大模型TeleChat是由中国电信人工智能科技有限公司研发训练的大语言模型，采用1.5万亿 Tokens中英文高质量语料进行训练。
     
- 参考实现：
  ```
  https://github.com/Tele-AI/Telechat
  ```

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表
  
  | 配套                                                         | 版本    | 下载链接                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 23.0.0  | [固件与驱动](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743/software/258924109?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C252764743) |
  | CANN（toolkit+kernels）                                     | 7.0.0   | [CANN](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/258923273?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C251168373) |
  | FrameworkPTAdapter (pytorch2.0.1)                                        | 5.0.0   | [PTA](https://gitee.com/ascend/pytorch/releases/tag/v5.0.0-pytorch2.0.1) | 
  | Python                                                     | 3.9.18   | -                                                            |            


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 基础环境搭建<a name="section4622531142816"></a>

1. 下载安装配套表中驱动固件

   根据芯片型号选择相应的安装包安装

   ```bash
   # 安装驱动，以arm为例
   chmod +x ./Ascend-hdk-{型号}-npu-driver_23.0.0_linux-aarch64.run
   ./Ascend-hdk-{型号}-npu-driver_23.0.0_linux-aarch64.run --full
   
   # 安装固件
   chmod +x ./Ascend-hdk-{型号}-npu-firmware_7.1.0.3.220.run
   ./Ascend-hdk-{型号}-npu-firmware_7.1.0.3.220.run --full
   ```

2. 下载安装cann-toolkit和cann-kernels

   ```bash
   # 安装toolkit，以arm为例
   chmod +x ./Ascend-cann-toolkit_7.0.0_linux-aarch64.run
   ./Ascend-cann-toolkit_7.0.0_linux-aarch64.run --full
   
   # 安装kernels
   chmod +x ./Ascend-cann-kernels-{型号}_7.0.0_linux.run
   ./Ascend-cann-kernels-{型号}_7.0.0_linux.run --install
   
   # 激活环境变量
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

3. 下载安装PytorchAdapter及配套python依赖

   首先安装torch，其次安装torch_npu，版本为2.0.1

   ```bash
   # 安装torch_npu，以arm为例
   pip3 install torch_npu-2.0.1.post1-cp39-cp39-linux_aarch64.whl
   
   # 安装requirements
   pip3 install -r requirements.txt
   ```

4. 下载安装AscendTransformerBoost

   具体使用cxx11abi0还是cxx11abi1，可通过python命令查询

   ```python
   import torch

   torch.compiled_with_cxx11_abi()
   ```

   若返回True，则使用cxx11abi1，否则相反

   ```bash
   # 安装atb，以arm为例
   chmod +x Ascend-cann-atb_7.0.0_linux-aarch64_abi0.run
   ./Ascend-cann-atb_7.0.0_linux-aarch64_abi0.run --install
   
   # 激活环境变量
   source /usr/local/Ascend/atb/set_env.sh
   ```

5. 编译ModelLink代码仓

   ```bash
   # clone代码仓
   git clone https://gitee.com/ascend/ModelLink.git

   # 编译
   cd ModelLink/mindie_ref/mindie_llm/atb_models
   bash scripts/build.sh

   # 激活环境变量
   source output/atb_speed/set_env.sh

   # 打开性能优化和内存优化开关
   export TASK_QUEUE_ENABLE=1 # 开启TaskQueue
   export ATB_OPERATION_EXECUTE_ASYNC=1 # Operation异步运行
   export ATB_USE_TILING_COPY_STREAM=1 # 开启多stream功能
   export ATB_LAYER_INTERNAL_TENSOR_REUSE=1 # 中间tensor内存复用
   ```

6. 下载权重

   |         权重类型            |        下载目录             | 提取码 |
   |---------------------------|---------------------------|-------|
   | [开源FP16权重](https://huggingface.co/Tele-AI/Telechat-7B/tree/main) | Telechat_float_path | - |
   | [anti outlier调优后的FP16权重](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=q9eqiGiuoKHuzT81qWOmCD0E1yNF9ApR/0pY3SD8NFzB2hq9LVTrvTX384DzbRe4kUCthwINB2AqYjojV/vBnsYIRSi3DH2M9fKvCcoxGIriP8L/m+24ZLLMvYvS1J2YIE4pRLbYigo40eg1WQRrM8D0yl/rfm4VtpmIEqMgpyzgxn+lA4QIVeBQFlESInHhWumXjIsaZkBdq8EFKL339la62w2vBcO4tG7Hs0Cav2Y9sd0nue5s9J6R5ItAg8wDKt1hD9+6fodv/08sSr+d4L2U+TJb8+yirGc5lAwZjkJSEW34tyxGH4pPl3sInyQX65apZe8sEbSgBQth8SPoKk6aHXIi7yi+QMnv8rmyD1ShHklPAxdWlCu1GNZKAnSDbVheN7Efo5rxK1YibblMPFTCco+Ts1ixhHfU5ANVnbaOstlemVW3HrBqXoDS8Lpe2iqF+4Rq+4Ti7/WkO9QQV6Xaw0gX44EGQMEFSoTZjEdE6FV3IYIu2UjbpYShU7JZof3LJ0fwDlZl8eI4V39mRP8bzqtBNfuC+irhRswUgtS4CBQWntvxfylGFVmN1pavux05e8wpu0pR7CzRWvoqTw3cbUqad6LgNCg6KAa5ZLXuUtaVUM7TQpADB0Tj+X5dLBMVYF8/vxVG3Ig7unlVC6FY6iD7bPLpXChCB3a6Ihw=) | Telechat_anti_float_path | lm089j |
   | [int8量化权重](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=IhnIgsIFijGUmoRXfi+V3dE/Yk655wSiLEXk5brO4lg0+i06AZBtcWSzhkhM7MJFZBzKnbtQIfF1YTEINITs9MWuLZd7BqONhzK+UxfQKBy6ho8nKxLVHNbqz3/4oRM+vYmDAdX3qID9h2UWoOcVmGFFmfaerurmCjW9+4wh1Au58beAgxrNOpXCwdfvJWoIfUQl/p85JpsI8eLzedtFzu3e7XtjL1rw+fdHrH36h4nRInao29feCNVVheDrGOz9v9YmtpYvaHclobd+bpQxoG9ZXT2+nejp9yi8CftTfmQqlBzHsSvTdNC9h8/HP0K1AKAY/ImOUERMJsgXo83c6QcE5Op1NanzIrkzY30YKL9dYr1nE2iqJayi0Skq+N2148/EU1xcUurIaKMIMgVcxdaPnWEkixZqMFnpOvV9WnxO4ixIE++LbGiR1n3Yeevol1Is+JrtVmBb5OsSaiup50B4iRpbv4Srq2FxyyOojhPZwfcFNrVy11O4725NUVfhImZ+8ztEsR4BSXPIaYYlmu+jmle75TonhlufCjBWLvbnYELf+Y3jbsZsxfifLSSztkMiXlncedXig7ztax5ZJEJRTOGCyF5niisJTPIQIftZSLQvNUiDGYXj8JmSnxCUIpVvzooubI/JZ2Xv8jqm9M5r5jSj8WGHy79+PC88WHxqymG4/00Q1TmFlcDW7rgT) | Telechat_quant_path | xf9jl2 |


## 模型推理<a name="section741711594517"></a>

1. 开始精度验证。

   - 场景一：使用提供的fp16权重，自行量化

       该模型在昇腾适配使用的技术栈为anti_outlier + ptq
       
       ```bash
       cd ModelLink/mindie_ref/mindie_llm/atb_models/pytorch/examples/telechat
       patch -p0 /usr/local/Ascend/ascend-toolkit/latest/tools/modelslim/pytorch/llm_ptq/anti_outlier/dag_utils/model_structure_process.py < model_structure_process.patch
       cp modeling_telechat_torch_npu.py $Telechat_float_path/modeling_telechat.py
       python3 quant_calib_anti_outlier.py --level=xx --jsonl_path=xx --checkpoint_path=xx
       ```
       - 命令参数说明：       
         -   `--level`：量化回退等级，默认为L5
         -   `--jsonl_path`：量化校准集路径
         -   `--checkpoint_path`：开源FP16权重路径，即$Telechat_float_path
     
   - 场景二：使用提供的int8量化权重
   
       新建test.jsonl文件，按以下格式写入您想要提问的问题
       ```
       # example
       {"input": "编写一首与春天有关的诗歌。"}
       {"input": "根据给定的故事情节编写结局。以下是一篇故事的示例，故事情节应该具有足够的信息以使参与者编写自己的结局。“小明和小红是俩好朋友，他们一起长大，彼此之间的友谊非常稳固。他们的家庭都住在同一个小区，而且也是邻居。自从小明的父亲因为工作原因要外派到国外，小明便非常的寂寞。夜里，小明会偷偷地默默地流泪，每当他想起小红时，他就更加难过。小红察觉到小明的变化之后，决定在小明的父亲离开之前做点什么来让他感到开心一些。"}
       {"input": "以下是一段两人的对话内容：小明：老师李，我这次数学考试考了多少分？老师李：小明，你考了60分。小明：60分？我怎么可能考这么低分啊！老师李：你平时上课不认真，不做作业，只顾着玩游戏，这是应得的结果。小明：可是我可以补考吗？老师李：可以，但是你必须得用这段时间好好复习，不能再浪费时间了。根据这段对话内容回答问题：小明为什么会考这么低的分数？"}
       ```
       **执行推理脚本**
       
       1. 修改`ModelLink/mindie_ref/mindie_llm/atb_models/pytorch/examples/telechat/run.sh`脚本中
            - 浮点权重路径: `FLOAT_MODEL_PATH`
            - 浮点切分权重路径: `FLOAT_PART_MODEL_PATH`
            - 量化权重路径: `QUANT_MODEL_PATH`
            - 量化切分权重路径: `QUANT_PART_MODEL_PATH`
            - 问题文件路径: `INPUT_JSON_PATH`
            - 答案文件路径: `OUTPUT_JSON_PATH`
            - 运行卡ID: `DEVICE` 
       ```bash
       # 单卡量化推理
       bash run.sh --run_performance=0 --run_parallel=0 --run_quant=1
      
       # 多卡量化切分并推理
       bash run.sh --run_performance=0 --run_parallel=1 --run_quant=1 --cut_quant
       
       # 多卡量化推理（跳过切分）
       bash run.sh --run_performance=0 --run_parallel=1 --run_quant=1

       # 单卡浮点推理
       bash run.sh --run_performance=0 --run_parallel=0 --run_quant=0
       # 多卡浮点切分并推理
       bash run.sh --run_performance=0 --run_parallel=1 --run_quant=0 --cut_float
       # 多卡浮点推理（跳过切分）
       bash run.sh --run_performance=0 --run_parallel=1 --run_quant=0
       ```
       - 命令参数说明：       
         -   `--run_performance`：为0时运行精度测试，为1时运行性能测试
         -   `--run_parallel`：为0时单卡运行，为1时双卡运行
         -   `--run_quant`：为0时运行浮点模型，为1时运行量化模型
         -   `--cut_quant`: 将antioutlier浮点权重切分，并同时切分量化权重
         
   **效果方面，完成了和基于A10卡模型量化效果对齐**
 
2. 开始性能验证

   a. 修改`ModelLink/mindie_ref/mindie_llm/atb_models/pytorch/examples/telechat/run.sh`脚本中
      - 浮点权重路径: `FLOAT_MODEL_PATH`
      - 浮点切分权重路径: `FLOAT_PART_MODEL_PATH`
      - 量化权重路径: `QUANT_MODEL_PATH`
      - 量化切分权重路径: `QUANT_PART_MODEL_PATH`
      - 问题文件路径: `INPUT_JSON_PATH`
      - 答案文件路径: `OUTPUT_JSON_PATH`
      - 运行卡ID: `DEVICE` 
      ```bash
      # 单卡量化推理
      bash run.sh --run_performance=1 --run_parallel=0 --run_quant=1
   
      # 多卡量化切分并推理
      bash run.sh --run_performance=1 --run_parallel=1 --run_quant=1 --cut_quant
      
      # 多卡量化推理（跳过切分）
      bash run.sh --run_performance=1 --run_parallel=1 --run_quant=1

      # 单卡浮点推理
      bash run.sh --run_performance=1 --run_parallel=0 --run_quant=0
      # 多卡浮点切分并推理
      bash run.sh --run_performance=1 --run_parallel=1 --run_quant=0 --cut_float
      # 多卡浮点推理（跳过切分）
      bash run.sh --run_performance=1 --run_parallel=1 --run_quant=0
      ```
      - 命令参数说明：       
         -   `--run_performance`：为0时运行精度测试，为1时运行性能测试
         -   `--run_parallel`：为0时单卡运行，为1时双卡运行
         -   `--run_quant`：为0时运行浮点模型，为1时运行量化模型
         -   `--cut_quant`: 将antioutlier浮点权重切分，并同时切分量化权重
 
   脚本中一共测试了25组case，其中部分case具体比对效果如下：
   
    | 输入输出信息                           | NPU (tokens/s)    |
    | ------------------------------------ | ------- |
    | 输入100输出100                         | 15  |
    | 输入1000输出100                        | 13   |
    | 输入2000输出100                        | 11   |
    | 25组case平均                           |  13  |
