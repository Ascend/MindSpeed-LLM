[TOC]

# Qwen-14B模型-推理指导

# 快速上手
### 路径变量解释

| 变量名                 | 含义                                                                   |  
|---------------------|----------------------------------------------------------------------|
| model_download_path | 开源权重放置目录                                                             | 
| llm_path            | 加速库及模型库下载后放置目录                                                       |
| model_path          | 工作时模型所在的目录，可以和model_download_path相同，但一般模型是公共的，为了避免影响其他用户，单独建一个模型工作目录 |
| script_path         | 工作脚本所在路径，本文为${llm_path}/pytorch/examples/qwen/14b                    |
| ceval_work_dir      | ceval数据集、及结果保存所在目录，不必和模型脚本在相同目录                                      |


## 获取源码及依赖
#### python requirements

| 包名                            | 推荐版本   |  
|-------------------------------|--------|
| transformers                  | 4.30.2 | 
| decorator                     | 5.1.1  |
| sympy                         | 1.11.1 |
| scipy                         | 1.11.3 |
| attrs                         | 23.1.0 |
| psutil                        | 5.9.6  |
| sentencepiece                 | 0.1.99 |
| tiktoken                      | 0.5.2  |
| transformers-stream-generator | 0.0.4  |
| einops                        | 0.7.0  |
| pandas                        | 0.8.2  |

### 下载模型权重
下载模型权重，放置到自定义`${model_download_path}` (请下载链接中'Files and versions'页签下的所有文件)

```
https://huggingface.co/Qwen/Qwen-14B-Chat
```

### 拷贝文件

### 准备

#### 1. 将开源模型拷贝到模型工作目录，权重文件使用软链接即可,同时将modeling文件拷贝到模型，并修改开源的config.json,

```shell
cd ${model_path}
cp ${model_download_path}/*.py ./
cp ${model_download_path}/*.json ./
cp ${model_download_path}/*.tiktoken ./
cp -s ${model_download_path}/*.safetensors ./
```

#### 2. 安装 atb_speed_sdk

```shell
cd ${llm_path}/pytorch/examples/atb_speed_sdk
pip install .
```

#### 3. 张量并行模型切分（仅在模型需要多卡并行时使用）

```shell
cp ${script_path}/modeling_qwen_cut.py ${model_path}
cp ${script_path}/modeling_qwen_ascend.py ${model_path}
```

修改 ${model_path}里的config.json中的kv对，改成`"AutoModelForCausalLM": "modeling_qwen_cut.QWenLMHeadModel"`

```text
修改`${script_path}/cut_model_and_run.sh`    
将 `input_dir` 修改为模型所在路径 `${model_path}` 
将 `output_dir` 修改为切分后的模型所存储的路径，如： `${model_path}/part_model`。模型切分成功后，会自动生成新目录part_model(用户无需新建该文件夹)
将 `rank_size` 修改为期望切分的份数。例如rank_size=4表示模型切分为4份。

```

目录结构示例建议

```
--model_path
  *.py(模型源文件)
  *.json(模型源文件)
  *.tiktoken(模型源文件)
  *.bin(模型源文件,软链接)
  modeling_qwen_cut.py(权重切分脚本)
  --part_model(以双卡为例，权重切分成功后文件夹)
    --0
    --1
  ......(其他)
--script_path
  cut_model_and_run.sh
  cut_model_util.py
  main.py
  config.ini
  ......(其他)
```

执行

```shell
cd ${script_path}
bash cut_model_and_run.sh
```

切分所需时间较长，切分完成后，将会打印 'Tensor parallelism weights have been successfully saved.'。

#### 4.修改config.json配置

- 单卡运行时**必须**修改
- 多卡运行时，会在切分阶段会自动修改，没有定制的情况下，可以不操作

##### 单卡
修改${model_path}/config.json中的kv对，改成

```
"AutoModelForCausalLM": "modeling_qwen_ascend.QWenLMHeadModel"
```

##### 多卡

修改
${model_path}/part_model/{rank_id}/config.json中的kv对，改成

```
"AutoModelForCausalLM": "modeling_qwen_ascend.QWenLMHeadModel"
```

# CPU高性能模式

可开启CPU Performance模式以提高模型推理性能。

```

cpupower frequency-set -g performance

```

### 执行推理

#### 修改 ${script_path}/config.ini

[config文件配置参考](../../atb_speed_sdk/README.md)  
提示：多卡并行推理时，config.ini中model_path路径为part_model父文件夹。例如：

```
# 正确示例：

model_path=../model

# 错误示例：

model_path=../model/part_model
```

#### main.py

提供了demo推理，精度测试，性能测试三种下游任务。  
task_name可选inference、precision、performance。

- 单卡
  修改 ${model_path}里的config.json中的kv对，改成`"AutoModelForCausalLM": "modeling_qwen_ascend.QWenLMHeadModel"`

```shell
python main.py --task ${task_name}
```

- 多卡
```shell
bash cut_model_and_run.sh ${task_name}
```

**注意**
1.docker环境与conda环境有所不同，docker环境中启动模型时需要修改环境变量"ATB_OPERATION_EXECUTE_ASYNC=0"、"TASK_QUEUE_ENABLE=0"，否则可能出现算子下发同步失败。

**可以使用 MAX_SEQ_LEN 环境变量来设置model支持的最大长度以优化显存占用, 默认使用config里面的max_model_length**  
如

```shell
MAX_SEQ_LEN=2048 python main.py --task ${task_name}
```

或

```shell
MAX_SEQ_LEN=2048 bash cut_model_and_run.sh ${task_name}
```

如果遇到

```text
Traceback (most recent call last):
  File "/root/miniconda3/envs/wqh39/lib/python3.9/site-packages/torch_npu/__init__.py", line 31, in <module>
    import torch_npu.npu
  File "/root/miniconda3/envs/wqh39/lib/python3.9/site-packages/torch_npu/npu/__init__.py", line 46, in <module>
    from .utils import (is_initialized, _lazy_call, _lazy_init, init, set_dump,
  File "/root/miniconda3/envs/wqh39/lib/python3.9/site-packages/torch_npu/npu/utils.py", line 27, in <module>
    import torch_npu._C
ImportError: /root/miniconda3/envs/wqh39/bin/../lib/libgomp.so.1: cannot allocate memory in static TLS block
Segmentation fault (core dumped)
```

则在命令行前加上`LD_PRELOAD=上面的error路径`。如

```shell
LD_PRELOAD=/root/miniconda3/envs/wqh39/bin/../lib/libgomp.so.1 MAX_SEQ_LEN=2048 python main.py --task ${task_name}  --is_quant ${is_quant}
```
# 竞品对比

# Atlas 800T A2

## 精度

| 精度             | NPU                | GPU                | 对比                  |
|----------------|--------------------|--------------------|---------------------|
| STEM           | 0.6534883720930232 | 0.6534883720930232 | 0.0                 |
| Social Science | 0. 8               | 0.8                | 0.0                 |
| Humanities     | 0.7509727626459144 | 0.7470817120622568 | +0.0038910505836576 |
| Other          | 0.6458333333333334 | 0.6458333333333334 | 0.0                 |
| Avg acc        | 0.6998514115898959 | 0.6991084695393759 | +0.00074294205052   |

## 性能

### NPU

| batch_size | input length | output length | 首token耗时（ms） | 非首token平均耗时（ms） | E2E吞吐（token/s） |
|------------|--------------|---------------|--------------|-----------------|----------------|
| 1          | 256          | 64            | 51.19872284  | 27.03416824     | 32.48730964    |
| 1          | 512          | 128           | 87.46957397  | 27.23144722     | 33.33333333    |
| 1          | 1024         | 256           | 170.4371033  | 27.77638435     | 33.81770145    |
| 1          | 3584         | 512           | 658.0726929  | 30.38709259     | 30.34973325    |
| 16         | 256          | 64            | 605.9732666  | 36.35633469     | 341.3333333    |
| 8          | 512          | 128           | 618.2112427  | 35.08851242     | 190.689013     |
| 4          | 1024         | 256           | 643.7208862  | 33.481987       | 106.6666667    |
| 2          | 3584         | 512           | 1373.592163  | 35.52774429     | 50.22069642    |

注：E2E吞吐近似计算为 batch_size * output_length / (首token耗时 + 非首token平均耗时 * output_length)

- NPU max batch

| batch_size | input length | output length | 首token耗时（ms） | 非首token平均耗时（ms） | E2E吞吐（token/s） |
|------------|--------------|---------------|--------------|-----------------|----------------|
| 28         | 256          | 64            | 1095.40918   | 42.24241257     | 463.0490956    |
| 12         | 512          | 128           | 947.1630859  | 37.5304451      | 259.0219224    |
| 7          | 1024         | 256           | 1143.512451  | 37.42843246     | 161.5870153    |
| 3          | 3584         | 512           | 2014.703735  | 39.86476898     | 66.20689655    |

### GPU

| batch_size | input length | output length | 首token耗时（ms） | 非首token平均耗时（ms） | E2E吞吐（token/s） |
|------------|--------------|---------------|--------------|-----------------|----------------|
|            |              |               |              |                 |                |

### 对比

| batch_size | input length | output length | 首token耗时对比 | 非首token耗时对比 | E2E吞吐对比 |
|------------|--------------|---------------|------------|-------------|---------|
|            |              |               |            |             |         |

# Atalas推理系列产品

## 性能

浮点

| 硬件形态  | 批大小 | 输入长度     | 输出长度     | 首次推理（ms/token） | 非首次推理(ms/token) |
|-------|-----|----------|----------|----------------|-----------------|
| Duo双芯 | 1   | 2^5~2^10 | 2^5~2^10 | 327            | 103             |

量化

| batch_size | input length | output length | 首token耗时(ms) | 非首token平均耗时(ms) | E2E吞吐(token/s) |
|------------|--------------|---------------|--------------|-----------------|----------------|
| 1          | 256          | 64            | 229.4333038  | 100.8819351     | 9.523809524    |
| 1          | 512          | 128           | 389.5070496  | 101.4488068     | 9.453471196    |
| 1          | 1024         | 256           | 812.1712036  | 102.6267548     | 9.309090909    |
| 1          | 3584         | 512           | 3658.04126   | 110.3474197     | 8.376963351    |
| 16         | 256          | 64            | 2622.353027  | 121.5807648     | 98.17833174    |
| 16         | 512          | 128           | 5759.428711  | 134.2488403     | 88.61964518    |
| 12         | 1024         | 256           | 9213.712891  | 142.4198303     | 66.45035691    |
| 3          | 3584         | 512           | 9857.051758  | 132.3894196     | 19.49733435    |

## 精度

| 精度             | NPU         | GPU               | 对比                |
|----------------|-------------|-------------------|-------------------| 
| STEM           | 0.655813953 | 0.653488372093023 | 1.00355871811388  |
| Social Science | 0.8         | 0.8               | 1                 |
| Humanities     | 0.750972763 | 0.747081712062256 | 1.00520833380729  |
| Other          | 0.645833333 | 0.645833333333333 | 0.999999999483871 |
| Avg acc        | 0.700594354 | 0.699108469539375 | 1.00212539902657  |

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

- 多芯

```shell
cd ${script_path}
bash cut_model_and_run.sh precision
```

结束后在${ceval_work_dir}/test_result目录下查看测试结果。[双芯结果每个两份，只需看其中一份即可]。

| 文件                        | 用途                   | 
|---------------------------|----------------------| 
| device0.log               | 运行过程日志               |
| cache0.csv                | 结果详情，C列为预期答案，D列为测试答案 |
| result_0_classes_acc.json | 测试数据下按不同维度统计准确率      |
| result_0_subject_acc.json | 测试数据下按不同学科统计准确率      |

**注意：后续重新运行， 需要删除当前目录下生成的test_result文件夹，否则只会读取当前的目录下的测试结果**

# 性能测试

在功能运行正常的基础下，执行以下步骤进行性能测试

## 按照推理指导,下载模型及配置路径，并安装atb_speed_sdk

## 1. 准备

参考 [SDK性能测试指南精确打点法章节](../../atb_speed_sdk/README.md) 进行准备

## 2. 修改配置文件

- 配置config.ini中[performance]属性， 如下：
  ```
  model_name=qwen_14b
  perf_mode=detail
  ```

## 3. 执行测试脚本

- 单芯

```shell
cd ${script_path}
TIMEIT=1 python main.py --task performance
```

- 多芯  
```shell
cd ${script_path}
TIMEIT=1 bash cut_model_and_run.sh performance
```

为了不影响正常使用，将`TIMEIT`设置成1来返回具体的性能测试的值，默认是0

### 性能测试结果

得到性能测试结果csv `performance_test_npu_${model_name}_xxx.csv`

### 结果分析

| 列名                            | 含义         |
|-------------------------------|------------|
| batch_size                    | batch大小    |
| input_seq_len(Encoding)       | 输入长度       |
| output_seq_len(Decoding)	     | 输出长度       |
| ResponseTime(s)	              | 总响应时间      |
| forward_first_token_time(ms)  | 首token推理时长 |
| forward_next_token_time(ms)   | 增量推理时长     |
| pre_next_token_time(ms)	      | 前处理时长      |
| post_next_token_time_post(ms) | 后处理时长      |

