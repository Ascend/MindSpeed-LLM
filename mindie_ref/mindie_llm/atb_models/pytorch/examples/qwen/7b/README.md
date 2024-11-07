[TOC]

# Qwen-7B模型-推理指导

注意，QWen-7b与14b版本模型结构一致，因此加速库及modeling等文件可复用，此处不再重复归档

# 快速上手
### 路径变量解释

| 变量名                 | 含义                                                                  |  
|---------------------|---------------------------------------------------------------------|
| model_download_path | 开源权重放置目录                                                            | 
| llm_path            | 加速库及模型库下载后放置目录                                                      |
| model_path          | 工作时模型所在的目录，可以和model_download_path相同，但一般模型是公共的，为了避免影响其他用户，单独建一个模型工作目录 |
| script_path         | 工作脚本所在路径，本文为${llm_path}/pytorch/examples/qwen/7b                    |
| ceval_work_dir      | ceval数据集、及结果保存所在目录，不必和模型脚本在相同目录                                     |


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
https://huggingface.co/Qwen/Qwen-7B
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
将 `output_dir` 修改为切分后的模型所存储的路径，如：`${model_path/part_model}`。模型切分成功后，会自动生成新目录part_model(用户无需新建该文件夹)
将 `rank_size` 修改为期望切分的份数。例如rank_size=4表示模型切分为4份。

```

目录结构示例建议

```
--model_path
  *.py(模型源文件)
  *.json(模型源文件)
  *.tiktoken(模型源文件)
  *.bin(模型源文件，软链接，部分模型权重为其它格式，如*.safetensors等)
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

注意，由于本模型体量较大，受硬件限制，单卡很可能无法跑起。

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

| 精度           | NPU                | GPU                | 对比 |
|----------------|--------------------|--------------------|-----|
| STEM           | 0.5302325581395348 | 0.5302325581395348 | 0.0 |
| Social Science | 0.6981818181818182 | 0.6981818181818182 | 0.0 |
| Humanities     | 0.6459143968871596 | 0.6459143968871596 | 0.0 |
| Other          | 0.5755208333333334 | 0.5755208333333334 | 0.0 |
| Avg acc        | 0.5995542347696879 | 0.5995542347696879 | 0.0 |

## 性能

### NPU

| batch_size | input length | output length | 首token耗时（ms）  | 非首token平均耗时（ms）| E2E吞吐（token/s）  |
|------------|--------------|---------------|-------------------|-----------------------|--------------------|
| 1          | 256          | 256           | 35.63896179199219 | 15.51435661315918     | 56.888888888888886 |
| 1          | 512          | 512           | 56.39886856079102 | 15.620501518249512    | 56.76274944567628  |
| 1          | 1024         | 1024          | 99.1506576538086  | 16.131181716918945    | 56.95216907675194  |
| 1          | 2048         | 2048          | 200.5741577148438 | 17.240901947021484    | 53.58451072736787  |
| 2          | 256          | 256           | 52.08611297607422 | 16.329647064208984    | 107.78947368421052 |
| 2          | 512          | 512           | 92.5691146850586  | 16.70061683654785     | 106.22406639004149 |
| 2          | 1024         | 1024          | 181.1108551025391 | 17.5529842376709      | 104.48979591836734 |
| 2          | 2048         | 2048          | 387.6204528808594 | 19.275894165039062    | 95.30013959981387  |
| 4          | 256          | 256           | 90.84725189208984 | 16.85601806640625     | 210.26694045174537 |
| 4          | 512          | 512           | 171.7660369873047 | 17.603639602661133    | 201.7733990147783  |
| 4          | 1024         | 1024          | 358.7157592773438 | 19.047927856445312    | 192.66227657572907 |
| 4          | 2048         | 2048          | 774.1217651367188 | 22.109966278076172    | 166.8092038281409  |
| 8          | 256          | 256           | 169.1129150390625 | 18.627500534057617    | 371.68784029038113 |
| 8          | 512          | 512           | 337.9776611328125 | 20.07361602783203     | 349.7865072587532  |
| 8          | 1024         | 1024          | 716.6774291992188 | 22.932018280029297    | 317.51937984496124 |
| 8          | 2048         | 2048          | 1540.99267578125  | 29.060359954833984    | 252.17792827458828 |

注：E2E吞吐近似计算为 batch_size * output_length / ResponseTime

- NPU max batch

暂无数据

### GPU

暂无数据

# Atalas推理系列产品

## 精度

| 精度           | NPU                | GPU                | 对比                  |
|----------------|--------------------|--------------------|-----------------------|
| STEM           | 0.5325581395348837 | 0.5302325581395348 | 0.0023255813953488857 |
| Social Science | 0.6981818181818182 | 0.6981818181818182 | 0.0 |
| Humanities     | 0.6459143968871596 | 0.6459143968871596 | 0.0 |
| Other          | 0.5755208333333334 | 0.5755208333333334 | 0.0 |
| Avg acc        | 0.600297176820208  | 0.5995542347696879 | 0.0007429420505200568 |

## 性能

### NPU

| batch_size | input length | output length | 首token耗时（ms）  | 非首token平均耗时（ms）| E2E吞吐（token/s）  |
|------------|--------------|---------------|-------------------|-----------------------|--------------------|
| 1          | 256          | 256           | 224.4498748779297 | 105.22329711914062    | 9.295570079883806  |
| 1          | 512          | 512           | 381.6845397949219 | 106.7011489868164     | 9.154300017879493  |
| 1          | 1024         | 1024          | 847.4793701171875 | 108.98991394042969    | 8.958096404514041  |
| 1          | 2048         | 2048          | 1992.810302734375 | 113.78995513916016    | 8.565453785027186  |
| 2          | 256          | 256           | 324.6171569824219 | 105.1316909790039     | 18.48375451263538  |
| 2          | 512          | 512           | 709.3186645507812 | 107.80441284179688    | 17.98068481123793  |
| 2          | 1024         | 1024          | 1556.345458984375 | 112.66254425048828    | 17.20574645047467  |
| 2          | 2048         | 2048          | 3393.751953125    | 122.02436828613281    | 15.894450911913077 |
| 4          | 256          | 256           | 616.9865112304688 | 111.66754913330078    | 34.52461227242077  |
| 4          | 512          | 512           | 1309.432006835938 | 116.58833312988281    | 33.05358295674629  |
| 4          | 1024         | 1024          | 2811.6142578125   | 126.20862579345703    | 30.530709600477042 |
| 4          | 2048         | 2048          | 6232.23046875     | 145.92770385742188    | 26.472774276942964 |

- NPU max batch

| batch_size | input length | output length | 首token耗时（ms） | 非首token平均耗时（ms） | E2E吞吐（token/s） |
|------------|--------------|---------------|------------------|------------------------|-------------------|
| 22         | 256          | 256           | 3162.53833007813 | 146.51502990722656     | 137.0983446932814 |
| 14         | 512          | 512           | 4227.68017578125 | 145.74563598632812     | 89.80205462290154 |
| 8          | 1024         | 1024          | 5255.033203125   | 145.74539184570312     | 52.3082817189196  |
| 6          | 2048         | 2048          | 4334.11669921875 | 136.3438720703125      | 42.01887566680345 |

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
  model_name=qwen_7b
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

