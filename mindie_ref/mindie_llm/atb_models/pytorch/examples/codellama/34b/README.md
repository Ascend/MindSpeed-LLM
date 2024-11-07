# CodeLlama-34b 推理指导

## 概述

CodeLlama是一组经过预训练和微调的生成文本模型，其参数量从70亿到340亿不等。

模型下载：https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf

### 模型功能:

- [√] Code completion.
- [ ] Infilling.
- [√] Instructions / chat.
- [ ] Python specialist.

### 输入输出

输入：仅对文本输入进行建模
输出：仅生成文本

### 状态

这是在离线数据集上训练的静态模型。Code Llama - Instruct 的未来版本将发布，因为我们根据社区反馈改进了模型安全性。

### 预期用途

#### 预期用例

CodeLlama及其变体旨在用于英语和相关编程语言的商业和研究用途；
基本模型CodeLlama可以适应各种代码合成和理解任务，CodeLlama - Python专门设计用于处理Python编程语言，CodeLlama - Instruct旨在更安全地用于代码助手和生成应用程序。

#### 超出范围的使用

以任何违反适用法律或法规（包括贸易合规法）的方式使用；
以英语以外的语言使用；
以CodeLlama及其变体的可接受使用政策和许可协议禁止的任何其他方式使用。

### 道德考虑和限制

CodeLlama及其变体是一种新技术，使用时存在风险。迄今为止进行的测试是用英语进行的，没有涵盖，也不能涵盖所有情况。由于这些原因，与所有LLM一样，CodeLlama的潜在输出无法提前预测，并且在某些情况下，该模型可能会对用户提示产生不准确或令人反感的响应。因此，在部署CodeLlama的任何应用程序之前，开发人员应执行针对其模型的特定应用程序量身定制的安全测试和调整。

## 加速库接入（暂不支持model级）

### 代码结构：

#### 加速库
model/codellama/34b
|---operation
    |---position_embedding.h
    |---position_embedding.cpp
    |---self_attention.h
    |---self_attention.cpp
    |---self_attention_kv_cache.cpp
|---layer
    |---layer_parallel.h
    |---encoder_parallel_layer.cpp
    |---decoder_parallel_layer.cpp

#### 模型脚本
pytorch/examples/codellama_34b
|---patches
    |---modelling_llama_layer_performance.py
    |---modelling_llama_layer_precision.py
|---cut_model_and_run_code_llama.sh
|---cut_model_util.py
|---modeling_llama_parallel.py
|---run_codellama_half_parallel.py

### 比较加速库精度（单层）
modelling_llama_layer_precision.py替换modeling_llama_parallel.py
>>bash cut_model_and_run_code_llama.sh

### 加速库替换及性能测试
modelling_llama_layer_performance.py替换modeling_llama_parallel.py
>>bash cut_model_and_run_code_llama.sh
（测试模型性能时需注意cpu和npu的争用问题）

## 竞品对比

### 精度(HumanEval)

|          | Atlas 800T A2               | 参考               | 对比              |
|----------|---------------------|--------------------|-------------------| 
| 问题总数  | 164                 | 164                | 1                 |
| 通过总数  | 54                  | 53                 | 1.018867924528302 |
| pass@1   | 0.32926829268292684 | 0.3231707317073171 | 1.018867924528302 |

### 性能

| 设备         | batch_size | 首token推理速度(token/s) | 增量推理速度(token/s) |
|--------------|------------|-------------------------|----------------------|
| Atlas 800T A2        | 1          | 1.238550074221983       | 8.932253698655556    |
| 参考         | 1          | 18.12342392948192       | 16.55063728397924    |
| 对比         | 1          | 0.06833973972253644     | 0.5396924327078233   |

## 精度测试指南

### 1. 下载测试数据集

https://github.com/openai/human-eval/tree/master/data

环境：transformers版本：4.33.1

### 2. 下载测试脚本

下载\human_eval
下载数据脚本：
https://github.com/openai/human-eval/blob/master/data/HumanEval.jsonl.gz

1、新建文件夹{CodeLlama-34b-Instruct-hf}，将模型内容上传至该文件夹，切割后的模型放在/CodeLlama-34b-Instruct-hf_test

2、将下列测试文件放在CodeLlama-34b-Instruct-hf_test的上一级目录
--execute.py
--human-eval-v2-20210705.jsonl
--parallel_model_test.py

### 3. 启动测试

>>torchrun --nproc_per_node 2 parallel_model_test.py --model_path={model_path}
可指定端口：--master_port={端口号}
    如：torchrun --nproc_per_node 2 --master_port=29501 parallel_model_test.py --model_path={model_path}

结束后在终端查看测试结果。

### 4. 精度计算

运行后得到评测分数：  
pass@1：human_eval数据集评测分数

