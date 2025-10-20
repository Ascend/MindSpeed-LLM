# GSM8k评估

## 使用场景

### 问题描述

**GSM8K（小学数学8K）**数据集包含8500道高质量、语言多样化的小学数学应用题，旨在支持对需要多步推理的基础数学问题进行问答任务。

### 特性介绍

GSM8K评估集专注于以下方面：

 - 这些题目通常需要经过2到8个步骤来求解，其解决过程主要依赖于依次进行一系列简单的计算，利用基本的算术运算（加、减、乘、除）得到最终答案。
 - 此外，解题过程采用自然语言来描述，而非单纯的数学表达式。


目前MindSpeed-LLM仓库对GSM8K评估有两种评估模式：

## 使用方法

### 1. 直接评估模式（默认）

#### 使用影响

 - 此模式将会直接将[模板提示](../../../../../mindspeed_llm/tasks/evaluation/eval_impl/fewshot_template/gsm8k_3shot_template.json)和需要回答的问题输入到模型中，进行评估

#### 推荐参数配置

【--max-new-tokens】

设置为512。

### 2. 思维链（COT）评估模式

#### 使用影响

- 该模式会将思维链 （CoT） 提示应用于 GSM8K 任务。
- 此模式将会读取GSM8K评估的 COT 模板的 prompt 作为评估模板，在与需要模型回答的问题连接后，输入到模型中，进行评估。

#### 推荐参数配置

【--max-new-tokens】

设置为512或者以上。

【--chain-of-thought】

使能`思维链（COT）评估模式`

## 参考文献

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., & Schulman, J. (2021). *Training verifiers to solve math word problems* [Preprint]. arXiv preprint arXiv:2110.14168.