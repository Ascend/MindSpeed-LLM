# AGI评估

## 使用场景

### 问题描述

**AGIEval** 是一个以人为本的基准测试，专门用于评估基础模型在与人类认知和问题解决相关任务中的综合能力。该基准测试源自 20 项面向普通人类考生的官方、公开且高标准的招生和资格考试，例如普通大学入学考试（如中国的高考和美国的学术能力评估测试（SAT））、法学院入学考试、数学竞赛、律师资格考试以及国家公务员考试。

### 特性介绍

新版本更新了中国高考（化学、生物、物理）数据集，纳入了2023年的题目，并处理了标注方面的问题。为了便于评估，现在所有选择题任务都只有一个答案（高考物理和JEC-QA数据集以前有多个正确答案）。AGIEval 包含20项任务，其中包括18项选择题任务和两项完形填空任务（高考数学完形填空和MATH）。

![AGIEval](../../../../../sources/images/evaluation/AGIEval_tasks.png)


目前MindSpeed-LLM仓库对 AGI 评估有两种评估模式：

## 使用方法

### 1. 直接评估模式（默认）

#### 使用影响

 - 此模式将会直接将[模板提示](../../../../../mindspeed_llm/tasks/evaluation/eval_impl/fewshot_template/AGI_fewshot.json)和需要回答的问题连接起来输入到模型中，进行评估

#### 推荐参数配置

【--max-new-tokens】

设置为3

### 2. 平替模板输出模式

#### 使用影响

- 与`直接评估模式`不相同的是，该模式也会使用[agi_utils](../../../../../mindspeed_llm/tasks/evaluation/eval_utils/agi_utils.py)中的`template_mapping`中的固定模板和需要回答的问题连接起来输入到模型中，进行评估

#### 推荐参数配置

【--max-new-tokens】

设置为5或者以上

【--alternative-prompt】

使能`平替模板输出模式`

## 参考文献

Zhong, W., Cui, R., Guo, Y., Liang, Y., Lu, S., Wang, Y., Saied, A., Chen, W., & Duan, N. (2023). AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models [Preprint]. arXiv. https://arxiv.org/abs/2304.06364