# BoolQ评估

## 使用场景

### 问题描述

BoolQ 是由 Google 研究院构建的布尔型问答数据集，具有以下核心特征：

 - 数据规模：包含 15,942 个（问题，段落，答案）三元组

 - 自然生成：问题均来自真实搜索场景，而非人工构造

 - 复杂语境：平均段落长度达 163 个单词，需要深度语义理解

 - 领域分布：涵盖 650+ 不同主题，包含多领域知识

### 特性介绍

BoolQ 数据集版本由三个 `.jsonl` 文件组成，其中每行是具有以下格式的 JSON 字典：

```json
{
  "question": "is france the same timezone as the uk",
  "passage": "At the Liberation of France in the summer of 1944, Metropolitan France kept GMT+2 as it was the time then used by the Allies (British Double Summer Time). In the winter of 1944--1945, Metropolitan France switched to GMT+1, same as in the United Kingdom, and switched again to GMT+2 in April 1945 like its British ally. In September 1945, Metropolitan France returned to GMT+1 (pre-war summer time), which the British had already done in July 1945. Metropolitan France was officially scheduled to return to GMT+0 on November 18, 1945 (the British returned to GMT+0 on October 7, 1945), but the French government canceled the decision on November 5, 1945, and GMT+1 has since then remained the official time of Metropolitan France.",
  "answer": false,
  "title": "Time in France"
}
```

这些文件分别是：

- **train.jsonl**: 9427 个带标签的训练示例
- **dev.jsonl**: 3270 个带标签的开发示例
- **test.jsonl**: 3245 个未标记的测试示例

MindSpeed-LLM 会对`dev`问题集中的内容进行评估。

## 使用方法

### 1. 直接评估模式（默认）

#### 使用影响

 - MindSpeed-LLM 对BoolQ的评估不会使用任何的提示板，而是直接对目标问题进行评估和输出最终答案。即模型直接接收问题-段落对，无需任何提示模板。

 - 输出层计算"Yes"/"No"的token概率，通过概率比较确定最终预测：P(Yes) > P(No) → 标记为 True  |  其他情况 → 标记为 False。

 - 注意：准确率可能虚高3-5%

#### 推荐参数配置

【--max-new-tokens】

设置为3或者4

### 2. 平替模板输出模式

#### 使用影响

 - 与`直接评估模式`相同的是，此种评估模式也不会使用任何的提示。与直接评估模式不同的是，该模式会希望模型输出选项`A`或`B`，分别对应答案的`True`和 `False`。即A → 对应 True，B → 对应 False

#### 推荐参数配置

【--alternative-prompt】

使能`平替模板输出模式`

【--max-new-tokens】

设置为3或者4

【--origin-postprocess】

若使能该参数，模型输出的答案会经过答案的映射mapping。例如，模型没有按照预期输出选项`A`或`B`，而是输出了`True`。若使能该参数，则会将`True`重新映射回`A`。

