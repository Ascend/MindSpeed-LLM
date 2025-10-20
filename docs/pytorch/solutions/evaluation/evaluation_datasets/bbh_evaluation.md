# BBH评估

## 使用场景

### 问题描述

**BIG-Bench Hard （BBH）**专注于一套 23 项具有挑战性的 BIG-Bench 任务。BIG-Bench Hard（BBH）是 BIG-Bench 评测套件的一个子集，专门挑选出了 23 项特别具有挑战性的任务，这些任务在过去的评估中往往未能达到或超过人类平均水平。这些任务的设计初衷在于：

### 特性介绍

BBH评估集专注于以下几个方面：

 - **多步骤推理要求**：BBH 中的许多任务要求模型进行复杂的、多步骤的逻辑推理和推断，而简单的“仅回答”提示往往无法充分展现模型的最佳能力。
- **评价现有模型的潜力**：在最初的 BIG-Bench 评测中，通过少样本提示（few-shot prompting）测试时，大多数模型的表现未能超越人类平均水平。然而，研究发现，当引入链式思维（Chain-of-Thought, CoT）提示时，模型可以在任务中展现出更深层次的推理能力。例如，PaLM 模型在 23 个任务中有 10 项超越了人类评分标准，而 Codex 模型在 17 项上超过了这一基线。
- **挖掘隐含能力**：BBH 的设计揭示了一个重要事实：传统的提示方法可能低估了模型在复杂推理任务中的潜力，而 CoT 提示则能够让模型通过“思考过程”逐步拆解问题，最终取得更好的表现。
- **指导未来研究**：BBH 为 AI 研究者提供了一个专门的工具，帮助他们深入分析和改进大语言模型在多步骤逻辑推理上的表现。这不仅为模型架构的改进提供了依据，也推动了提示设计和推理技术的发展。


目前MindSpeed-LLM仓库对BBH评估有三种评估模式：

## 使用方法

### 1. 直接评估模式（默认）

#### 使用影响

 - 此模式将会把需要回答的问题直接输入到模型中，进行评估


#### 推荐参数配置

【--max-new-tokens】

设置为32，确保[word_sorting](https://github.com/suzgunmirac/BIG-Bench-Hard/blob/main/bbh/word_sorting.json)等需要长输出任务可以输出完全。

### 2. 微调模板评估模式

#### 使用影响

 - 此模式将会读取bbh评估的[模板的文件](../../../../../mindspeed_llm/tasks/evaluation/eval_impl/fewshot_template/bbh_template.json)作为评估模板，在与需要模型回答的问题连接后，输入到模型中，直接进行评估。

 - 请注意该模式使用的模板，不是思维链（chain of thought）的模板风格


#### 推荐参数配置

【--max-new-tokens】

设置为32，确保[word_sorting](https://github.com/suzgunmirac/BIG-Bench-Hard/blob/main/bbh/word_sorting.json)等需要长输出任务可以输出完全。

【--prompt-type】

该参数用于指定模型模板类型，应与您在使用MindSpeed-LLM仓进行微调时配置的`--prompt-type`参数保持一致。

### 3. 思维链（CoT）评估模式

#### 使用影响

- 该模式会将思维链 （CoT） 提示应用于 BBH 任务。
- 此模式将会读取bbh评估的[CoT模板的文件](../../../../../mindspeed_llm/tasks/evaluation/eval_impl/fewshot_template/bbh_cot_template.json)作为评估模板，在与需要模型回答的问题连接后，输入到模型中，进行评估。

#### 推荐参数配置

【--max-new-tokens】

设置为512或者以上

【--chain-of-thought】

使能`思维链（CoT）评估模式`

## 参考文献

Srivastava, A., Rastogi, A., Rao, A., Shoeb, A. A. M., Abid, A., Fisch, A., Brown, A. R., Santoro, A., Gupta, A., Garriga-Alonso, A., *et al.* (2022). *Beyond the imitation game: Quantifying and extrapolating the capabilities of language models* [Preprint]. arXiv. https://arxiv.org/abs/2206.04615