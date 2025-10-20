# MMLU评估

## 使用场景

### 问题描述

MMLU（Massive Multitask Language Understanding）评估包含一系列多样化的任务和领域，旨在全面测试模型的理解能力和知识广度。具体来说，MMLU涵盖了以下主要领域：

1. **人文学科**：如历史、哲学、文学等。
2. **社会科学**：如心理学、社会学、经济学等。
3. **STEM（科学、技术、工程和数学）**：如数学、物理、化学、生物等。
4. **其他专业领域**：如法律、医学、商业等。

每个领域下又包含多个具体的任务和问题，通过这些多样化的任务，MMLU能够评估模型在不同领域的知识掌握情况和跨领域的泛化能力。

其中，[STEM](#STEM)，[人文学科](#人文学科)和[社会科学](#社会科学)所包含的学科问题集会在文末说明。

目前MindSpeed-LLM仓库对MMLU评估有三种评估模式：

## 使用方法

### 1. 直接评估模式

#### 使用影响

此模式将会读取对外公开的[MMLU评估模板的文件](../../../../../mindspeed_llm/tasks/evaluation/eval_impl/fewshot_template/mmlu_5shot_template.json)，在与需要模型回答的问题连接后，输入到模型中，直接进行评估。

此种模式下，模型的第一个输出将会作为答案。

该模式的优势是直接且速度快，可以直接对模型的预训练权重进行评估。

#### 推荐参数配置

【--max-new-tokens】

设置为1或者2

### 2. 微调模板评估模式

#### 使用影响

此模式将会读取您的启动脚本中的`DATA_PATH`路径中的同级`dev`文件夹中对应问题的以`_dev.csv`为后缀的文件，作为模板问题，经处理后输入到模型中。

与直接评估模式不同的是，该模式会根据种子数，打乱dev文件中的模板问题的顺序。再与需要模型回答的问题连接后，再进行对话模板处理后，将得到的对话字典输入到模型中进行评估。

该模式的优势是评估速度较快，适用于对模型的微调权重进行评估。

#### 推荐参数配置

【--max-new-tokens】

设置为1或者2

【--prompt-type】

该参数用于指定模型模板类型，应与您在使用MindSpeed-LLM仓进行微调时配置的`--prompt-type`参数保持一致。

### 3. 平替模板输出模式

#### 使用影响

与`微调模板评估模式`相同的是，该模式也会使用您评估脚本中的`DATA_PATH`路径中的同级`dev`文件夹中对应问题的以`_dev.csv`为后缀的文件，并作为模板问题。

与其他模式不同的是，该模式不会打乱模板问题的顺序。模板问题与需要模型回答的问题连接后，不进行对话字典的处理，直接输入到模型，得到前向输出。

该模式的优势是可以使用与业界优秀评估方案相同的评估模板进行评估，并获得较好的评估分数。

#### 推荐参数配置

【--max-new-tokens】

设置为128或者以上

【--alternative-prompt】

使能`平替模板输出模式`

### 4. PPL模式

#### 使用影响

`ppl` 是困惑度 (perplexity) 的缩写，是一种评价模型进行语言建模能力的指标。该模式也会使用您评估脚本中的`DATA_PATH`路径中`dev`文件夹中对应问题的以`_dev.csv`为后缀的文件，并作为模板问题。此时，我们会将 n 个选项拼接上上下文后，形成 n 个序列，然后计算模型对这 n 个序列的 perplexity，我们认为其中 perplexity 最低的序列所对应的选项即为模型在这道题上面的推理结果，该种评测方法的后处理简单直接、确定性高。

#### 推荐参数配置

【--task】

设置为 `mmlu_ppl`

## MMLU评估的子集

<a name="STEM"></a>
### **STEM** 包含的问题集：

1. **abstract_algebra**（抽象代数）
2. **astronomy**（天文学）
3. **college_biology**（大学生物学）
4. **college_chemistry**（大学化学）
5. **college_computer_science**（大学计算机科学）
6. **college_mathematics**（大学数学）
7. **college_physics**（大学物理学）
8. **computer_security**（计算机安全）
9. **conceptual_physics**（概念物理学）
10. **electrical_engineering**（电气工程学）
11. **elementary_mathematics**（初等数学）
12. **high_school_biology**（高中生物学）
13. **high_school_chemistry**（高中化学）
14. **high_school_computer_science**（高中计算机科学）
15. **high_school_mathematics**（高中数学）
16. **high_school_physics**（高中物理学）
17. **high_school_statistics**（高中统计学）
18. **machine_learning**（机器学习）

<a name="人文学科"></a>
### **人文学科** 包含的问题集：

1. **formal_logic**（形式逻辑）
2. **high_school_european_history**（高中欧洲历史）
3. **high_school_us_history**（高中美国历史）
4. **high_school_world_history**（高中世界历史）
5. **international_law**（国际法）
6. **jurisprudence**（法理学）
7. **logical_fallacies**（逻辑谬误）
8. **moral_disputes**（道德争议）
9. **moral_scenarios**（道德场景）
10. **philosophy**（哲学）
11. **prehistory**（史前史）
12. **professional_law**（职业法律）
13. **world_religions**（世界宗教）

<a name="社会科学"></a>
### **社会科学** 包含的问题集：

1. **clinical_knowledge**（临床知识）
2. **college_medicine**（大学医学）
3. **global_facts**（全球事实）
4. **human_aging**（人类衰老）
5. **human_sexuality**（人类性行为）
6. **marketing**（市场营销）
7. **medical_genetics**（医学遗传学）
8. **miscellaneous**（杂项）
9. **nutrition**（营养学）
10. **professional_accounting**（专业会计）
11. **professional_medicine**（专业医学）
12. **public_relations**（公共关系）
13. **security_studies**（安全研究）
14. **sociology**（社会学）
15. **us_foreign_policy**（美国外交政策）