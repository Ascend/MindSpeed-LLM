# HellaSwag Evaluation

## Use Cases

### Problem Description

The HellaSwag evaluation set mainly tests the performance of natural language processing models on common-sense reasoning tasks. By providing context and multiple options, the model needs to choose the most reasonable next event or sentence ending. This task design evaluates how well the model understands complex situations and reasons with common sense.

### Features

The HellaSwag evaluation set focuses on the following aspects:

1. **Common-sense reasoning ability**: It tests whether the model can understand common sense in everyday situations and choose the most reasonable answer.
2. **Adversarial filtering**: It uses adversarial filtering to ensure that the questions in the dataset are challenging and can expose model weaknesses.
3. **Diverse scenarios**: The dataset covers a variety of real-life scenarios, such as sports, cooking, and family life, which ensures diversity and complexity in the data.
4. **Multiple-choice design**: Each question contains one context description and four options, and the model must choose the most reasonable answer from the options.
5. **High-quality annotation**: The dataset undergoes strict human annotation and verification to ensure accuracy and consistency.
6. **Suitable for multiple tasks**: It can be used for model training, evaluation, and optimization, and it performs especially well in practical applications such as dialogue systems and intelligent assistants.

Currently, the MindSpeed LLM repository provides the following HellaSwag evaluation mode.

## Usage

### Direct Evaluation Mode

#### Impact

- This mode sends the question to be answered directly to the model for evaluation.

#### Recommended Parameters

`--max-new-tokens`

Set it to 32 to ensure that the task output is complete.
