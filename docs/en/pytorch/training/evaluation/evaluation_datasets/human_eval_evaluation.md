# HumanEval Evaluation

## Use Cases

### Problem Description

- In the field of evaluating code generation in LLMs, accurately measuring the quality and correctness of generated code is a key problem. The HumanEval dataset provides an effective solution to this challenge.
- It aims to evaluate code generation models objectively and systematically, determine how reliable and practical their generated code is in real scenarios, and help optimize and improve the model so that it better meets the needs of practical applications such as software development.

### Features

The HumanEval dataset has several notable features:

- It contains a series of representative programming problems that cover many common programming task scenarios, such as data structure operations, algorithm implementation, and string processing. These problems can comprehensively assess code generation models across different domains.
- Each problem in the dataset includes a detailed description and corresponding test cases, which provide a clear standard and basis for evaluation and ensure that the evaluation results are objective and accurate.
- Its problem design emphasizes code readability and consistency, which encourages the model to generate high-quality, maintainable code that matches the requirements of real development. Therefore, the dataset provides a comprehensive assessment of code generation models from multiple angles and helps improve model performance so that it can adapt to complex and changing programming application scenarios.

MindSpeed LLM evaluates the content in the HumanEval question set.

## Usage

### 1. Direct Evaluation Mode (Default)

#### Impact

- MindSpeed LLM does not use any prompt template for HumanEval evaluation. Instead, it evaluates the target question directly and outputs the final answer. In other words, the model receives the question directly, without any prompt template.
- This mode uses beam search for model inference.
- After the LLM outputs an answer for each question, the program uses `logger.info` to indicate whether the question passes evaluation.

#### Recommended Parameters

`--max-new-tokens`

Set this to 1024 to ensure that the code is output completely.

### 2. Alternative Template Output Mode

#### Impact

- Like direct evaluation mode, this evaluation mode also does not use a prompt template.
- This mode uses the standard inference method for model output. Therefore, it does not use sampling or beam search for inference.
- Unlike direct evaluation mode, this mode judges the inference results of the model only after all questions have finished inference.

#### Recommended Parameters

`--alternative-prompt`

Enable alternative template output mode.

`--max-new-tokens`

Set this to 1024 to ensure that the code is output completely.
