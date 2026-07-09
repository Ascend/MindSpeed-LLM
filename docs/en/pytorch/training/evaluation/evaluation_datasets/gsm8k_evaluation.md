# GSM8K Evaluation

## Use Cases

### Problem Description

The **Grade School Math 8K (GSM8K)** dataset contains 8,500 high-quality, linguistically diverse grade-school math word problems. It is designed to support question-answering tasks for basic math problems that require multi-step reasoning.

### Features

The GSM8K evaluation set focuses on the following aspects:

- These problems usually require 2 to 8 steps to solve. The solution process mainly depends on carrying out a series of simple calculations in sequence and using basic arithmetic operations, including addition, subtraction, multiplication, and division, to obtain the final answer.
- In addition, the solution process is described in natural language rather than as a simple mathematical expression.

Currently, the MindSpeed LLM repository provides two GSM8K evaluation modes.

## Usage

### 1. Direct Evaluation Mode (Default)

#### Impact

- This mode concatenates the [template prompt](../../../../../../mindspeed_llm/tasks/evaluation/eval_impl/fewshot_template/gsm8k_3shot_template.json) with the question to be answered, then feeds the result to the model for evaluation.

#### Recommended Parameters

`--max-new-tokens`

Set this to 512.

### 2. Chain-of-Thought (CoT) Evaluation Mode

#### Impact

- This mode applies CoT prompts to GSM8K tasks.
- This mode uses the CoT template prompt as the evaluation template, concatenates it with the question the model needs to answer, and then feeds the result to the model for evaluation.

#### Recommended Parameters

`--max-new-tokens`

Set this to 512 or higher.

`--chain-of-thought`

Enable CoT evaluation mode.

## References

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., & Schulman, J. (2021). *Training verifiers to solve math word problems* [Preprint]. arXiv preprint arXiv:2110.14168.
