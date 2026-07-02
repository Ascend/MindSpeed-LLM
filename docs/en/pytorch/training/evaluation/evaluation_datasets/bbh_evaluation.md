# BBH Evaluation

## Use Cases

### Problem Description

**BIG-Bench Hard (BBH)** focuses on a set of 23 challenging BIG-Bench tasks. BBH is a subset of the BIG-Bench suite that includes 23 especially difficult tasks, on which models have historically failed to match or surpass average human performance.

### Features

The BBH evaluation set focuses on the following aspects:

- **Requiring multi-step reasoning**: Many tasks in BBH require the model to perform complex, multi-step logical reasoning and inference, and simple "answer only" prompts often fail to show best capabilities of the the model.
- **Revealing the potential of existing models**: In the original BIG-Bench evaluation, most models failed to outperform the human average when tested with few-shot prompting. However, research found that when Chain-of-Thought (CoT) prompting was introduced, models could show deeper reasoning abilities on the tasks. For example, the PaLM model surpassed the human benchmark on 10 of the 23 tasks, and the Codex model surpassed that baseline on 17 tasks.
- **Uncovering latent capabilities**: The design of BBH highlights an important fact. Traditional prompting methods may underestimate the potential of a model on complex reasoning tasks, while CoT prompting lets the model break down problems step by step through a "thinking process" and ultimately achieve better performance.
- **Guiding future research**: BBH gives AI researchers a dedicated tool that helps them analyze and improve the performance of LLMs on multi-step logical reasoning. This not only provides a basis for improving model architectures, but also drives the development of prompt design and reasoning techniques.

Currently, the MindSpeed LLM repository provides the following BBH evaluation modes.

## Usage

### 1. Direct Evaluation Mode (Default)

#### Impact

- This mode feeds the question directly into the model for evaluation.

#### Recommended Parameters

`--max-new-tokens`

Set this to 32 to ensure that tasks requiring long outputs, such as [word_sorting](https://github.com/suzgunmirac/BIG-Bench-Hard/blob/main/bbh/word_sorting.json), can produce complete outputs.

### 2. Fine-Tuned Template Evaluation Mode

#### Impact

- This mode uses the [template file](../../../../../../mindspeed_llm/tasks/evaluation/eval_impl/fewshot_template/bbh_template.json) as the evaluation template, concatenates it with the question the model must answer, and then feeds the result directly to the model for evaluation.
- Note that the template used in this mode is not a CoT template style.

#### Recommended Parameters

`--max-new-tokens`

Set this to 32 to ensure that tasks requiring long outputs, such as [word_sorting](https://github.com/suzgunmirac/BIG-Bench-Hard/blob/main/bbh/word_sorting.json), can produce complete outputs.

`--prompt-type`

This parameter specifies the model template type. It should match the `--prompt-type` parameter that you configure when you fine-tune the model in the MindSpeed LLM repository.

### 3. CoT Evaluation Mode

#### Impact

- This mode uses CoT prompts for BBH tasks.
- This mode uses the [CoT template file](../../../../../../mindspeed_llm/tasks/evaluation/eval_impl/fewshot_template/bbh_cot_template.json) as the evaluation template, concatenates it with the question the model must answer, and then feeds the result to the model for evaluation.

#### Recommended Parameters

`--max-new-tokens`

Set this to 512 or higher.

`--chain-of-thought`

Enable CoT evaluation mode.

## References

Srivastava, A., Rastogi, A., Rao, A., Shoeb, A. A. M., Abid, A., Fisch, A., Brown, A. R., Santoro, A., Gupta, A., Garriga-Alonso, A., and others (2022). *Beyond the imitation game: Quantifying and extrapolating the capabilities of language models* [Preprint]. arXiv. <https://arxiv.org/abs/2206.04615>
