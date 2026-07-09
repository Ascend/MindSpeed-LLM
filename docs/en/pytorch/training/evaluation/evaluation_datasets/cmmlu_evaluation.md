# CMMLU Evaluation

## Use Cases

### Problem Description

Chinese Multi-Modal Large-scale Understanding (CMMLU) is an evaluation set designed specifically for LLMs. It aims to comprehensively evaluate knowledge mastery, reasoning ability, and interdisciplinary understanding of an LLM in Chinese contexts. The following sections provide a detailed introduction to the CMMLU evaluation set, especially its subject classifications.

### Features

The CMMLU evaluation set focuses on the following aspects:

- Chinese language environment: All tasks are based on Chinese contexts and cover Chinese history, culture, society, science, and other fields.
- Multidisciplinary coverage: The tasks involve multiple subject areas and test the breadth and depth of knowledge of the model.
- Combination of knowledge and reasoning: The benchmark not only tests the ability of the model to remember factual knowledge, but also its logical reasoning and complex problem-solving abilities.
- Standardized evaluation: It provides a unified task format and scoring standard, which makes it easier to compare performance across different models.

Each domain also contains multiple specific tasks and questions. Through these diverse tasks, CMMLU can evaluate mastery of knowledge of a model in different fields and its ability to generalize across domains.

At present, the MindSpeed LLM repository provides three evaluation modes for CMMLU evaluation.

## Usage

### 1. Direct Evaluation Mode (Default)

#### Impact

- This mode reads the [template file](../../../../../../mindspeed_llm/tasks/evaluation/eval_impl/fewshot_template/cmmlu_5shot_template.json) for public CMMLU evaluation as the evaluation template. It concatenates the template with the question the model needs to answer, then feeds the result to the model for direct evaluation.
- In this mode, the first output of the model is taken as the answer.
- The advantage of this mode is that it is direct and fast. Therefore, it can evaluate the pretrained weights of the model directly.

#### Recommended Parameters

`--max-new-tokens`

Set this to 1 or 2.

### 2. Fine-Tuned Template Evaluation Mode

#### Impact

- This mode reads the files with the `_dev.csv` suffix for the corresponding questions in the sibling `dev` folder under the `DATA_PATH` path in your startup script. It uses those files as template questions and processes them before feeding them to the model.
- Unlike direct evaluation mode, this mode shuffles the template questions in the `dev` file based on the seed. After they are concatenated with the questions the model needs to answer and processed through the chat template, the resulting dialogue dictionary is fed into the model for evaluation.
- The advantage of this mode is that evaluation is faster, and it is suitable for evaluating fine-tuned model weights.

#### Recommended Parameters

`--max-new-tokens`

Set this to 1 or 2.

`--prompt-type`

This parameter specifies the model template type. It should match the `--prompt-type` parameter configured when you fine-tune the model in the MindSpeed LLM repository.

`--eval-language`

Set it to `zh` by default. For English models, you can set it to `en` for evaluation.

### 3. Alternative Template Output Mode

#### Impact

Like fine-tuned template evaluation mode, this mode also uses the `_dev.csv` files for the corresponding questions in the sibling `dev` folder under the `DATA_PATH` path in your evaluation script, and uses them as template questions.

Unlike the other modes, this mode does not shuffle the template questions. After the template questions are concatenated with the questions the model needs to answer, the dialogue dictionary is not processed, and the result is fed to the model directly to obtain the forward-pass output.

The advantage of this mode is that it uses the same evaluation template as leading industry evaluation solutions, and it can achieve better evaluation scores.

#### Recommended Parameters

`--max-new-tokens`

Set this to 128 or higher.

`--alternative-prompt`

Enable Alternative Template Output Mode.

`--eval-language`

Set it to `zh` by default. For English models, you can set it to `en` for evaluation.

## Detailed Subject Classification for CMMLU

The tasks in the CMMLU evaluation set are classified by subject area and cover a wide range of academic and practical fields. The following list shows the detailed classifications and representative tasks.

### Humanities and Social Sciences

- History: Tests the understanding of the model of Chinese historical events, figures, and timelines.
- Literature: Evaluates the understanding of the model of classic literary works, poetry, and idioms.
- Philosophy: Tests the understanding of the model of philosophical ideas, ethics, and logical reasoning.
- Law: Evaluates the understanding of the model of legal provisions, cases, and legal theory.

### Natural Sciences

- Mathematics: Tests the mathematical calculation and logical reasoning abilities of the model.
- Physics: Evaluates the understanding of the model of physical laws and phenomena.
- Chemistry: Tests the understanding of the model of chemical elements, reactions, and molecular structures.
- Biology: Evaluates the understanding of the model of biological concepts and ecosystems.

### Engineering and Technology

- Computer science: Tests the understanding of the model of programming, algorithms, and computer system principles.
- Engineering: Evaluates the understanding of the model of mechanical, electrical, and civil engineering principles.

### Medicine and Health

- Medicine: Tests the understanding of the model of diseases, diagnosis, and treatment methods.
- Psychology: Evaluates the understanding of the model of psychological phenomena and behavioral patterns.

### Economics and Management

- Economics: Tests the understanding of the model of economic theory, market mechanisms, and policy.
- Management: Evaluates the understanding of the model of business management, organizational behavior, and strategic planning.

### Arts and Culture

- Art: Tests the understanding of the model of art forms such as painting, music, and film.
- Culture: Evaluates the understanding of the model of Chinese traditional culture, customs, and social phenomena.

### General Knowledge and Interdisciplinary Topics

- Common-sense reasoning: Tests the understanding of the model of everyday common sense and logical reasoning.
- Interdisciplinary questions: Evaluates the ability of the model to perform integrated analysis across multiple subject areas.
