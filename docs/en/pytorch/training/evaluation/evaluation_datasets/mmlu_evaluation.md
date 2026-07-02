# MMLU Evaluation

## Use Cases

### Problem Description

Massive Multitask Language Understanding (MMLU) evaluation includes a wide range of tasks and domains that are intended to comprehensively test the comprehension ability and breadth of knowledge of a model. Specifically, MMLU covers the following main domains:

1. **Humanities**: Such as history, philosophy, and literature.
2. **Social sciences**: Such as psychology, sociology, and economics.
3. **Science, technology, engineering, and mathematics (STEM)**: Such as mathematics, physics, chemistry, and biology.
4. **Other professional fields**: Such as law, medicine, and business.

Each domain also contains multiple specific tasks and questions. Through these diverse tasks, MMLU can evaluate mastery of knowledge and cross-domain generalization ability of a model.

The subject sets included in [STEM](#stem-question-sets), [Humanities](#humanities-question-sets), and [Social Sciences](#social-sciences-question-sets) are listed at the end of this document.

The MindSpeed LLM repository currently provides four evaluation modes for MMLU evaluation.

## Usage

### 1. Direct Evaluation Mode

#### Impact

This mode reads the publicly available [MMLU evaluation template file](../../../../../../mindspeed_llm/tasks/evaluation/eval_impl/fewshot_template/mmlu_5shot_template.json). It concatenates the template with the question the model needs to answer, then feeds the result to the model for direct evaluation.

In this mode, the first output of the model is taken as the answer.

The advantage of this mode is that it is direct and fast, and it can evaluate the pretrained weights of the model directly.

#### Recommended Parameters

`--max-new-tokens`

Set this to 1 or 2.

### 2. Fine-Tuned Template Evaluation Mode

#### Impact

This mode reads the files with the `_dev.csv` suffix for the corresponding questions in the sibling `dev` folder under the `DATA_PATH` path in your startup script. It uses those files as templates and preprocesses them before feeding them to the model.

Unlike direct evaluation mode, this mode shuffles the template questions in the `dev` file based on the seed. After they are concatenated with the questions the model needs to answer and processed through the chat template, the resulting dialogue dictionary is fed into the model for evaluation.

The advantage of this mode is that evaluation is faster, and it is suitable for evaluating fine-tuned model weights.

#### Recommended Parameters

`--max-new-tokens`

Set this to 1 or 2.

`--prompt-type`

This parameter specifies the model template type. It should match the `--prompt-type` parameter configured when you fine-tune the model in the MindSpeed LLM repository.

### 3. Alternative Template Output Mode

#### Impact

Like fine-tuned template evaluation mode, this mode also uses the `_dev.csv` files for the corresponding questions in the sibling `dev` folder under the `DATA_PATH` path in your evaluation script as template questions.

Unlike the other modes, this mode does not shuffle the template questions. After the template questions are concatenated with the questions the model needs to answer, the dialogue dictionary is not processed, and the result is fed directly to the model to obtain the forward-pass output.

The advantage of this mode is that it uses the same evaluation template as leading industry evaluation solutions, and it can achieve better evaluation scores.

#### Recommended Parameters

`--max-new-tokens`

Set this to 128 or higher.

`--alternative-prompt`

Enable `Alternative Template Output Mode`.

### 4. PPL Mode

#### Impact

`ppl` is short for perplexity, which is a metric used to evaluate the language modeling ability of a model. This mode also uses the `_dev.csv` files for the corresponding questions in the sibling `dev` folder under the `DATA_PATH` path in your evaluation script as template questions. At this point, we concatenate the n options with the context to form n sequences, then compute the perplexity of the model over those n sequences. We treat the option corresponding to the sequence with the lowest perplexity as the inference result of the model for the question. This evaluation method uses simple post-processing and is highly deterministic.

#### Recommended Parameters

`--task`

Set this to `mmlu_ppl`.

## MMLU Subsets

### STEM Question Sets

1. **abstract_algebra** (abstract algebra)
2. **astronomy** (astronomy)
3. **college_biology** (college biology)
4. **college_chemistry** (college chemistry)
5. **college_computer_science** (college computer science)
6. **college_mathematics** (college mathematics)
7. **college_physics** (college physics)
8. **computer_security** (computer security)
9. **conceptual_physics** (conceptual physics)
10. **electrical_engineering** (electrical engineering)
11. **elementary_mathematics** (elementary mathematics)
12. **high_school_biology** (high school biology)
13. **high_school_chemistry** (high school chemistry)
14. **high_school_computer_science** (high school computer science)
15. **high_school_mathematics** (high school mathematics)
16. **high_school_physics** (high school physics)
17. **high_school_statistics** (high school statistics)
18. **machine_learning** (machine learning)

### Humanities Question Sets

1. **formal_logic** (formal logic)
2. **high_school_european_history** (high school European history)
3. **high_school_us_history** (high school US history)
4. **high_school_world_history** (high school world history)
5. **international_law** (international law)
6. **jurisprudence** (jurisprudence)
7. **logical_fallacies** (logical fallacies)
8. **moral_disputes** (moral disputes)
9. **moral_scenarios** (moral scenarios)
10. **philosophy** (philosophy)
11. **prehistory** (prehistory)
12. **professional_law** (professional law)
13. **world_religions** (world religions)

### Social Sciences Question Sets

1. **clinical_knowledge** (clinical knowledge)
2. **college_medicine** (college medicine)
3. **global_facts** (global facts)
4. **human_aging** (human aging)
5. **human_sexuality** (human sexuality)
6. **marketing** (marketing)
7. **medical_genetics** (medical genetics)
8. **miscellaneous** (miscellaneous)
9. **nutrition** (nutrition)
10. **professional_accounting** (professional accounting)
11. **professional_medicine** (professional medicine)
12. **public_relations** (public relations)
13. **security_studies** (security studies)
14. **sociology** (sociology)
15. **us_foreign_policy** (US foreign policy)
