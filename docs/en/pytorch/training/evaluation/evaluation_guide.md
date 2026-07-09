# Evaluation Guide

MindSpeed LLM supports accurate evaluation of large models on public benchmark datasets. For detailed statistics on the currently supported benchmarks, see [MindSpeed LLM Evaluation Support](./models_evaluation.md).

## Environment Variables

Refer to [Model Script Environment Variables](../../features/mcore/environment_variable.md) for configuration of the environment variables used in the script.

## Introduction to Distributed LLM Evaluation

### 1. Benchmark Evaluation

The naming convention and startup method for MindSpeed LLM benchmark evaluation scripts are as follows:

```shell
# Naming and startup: examples/mcore/model_name/evaluate_xxx.sh
bash examples/mcore/llama2/evaluate_llama2_7b_mmlu_ptd.sh
```

```shell
# Modify the model parameter path and tokenizer path.
TOKENIZER_PATH="./model_from_hf/llama-2-7b-hf/"  # Tokenizer path
CHECKPOINT="./model_weights/llama-2-7b-mcore"    # Checkpoint path
# Configure the task and dataset path
DATA_PATH="./mmlu/data/test/"
TASK="mmlu"  # Supports mmlu, ceval, agieval, bbh, boolq, human_eval.

# Start the evaluation script.
bash examples/mcore/llama2/evaluate_llama2_7b_mmlu_ptd.sh
```

`--max-new-tokens`

This parameter controls the model generation length. The output length for multiple-choice questions is shorter than that for code tasks. This parameter has a significant impact on evaluation performance.

`--evaluation-batch-size`

You can enable multi-batch inference to improve evaluation performance.

`--broadcast`

When expert parallelism is enabled, you must set this flag for evaluation.

### 2. Instruction-Tuning Evaluation

Use the following naming convention and startup method for evaluation scripts with instruction-tuned weights:

```shell
bash examples/mcore/llama2/evaluate_llama2_7b_full_mmlu_ptd.sh
```

`--prompt-type`

The model chat template. Use it to select the corresponding chat template for evaluation.

`--hf-chat-template`

If the model's tokenizer already has the `chat_template` attribute, you can add `--hf-chat-template` to use the model's built-in chat template for evaluation.

`--eval-language`

Set this based on the language of the evaluation dataset. The default is `en`. If the evaluation dataset is Chinese, set this to `zh`.

### 3. LoRA Weight Evaluation

Use the following naming convention and startup method for evaluation scripts with LoRA weights:

```shell
# Start the evaluation script with LoRA weights loaded. Naming style and startup method:
bash examples/mcore/codellama/evaluate_codellama_34b_lora_ptd.sh
```
