# MindSpeed LLM Chat

## Chat Example

### Instructions

You can use the model [chat script](../../../../../examples/mcore/llama2/chat_llama2_13b_ptd.sh) and load the preset model weights to conduct a multi-turn conversation with the model. The preset model weights can be either the chat weights of an LLM or the weights after [Instruction Fine-Tuning for LLMs](../finetune/mcore/instruction_finetune.md).

### Initialize Environment Variables

```shell
source /usr/local/Ascend/cann/set_env.sh # Change this to the actual Toolkit package installation path.
source /usr/local/Ascend/nnal/atb/set_env.sh # Change this to the actual nnal package installation path.
```

### Running the Script

Use the [chat script](../../../../../examples/mcore/llama2/chat_llama2_13b_ptd.sh) in the Llama-2-13B model directory.

#### Filling in the Paths

`CHECKPOINT`: Path where the converted weights are saved

`TOKENIZER_PATH`: Directory where the model tokenizer is stored

`TOKENIZER_MODEL`: Tokenizer file path for the model, for example `tokenizer.model`

Therefore, for the preceding example, set the paths as follows:

```shell
CHECKPOINT="./model_weights/llama-2-13b-mcore/"
TOKENIZER_PATH="./model_from_hf/llama-2-13b-hf-chat/"
TOKENIZER_MODEL="./model_from_hf/llama-2-13b-hf-chat/tokenizer.model"
```

`--tokenizer-type`

When the value is `PretrainedFromHF`, you only need to point `TOKENIZER_PATH` to the model folder. You do not need to point `TOKENIZER_PATH` to the `tokenizer.model` file.

**Example**

```shell
    TOKENIZER_PATH="./model_from_hf/llama-2-hf/"
    --tokenizer-name-or-path ${TOKENIZER_PATH}
```

When the value is not `PretrainedFromHF`, for example `Llama2Tokenizer`, you must point `TOKENIZER_MODEL` to the `tokenizer.model` file.

**Example**

```shell
    TOKENIZER_MODEL="./model_from_hf/llama-2-hf/tokenizer.model"
    --tokenizer-model ${TOKENIZER_MODEL} \
```

#### Running the Script

```shell
bash examples/mcore/llama2/chat_llama2_13b_ptd.sh
```

#### Parameters for the Chat Script

You can find the parameter settings for the multi-turn conversation script and streaming inference in the [`MindSpeed LLM Streaming Inference`](inference.md) file.

`--task`

The default value is `chat`, which specifies a multi-turn conversation task.

`--history-turns`

In multi-turn conversations, you can set `--history-turns` to change the number of previous conversation turns. The default is `3` turns.

`--hf-chat-template`

If the model tokenizer already has the `chat_template` attribute, you can add `--hf-chat-template` to use the built-in chat template of the model.

`--prompt-type`

The conversation template of the model has the same effect as `--hf-chat-template`, but it does not require the model tokenizer to have the `chat_template` attribute. When you run inference for a fine-tuned model, you should choose the conversation template that matches the model. You can find the available `prompt-type` values in the [`templates`](../../../../../configs/finetune/templates.json) file.

#### Commands for the Chat Program

After the chat program displays the `You >>` prompt, you can enter text to have a multi-turn conversation with the model.

**Clearing chat history**

If you need to clear the previous conversation history and start a new chat session, enter one of the `[clear, new]` commands in the dialog box.

**Exit the chat program**

If you need to exit the chat program, enter one of the `[q, quit, exit]` commands in the dialog box.
