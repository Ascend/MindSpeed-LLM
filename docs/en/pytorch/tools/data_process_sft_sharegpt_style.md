# ShareGPT Datasets

## Common ShareGPT Datasets

Common instruction-tuning datasets for dialogue include:

- [Roleplay multi-turn dialogue: ShareGPT dataset](https://huggingface.co/datasets/shibing624/roleplay-zh-sharegpt-gpt4-data)
- [Chain-of-thought: ShareGPT dataset](https://huggingface.co/datasets/isaiahbjork/chain-of-thought-sharegpt)
- [Capybara: ShareGPT dataset](https://huggingface.co/datasets/Undi95/Capybara-ShareGPT)

## Downloading ShareGPT-Style Datasets

You can download ShareGPT-style fine-tuning datasets directly from the web page or from the CLI. For example:

```shell
mkdir -p dataset
cd dataset/
wget https://huggingface.co/datasets/shibing624/roleplay-zh-sharegpt-gpt4-data/resolve/main/sharegpt_formatted_data-evol-gpt4.jsonl
cd ..
```

## Processing ShareGPT-Style Datasets

The ShareGPT format supports more role types, such as `human`, `gpt`, `observation`, and `function_call`. These roles make up a list of objects in the `conversations` column.

ShareGPT-style example:

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "Human instruction"
      },
      {
        "from": "function_call",
        "value": "Tool parameters"
      },
      {
        "from": "observation",
        "value": "Tool result"
      },
      {
        "from": "gpt",
        "value": "Model response"
      }
    ],
    "system": "System prompt (optional)",
    "tools": "Tool description (optional)"
  }
]
```

Preprocessing script for ShareGPT-style data:

```shell
source /usr/local/Ascend/cann/set_env.sh # Replace this with the actual Toolkit installation path.
mkdir ./finetune_dataset

python ./preprocess_data.py \
    --input ./dataset/sharegpt_formatted_data-evol-gpt4.jsonl \
    --tokenizer-name-or-path ./model_from_hf/llama-2-7b-hf/ \
    --output-prefix ./finetune_dataset/sharegpt \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF \
    --handler-name SharegptStyleInstructionHandler \
    --prompt-type llama2  # <-- Fill in the model template here.
    # --map-keys '{"messages":"conversations", "tags":{"role_tag": "from","content_tag": "value","user_tag": "human","assistant_tag": "gpt","system_tag": "system", "observation_tag":"observation", "function_tag":"function_call"}}' # Default value. You can omit it.
```

`--prompt-type`

This option specifies the model template and helps the base model develop stronger conversational ability after fine-tuning. You can find the available `prompt-type` options in the [templates](../../../../configs/finetune/templates.json) file.

`--map-keys`

The `--map-keys` parameter configures field mappings for dataset usage. The default value is

`'{"messages":"conversations", "tags":{"role_tag": "from","content_tag": "value","user_tag": "human","assistant_tag": "gpt","system_tag": "system", "observation_tag":"observation", "function_tag":"function_call"}}'`

The key values `"messages"` and `"tags"` represent the mapped dataset attributes. These names are fixed in the code and must not change. Within the value, `"conversations"` corresponds to the dataset column name, `"from"` corresponds to the role tag, `"human"`, `"gpt"`, `"system"`, `"observation"`, and `"function_call"` correspond to the role types, and `"value"` corresponds to the content field.

The OpenAI format is a special case of the ShareGPT format. The first message can be a system prompt.

Example of the OpenAI format:

```json
[
  {
    "messages": [
      {
        "role": "system",
        "content": "System prompt (optional)"
      },
      {
        "role": "user",
        "content": "Human instruction"
      },
      {
        "role": "assistant",
        "content": "Model response"
      }
    ]
  }
]
```

OpenAI-format data preprocessing script:

```shell
source /usr/local/Ascend/cann/set_env.sh # Replace this with the actual Toolkit installation path.
mkdir ./finetune_dataset

python ./preprocess_data.py \
    --input ./dataset/xxx.json \
    --tokenizer-name-or-path ./model_from_hf/llama-2-7b-hf/ \
    --output-prefix ./finetune_dataset/openai \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF \
    --handler-name SharegptStyleInstructionHandler \
    --prompt-type llama2 \
    --map-keys '{"messages":"messages", "tags":{"role_tag": "role","content_tag": "content","user_tag": "user","assistant_tag": "assistant","system_tag": "system"}}'
```

`--handler-name`

When you preprocess fine-tuning data in ShareGPT style, set this to `SharegptStyleInstructionHandler` and extract the corresponding dataset columns according to the `--map-keys` parameter.

**Example 1**

```bash
--map-keys '{"messages":"chat"}'
```

This extracts the `"chat"` column from the dataset. The `"tags"` attribute contains the role format and the content format, and it exists implicitly as the default value. The role format can be `"from": "human"`, `"from": "gpt"`, `"from": "observation"`, or `"from": "function_call"`. The content format is `"value": "specific content"`.

**Example 2**

```bash
--map-keys '{"messages":"messages", "tags":{"role_tag": "role","content_tag": "content","user_tag": "user","assistant_tag": "assistant"}}'
```

This extracts the `"messages"` column from the dataset. The role format can be `"role": "user"` or `"role": "assistant"`. The content format is `"content": "specific content"`.

### Launching the Script

The naming convention and startup method for MindSpeed LLM fine-tuning data processing scripts are as follows:

```shell
# MCore
# Naming and startup: examples/mcore/model_name/data_convert_xxx_instruction.sh
bash examples/mcore/llama2/data_convert_llama2_instruction.sh
```

The instruction-tuning dataset processing results are as follows:

```shell
./finetune_dataset/openai_packed_attention_mask_document.bin
./finetune_dataset/openai_packed_attention_mask_document.idx
./finetune_dataset/openai_packed_input_ids_document.bin
./finetune_dataset/openai_packed_input_ids_document.idx
./finetune_dataset/openai_packed_labels_document.bin
./finetune_dataset/openai_packed_labels_document.idx
```

When you fine-tune, use `./finetune_dataset/openai` as the dataset path.
