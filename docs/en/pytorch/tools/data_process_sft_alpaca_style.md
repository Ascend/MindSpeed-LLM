# Alpaca-Style Datasets

## Common Alpaca Datasets

Common instruction-tuning datasets for dialogue include:

- [Single-turn dialogue: Alpaca English dataset](https://huggingface.co/datasets/tatsu-lab/alpaca)
- [Single-turn dialogue: Alpaca Chinese dataset](https://huggingface.co/datasets/llm-wizard/alpaca-gpt4-data-zh/tree/main)
- [Multi-turn dialogue: AlpacaHistory dataset](https://huggingface.co/datasets/kimnt93/oaast-selected)
- [Chain-of-thought (CoT): Alpaca dataset](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/Auto-CoT)
- [BELLE: instruction-tuning dataset](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)

## Alpaca-Style Dataset Processing

### Downloading Alpaca-Style Datasets

You can download Alpaca-style fine-tuning datasets directly from the web or through the CLI. For example:

```shell
cd dataset/
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..
```

### Processing Alpaca-Style Datasets

During instruction-supervised fine-tuning, the content in the `instruction` column is concatenated with the content in the `input` column and used as the human instruction. That is, the human instruction is `instruction\ninput`, where `\n` is the newline separator. The content in the `output` column becomes the model response. If you specify `history`, the historical conversation content is also included. If you specify the `system` column, its content is used as the system prompt.

```shell
source /usr/local/Ascend/cann/set_env.sh # Change this to the actual Toolkit installation path.
mkdir ./finetune_dataset

python ./preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./model_from_hf/llama-2-7b-hf/ \
    --output-prefix ./finetune_dataset/alpaca \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF \
    --handler-name AlpacaStyleInstructionHandler \
    --prompt-type llama2  # <-- Specify the model template here.
    # --map-keys '{"prompt":"instruction","query":"input","response":"output"}' # Default value, optional.
```

`--input`

You can pass either a dataset directory or a specific file. If you pass a directory, the script processes all files in it. It supports the `.parquet`, `.csv`, `.json`, `.jsonl`, `.txt`, and `.arrow` formats. Keep the data format consistent within the same folder.

`--map-keys`

The `--map-keys` parameter configures the field mappings used to ingest the dataset.

Alpaca-style example:

```json
[
{
    "instruction": "Human instruction (required)",
    "input": "Human input (optional)",
    "output": "Model response (required)",
    "system": "System prompt (optional)",
    "history": [
    ["First-round instruction (optional)", "First-round response (optional)"],
    ["Second-round instruction (optional)", "Second-round response (optional)"]
    ]
}
]
```

For data in the format above, the complete `--map-keys` parameter should be:

`'{"prompt":"instruction","query":"input","response":"output","system":"system","history":"history"}'`

The key values `prompt`, `query`, `response`, `system`, and `history` represent the mapped dataset attributes. These names are fixed in the code and must not change. The values `instruction`, `input`, `output`, `system`, and `history` correspond to the dataset column names.

Because most Alpaca datasets use the `instruction` / `input` / `output` format, we set default values for the `prompt`, `query`, and `response` keys. Therefore, the `--map-keys` parameter above can be simplified to `'{"system": "system","history": "history"}'`.

If the dataset does not contain `system` or `history` columns, you can omit `--map-keys`.

`--prompt-type`

Use this to specify the model template. It helps a base model achieve stronger dialogue capabilities after fine-tuning. You can find the available `prompt-type` options in [templates](../../../../mindspeed_llm/tasks/preprocess/templates.py).

`--handler-name`

When preprocessing Alpaca-style fine-tuning datasets, set this to `AlpacaStyleInstructionHandler`, which extracts the corresponding columns according to the `--map-keys` parameter.

**Example 1**

```bash
--map-keys '{"prompt":"notice","query":"question","response":"answer","system":"system_test","history":"histories"}'
```

This extracts the `notice`, `question`, `answer`, `system_test`, and `histories` columns from the dataset.

**Example 2**

```bash
--map-keys '{"history":"histories"}'
```

This extracts the `instruction`, `input`, `output`, and `histories` columns from the dataset. The `instruction`, `input`, and `output` columns are available implicitly as the default values.

### Launching the Script

The naming convention and launch method for the MindSpeed LLM fine-tuning data processing scripts are as follows:

```shell
# Naming and launch: examples/mcore/model_name/data_convert_xxx_instruction.sh
bash examples/mcore/llama2/data_convert_llama2_instruction.sh
```

The instruction-tuning dataset processing output is as follows:

```shell
./finetune_dataset/alpaca_packed_attention_mask_document.bin
./finetune_dataset/alpaca_packed_attention_mask_document.idx
./finetune_dataset/alpaca_packed_input_ids_document.bin
./finetune_dataset/alpaca_packed_input_ids_document.idx
./finetune_dataset/alpaca_packed_labels_document.bin
./finetune_dataset/alpaca_packed_labels_document.idx
```

When you fine-tune, use `./finetune_dataset/alpaca` as the dataset path.
