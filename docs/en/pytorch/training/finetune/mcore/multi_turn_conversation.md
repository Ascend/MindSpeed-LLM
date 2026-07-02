# MindSpeed LLM Multi-Turn Conversation Fine-Tuning

## Feature Overview

The core idea of multi-turn conversation is to enable a trained or fine-tuned model to retain conversation context while interacting with users. This helps the model stay consistent across turns and provide more coherent answers.

By retaining previous conversation context, the model can better understand user intent, adapt to user needs, and provide more personalized responses in later exchanges. This feature not only improves the user experience, but also gives the model stronger reasoning ability when handling complex problems. For example, users can ask questions step by step across multiple turns, and the model can provide more accurate and targeted suggestions or solutions based on the preceding conversation. In addition, multi-turn conversation can analyze user feedback and preferences to gradually optimize the response strategy. Therefore, the model continues to learn and improve during long-term interaction.

### Usage

You can directly use the [chat](../../../../../../examples/mcore/gemma2/chat_gemma2_9b_ptd.sh) or [generate](../../../../../../examples/mcore/gemma2/generate_gemma2_9b_ptd.sh) script, load the preset weights, and carry out multi-turn conversation with the model to achieve the expected result. The preset weights can be chat weights, base weights, or fine-tuned weights of an LLM.

### Demo

```text
You >> My sister's birthday is tomorrow and I need a birthday bouquet.
MindSpeed-LLM:
I'd be happy to help you with that! Do you have any specific preferences or theme in mind for the bouquet? Are there any particular flowers or color scheme that you would like to incorporate?

You >> She likes pink roses. The color is pink.
MindSpeed-LLM:
Pink roses it is then! Do you have any other preferences or restrictions, such as budget or flower availability in your area?

You >> Here I go again. Remember why I came to buy flowers yesterday?
MindSpeed-LLM:
Yes, I do remember. You came to buy flowers for your sister's birthday, and you mentioned that she likes pink roses. Is there anything else related to that purchase that I can help you with today?
```

## Starting Multi-Turn Conversation Instruction Fine-Tuning

For a detailed introduction to the Alpaca dataset, see the [Alpaca-Style Datasets](../../../tools/data_process_sft_alpaca_style.md).

### Multi-Turn Conversation Data Processing

Use the Alpaca dataset with the `history` field to run the following [script](../../../../../../examples/mcore/gemma2/data_convert_gemma2_instruction.sh):

```shell
# Download the Alpaca dataset according to the "Dataset Preparation and Processing" section in examples/README.md.
source /usr/local/Ascend/cann/set_env.sh # Replace this with the actual Toolkit installation path.
mkdir ./finetune_dataset

python ./preprocess_data.py \
    --input ./dataset/train-00000-of-00001.parquet \
    --tokenizer-name-or-path ./model_from_hf/gemma2_hf/ \
    --output-prefix ./finetune_dataset/alpaca \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF \
    --handler-name AlpacaStyleInstructionHandler \
    --prompt-type gemma \
    --map-keys '{"history":"histories"}'
    # --map-keys '{"prompt":"instruction","query":"input","response":"output"}' # Default value. You can omit it.
```

`--map-keys '{"history":"histories"}'`

This extracts the `instruction`, `input`, `output`, and `histories` columns from the dataset. The `instruction`, `input`, and `output` columns exist by default.

### Starting the Script

The naming style and startup method for MindSpeed LLM fine-tuning data processing scripts are as follows:

```shell
# Mcore
# Naming and startup: examples/mcore/gemma2/data_convert_gemma2_instruction.sh
bash examples/mcore/gemma2/data_convert_gemma2_instruction.sh
```

The instruction fine-tuning dataset processing results are as follows:

```shell
./finetune_dataset/alpaca_packed_attention_mask_document.bin
./finetune_dataset/alpaca_packed_attention_mask_document.idx
./finetune_dataset/alpaca_packed_input_ids_document.bin
./finetune_dataset/alpaca_packed_input_ids_document.idx
./finetune_dataset/alpaca_packed_labels_document.bin
./finetune_dataset/alpaca_packed_labels_document.idx
```

When you fine-tune, you only need to specify `./finetune_dataset/alpaca` as the dataset path.

#### Filling in Relevant Paths

`DATA_PATH`: Specifies the save path after data preprocessing.

`TOKENIZER_MODEL`: Specifies the model tokenizer path, for example `tokenizer.model`.

`CKPT_LOAD_DIR`: Points to the path where the converted weights are saved.

`CKPT_SAVE_DIR`: Points to the user-specified save path for the fine-tuned weights, for example a directory named `ckpt`.

Therefore, based on the preceding example, fill in the paths as follows:

```shell
DATA_PATH="./finetune_dataset/alpaca"
CKPT_SAVE_DIR="./ckpt/"
TOKENIZER_MODEL="./model_from_hf/gemma2_hf/"
CKPT_LOAD_DIR="./model_weights/gemma2-9b-mcore/"
```

**Run the multi-turn conversation instruction fine-tuning script.**

```shell
bash examples/mcore/gemma2/chat_gemma2_9b_ptd.sh
```

## Starting Multi-Turn Conversation

Use the [conversation script](../../../../../../examples/mcore/gemma2/chat_gemma2_9b_ptd.sh), load the model tokenizer and the fine-tuned weights, and carry out multi-turn conversation.

Fill in the relevant paths:

`TOKENIZER_PATH`: Specifies the model tokenizer path, for example `tokenizer.model`.

`CHECKPOINT`: Points to the path where the fine-tuned weights are saved.

```shell
TOKENIZER_PATH="./model_from_hf/gemma2_hf/"
CHECKPOINT="./ckpt/"
```

**Run the multi-turn conversation script.**

```shell
bash examples/mcore/gemma2/chat_gemma2_9b_ptd.sh
```

After you confirm that the fine-tuned weights load successfully, you can start a conversation when the `You >>` prompt appears.

## Usage Constraints

- The default template used by the current fine-tuning data preprocessing aligns with LLaMA Factory 0.8.2. If you need alignment with a later version, set the `prompt-type` parameter to `qwen_lf` during the fine-tuning data preprocessing stage.
