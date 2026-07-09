# Multi-Sample Pack Fine-Tuning

## Use Cases

Because compute resources are limited, the samples loaded in each batch during training vary in length. Most data must be padded to the `seq-length` at the end, which reduces training efficiency and wastes compute resources. Multi-sample pack combines multiple samples, such as text sequences, into one "pack." It combines data of different lengths from the specified dataset into a sequence of the target length and fills it with valid data content as much as possible. If the concatenated data does not reach the target `seq-length`, it is padded to the target length. Therefore, **the length of each pack dataset sample is the same**, which reduces the number of samples processed during training and improves training efficiency.

As shown in the figure, a new sample is formed by merging multiple original samples, and each block is a new sample:

<img src="../../../figures/multi_sample_pack_fine_tuning/samples_pack.png" width="60%"/>

Each sample in the fine-tuning data consists of one problem (`p`) paired with one response (`r`), that is, a one-question-one-answer pair such as `p1r1` and `p2r2`.

The `attn_mask` used in the computation also becomes a zigzag matrix:

<img src="../../../figures/multi_sample_pack_fine_tuning/zigzag_attn_mask.png" width="20%"/>

In the non-pack scenario, the matrix is a complete lower-triangular matrix. Self-attention between multiple samples is not masked, and all tokens participate in computation. In the pack scenario, the data form a zigzag white triangle and the remaining area is masked. Samples stay independent and cannot perform self-attention across samples. This reduces the amount of fine-tuning data processing and preserves sample independence, which improves training efficiency.

Other pack modes are also used in the industry, such as lower-triangular pack modes. Support for those modes is coming soon.

## Usage

This section uses the [Qwen3-32B Pack fine-tuning script](../../../../../../examples/mcore/qwen3/tune_qwen3_32b_32K_full_pack_A3_ptd.sh) with the Alpaca dataset as an example to introduce multi-sample pack fine-tuning. **For more models that support pack mode, see the examples directory.**

Multi-sample pack fine-tuning mainly includes the following steps:

1. Environment setup.

    Before you start fine-tuning, refer to the [MindSpeed LLM Installation Guide](../../install_guide.md) to complete environment setup, and ensure that you have configured the Ascend NPU suite environment variables as follows:

    ```shell
    source /usr/local/Ascend/cann/set_env.sh     # Replace this with the actual Toolkit installation path.
    source /usr/local/Ascend/nnal/atb/set_env.sh # Replace this with the actual nnal package installation path.
    ```

2. Configure the instruction fine-tuning script.

    For detailed parameter configuration, see the [Qwen3-32B pack fine-tuning script](../../../../../../examples/mcore/qwen3/tune_qwen3_32b_32K_full_pack_A3_ptd.sh).

    You need to modify the following path parameters in the script:

    ```shell
    CKPT_LOAD_DIR="your model ckpt path"        # Hugging Face weight path.
    CKPT_SAVE_DIR="your model save ckpt path"   # Path to the user-specified fine-tuned weight save directory, for example a directory under `ckpt`.
    DATA_PATH="your data path"                  # Original dataset path.
    TOKENIZER_PATH="your tokenizer path"        # Model tokenizer path, for example `tokenizer.model`.
    ```

    Parameter descriptions for fine-tuning:

    - `--is-instruction-dataset`: Specifies that instruction fine-tuning data is used during fine-tuning so the model can be fine-tuned according to the specified instruction data.
    - `--prompt-type`: Specifies the model template so the base model can gain better conversational ability after fine-tuning. You can find the available `prompt-type` options in the [templates.json](../../../../../../configs/finetune/templates.json) file.
    - `--reset-attention-mask`: When you enable this parameter, the system calculates the sentence boundary from the EOD token, generates `actual_seq_len`, and passes it to the FA operator to achieve a masking effect similar to a zigzag pattern. The FA operator then performs the computation in the TND format.
    - `--neat-pack`: Enables the zigzag `attention_mask` in pack scenarios. It uses the `attention_mask` generated during dataset processing to create the corresponding `actual_seq_len`.
    - `--padded-samples` [optional]: Pads the total number of samples to an integer multiple of `batch-size`.
    - `--no-shuffle` [optional]: Loads the data in order.

3. Start fine-tuning.

    After you finish configuring the fine-tuning script, run the script to start fine-tuning:

    ```shell
    bash examples/mcore/qwen3/tune_qwen3_32b_32K_full_pack_A3_ptd.sh
    ```

> [!NOTE]
>
> The [Qwen3-32B pack fine-tuning script](../../../../../../examples/mcore/qwen3/tune_qwen3_32b_32K_full_pack_A3_ptd.sh) supports online data and weight loading during training. It integrates data preprocessing, weight conversion, and training into one script. Therefore, you can start a training job with a single command. For more details, see [Training With Online Data and Weight Loading (Train_from_HF)](../../pretrain/mcore/train_from_hf.md).
>
> - Integrated weight conversion and training: This provides bidirectional automatic conversion between Hugging Face weights and the Megatron format, together with training. Therefore, you do not need to run the weight conversion step separately. This enables one-click integration from Hugging Face weights to the training task.
> - Automatic raw data conversion: The data preprocessing feature automatically identifies and converts raw data files during model training. Therefore, you do not need to convert raw data manually. The system automatically determines whether the input path points to raw data, such as `.jsonl` or `.parquet` files, and completes the data format conversion during training initialization.

## Usage Constraints

- When you use `--neat-pack` during data preprocessing, you must also use `--pack`.

- When you use `--neat-pack` during fine-tuning, you must also use `--reset-attention-mask`.

- The default template used by the current fine-tuning data preprocessing is aligned with LLaMA Factory 0.8.2. If you need alignment with a later version, set the `prompt-type` parameter to `qwen_lf` during the fine-tuning data preprocessing stage.
