# PMCC Obfuscation for Fine-Tuning

## Use Cases

Privacy and Model Confidential Computing (PMCC) obfuscation refers to obfuscating the model files and datasets used in the fine-tuning process to prevent potential leakage of model or data during storage or computation and to protect the confidentiality of both.

## Usage

Because the PMCC obfuscation feature currently supports only the Qwen3-32B model, this document uses that model as an example to describe how to enable PMCC. Perform the following steps:

1. Refer to the [MindSpeed LLM Installation Guide](../../install_guide.md) to complete the environment setup.

    Before training begins, configure the Ascend NPU suite environment variables as follows:

    ```shell
    source /usr/local/Ascend/cann/set_env.sh     # Replace this with the actual Toolkit installation path.
    source /usr/local/Ascend/nnal/atb/set_env.sh # Replace this with the actual nnal package installation path.
    ```

2. Prepare the model weights and fine-tuning dataset.

    The complete [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B/tree/main) model directory should contain the following files:

    ```shell
    .
    ├── README.md                    # Model documentation.
    ├── config.json                  # Model architecture configuration file.
    ├── generation_config.json       # Configuration for text generation.
    ├── merges.txt                   # Tokenizer merge rules file.
    ├── model-00001-of-00017.safetensors  # Part 1 of the model weight files, 17 parts in total.
    ├── model-00002-of-00017.safetensors  # Part 2 of the model weight files.
    ├── ...
    ├── model-00016-of-00017.safetensors  # Part 16 of the model weight files.
    ├── model-00017-of-00017.safetensors  # Part 17 of the model weight files.
    ├── model.safetensors.index.json      # Weight shard index file that maps each parameter to its file.
    ├── tokenizer.json              # Tokenizer in the Hugging Face format.
    ├── tokenizer_config.json       # Tokenizer-related configuration.
    └── vocab.json                  # Model vocabulary file.
    ```

3. Install the PMCC obfuscation package.

    ```shell
    pip3 install ai_asset_obfuscate
    ```

4. Obfuscate the model.

    Create a new model obfuscation script named `obf_model.py` with the following content. Set the model weight path and the obfuscation seed content:

    ```python
    from sys import argv
    from ai_asset_obfuscate import ModelAssetObfuscation, ModelType

    # Model obfuscation.
    seed_content = "xxxxxx"    # Obfuscation seed content. It must be a 32-character string.
    model_path = "./model_from_hf/qwen3_hf/"     # Path to the original Hugging Face model weights.
    save_path = "./model_from_hf/qwen3_obf_hf/"  # Path to the obfuscated Hugging Face model weights.
    model = ModelAssetObfuscation.create_model_obfuscation(model_path, ModelType.QWEN3) # ModelType.QWEN3 indicates the model type to obfuscate.

    res = model.set_seed_content(seed_type=2, seed_content=seed_content) # seed_type=1 indicates a model obfuscation seed and 2 indicates a data obfuscation seed. Fine-tuning uses only data obfuscation seeds to protect the model.
    print(res)

    res = model.model_weight_obf(obf_type=2, model_save_path=save_path, device_type="npu", device_id=[0, 1, 2, 3, 4, 5, 6, 7]) # obf_type=1 indicates model obfuscation for model protection, and 2 indicates model obfuscation for data protection. Fine-tuning uses only 2. device_type="npu" means NPU acceleration, and device_id specifies the NPU device IDs.
    print(res)
    ```

    After you verify that the configuration is correct, run the model obfuscation script:

    ```shell
    python obf_model.py
    ```

5. Convert the obfuscated Hugging Face weights to Megatron weights.

    Using the Qwen3-32B model with a TP=8 and PP=2 split as an example, see the [Qwen3 weight conversion script](../../../../../../examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh) for detailed configuration. Modify the related path parameters and model partitioning configuration:

    ```shell
    --target-tensor-parallel-size 8          # TP partition size.
    --target-pipeline-parallel-size 2        # PP partition size.
    --load-dir ./model_from_hf/qwen3_obf_hf/ # Path to the obfuscated Hugging Face model weights.
    --save-dir ./model_weights/qwen3_mcore/  # Path to save the Megatron weights.
    ```

    After you verify that the paths are correct, run the weight conversion script:

    ```shell
    bash examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh
    ```

6. Perform data preprocessing.

    Using the [Alpaca dataset](https://huggingface.co/datasets/tatsu-lab/alpaca/blob/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet) as an example, perform data preprocessing. For detailed configuration, see the [Qwen3 data preprocessing script](../../../../../../examples/mcore/qwen3/data_convert_qwen3_instruction.sh). Modify the paths in the script and add the data obfuscation parameters:

    ```shell
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet # Path to the original dataset.
    --tokenizer-name-or-path ./model_from_hf/qwen3_hf # Hugging Face tokenizer path.
    --output-prefix ./finetune_dataset/alpaca         # Save path.
    ......
    --data-obfuscation             # Data obfuscation switch.
    --obf-seed-content "xxxxxx"    # Obfuscation seed content. It must be a 32-character string and must match the value used during model obfuscation.
    ```

    After you finish setting the relevant parameters, run the data preprocessing script:

    ```shell
    bash examples/mcore/qwen3/data_convert_qwen3_instruction.sh
    ```

7. Start fine-tuning.

    Configure the model fine-tuning script. For detailed configuration, see the [Qwen3-32B fine-tuning script](../../../../../../examples/mcore/qwen3/tune_qwen3_32b_4K_full_ptd.sh). Modify the related path parameters and model partitioning configuration. Note that the parallel configuration of training parameters, such as TP and PP, must match the configuration used during weight conversion in Step 5.

    ```shell
    CKPT_LOAD_DIR="your model ckpt path"      # Weight load path. Enter the weight path saved during conversion.
    CKPT_SAVE_DIR="your model save ckpt path" # Weight save path after fine-tuning finishes.
    DATA_PATH="your data path"                # Dataset path. Enter the path saved during data preprocessing, and note that you need to add the suffix.
    TOKENIZER_PATH="your tokenizer path"      # Vocabulary path. Enter the vocabulary path from the downloaded open-source weights.
    TP=8                                      # The value of target-tensor-parallel-size used during weight conversion.
    PP=2                                      # The value of target-pipeline-parallel-size used during weight conversion.
    ```

    After you finish setting the relevant parameters, run the fine-tuning script:

    ```shell
    bash examples/mcore/qwen3/tune_qwen3_32b_4K_full_ptd.sh
    ```

8. Deobfuscate the model.

    The fine-tuned model remains obfuscated and must be deobfuscated back to plaintext. To deobfuscate it, first convert the fine-tuned model weights back to the Hugging Face format. For detailed configuration, see the [Qwen3 weight conversion script](../../../../../../examples/mcore/qwen3/ckpt_convert_qwen3_mcore2hf.sh). Modify the related path parameters and model partitioning configuration:

    ```shell
    --load-dir ./ckpt/qwen3_obf_mg/           # Weight load path. Enter the weight path saved during conversion.
    --save-dir ./model_from_hf/qwen3_obf_hf/  # Hugging Face weight save directory when converting Megatron to Hugging Face weights.
    --hf-cfg-dir ./model_from_hf/qwen3_hf/    # Hugging Face configuration file directory.
    ```

    Parameter description:

    - `hf-cfg-dir`: The Megatron-to-Hugging Face weight conversion generates only the weights and `model.safetensors.index.json` and does not generate the configuration file required for deobfuscation. By specifying this parameter, the configuration files from the original Hugging Face model are copied to the Hugging Face weight directory generated by the weight conversion.

    After you verify that the paths are correct, run the weight conversion script:

    ```shell
    bash examples/mcore/qwen3/ckpt_convert_qwen3_mcore2hf.sh
    ```

    Create a new model deobfuscation script named `unobf_model.py` with the following content. Set the model weight path and the obfuscation seed content:

    ```python
    from sys import argv
    from ai_asset_obfuscate import ModelAssetObfuscation, ModelType

    # Model deobfuscation.
    seed_content = "xxxxxx"    # Obfuscation seed content. It must match the value used during model obfuscation.
    model_path = "./model_from_hf/qwen3_obf_hf/"     # Hugging Face weight save directory when converting Megatron to Hugging Face weights.
    save_path = "./model_from_hf/qwen3_un_obf_hf/"   # Path to save the deobfuscated Hugging Face model weights.
    model = ModelAssetObfuscation.create_model_obfuscation(model_path, ModelType.QWEN3, is_obfuscation = False) # is_obfuscation = False indicates deobfuscation.

    res = model.set_seed_content(seed_type=2, seed_content=seed_content)
    print(res)

    res = model.model_weight_obf(obf_type=2, model_save_path=save_path, device_type="npu", device_id=[0, 1, 2, 3, 4, 5, 6, 7])
    print(res)
    ```

    After you verify that the configuration is correct, run the model deobfuscation script:

    ```shell
    python unobf_model.py
    ```
