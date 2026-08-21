# Guide to Using the MindSpeed LLM FSDP2 Backend DCP Weight Conversion Tool

## Use Cases

During LLM training and deployment, the DCP weight conversion tool is commonly used in the following scenarios:

- Train in the DCP format, then convert the resulting model weights to the Hugging Face standard format for inference or downstream tasks.
- Handle models with so many parameters that the full weights cannot be loaded into memory at once for format conversion.

The `merge_dcp_to_hf.py` script merges shards incrementally. This preserves correctness while minimizing memory usage. Therefore, it is suitable for LLM weight format conversion.

## Usage

### 1. CLI Parameters

| Parameter | Type | Default | Description |
|------|------|--------|------|
| --load-dir | str | None (required) | Directory that stores the DCP weights. It must contain valid PyTorch distributed checkpoint files. |
| --save-dir | str | `<load-dir>/hf_ckpt` | Output directory for Hugging Face-format weights. |
| --model-configs | str | None | Directory that contains model configuration files. The script copies it to the output directory. |
| --shard-size | int | `5000000000` (5 GB) | Maximum byte size of a single weight shard file. |

### 2. Basic Conversion

The simplest usage is as follows:

```bash
python mindspeed_llm/fsdp2/checkpoint/merge_dcp_to_hf.py \
        --load-dir <DCP_weight_path>
```

After the command runs, the Hugging Face-format weights are generated in the `<DCP_weight_path>/hf_ckpt` directory.

### 3. Specifying an Output Directory

If you need to save the converted model to a specific location, use the `--save-dir` parameter to specify the output directory:

```bash
python mindspeed_llm/fsdp2/checkpoint/merge_dcp_to_hf.py \
        --load-dir checkpoints/my_model/dcp_checkpoint \
        --save-dir hf_models/my_model
```

### 4. Copying Model Configuration Files at the Same Time

If you want the generated Hugging Face weights to load directly with `from_pretrained()`, provide the model configuration files, such as `config.json` and `tokenizer.json`:

```bash
python mindspeed_llm/fsdp2/checkpoint/merge_dcp_to_hf.py \
        --load-dir checkpoints/my_model/dcp_checkpoint \
        --save-dir hf_models/my_model \
        --model-configs pretrained_models/qwen3-8b
```

The script also copies the configuration files in that directory to the output directory.

### 5. Controlling the Weight Shard Size

For environments with limited memory resources, you can use `--shard-size` to control the maximum size of a single weight shard:

```bash
python mindspeed_llm/fsdp2/checkpoint/merge_dcp_to_hf.py \
        --load-dir checkpoints/my_model/dcp_checkpoint \
        --shard-size 2000000000
```

This example limits the size of each shard to 2 GB. Therefore, it helps reduce the memory peak during conversion.

## Outputs

1. When the total model size does not exceed `shard-size`, the output is a single file:

    ```text
    output_directory/
    ├── model.safetensors
    ├── config.json
    └── tokenizer.json
    ```

2. When the total model size exceeds `shard-size`, the output is split into shards:

    ```text
    output_directory/
    ├── model-00001-of-00005.safetensors
    ├── model-00002-of-00005.safetensors
    ├── model-00003-of-00005.safetensors
    ├── model-00004-of-00005.safetensors
    ├── model-00005-of-00005.safetensors
    ├── model.safetensors.index.json
    ├── config.json
    └── tokenizer.json
    ```

## Notes

- `--load-dir` must point to a complete and valid DCP weight directory.
- The script does not generate model configuration files automatically. If you need to load the model directly in Hugging Face, explicitly specify `--model-configs`.
- The conversion process uses shard-based loading, but each shard must still be loadable in the current environment.
