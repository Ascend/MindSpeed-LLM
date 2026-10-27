# Model Input and Output Collection 

This document applies to simple troubleshooting scenarios such as quickly viewing the module forward and backward inputs and outputs and the batch fields at a specified training step. When you need complex precision troubleshooting capabilities such as layered collection by Module/API, call stacks, or graph construction information, you are advised to use [msProbe Precision Data Collection](msprobe.md).

## Overview

MindSpeed LLM provides a source-level model input and output collection capability that is independent of msProbe. It applies to scenarios where you cannot install msProbe, where you need to view
raw batch fields, or where you need to directly locate forward and backward data by module name. This feature supports the following operations:

- Registering full-model module forward hooks and full backward hooks for target ranks and target training steps
- Collecting the inputs, outputs, and gradient statistics of all leaf modules by default
- Collecting previews of batch fields such as `input_ids`, `tokens`, `labels`, `attention_mask`, and `position_ids`
- Saving complete tensors as `.pt` files as required

Hooks exist only in the training steps that the configuration matches, and the system removes them immediately after the step ends. A step represents one complete training iteration, which includes all
micro-batches and optimizer updates of that iteration.

## `config.json` File

MindSpeed LLM provides a default configuration file [`model_io_trace_config.json`](../../../../mindspeed_llm/tools/model_io_trace_config.json), and uses this configuration automatically when you do not specify a custom configuration file.

The default configuration explicitly lists all supported items and collects the leaf module statistics and common batch fields of rank 0 and step 0. The configuration example is as follows:

```json
{
  "output_format": "text",
  "ranks": [0],
  "steps": [0],
  "module": {
    "leaf_only": true,
    "include": [],
    "exclude": [],
    "forward": true,
    "backward": true
  },
  "tensor": {
    "mode": "statistics",
    "statistics": ["abs_sum", "abs_mean", "max", "min"]
  },
  "batch": {
    "enabled": true,
    "fields": ["input_ids", "tokens", "labels", "attention_mask", "position_ids"],
    "max_rows": 2,
    "max_tokens": 64
  }
}
```

To change the collection ranks, steps, or module scope, copy the default file and pass the path of the custom configuration through a startup parameter. The root directory of the collection results is not written into `config.json`. It is passed in uniformly through the startup parameters of FSDP2 or Megatron.

| Configuration Item | Required | Default | Description |
| --- | --- | --- | --- |
| output_format | No | `"text"` | Output format. `text` generates human-readable text files. `jsonl` generates structured records that are easy for programs to parse. `both` generates both formats. |
| ranks | No | `[0]` | List of target global ranks. `[-1]` indicates all ranks. |
| steps | No | `[0]` | Target global steps. Integer values, closed-interval strings, and `"all"` are supported. |
| module.leaf_only | No | `true` | When set to `true`, collects only leaf nodes that have no child modules. |
| module.include | No | `[]` | Glob allowlist of full module names. An empty list indicates all modules. |
| module.exclude | No | `[]` | Glob denylist of full module names. |
| module.forward | No | `true` | Whether to register forward hooks for modules. |
| module.backward | No | `true` | Whether to register full backward hooks for modules. |
| tensor.mode | No | `"statistics"` | `statistics` collects only statistics. `tensor` saves complete tensors. |
| tensor.statistics | No | `["abs_sum", "abs_mean", "max", "min"]` | Statistics to compute. All of them are calculated based on absolute values. |
| batch.enabled | No | `true` | Whether to collect batch field previews. |
| batch.fields | No | `["input_ids", "tokens", "labels", "attention_mask", "position_ids"]` | Batch fields to record. The field names must be consistent with the input parameter names of `model.forward`. Positional arguments and keyword arguments are both matched by name. |
| batch.max_rows | No | 2 | Maximum number of rows recorded in a batch preview. |
| batch.max_tokens | No | 64 | Maximum number of elements recorded per row in a batch preview. |

- Module names used in `include`, `exclude`, and JSONL records start with `model_chunk_0` and `model_chunk_1`, which helps distinguish the model chunks on the current rank of Megatron PP/VPP.

- For a single model, the `text` output removes the `model_chunk_0` prefix to make module names more concise. Multi-model-chunk scenarios retain the prefix to prevent duplicate names.

- `include` and `exclude` use globs, for example:

    ```json
    {
      "include": ["model_chunk_0.*.mlp.*"],
      "exclude": ["*.dropout*"]
    }
    ```

## Enabling Collection on the FSDP2 Backend

Add the following example code to the `training` field in the YAML configuration file:

```yaml
training:
  model_io_trace: true
  model_io_trace_output_path: /data/model_io_trace/qwen3_8b
  model_io_trace_config_path: /home/custom_config.json
```

You can also override the settings from the CLI:

```bash
--training.model_io_trace True \
--training.model_io_trace_output_path /data/model_io_trace/qwen3_8b \
--training.model_io_trace_config_path /home/custom_config.json
```

- `model_io_trace`: Optional parameter, whether to enable model input and output collection. MindSpeed LLM does not enable it by default, and you must set this parameter when you enable the feature.

- `model_io_trace_output_path`: Required parameter, the path for saving the model input and output collection results.

- `model_io_trace_config_path`: Optional parameter, the path to a custom configuration file for model input and output collection.

  > [!NOTE]
  >
  > - When the `model_io_trace_config_path` parameter is not configured, MindSpeed LLM uses [`model_io_trace_config.json`](../../../../mindspeed_llm/tools/model_io_trace_config.json) by default.
  > - The custom configuration file `custom_config.json` must be a file accessible to all training processes.

## Qwen3-8B FSDP2 Example

Take the [Qwen3-8B FSDP2 pretraining configuration](../../../../examples/fsdp2/qwen3/pretrain_qwen3_8b_4k_fsdp2_A3.yaml) as an example. Assume that the loss of the first training step is abnormal, and you need to quickly view the module forward and backward inputs and outputs and the batch fields of rank 0 and step 0.

1. Enable model input and output collection

    The default configuration already collects rank 0 and step 0. Therefore, you only need to enable the feature and specify the output directory in the `training` field of the pretraining YAML file.

    ```yaml
    training:
      model_io_trace: true
      model_io_trace_output_path: /data/model_io_trace/qwen3_8b
    ```

2. Edit the training example script

    Modify and save the model and data paths according to your actual environment. Run the following command:

    ```bash
    vi examples/fsdp2/qwen3/pretrain_qwen3_8b_4k_fsdp2_A3.sh
    ```

3. Run the training script

    ```bash
    bash examples/fsdp2/qwen3/pretrain_qwen3_8b_4k_fsdp2_A3.sh
    ```

After the first training step is complete, the collection results of rank 0 are located in `/data/model_io_trace/qwen3_8b/rank0/step0/`. You can first view `output.txt` and `token_ids.log`. If you locate a suspicious module, narrow the collection scope with `module.include`.

## Enabling Collection on the Megatron Backend

When you train a model on the Megatron backend, add the following example code to the training startup parameters:

```bash
--model-io-trace \
--model-io-trace-output-path /data/model_io_trace/qwen3_8b \
--model-io-trace-config-path /home/custom_config.json
```

- `--model-io-trace`: Optional parameter, whether to enable model input and output collection. MindSpeed LLM does not enable it by default, and you must set this parameter when you enable the feature.

- `--model-io-trace-output-path`: Required parameter, the path for saving the model input and output collection results.

- `--model-io-trace-config-path`: Optional parameter, the path to a custom configuration file for model input and output collection.

  > [!NOTE]
  >
  > - When the `--model-io-trace-config-path` parameter is not configured, MindSpeed LLM uses [`model_io_trace_config.json`](../../../../mindspeed_llm/tools/model_io_trace_config.json) by default.
  > - The custom configuration file `custom_config.json` must be a file accessible to all training processes.

The Megatron step uses the restored `args.iteration`, while FSDP2 uses the restored `global_step`. Therefore, the first iteration of a fresh training run corresponds to step 0.

## Output

A typical output structure is as follows:

```text
output_path/
└── rank0/
    └── step10/
        ├── output.txt
        ├── token_ids.log
        └── tensors/
```

- `output.txt`: Records the module forward and backward inputs and outputs and the tensors in call order, as well as the absolute-value `sum/mean/max/min`.
- Single-model scenarios use concise module names. Megatron PP/VPP multi-model-chunk scenarios add a `model_chunk_N` prefix to prevent duplicate names.
- `token_ids.log` saves the shape and truncated preview of the batch fields for each call to the model entry point, so that the NPU and the GPU can use the same text processing script.

When you set `"output_format": "text"`, the tool converts tensors to CPU `float32` and writes out the PyTorch text representation. Even if `tensor.mode` is set to `statistics`, the output still includes truncated numeric previews.

When you set `"output_format": "jsonl"`, the tool generates `module_io.jsonl` and `batch_inputs.jsonl`. Each line of the former records one module input, output, or gradient tensor, including the module name, phase, slot, shape, dtype, device, and statistics. The latter saves the batch field shapes and truncated previews.

When you set `both`, the tool generates both sets of files.

The tool creates `tensors/` only when `tensor.mode` is set to `tensor`.

## Usage Notes

Full-model forward and backward hooks significantly reduce training performance, and complete tensors also introduce device-to-host synchronization and quickly fill the drive. For initial troubleshooting, you are advised to select only one rank and one step and to keep `leaf_only: true` and `tensor.mode: statistics`. After you identify a suspicious module, narrow the scope with `include` and change `tensor.mode` to `tensor`. Batch previews and complete tensors may contain training data. Therefore, do not write them to a public directory.
