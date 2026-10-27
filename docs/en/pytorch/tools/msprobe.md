# msProbe Precision Data Collection 

## Overview

The FSDP2 and Megatron backends of MindSpeed LLM support calling msProbe during training to collect statistics or actual tensors of the Module/API forward and backward inputs and outputs. One training iteration (including all its micro-batches and optimizer updates) corresponds to one msProbe step.

The `rank`, `step`, `level`, `task`, and output path of msProbe continue to be controlled by its native configuration file. MindSpeed LLM provides a ready-to-use default configuration file [`msprobe_config.json`](../../../../mindspeed_llm/tools/msprobe_config.json), which collects L0-level statistics for rank 0 and step 0 by default and writes the results to `msprobe_dump` in the current working directory.

msProbe applies to scenarios where you need to collect statistics or actual tensors hierarchically by Module/API and to troubleshoot complex precision problems in combination with call stack and graph construction information. If you only need to quickly view the Module forward and backward inputs and outputs or the batch fields of a specified training step, you can use [Model Input and Output Collection](model_io_trace.md).

## Tool Preparation

Install an msProbe version that is compatible with the current PyTorch, TorchNPU, and CANN versions. To prevent pre-release version changes from affecting the collection results, refer to the [msProbe Tool Installation Guide](https://gitcode.com/Ascend/msprobe/blob/26.0.0/docs/en/msprobe_install_guide.md) and install a stable version.

For online installation, run the following command:

```bash
pip install mindstudio-probe==26.0.0
```

For the version mapping of other software, see [msProbe Release](https://gitcode.com/Ascend/msprobe/releases).

When msProbe is not enabled, MindSpeed LLM does not import or depend on this package.

## Enabling Collection on the FSDP2 Backend

When you train a model on the FSDP2 backend, add the following example code to the `training` field of the YAML configuration file:

```yaml
training:
  msprobe: true
  msprobe_config_path: /home/custom_msprobe_config.json
```

You can also override the settings from the CLI:

```bash
--training.msprobe True \
--training.msprobe_config_path /home/custom_msprobe_config.json
```

- `msprobe`: Optional parameter, whether to enable msProbe precision data collection. MindSpeed LLM does not enable it by default, and you must set this parameter when you enable the tool.

- `msprobe_config_path`: Optional parameter, used for custom configuration when you need to override the default rank, step, collection level, or output path.

> [!NOTE]
>
> - When the `msprobe_config_path` parameter is not configured, MindSpeed LLM uses [`msprobe_config.json`](../../../../mindspeed_llm/tools/msprobe_config.json) by default.
> - The custom configuration file `custom_msprobe_config.json` must be a file accessible to all training processes.
> - When you resume training, the collection step aligns with the training global step after recovery.

## Qwen3-8B FSDP2 Example

Take the [Qwen3-8B FSDP2 pretraining configuration](../../../../examples/fsdp2/qwen3/pretrain_qwen3_8b_4k_fsdp2_A3.yaml) as an example. Assume that the loss of the first training step is abnormal, and you need to collect Module-level statistics for rank 0 and step 0 to troubleshoot the issue.

1. Enable msProbe

    The default configuration already collects rank 0 and step 0. Therefore, you only need to enable msProbe in the `training` field of the pretraining YAML file.

    ```yaml
    training:
      msprobe: true
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

After the first training step is complete, the collection results of rank 0 are located in `./msprobe_dump/step0/rank0/`. First, use the statistics in `dump.json` to locate suspicious APIs. If you need to change the output directory or collect actual tensors, you can copy the default configuration and pass the custom configuration file through `msprobe_config_path`.

## Enabling Collection on the Megatron Backend

When you train a model on the Megatron backend, add the following example code to the training startup parameters:

```bash
--msprobe \
--msprobe-config-path /home/custom_msprobe_config.json
```

- `--msprobe`: Optional parameter, whether to enable msProbe precision data collection. MindSpeed LLM does not enable it by default, and you must set this parameter when you enable the tool.

- `--msprobe-config-path`: Optional parameter, used for custom configuration when you need to override the default rank, step, collection level, or output path.

  > [!NOTE]
  >
  > When the `--msprobe-config-path` parameter is not configured, MindSpeed LLM uses [`msprobe_config.json`](../../../../mindspeed_llm/tools/msprobe_config.json) by default.

The Megatron backend enables collection before each `train_step()` and calls `stop()` and `step()` in sequence after that training iteration ends. In PP/VPP scenarios, all model chunks on the current rank are passed to msProbe as a list.

When you resume training, the start step of msProbe uses `args.iteration` after checkpoint recovery. The first iteration of a fresh training run corresponds to msProbe `step0`, and the training log shows `iteration 1` after that iteration is complete.

## `config.json` Example

For the first troubleshooting attempt on an LLM, you are advised to collect only the Module-level statistics of the target rank and the target step. The following shows a `config.json` configuration example:

```json
{
  "task": "statistics",
  "dump_path": "/data/msprobe/statistics",
  "rank": [0],
  "step": [0],
  "level": "L0",
  "async_dump": false,
  "precision": "low",
  "extra_info": true,
  "statistics": {
    "scope": [],
    "list": [],
    "tensor_list": [],
    "data_mode": ["all"],
    "summary_mode": "statistics"
  }
}
```

After you locate suspicious APIs, collect actual tensors separately. The following shows a `config.json` configuration example:

```json
{
  "task": "tensor",
  "dump_path": "/data/msprobe/tensor",
  "rank": [0],
  "step": [0],
  "level": "L1",
  "async_dump": false,
  "tensor": {
    "scope": [],
    "list": ["Functional.linear.0.forward"],
    "data_mode": ["all"],
    "summary_mode": "statistics"
  }
}
```

- Replace the names in `list` with API names that actually exist in the `dump.json` generated by the first collection. Do not use an empty `list` to collect tensors for the entire network on an LLM. Otherwise, it generates a large amount of data.

- `step` supports a single step, discrete steps, and a closed interval, for example, `[10]`, [10, 15], and `["10-19"]`. Both the FSDP2 and Megatron backends continue to advance the internal msProbe step in training iterations that are not matched, but they do not write data to drive for those iterations.

## Output

A typical output structure of a multi-device task is as follows:

```text
dump_path/
└── step0/
    └── rank0/
        ├── construct.json
        ├── dump.json
        ├── stack.json
        └── dump_tensor_data/
```

`task: statistics` saves only statistics such as shape, dtype, max, min, mean, and norm, while `task: tensor` also saves actual tensors in `dump_tensor_data`.

For more information about configuration and the troubleshooting process, see:

- [msProbe Configuration File Introduction](https://gitcode.com/Ascend/msprobe/blob/26.2.0/docs/en/user_guide/dump/config_json_introduct.md)
- [msProbe LLM Training Precision Troubleshooting Guide](https://gitcode.com/Ascend/msprobe/blob/26.2.0/docs/en/best_practices/train_debug_guide.md)
