# msProbe 精度数据采集

## 概述

MindSpeed LLM FSDP2与Megatron后端支持在训练过程中调用msProbe采集Module/API前反向输入输出的统计量或真实Tensor。一次训练迭代（包括其全部micro-batch和optimizer update）对应一个msProbe step。

msProbe的`rank`、`step`、`level`、`task`和输出路径继续由其原生配置文件控制。MindSpeed LLM提供了可直接使用的默认配置文件[`msprobe_config.json`](../../../../mindspeed_llm/tools/msprobe_config.json)，默认采集rank 0、step 0的L0级统计量，并将结果写入当前工作目录下的`msprobe_dump`。

msProbe适用于需要按Module/API分层采集统计量或真实Tensor，并结合调用栈、构图信息开展复杂精度问题定位的场景。如果只需要快速查看指定训练step的Module前反向输入输出或batch字段，可使用[模型输入输出采集](model_io_trace.md)。

## 工具准备

安装与当前PyTorch、TorchNPU和CANN版本兼容的msProbe。为避免预发布版本变化影响采集结果，请参考[msProbe工具安装指南](https://gitcode.com/Ascend/msprobe/blob/26.0.0/docs/zh/msprobe_install_guide.md)安装稳定版本。

在线安装可执行如下命令：

```bash
pip install mindstudio-probe==26.0.0
```

其他软件版本配套关系请参见[msProbe Release](https://gitcode.com/Ascend/msprobe/releases)。

未开启msProbe时，MindSpeed LLM不会导入或依赖该软件包。

## FSDP2后端开启采集

基于FSDP2后端进行模型训练时，需在YAML配置文件的`training`字段中增加如下示例代码：

```yaml
training:
  msprobe: true
  msprobe_config_path: /home/custom_msprobe_config.json
```

也可以通过命令行覆盖：

```bash
--training.msprobe True \
--training.msprobe_config_path /home/custom_msprobe_config.json
```

- `msprobe`：可选参数，是否启用msProbe精度数据采集。MindSpeed LLM默认未开启，启用工具时必选该参数。

- `msprobe_config_path`：可选参数，需要覆盖默认rank、step、采集级别或输出路径时可进行自定义配置。

> [!NOTE]
>
> - `msprobe_config_path`参数未配置时，MindSpeed LLM默认使用[`msprobe_config.json`](../../../../mindspeed_llm/tools/msprobe_config.json)。
> - 自定义配置文件`custom_msprobe_config.json`必须是所有训练进程均可访问的文件。
> - 断点续训时，采集step会与恢复后的训练global step对齐。

## Qwen3-8B FSDP2使用案例

以 [Qwen3-8B FSDP2预训练配置](../../../../examples/fsdp2/qwen3/pretrain_qwen3_8b_4k_fsdp2_A3.yaml) 为例，假设首个训练step的loss异常，需要采集rank 0、step 0的Module级统计量进行定位。

1. 启用msProbe

    默认配置已经采集rank 0、step 0，此时只需在预训练YAML文件的`training`字段中启用msProbe。

    ```yaml
    training:
      msprobe: true
    ```

2. 编辑训练示例脚本

    根据实际环境修改并保存模型和数据路径，命令如下：

    ```bash
    vi examples/fsdp2/qwen3/pretrain_qwen3_8b_4k_fsdp2_A3.sh
    ```

3. 执行训练脚本

    ```bash
    bash examples/fsdp2/qwen3/pretrain_qwen3_8b_4k_fsdp2_A3.sh
    ```

首个训练step完成后，rank 0的采集结果位于`./msprobe_dump/step0/rank0/`。先根据`dump.json`中的统计量定位可疑API，若需修改输出目录或采集真实Tensor，可复制默认配置，并通过`msprobe_config_path`传入自定义配置文件。

## Megatron后端开启采集

基于Megatron后端进行模型训练时，需在训练启动参数中增加如下示例代码：

```bash
--msprobe \
--msprobe-config-path /home/custom_msprobe_config.json
```

- `--msprobe`：可选参数，是否启用msProbe精度数据采集。MindSpeed LLM默认未开启，启用工具时必选该参数。

- `--msprobe-config-path`：可选参数，需要覆盖默认rank、step、采集级别或输出路径时可进行自定义配置。

  > [!NOTE]
  >
  > `--msprobe-config-path`参数未配置时，MindSpeed LLM默认使用[`msprobe_config.json`](../../../../mindspeed_llm/tools/msprobe_config.json)。

Megatron后端会在每次`train_step()`前开启采集，并在该训练迭代结束后依次调用`stop()`和`step()`。PP/VPP场景会将当前 rank上的全部模型分块作为一个列表传入msProbe。

断点续训时，msProbe的起始step会使用checkpoint恢复后的`args.iteration`。全新训练的第一次迭代对应msProbe `step0`，训练日志在该次迭代完成后显示`iteration 1`。

## config.json示例

大模型进行首次定位时，建议只采集目标rank、目标step的Module级统计量。config.json配置示例如下：

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

定位到可疑API后，再单独采集真实Tensor。config.json配置示例如下：

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

- `list`中的名称应替换为首次采集生成的`dump.json`中实际存在的API名称。不要在大模型上使用空`list`采集整网Tensor，否则会产生大量数据。

- `step`支持单步、离散步数和闭区间，例如`[10]`、`[10, 15]`和`["10-19"]`。FSDP2与Megatron后端都会在未命中的训练迭代继续推进msProbe内部step，但不会为该迭代落盘。

## 输出

多卡任务的典型输出结构如下：

```text
dump_path/
└── step0/
    └── rank0/
        ├── construct.json
        ├── dump.json
        ├── stack.json
        └── dump_tensor_data/
```

`task: statistics`仅保存shape、dtype、max、min、mean、norm等统计量；`task: tensor`还会在`dump_tensor_data`中保存真实Tensor。

更多配置与定位流程请参考：

- [msProbe 配置文件介绍](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/user_guide/dump/config_json_introduct.md)
- [msProbe 大模型训练精度定位指南](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/best_practices/train_debug_guide.md)
