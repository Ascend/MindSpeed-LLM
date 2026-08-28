# msProbe 精度数据采集

## 概述

MindSpeed LLM FSDP2 与 Megatron 后端支持在训练过程中调用 msProbe 采集 Module/API 前反向输入输出的统计量或真实 Tensor。一次训练迭代（包括其全部 micro-batch 和 optimizer update）对应一个 msProbe step。

msProbe 的 `rank`、`step`、`level`、`task` 和输出路径继续由其原生配置文件控制。仓库提供了可直接使用的[默认配置文件](../../../../mindspeed_llm/tools/msprobe_config.json)，默认采集 rank 0、step 0 的 L0 级统计量，并将结果写入当前工作目录下的 `msprobe_dump`。

msProbe 适用于需要按 Module/API 分层采集统计量或真实 Tensor，并结合调用栈、构图信息开展复杂精度问题定位的场景。如果只需要快速查看指定训练 step 的 Module 前反向输入输出或 batch 字段，可使用[模型输入输出采集](model_io_trace.md)。

## 环境准备

安装与当前 PyTorch、torch_npu 和 CANN 版本兼容的 msProbe。为避免预发布版本变化影响采集结果，建议固定安装稳定版本：

```bash
pip install mindstudio-probe==26.0.0
```

其他软件版本配套关系请参考 msProbe 对应版本的发布说明。

未开启 msProbe 时，MindSpeed LLM 不会导入或依赖该软件包。

## FSDP2 后端开启采集

在 FSDP2 训练 YAML 的 `training` 字段中增加：

```yaml
training:
  msprobe: true
  # 可选；不配置时使用仓库默认 msprobe_config.json
  msprobe_config_path: /absolute/path/to/custom_msprobe_config.json
```

也可以通过命令行覆盖：

```bash
--training.msprobe True \
--training.msprobe_config_path /absolute/path/to/custom_msprobe_config.json
```

`msprobe_config_path` 为可选参数；需要覆盖默认 rank、step、采集级别或输出路径时再传入自定义配置。自定义配置文件必须是所有训练进程均可访问的文件。断点续训时，采集 step 会与恢复后的训练 global step 对齐。

## Qwen3-8B FSDP2 使用案例

以仓库中的 [Qwen3-8B FSDP2 预训练配置](../../../../examples/fsdp2/qwen3/pretrain_qwen3_8b_4k_fsdp2_A3.yaml) 为例，假设首个训练 step 的 loss 异常，需要采集 rank 0、step 0 的 Module 级统计量进行定位。

默认配置已经采集 rank 0、step 0，因此只需在预训练 YAML 的 `training` 字段中开启功能：

```yaml
training:
  msprobe: true
```

根据实际环境修改模型和数据路径后，运行仓库中的启动脚本：

```bash
bash examples/fsdp2/qwen3/pretrain_qwen3_8b_4k_fsdp2_A3.sh
```

首个训练 step 完成后，rank 0 的采集结果位于 `./msprobe_dump/step0/rank0/`。先根据 `dump.json` 中的统计量定位可疑 API；需要修改输出目录或采集真实 Tensor 时，复制默认配置并通过 `msprobe_config_path` 传入自定义文件。

## Megatron 后端开启采集

在 Megatron 训练启动参数中增加：

```bash
--msprobe \
--msprobe-config-path /absolute/path/to/custom_msprobe_config.json
```

`--msprobe-config-path` 可省略；省略时使用仓库默认 `msprobe_config.json`。

Megatron 后端会在每次 `train_step()` 前开启采集，并在该训练迭代结束后依次调用 `stop()` 和 `step()`。PP/VPP 场景会将当前 rank 上的全部模型分块作为一个列表传入 msProbe。

断点续训时，msProbe 的起始 step 会使用 checkpoint 恢复后的 `args.iteration`。全新训练的第一次迭代对应 msProbe `step0`，训练日志在该次迭代完成后显示 `iteration 1`。

## config.json 示例

大模型首次定位建议只采集目标 rank、目标 step 的 Module 级统计量：

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

定位到可疑 API 后，再单独采集真实 Tensor：

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

`list` 中的名称应替换为首次采集生成的 `dump.json` 中实际存在的 API 名称。不要在大模型上使用空 `list` 采集整网 Tensor，否则会产生大量数据。

`step` 支持单步、离散步数和闭区间，例如 `[10]`、`[10, 15]` 和 `["10-19"]`。FSDP2 与 Megatron 后端都会在未命中的训练迭代继续推进 msProbe 内部 step，但不会为该迭代落盘。

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

`task: statistics` 只保存 shape、dtype、max、min、mean、norm 等统计量；`task: tensor` 还会在 `dump_tensor_data` 中保存真实 Tensor。

更多配置与定位流程请参考：

- [msProbe 配置文件介绍](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/user_guide/dump/config_json_introduct.md)
- [msProbe 大模型训练精度定位指南](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/best_practices/train_debug_guide.md)
