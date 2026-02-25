# 断点续训功能使用介绍

在大规模模型预训练过程中，训练可能因硬件故障、资源调度等原因中断。为支持从中断处恢复训练，系统提供了 **断点续训（Checkpoint Resume Training）** 功能。本文档简要说明如何配置和使用该功能。

---

## 一、启用断点续训的前提条件

为了支持断点续训，需在启动预训练脚本时正确设置相关参数，确保优化器状态、模型参数和训练进度均被完整保存。

### ✅ 正确设置 `pretrain` 脚本参数

在预训练脚本的 `GPT_ARGS` 里要注意：

```
GPT_ARGS="
    [其他参数...] \
    --use-distributed-optimizer \  # 使用分布式优化器（必选）
"
```

#### 关键参数说明：

| 参数 | 说明 |
|------|------|
| `--use-distributed-optimizer` | 必须开启，使优化器状态也按数据并行方式分布保存，便于后续恢复 |
| `--finetune` | ❌ 不可设置，否则会跳过优化器状态加载 |
| `--no-load-optim` | ❌ 不可设置，否则不会恢复优化器状态（如学习率、动量等） |

> ⚠️ 若设置了 `--finetune` 或 `--no-load-optim`，系统将不恢复优化器状态，导致无法真正“续训”。

---

## 二、保存训练检查点（Checkpoint）

当训练运行时，只要配置了 `--save $SAVE_PATH` 和保存频率（如 `--save-interval`），系统会自动定期保存完整检查点，包括：

- 模型权重（model weights）
- 优化器状态（optimizer states）
- 训练步数（iteration）
- 随机状态（random states）等

示例：
```
--save /your/checkpoint/path \
--save-interval 500   # 每 500 步保存一次
```

每次保存生成如下结构：
```
/your/checkpoint/path/
├── latest_checkpointed_iteration.txt
├── iter_0000001/
│   ├── mp_rank_00_000
|   |    |—— distrib_optim.pt
|   |    |—— model_optim_rng.pt
│   └── ...
└── iter_0000500/
    ├── mp_rank_00_000
    |    |—— distrib_optim.pt
    |    |—— model_optim_rng.pt
    └── ...
```

---

## 三、从检查点恢复训练（加载权重）

要从中断处继续训练，在预训练脚本中的启动命令中指定 `--load` 为之前的保存路径：

```
GPT_ARGS="
    [其他参数...] \
    --use-distributed-optimizer \  # 使用分布式优化器（必选）
"

...

msrun ${DISTRIBUTED_ARGS} pretrain_gpt.py \
    [其他参数...] \
    --load $CHECKPOINT_PATH \
```

> 系统会自动读取 `latest_checkpointed_iteration.txt` 文件，找到最新的迭代步数，并恢复模型和优化器状态。

#### 自动恢复内容包括：
- 模型参数
- 优化器状态（Adam momentum, variance 等）
- 学习率调度器状态
- 已完成的训练步数（避免重复训练）

恢复后训练将继续从断点开始，打印如下日志内容
```
successfully loaded checkpoint from xx at iteration x
(min, max) time across ranks(ms):
load-checkpoint ....................:(9289.88, 9288.22)
```

---

## 四、注意事项

1. **文件完整性**：确保 `$CHECKPOINT_PATH` 下的检查点文件完整无损。
2. **训练配置一致性**：恢复训练时，batch size、模型结构、训练步数、优化器类型等应与保存时一致。
3. **分布式环境匹配**：NPU 数量和并行策略（TP/DP）要保持不变，否则会导致加载失败。
