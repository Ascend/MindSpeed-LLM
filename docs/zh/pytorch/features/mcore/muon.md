# Muon 优化器

## 使用场景

Muon（Momentum + Orthogonalization Update）是一种面向大语言模型预训练的高效优化器。其核心思路是对动量梯度通过 Newton-Schulz 迭代进行正交化处理，使参数更新矩阵近似正交，适用于期望在相同计算预算下获得比 Adam 更优收敛效率的训练任务。

## 使用说明

通过在训练脚本中设置 `--optimizer muon` 启用 Muon 优化器。以下为完整参数说明：

### 基础参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--optimizer muon` | str | — | 启用 Muon 优化器 |
| `--muon-momentum` | float | `0.9` | Muon 内部 SGD 的动量系数 |
| `--muon-use-nesterov` | flag | `False` | 在内部 SGD 中启用 Nesterov 动量 |
| `--muon-no-split-qkv` | flag | `True` | 禁用 QKV 参数的分块独立正交化（默认启用分块）|
| `--muon-extra-scale-factor` | float | `1.0` | 对 Muon 更新量施加额外的全局缩放因子 |

### Newton-Schulz 迭代参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--muon-num-ns-steps` | int | `5` | Newton-Schulz 迭代步数，步数越多正交化越精确，计算开销越大 |
| `--muon-fp32-matmul-prec` | str | `medium` | NS 迭代中 FP32 矩阵乘精度，影响正交化数值稳定性 |
| `--muon-scale-mode` | str | `spectral` | 正交化后更新量的缩放模式 |

### 张量并行模式参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--muon-tp-mode` | str | `blockwise` | 张量并行权重的 NS 正交化计算方式 |

## 使用约束

使用 Muon 优化器时，以下特性**不可同时启用**：

| 不兼容特性 | 对应参数 |
|------------|----------|
| 梯度规约重叠 | `--overlap-grad-reduce` |
| 参数 Gather 重叠 | `--overlap-param-gather` |
| 分布式优化器 | `--use-distributed-optimizer` |
| Torch FSDP2 | `--use-torch-fsdp2` |
