# GatedDeltaNet (dense / hybrid linear attention)

稠密（dense）版的 GatedDeltaNet 混合线性注意力模型。本目录用于在昇腾 NPU 上
**复现论文 [A Systematic Analysis of Hybrid Linear Attention](https://arxiv.org/abs/2507.06457)**
（arXiv:2507.06457）中的 GatedDeltaNet 混合架构：在不同的「线性注意力 : 全注意力」
混合比下从头预训练，并与 GPU 参考实现对齐。提供 **340M** 与 **1.3B** 两个规格的
预训练与评估脚本。

## 架构与 spec

- 复用 Qwen3-Next 的 GatedDeltaNet / full-attention 层实现与 `qwen3_next_spec`
  （`qwen3_next_gated_deltanet_attention`、`qwen3_next_full_attention`）。
- `qwen3_next_spec` 扩展为按 `--num-experts` **动态选择 MLP**：不传 MoE 参数即为
  **dense**（标准 MLP），传 `--num-experts>0` 则为 MoE。
- 训练用 `--spec mindspeed_llm.tasks.models.spec.qwen3_next_spec layer_spec`，
  dense 由"不传任何 MoE 参数"触发。

## 混合比（hybrid ratio）

由 `--full-attention-interval N` 控制：每 N 层放 1 层 full-attention，其余为
GatedDeltaNet，即 `(N-1):1` 的 GDN:attn 比例。论文评估了 24:1 / 12:1 / 6:1 / 3:1
等多种比例，对应 `--full-attention-interval` = 25 / 13 / 7 / 4 …
示例脚本用 `3`（即 2:1）；改这一个值即可切换不同混合比。

## 规格

| 规格 | num-layers | hidden | ffn | linear heads / dim |
|---|---|---|---|---|
| 340M | 24 | 1024 | 2816 | 4 / 256 |
| 1.3B | 24 | 2048 | 5632 | 8 / 256 |

## 训练方案

**数据**：[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)。

- 1.3B 使用 [`sample/100BT`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/main/sample/100BT) 子集（~100B tokens）。
- 340M 取 `sample/100BT` 的前若干个 parquet 分片累计到 ~20B tokens。

把原始文件（`.parquet`/`.jsonl`）填进 `pretrain_*.sh` 的 `DATA_PATH`，训练初始化时自动预处理
（参考[train_from_hf](../../../docs/zh/pytorch/training/pretrain/mcore/train_from_hf.md)），无需单独的数据转换脚本。
tokenizer 与模型一致（`PretrainedFromHF`，`padded-vocab-size=32000`）。

**优化配置**（完整超参以各 `pretrain_*.sh` 为准）：

| 项 | 340M | 1.3B |
|---|---|---|
| 训练 tokens | ~20B | ~100B |
| 序列长度 | 4096 | 4096 |
| global / micro batch | 48 / 6 | 256 / 2 |
| 训练步数 | 101726 | 95368 |
| 学习率（峰值 / 末端） | 3e-4 / 3e-5 | 1.3e-3 / 1.3e-4 |
| 调度 / warmup | cosine / 1024 步 | cosine / 1024 步 |
| 并行（TP / PP / DP） | 1 / 1 / 8 | 1 / 1 / 16 |

- 优化器：AdamW（β1=0.9，β2=0.95，weight-decay=0.1，clip-grad=1.0，init-std=0.01），bf16 训练。
- 并行：TP=PP=1 的单机数据并行（DP = NPU 卡数，340M 8 卡 / 1.3B 16 卡）。
- token 数 = global batch × 序列长度 × 步数（48×4096×101726 ≈ 20B；256×4096×95368 ≈ 100B）。
- GDN/混合配置（`--full-attention-interval 3`、`--mamba-chunk-size 64`、`--mamba-d-conv 4` 等）见脚本。
- 训练合一：默认从头预训练（`CKPT_LOAD_DIR=None`），训练保存时自动 mcore→HF（`--enable-mg2hf-convert`）；
  如需从已有 HuggingFace 权重继续预训练，将脚本里的 `CKPT_LOAD_DIR` 指向 HF 权重目录即可（自动 HF→mcore）。

## 脚本

| 文件 | 用途 |
|---|---|
| `pretrain_gated_deltanet_340M_4K_A2_ptd.sh` | 340M 预训练（4K 序列，A2） |
| `pretrain_gated_deltanet_1.3B_4K_A3_ptd.sh` | 1.3B 预训练（4K 序列，A3） |
| `evaluate_gated_deltanet_lmeval.sh` | HF + lm-evaluation-harness 下游评估 |

> 预训练脚本已内置「训练合一」：默认从头训练、保存时自动 mcore→HF（`--enable-mg2hf-convert`）、
> 原始数据训练时自动预处理，无需单独的权重转换与数据预处理脚本。

## 评估

训练已自动将 Megatron 权重转为 HuggingFace 格式（`--enable-mg2hf-convert`，默认保存在
`{CKPT_SAVE_DIR}/mg2hf_iteration{N}/`），直接在该 HF 权重上用
`evaluate_gated_deltanet_lmeval.sh`（封装
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)）
跑 arc / hellaswag / lambada / openbookqa / piqa 等任务。

## 精度

- **框架级（GPU Flame 框架）**：GatedDeltaNet 的前向/反向已与 GPU 上的 Flame 框架
  （<https://github.com/fla-org/flame>）参考实现逐张量对齐。喂入同一份 bf16 输入
  （T=4096），对比前向输出 `o` 与 5 个输入梯度 `dq/dk/dv/dg/dbeta`：

  | 张量 | max_abs（最坏单点） | mean_abs（全张量平均） |
  |---|---|---|
  | `o`（前向输出） | 1.22e-4 | 1.6e-8 |
  | `dq` | 1.22e-4 | 9.7e-8 |
  | `dk` | 1.22e-4 | 1.9e-7 |
  | `dv` | 1.22e-4 | 8.6e-8 |
  | `dg` | 9.77e-4 | 4.6e-6 |
  | `dbeta` | 1.95e-3 | 6.9e-6 |

  `max_abs` 是全张量最坏单点误差，`mean_abs` 是逐元素平均（比最坏点低 1~3 个数量级，
  说明误差稀疏、仅个别离群点）；余弦相似度均 ≈1.0，为 bf16 下的正常累积量级。
- **收敛**：340M（混合比 2:1，20B tokens）与 GPU 基线 loss 轨迹全程误差 ~2%。
- **下游精度**：在严格对齐训练配置下，NPU 与开源基线相当；详见下表
  （开源基线 = FLAME/GPU，NPU = MindSpeed-LLM）。

**340M（20B tokens）**

| Dataset | 开源基线 | NPU |
|---|---|---|
| ARC-Easy | 0.583 | 0.579 |
| ARC-Challenge | 0.281 | 0.290 |
| HellaSwag | 0.426 | 0.428 |
| LAMBADA | 0.367 | 0.362 |
| OpenBookQA | 0.314 | 0.340 |
| PIQA | 0.677 | 0.671 |
| **Avg.** | 0.441 | **0.445** |

**1.3B（100B tokens）**

| Dataset | 开源基线 | NPU |
|---|---|---|
| ARC-Easy | 0.722 | 0.702 |
| ARC-Challenge | 0.399 | 0.385 |
| HellaSwag | 0.593 | 0.586 |
| LAMBADA | 0.506 | 0.498 |
| OpenBookQA | 0.398 | 0.402 |
| PIQA | 0.735 | 0.732 |
| **Avg.** | 0.559 | 0.551 |

全任务平均精度与开源基线相差 <1%。
