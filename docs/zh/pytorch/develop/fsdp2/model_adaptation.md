# MindSpeed LLM FSDP2 后端模型适配指南

本文以 GPT-OSS 为例，介绍如何将 Hugging Face 模型基于 MindSpeed LLM 接入 FSDP2 训练后端，并串联权重下载、数据集下载、模型适配、YAML 配置和训练启动的完整流程。

FSDP2 训练后端支持以下两种适配路径：

| 路径 | 适用场景 | 复杂度 | 优势 |
| --- | --- | --- | --- |
| 原生 Transformers 适配 | 模型已被当前环境中的 `transformers` 支持，且无需修改模型结构、算子或 `forward` 逻辑。 | 低 | 无需新增模型代码，通过 `AutoModelForCausalLM` 自动加载。 |
| 自定义注册适配 | 需要在原生 Transformers 实现上进行二次开发。例如，注入 NPU 融合算子、专家并行、上下文并行、特殊 Attention 或 MoE 路由逻辑。 | 中 | 可以直接控制模型实现，适合进行性能优化和硬件亲和适配。 |

GPT-OSS 在 MindSpeed LLM 中使用第二种路径：先基于 Transformers 原生实现迁移模型文件，然后在 `mindspeed_llm/fsdp2/models/gpt_oss/` 下适配 MoE 专家并行和 NPU grouped GEMM。

## 获取权重和数据集

如下命令以 `gpt-oss-20b` 和 Alpaca parquet 数据为例。请根据实际机器磁盘路径调整模型和数据目录。

### 下载 GPT-OSS 权重

| 名称 | 链接 | 用途 |
| --- | --- | --- |
| gpt-oss-20b | [Hugging Face](https://huggingface.co/openai/gpt-oss-20b) / [ModelScope](https://modelscope.cn/models/unsloth/gpt-oss-20b-BF16/) | 本文示例使用的 20B 权重和 tokenizer。 |
| gpt-oss-120b | [Hugging Face](https://huggingface.co/openai/gpt-oss-120b) / [ModelScope](https://modelscope.cn/models/unsloth/gpt-oss-120b-BF16) | 更大规模模型，可按同样方式适配训练配置。 |

以下两种下载方式二选一：

- 使用 Git LFS 下载：

```bash
mkdir -p /home/data
cd /home/data
git lfs install
git clone https://huggingface.co/openai/gpt-oss-20b gpt-oss-20b-hf
```

- 使用 Hugging Face CLI 下载：

```bash
hf download openai/gpt-oss-20b \
  --local-dir /home/data/gpt-oss-20b-hf
```

下载完成后，模型目录应包含 `config.json`、tokenizer 文件和 safetensors 权重文件。

### 下载示例数据集

| 名称 | 链接 | 用途 |
| --- | --- | --- |
| Alpaca parquet 文件 | [https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet) | 可直接下载到本地作为示例数据。 |

下载 parquet 文件：

```bash
mkdir -p /home/data/alpaca
wget -O /home/data/alpaca/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
  https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
```

## 模型适配

### 路径一：原生 Transformers 适配

如果当前 `transformers` 版本已经支持目标模型，例如 GPT-OSS、Qwen3 等，训练时无需改写模型源码，可直接使用 Hugging Face 模型目录启动 FSDP2 训练。

1. 检查模型目录

    模型目录至少需要包含 `config.json`、tokenizer 文件和权重文件。GPT-OSS 的 `config.json` 示例：

    ```jsonc
    {
      // Hugging Face 加载模型时使用的模型类名称。
      "architectures": [
        "GptOssForCausalLM"
      ],
      // Attention 线性层是否带 bias，attention_dropout 为 attention dropout 概率。
      "attention_bias": true,
      "attention_dropout": 0.0,
      // EOS 和 PAD token id。
      "eos_token_id": 200002,
      "pad_token_id": 199999,
      // 每个 token 路由到的专家数量；num_experts_per_tok 是同义配置。
      "experts_per_token": 4,
      "num_experts_per_tok": 4,
      // 单个 attention head 的维度。
      "head_dim": 64,
      "hidden_act": "silu",
      // 隐藏层维度和 MLP 中间层维度。
      "hidden_size": 2880,
      "intermediate_size": 2880,
      // 原始上下文长度，通常和 RoPE 扩展前长度对应。
      "initial_context_length": 4096,
      "initializer_range": 0.02,
      // 模型最大位置长度。
      "max_position_embeddings": 131072,
      // 模型类型标识，GPT-OSS 使用 gpt_oss。
      "model_type": "gpt_oss",
      // Attention 查询头数和 KV 头数。
      "num_attention_heads": 64,
      "num_key_value_heads": 8,
      // Transformer 层数。
      "num_hidden_layers": 24,
      // 本地专家总数。
      "num_local_experts": 32,
      // 是否输出 router logits，通常训练主链路可保持 false。
      "output_router_logits": false,
      // RMSNorm epsilon。
      "rms_norm_eps": 1e-05,
      // RoPE 扩展配置，示例使用 YaRN。
      "rope_scaling": {
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "factor": 32.0,
        "original_max_position_embeddings": 4096,
        "rope_type": "yarn",
        "truncate": false
      },
      // RoPE 频率基数。
      "rope_theta": 150000,
      // MoE router 辅助损失系数。
      "router_aux_loss_coef": 0.9,
      // 滑窗 attention 的窗口大小。
      "sliding_window": 128,
      // SwiGLU 激活裁剪上限。
      "swiglu_limit": 7.0,
      // 是否共享输入 embedding 和输出 head 权重。
      "tie_word_embeddings": false,
      // 权重默认 dtype，GPT-OSS 示例使用 bfloat16。
      "torch_dtype": "bfloat16",
      // 导出该配置时对应的 Transformers 版本。
      "transformers_version": "4.56.0.dev0",
      // 推理时是否启用 KV cache。
      "use_cache": true,
      // tokenizer 词表大小。
      "vocab_size": 201088
    }
    ```

2. 复用任务 YAML 启动

    原生 Transformers 适配不需要新增模型源码，准备好权重、数据集和 `config.json` 后，即可复用“配置任务脚本和 YAML”章节的任务脚本和 YAML 配置启动训练。MindSpeed LLM 现有 GPT-OSS 配置结构可参考 `examples/fsdp2/gpt_oss/pretrain_gpt_oss_20b_4k_fsdp2_A3.yaml`；如果使用原生 Transformers 路径，重点调整 `model.model_name_or_path`、数据路径、并行切分和训练参数即可。

    如果后续希望使用 NPU 融合算子、专家并行定制或替换 GPT-OSS 的 MoE 计算逻辑，则需要参见[路径二：自定义注册适配 GPT-OSS](#路径二自定义注册适配-gpt-oss)。

### 路径二：自定义注册适配 GPT-OSS

GPT-OSS 在 MindSpeed LLM 中通过自定义注册方式接入。推荐从 Transformers 原生实现出发，将模型文件迁移到 FSDP2 模型目录后再进行二次开发，这样可以尽量保留 Hugging Face 权重命名和模型结构，降低权重加载差异。

1. 放置模型源码

    从 Transformers 原生实现获取 GPT-OSS 模型文件：

    ```text
    transformers/models/gpt_oss/modeling_gpt_oss.py
    transformers/models/gpt_oss/configuration_gpt_oss.py
    ```

    在 MindSpeed LLM 中放到如下目录：

    ```text
    mindspeed_llm/fsdp2/models/
    └── gpt_oss/
        ├── __init__.py
        └── modeling_gpt_oss.py
    ```

    MindSpeed LLM 当前已经存在 `mindspeed_llm/fsdp2/models/gpt_oss/modeling_gpt_oss.py`，可在该文件内继续维护 GPT-OSS 的 FSDP2 适配逻辑。

2. 模块优化

    GPT-OSS 是一个 MoE 模型，其适配重点在 MLP/Expert 层。当前实现保留了原生专家计算路径，同时新增 `GptOssFusedExperts`。当启用 `moe_grouped_gemm` 或使用 `fused` 专家分发时，将调用 NPU grouped GEMM。

    ```python
    class GptOssMLP(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.router = GptOssTopKRouter(config)
            args = get_args()
            if args.moe_grouped_gemm or args.ep_dispatcher == "fused":
                self.experts = GptOssFusedExperts(config)
            else:
                self.experts = GptOssExperts(config)

        def forward(self, hidden_states):
            router_scores, router_indices = self.router(hidden_states)
            routed_out = self.experts(hidden_states, router_indices, router_scores)
            return routed_out, router_scores
    ```

    GPT-OSS 适配时通常需要重点检查以下位置：

    | 适配点 | 说明 |
    | --- | --- |
    | 模型类 | 需要导出 `GptOssForCausalLM`，用于注册到 `ModelRegistry`。 |
    | Attention | 如需 FlashAttention、长序列 CP 或 NPU 融合算子，需要替换对应 `forward`。 |
    | MLP/Expert | MoE 模型通常需要适配专家并行、token dispatch、grouped GEMM。 |
    | Norm/RoPE | 如果要使用融合 RMSNorm、融合 RoPE，需要在模型实现中接入对应算子。 |
    | 权重名 | 保持和 Hugging Face 权重文件中的 key 对齐，避免加载权重时缺失或 unexpected key。 |

3. 注册模型类

    修改 `mindspeed_llm/fsdp2/models/model_registry.py`，导入 GPT-OSS 模型类并加入 `_REGISTRY`。MindSpeed LLM 当前代码示例如下：

    ```python
    class ModelRegistry:
        from mindspeed_llm.fsdp2.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM
        from mindspeed_llm.fsdp2.models.step35.modeling_step3p5 import Step3p5ForCausalLM
        from mindspeed_llm.fsdp2.models.qwen3.qwen3 import Qwen3ForCausalLM
        from mindspeed_llm.fsdp2.models.qwen3.qwen3_moe import Qwen3MoEForCausalLM
        from mindspeed_llm.fsdp2.models.qwen3_next.qwen3_next import Qwen3NextForCausalLM
        from mindspeed_llm.fsdp2.models.mamba3.modeling_mamba3 import Mamba3ForCausalLM
        from mindspeed_llm.fsdp2.models.minimax_m27.modeling_minimax_m2 import MiniMaxM2ForCausalLM

        _REGISTRY = {
            "gpt_oss": GptOssForCausalLM,
            "step35": Step3p5ForCausalLM,
            "qwen3": Qwen3ForCausalLM,
            "qwen3_moe": Qwen3MoEForCausalLM,
            "qwen3_next": Qwen3NextForCausalLM,
            "mamba3": Mamba3ForCausalLM,
            "minimax_m27": MiniMaxM2ForCausalLM,
        }
    ```

4. 更新参数枚举

    `model_id` 的可选值定义在 `mindspeed_llm/fsdp2/utils/arguments.py` 的 `ModelArguments` 中。GPT-OSS 需要包含在 `Literal[...]` 内。MindSpeed LLM 当前代码示例如下：

    ```python
    model_id: Optional[Literal[
        "gpt_oss",
        "qwen3",
        "qwen3_moe",
        "qwen3_next",
        "step35",
        "mamba3",
        "minimax_m27",
    ]] = field(default=None)
    ```

    MindSpeed LLM 当前已经完成 GPT-OSS 的注册和参数枚举配置。

## 配置任务脚本和 YAML 文件

MindSpeed LLM 已经提供了 GPT-OSS FSDP2 预训练示例：

```text
examples/fsdp2/gpt_oss/
├── pretrain_gpt_oss_20b_4k_fsdp2_A3.sh
└── pretrain_gpt_oss_20b_4k_fsdp2_A3.yaml
```

### 修改预训练 YAML 文件

修改 `examples/fsdp2/gpt_oss/pretrain_gpt_oss_20b_4k_fsdp2_A3.yaml` 中的模型和数据路径：

下面示例使用路径二的自定义注册适配，因此需要配置 `model.model_id: gpt_oss`。如果使用路径一的原生 Transformers 适配，不要配置 `model_id`，框架会根据 `config.json` 通过 `AutoModelForCausalLM` 自动加载模型。

```yaml
model:
  model_id: gpt_oss                            # 启用 ModelRegistry 中注册的 GPT-OSS 自定义实现
  model_name_or_path: /home/data/gpt-oss-20b-hf/  # Hugging Face 权重目录
  trust_remote_code: false                     # 本地已有模型实现时通常设为 false
  train_from_scratch: false                    # false 表示加载已有权重
  tokenizer_name_or_path: null                 # tokenizer 路径；null 表示默认使用 model_name_or_path

data:
  dataset:
    file_name: /home/data/alpaca/train-00000-of-00001-a09b74b3ef9c3b56.parquet  # 原始 parquet 数据路径
  template: gpt                              # 数据模板，GPT-OSS 示例使用 gpt
  cutoff_len: 4096                           # 单条样本最大 token 长度
  max_samples: 100000                        # 最多读取样本数，调试时可改小
  overwrite_cache: true                      # 是否覆盖数据缓存
  preprocessing_num_workers: 1               # 数据预处理并发 worker 数
  data_manager_type: mg                      # 预训练示例使用 mg 数据管理

parallel:
  fsdp_size: 16                              # FSDP 切分规模，通常与训练卡数匹配
  fsdp_modules:
    - model.layers.{*}                       # Transformer 层按层 FSDP 包裹
    - model.embed_tokens                     # embedding 层参与 FSDP
    - lm_head                                # 输出头参与 FSDP
  tp_size: 1                                 # 张量并行规模
  ep_size: 4                                 # 专家并行规模
  ep_modules:
    - model.layers.{*}.mlp.experts           # 专家并行作用的专家模块路径
  ep_fsdp_size: 4                            # 专家模块内部 FSDP 切分规模
  ep_fsdp_modules:
    - model.layers.{*}.mlp.experts           # 专家模块内部 FSDP 包裹路径
  ep_dispatcher: eager                       # 专家 token 分发方式
  recompute: true                            # 是否开启重计算以节省显存
  recompute_modules:
    - model.layers.{*}                       # 开启重计算的模块
  cp_size: 1                                 # 上下文并行规模
  cp_type: ulysses                           # 上下文并行类型

training:
  stage: pt                                  # 训练阶段，pt 表示预训练
  per_device_train_batch_size: 1             # 单卡 batch size
  gradient_accumulation_steps: 1             # 梯度累积步数
  dataloader_num_workers: 1                  # dataloader worker 数
  disable_shuffling: 1                       # 是否关闭数据 shuffle
  seed: 42                                   # 随机种子
  output_dir: ./output                       # checkpoint 和输出目录
  optimizer: adamw                           # 优化器类型
  lr: 1.25e-06                               # 学习率
  weight_decay: 0.0                          # 权重衰减
  max_grad_norm: 1.0                         # 梯度裁剪阈值
  lr_scheduler_type: cosine                  # 学习率调度类型
  max_steps: 2000                            # 最大训练步数
  save_steps: 500                            # checkpoint 保存间隔
  logging_steps: 1                           # 日志打印间隔

optimization:
  use_fused_rmsnorm: true                    # 使用融合 RMSNorm
  moe_grouped_gemm: true                     # 使用 grouped GEMM 专家计算路径
  use_fused_rotary_pos_emb: true             # 使用融合 RoPE
```

常见参数可选项说明：

| 配置 | 常见取值 | 说明 |
| --- | --- | --- |
| `model.model_id` | `gpt_oss`、`qwen3`、`qwen3_moe`、`qwen3_next`、`step35`、`mamba3`、`minimax_m27` | 指定自定义注册模型。仅路径二需要配置；使用路径一原生 Transformers 适配时不要配置该项。 |
| `model.model_name_or_path` | 本地 Hugging Face 权重目录 | 必填，指向包含 `config.json`、tokenizer 和权重文件的目录。 |
| `model.trust_remote_code` | `true` / `false` | 模型依赖 Hub 上自定义代码时设为 `true`；使用 MindSpeed LLM 内置模型实现时通常为 `false`。 |
| `model.train_from_scratch` | `true` / `false` | `true` 表示按 config 随机初始化；`false` 表示加载已有权重。 |
| `model.tokenizer_name_or_path` | `null` 或 tokenizer 目录 | `null` 表示默认使用 `model_name_or_path`。 |
| `data.dataset.file_name` | 本地 parquet/json/jsonl 路径 | 原始数据路径；示例使用 Alpaca parquet。 |
| `data.template` | `gpt` 等模板名 | 指定样本拼接模板，GPT-OSS 示例使用 `gpt`。 |
| `data.data_manager_type` | `mg` / `lf` | `mg` 用于 Megatron 风格预训练数据；`lf` 用于 LlamaFactory 风格 SFT 数据。 |
| `training.stage` | `pt` / `sft` | `pt` 表示预训练；`sft` 表示监督微调。 |
| `parallel.fsdp_size` | 正整数 | FSDP 切分规模，通常需要和实际训练卡数、`ep_size` 等并行规模匹配。 |
| `parallel.fsdp_modules` | 模块路径列表 | FSDP 包裹的模块，常见写法为 `model.layers.{*}`、`model.embed_tokens`、`lm_head`。 |
| `parallel.tp_size` | 正整数 | 张量并行规模；无 TP 时设为 `1`。 |
| `parallel.ep_size` | 正整数 | 专家并行规模，MoE 模型按专家数量和设备数配置。 |
| `parallel.ep_modules` | 模块路径列表 | 专家并行作用的专家模块路径，GPT-OSS 为 `model.layers.{*}.mlp.experts`。 |
| `parallel.ep_dispatcher` | `eager` / `fused` / `mc2` | 专家 token 分发方式；`fused` 会触发 fused 专家路径。 |
| `parallel.recompute` | `true` / `false` | 是否开启重计算节省显存。 |
| `parallel.cp_size` | 正整数 | 上下文并行规模；不开启时设为 `1`。 |
| `parallel.cp_type` | `ulysses` / `ring` | 上下文并行类型。 |
| `training.optimizer` | `adamw` / `muon` | 优化器类型。 |
| `training.lr_scheduler_type` | `cosine` / `linear` / `constant` | 学习率调度策略。 |
| `optimization.moe_grouped_gemm` | `true` / `false` | 打开后调用 `GptOssFusedExperts` 中的 grouped GEMM 路径。 |
| `optimization.use_fused_rmsnorm` | `true` / `false` | 是否使用融合 RMSNorm。 |
| `optimization.use_fused_rotary_pos_emb` | `true` / `false` | 是否使用融合 RoPE。 |

### 修改启动脚本

确认 `examples/fsdp2/gpt_oss/pretrain_gpt_oss_20b_4k_fsdp2_A3.sh` 中的机器配置符合实际环境：

```bash
NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6499
NNODES=1
NODE_RANK=0
```

如果是多机训练，需要按实际集群修改 `NNODES`、`NODE_RANK`、`MASTER_ADDR` 和 `MASTER_PORT`。

建议将模型权重路径、数据路径、模型切分和调试步数等随机器或任务变化较多的参数放在 sh 脚本中覆盖 YAML 默认值，避免频繁修改同一个 YAML。点分参数会覆盖 YAML 中的同名字段，例如：

```bash
torchrun $DISTRIBUTED_ARGS train_fsdp2.py \
  examples/fsdp2/gpt_oss/pretrain_gpt_oss_20b_4k_fsdp2_A3.yaml \
  --model.model_name_or_path /home/data/gpt-oss-20b-hf/ \
  --data.dataset.file_name /home/data/alpaca/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
  --parallel.fsdp_size 16 \
  --parallel.ep_size 4 \
  --parallel.ep_fsdp_size 4 \
  --training.output_dir ./output/gpt_oss_20b \
  --training.max_steps 2000 \
  | tee logs/pretrain_gpt_oss_20b_4k_${TIMESTAMP}.log
```

如果使用路径二自定义注册适配，可以在 YAML 中保留 `model.model_id: gpt_oss`，也可以在 sh 中追加 `--model.model_id gpt_oss`。如果使用路径一原生 Transformers 适配，不要在 YAML 或 sh 中配置 `model_id`。

## 启动 GPT-OSS 训练

FSDP2 示例脚本会先加载公共环境变量：

```bash
source examples/fsdp2/env_config.sh
```

其中会设置：

```bash
export TRAINING_BACKEND=mindspeed_fsdp
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

如果你的 Ascend 环境需要手动加载 toolkit，请在启动前执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

启动单机 16 卡预训练：

```bash
bash examples/fsdp2/gpt_oss/pretrain_gpt_oss_20b_4k_fsdp2_A3.sh
```

脚本等价于执行：

```bash
torchrun \
  --nproc_per_node 16 \
  --nnodes 1 \
  --node_rank 0 \
  --master_addr localhost \
  --master_port 6499 \
  train_fsdp2.py \
  examples/fsdp2/gpt_oss/pretrain_gpt_oss_20b_4k_fsdp2_A3.yaml
```

日志会写入 `logs/pretrain_gpt_oss_20b_4k_<timestamp>.log`。

## 适配检查清单

完成 GPT-OSS 适配后，建议按下面顺序检查：

1. `mindspeed_llm/fsdp2/models/gpt_oss/modeling_gpt_oss.py` 中存在 `GptOssForCausalLM`。
2. `mindspeed_llm/fsdp2/models/model_registry.py` 已导入 `GptOssForCausalLM`，并在 `_REGISTRY` 中加入 `"gpt_oss"`。
3. `mindspeed_llm/fsdp2/utils/arguments.py` 的 `ModelArguments.model_id` 枚举包含 `"gpt_oss"`。
4. 选择路径二自定义注册适配时，YAML 中显式设置 `model.model_id: gpt_oss`；选择路径一原生 Transformers 适配时，不要配置 `model_id`。
5. YAML 中 `model.model_name_or_path` 指向本地 Hugging Face 权重目录。
6. MoE 模型正确配置 `parallel.ep_modules`、`parallel.ep_size`、`parallel.ep_fsdp_modules`。
7. 使用融合算子时，确认 `optimization` 中的开关与模型代码路径一致，例如 `moe_grouped_gemm: true` 会触发 `GptOssFusedExperts`。
8. 启动前先用较小的 `max_steps` 和 `max_samples` 验证链路，再放大训练规模。

## 常见问题

- 原生 Transformers 适配和自定义注册适配如何选择？

只验证训练链路、且不需要修改 GPT-OSS 模型源码时，可以使用原生 Transformers 适配。需要 NPU 融合算子、专家并行定制或修改 MoE 计算逻辑时，使用自定义注册适配。

- 为什么设置了 `model_id: gpt_oss` 之后找不到模型？

通常有三个原因：

1. `mindspeed_llm/fsdp2/models/model_registry.py` 没有注册 `"gpt_oss"`。
2. `mindspeed_llm/fsdp2/utils/arguments.py` 的 `Literal[...]` 没有加入 `"gpt_oss"`，参数解析阶段失败。
3. YAML 中的 `model_id` 和 `_REGISTRY` 中的 key 不完全一致。

- 权重加载出现 missing key 或 unexpected key 怎么处理？

优先检查 `mindspeed_llm/fsdp2/models/gpt_oss/modeling_gpt_oss.py` 中的参数名是否与 Hugging Face 权重一致。从原生 `modeling_gpt_oss.py` 二次开发时，建议保持模块层级和参数名稳定，只在需要优化的模块内部替换计算逻辑。

- GPT-OSS 的专家并行模块如何配置？

示例 YAML 中使用：

```yaml
parallel:
  ep_modules:
    - model.layers.{*}.mlp.experts
  ep_fsdp_modules:
    - model.layers.{*}.mlp.experts
```

该路径需要与模型实现中的模块命名一致。GPT-OSS 的 MLP 中专家模块挂在 `self.experts`，所以路径为 `model.layers.{*}.mlp.experts`。
