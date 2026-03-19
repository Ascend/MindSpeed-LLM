# MindSpeed LLM 文档导读

---

## 文档介绍

MindSpeed LLM 文档按照不同的训练框架进行组织，主要包含以下核心目录：

- **pytorch/**：基于 PyTorch 训练框架的文档，主要支持 Mcore和FSDP2 两种训练后端，包含安装指南、模型清单、特性说明、训练方案和工具链等
- **mindspore/**：基于 MindSpore 训练框架的文档，仅支持Mcore训练后端，提供 MindSpore 框架下的使用指南和特性说明

### 文档目录结构

MindSpeed LLM 文档目录层级介绍如下：

``` shell
docs/zh/

├── introduction.md         # 项目介绍
├── project_guide.md        # 项目导读
├── docs_guide.md           # 文档导读
├── install_guide.md        # 安装指南
├── appendixes.md           # 附录文档
├── pytorch/                # PyTorch 训练框架相关文档
│   ├── quick_start.md      # 快速入门指南
│   ├── develop/            # 开发指南
│   │   ├── mcore/          # Mcore 开发指南
│   │   │   └── lora_finetune_adaptation.md # LoRA微调迁移开发
│   │   └── fsdp2/          # FSDP2 开发指南
│   │       └── model_adaptation.md # FSDP2 模型适配
│   ├── features/           # 特性文档
│   │   ├── mcore/          # Mcore 特性文档
│   │   └── fsdp2/          # FSDP2 特性文档
│   │       ├── arguments.md           # FSDP2 参数说明
│   │       └── fsdp2_basic_features.md # FSDP2 特性说明
│   ├── figures/            # 图片资源
│   ├── models/             # PyTorch 框架支持的模型
│   │   └── supported_models.md
│   ├── training/           # 训练解决方案文档
│   │   ├── evaluation/     # 模型评估
│   │   │   ├── evaluation_guide.md
│   │   │   ├── models_evaluation.md
│   │   │   └── evaluation_datasets/  # 评估数据集
│   │   ├── finetune/       # 模型微调
│   │   │   ├── mcore/      # Mcore 微调方案
│   │   │   └── fsdp2/      # FSDP2 微调方案
│   │   │       └── finetune_fsdp2.md
│   │   ├── inference/      # 模型推理
│   │   │   ├── inference.md
│   │   │   └── chat.md
│   │   └── pretrain/       # 模型预训练
│   │       └── mcore/      # Mcore 预训练方案
│   │           ├── pretrain.md
│   │           ├── pretrain_eod.md
│   │           └── train_from_hf.md
│   └── tools/              # 工具文档
│       ├── data_process_sft_alpaca_style.md   # Alpaca格式数据处理
│       ├── data_process_sft_sharegpt_style.md # ShareGPT格式数据处理
│       ├── data_process_dpo_pairwise.md       # Pairwise数据处理
│       ├── data_process_pretrain.md           # 预训练数据处理
│       ├── checkpoint_convert_hf_mcore.md     # 权重转换
│       ├── checkpoint_convert_hf_mcore_large_params.md  # 权重转换V2
│       ├── checkpoint_convert_hf_dcp.md       # HF-DCP权重转换
│       ├── profiling.md                       # 性能分析
│       └── deterministic_computation.md       # 确定性计算
└── mindspore/              # MindSpore 训练框架相关文档
    ├── readme.md           # MindSpore 文档说明
    ├── quick_start.md      # 快速入门指南
    ├── features/           # MindSpore 特性文档
    └── models/             # MindSpore 框架支持的模型
```

## 核心文档导航

**快速跳转**：[入门指南](#入门指南) | [Mcore后端](#mcore后端) | [FSDP2后端](#fsdp2后端) | [工具链](#工具链)

### 入门指南

| 内容 | 说明 |
|------|------|
| [install_guide](install_guide.md) | 环境安装指导 |
| [quick_start_pytorch](./pytorch/training/quick_start.md) | Mcore后端的快速上手指导，基于PyTorch框架从环境安装到模型预训练和微调 |
| [quick_start_mindspore](./mindspore/quick_start.md) | Mcore后端的快速上手指导，基于MindSpore框架从环境安装到模型预训练和微调 |
| [finetune_fsdp2](pytorch/training/finetune/fsdp2/finetune_fsdp2.md) | FSDP2后端的快速上手指导，从环境安装到模型训练 |
| [supported_models](pytorch/models/supported_models.md) | 模型支持列表 |

### Mcore后端

**特性**

| 内容 | 说明 |
|------|------|
| [features](pytorch/features/mcore) | 收集了部分仓库支持的性能优化和显存优化的特性 |

**开发指南**

| 内容 | 说明 |
|------|------|
| [lora_finetune_adaptation](pytorch/develop/mcore/lora_finetune_adaptation.md) | LoRA微调迁移开发指南 |

**训练方案**

| 分类 | 内容 | 说明 |
|------|------|------|
| 预训练 | [pretrain](pytorch/training/pretrain/mcore/pretrain.md) | 多样本预训练方法 |
| | [pretrain_eod](pytorch/training/pretrain/mcore/pretrain_eod.md) | 多样本pack预训练方法 |
| 微调 | [instruction_finetune](pytorch/training/finetune/mcore/instruction_finetune.md) | 模型全参微调方案 |
| | [multi_sample_pack_finetune](pytorch/training/finetune/mcore/multi_sample_pack_finetune.md) | 多样本Pack微调方案 |
| | [multi_turn_conversation](pytorch/training/finetune/mcore/multi_turn_conversation.md) | 多轮对话微调方案 |
| | [lora_finetune](pytorch/training/finetune/mcore/lora_finetune.md) | 模型lora微调方案 |
| | [qlora_finetune](pytorch/training/finetune/mcore/qlora_finetune.md) | 模型qlora微调方案 |
| 推理 | [inference](pytorch/training/inference/inference.md) | 模型推理 |
| | [chat](pytorch/training/inference/chat.md) | 对话 |
| | [yarn](pytorch/features/mcore/yarn.md) | 使用yarn方案来扩展上下文长度，支持长序列推理 |
| 评估 | [evaluation_guide](pytorch/training/evaluation/evaluation_guide.md) | 模型评估方案 |
| | [models_evaluation](pytorch/training/evaluation/models_evaluation.md) | 仓库模型评估清单 |
| | [evaluation_datasets](pytorch/training/evaluation/evaluation_datasets) | 仓库支持评估数据集 |

### FSDP2后端

**特性**

| 内容 | 说明 |
|------|------|
| [fsdp2_basic_features](pytorch/features/fsdp2/fsdp2_basic_features.md) | FSDP2后端特性介绍 |
| [arguments](pytorch/features/fsdp2/arguments.md) | FSDP2后端全量参数说明 |

**开发指南**

| 内容 | 说明 |
|------|------|
| [model_adaptation](pytorch/develop/fsdp2/model_adaptation.md) | FSDP2后端模型适配指南 |

**训练方案**

| 分类 | 内容 | 说明 |
|------|------|------|
| 微调 | [finetune](pytorch/training/finetune/fsdp2/finetune_fsdp2.md) | 全参微调方法 |

### 工具链

| 内容 | 说明 |
|------|------|
| [checkpoint_convert_hf_mcore](pytorch/tools/checkpoint_convert_hf_mcore.md) | 支持Huggingface、Megatron-core两种格式的权重互转，支持LoRA权重合并 |
| [checkpoint_convert_hf_mcore_large_params](pytorch/tools/checkpoint_convert_hf_mcore_large_params.md) | 支持大参数模型mcore、hf等各种不同格式权重间的转换 |
| [checkpoint_convert_hf_dcp](pytorch/tools/checkpoint_convert_hf_dcp.md) | HF和DCP之间的权重转换工具 |
| [data_process_pretrain](pytorch/tools/data_process_pretrain.md) | 预训练任务的数据预处理 |
| [data_process_sft_alpaca_style](pytorch/tools/data_process_sft_alpaca_style.md) | 指令微调alpaca风格数据预处理 |
| [data_process_sft_sharegpt_style](pytorch/tools/data_process_sft_sharegpt_style.md) | 指令微调sharegpt风格数据预处理 |
| [data_process_dpo_pairwise](pytorch/tools/data_process_dpo_pairwise.md) | 偏好对齐pairwise数据处理 |
| [profiling](pytorch/tools/profiling.md) | 基于昇腾芯片采集profiling数据 |
| [deterministic_computation](pytorch/tools/deterministic_computation.md) | 基于昇腾芯片开启确定性计算 |
