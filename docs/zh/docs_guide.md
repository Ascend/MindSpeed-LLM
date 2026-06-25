# MindSpeed LLM 文档导航

---

## 文档介绍

MindSpeed LLM 文档按照不同的训练框架进行组织，主要包含以下核心目录：

- **pytorch/**：基于 PyTorch 训练框架的文档，主要支持 Mcore和FSDP2 两种训练后端，包含安装指南、模型清单、特性说明、训练方案和工具链等

### 文档目录结构

MindSpeed LLM 文档目录层级介绍如下：

``` shell
docs/zh/

├── introduction.md           # 项目介绍
├── project_guide.md          # 项目导读
├── docs_guide.md             # 文档导航
├── appendixes.md             # 附录文档
├── release_notes_llm.md      # 版本发布说明
├── replace_ascend_path_guide.md  # 昇腾路径替换指南
├── FAQ.md                    # 常见问题
└── pytorch/                  # PyTorch 训练框架相关文档
    ├── develop/              # 开发指南
    │   ├── mcore/            # Mcore 开发指南
    │   │   └── lora_finetune_adaptation.md # LoRA微调迁移开发
    │   ├── fsdp2/            # FSDP2 开发指南
    │   │   └── model_adaptation.md # FSDP2 模型适配
    │   └── precision_issue.md    # 精度问题指南
    ├── features/             # 特性文档
    │   ├── mcore/            # Mcore 特性文档
    │   │   ├── async_activation_offload.md       # 异步激活卸载
    │   │   ├── async_save_torch_dist.md          # 异步保存
    │   │   ├── cc_lora.md                        # CC-LoRA
    │   │   ├── checkpoint_resume.md              # 断点续训
    │   │   ├── chunk_loss.md                     # Chunk Loss
    │   │   ├── communication-over-computation.md  # 通信计算重叠
    │   │   ├── environment_variable.md            # 环境变量
    │   │   ├── fine-tuning-with-context-parallel.md # 上下文并行微调
    │   │   ├── high_availability.md               # 高可用
    │   │   ├── kvallgather-context-parallel.md    # KV AllGather上下文并行
    │   │   ├── layerwise_disaggregated_training.md # 逐层分离训练
    │   │   ├── mamba_context_parallel.md          # Mamba上下文并行
    │   │   ├── mc2.md                            # MC2通信
    │   │   ├── multi-latent-attention.md          # 多潜在注意力
    │   │   ├── o2.md                             # O2优化
    │   │   ├── recompute_relative.md              # 重计算策略
    │   │   ├── ring-attention-context-parallel.md # Ring Attention上下文并行
    │   │   ├── tensor_parallel_2d.md              # 2D张量并行
    │   │   ├── variable_length_flash_attention.md # 变长Flash Attention
    │   │   ├── virtual_pipeline_parallel.md       # 虚拟流水线并行
    │   │   └── yarn.md                           # YaRN上下文扩展
    │   └── fsdp2/            # FSDP2 特性文档
    │       ├── arguments.md            # FSDP2 参数说明
    │       ├── fsdp2_basic_features.md # FSDP2 特性说明
    │       └── quantization.md         # 量化特性
    ├── figures/              # 图片资源
    ├── models/               # PyTorch 框架支持的模型
    │   └── supported_models.md
    ├── training/             # 训练解决方案文档
    │   ├── install_guide.md    # 安装指南
    │   ├── quick_start.md      # 快速入门指南
    │   ├── evaluation/         # 模型评估
    │   │   ├── evaluation_guide.md
    │   │   ├── models_evaluation.md
    │   │   └── evaluation_datasets/  # 评估数据集
    │   ├── finetune/         # 模型微调
    │   │   ├── mcore/        # Mcore 微调方案
    │   │   │   ├── instruction_finetune.md      # 全参微调
    │   │   │   ├── lora_finetune.md             # LoRA微调
    │   │   │   ├── lu_lora_finetune.md          # LU-LoRA微调
    │   │   │   ├── qlora_finetune.md            # QLoRA微调
    │   │   │   ├── multi_sample_pack_finetune.md # 多样本Pack微调
    │   │   │   ├── single_sample_finetune.md    # 单样本微调
    │   │   │   ├── multi_turn_conversation.md   # 多轮对话微调
    │   │   │   ├── offline_dpo.md               # 离线DPO
    │   │   │   ├── layerwise_disaggregated_training.md # 逐层分离训练微调
    │   │   │   └── pmcc_obfuscation.md          # PMCC混淆
    │   │   └── fsdp2/        # FSDP2 微调方案
    │   │       └── finetune_fsdp2.md
    │   ├── inference/        # 模型推理
    │   │   ├── inference.md
    │   │   └── chat.md
    │   └── pretrain/         # 模型预训练
    │       └── mcore/        # Mcore 预训练方案
    │           ├── pretrain.md
    │           ├── pretrain_eod.md
    │           └── train_from_hf.md
    ├── tuning/               # 调优文档
    │   └── performance_tuning.md # 性能调优
    └── tools/                # 工具文档
        ├── data_process_sft_alpaca_style.md   # Alpaca格式数据处理
        ├── data_process_sft_sharegpt_style.md # ShareGPT格式数据处理
        ├── data_process_dpo_pairwise.md       # Pairwise数据处理
        ├── data_process_pretrain.md           # 预训练数据处理
        ├── checkpoint_convert_hf_mcore.md     # 权重转换
        ├── checkpoint_convert_hf_mcore_large_params.md  # 权重转换V2
        ├── checkpoint_convert_hf_dcp.md       # HF-DCP权重转换
        ├── profiling.md                       # 性能分析
        └── deterministic_computation.md       # 确定性计算
```

## 核心文档导航

**快速跳转**：[入门指南](#入门指南) | [Mcore后端](#mcore后端) | [FSDP2后端](#fsdp2后端) | [工具链](#工具链) | [调优指南](#调优指南) | [其他](#其他)

### 入门指南

| 内容 | 说明 |
|------|------|
| [install_guide_pytorch](./pytorch/training/install_guide.md) | 基于PyTorch框架环境安装指导 |
| [quick_start_pytorch](./pytorch/training/quick_start.md) | Mcore后端的快速上手指导，基于PyTorch框架从环境安装到模型预训练和微调 |
| [fsdp2_quick_start](./pytorch/training/fsdp2_quick_start.md) | FSDP2后端的快速上手指导，从环境安装到模型预训练和微调 |
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
| | [pretrain_eod](pytorch/training/pretrain/mcore/pretrain_eod.md) | 多样本Pack预训练方法 |
| 微调 | [instruction_finetune](pytorch/training/finetune/mcore/instruction_finetune.md) | 模型全参微调方案 |
| | [single_sample_finetune](pytorch/training/finetune/mcore/single_sample_finetune.md) | 单样本微调方案 |
| | [multi_sample_pack_finetune](pytorch/training/finetune/mcore/multi_sample_pack_finetune.md) | 多样本Pack微调方案 |
| | [multi_turn_conversation](pytorch/training/finetune/mcore/multi_turn_conversation.md) | 多轮对话微调方案 |
| | [lora_finetune](pytorch/training/finetune/mcore/lora_finetune.md) | 模型LoRA微调方案 |
| | [lu_lora_finetune](pytorch/training/finetune/mcore/lu_lora_finetune.md) | 模型LU-LoRA微调方案 |
| | [qlora_finetune](pytorch/training/finetune/mcore/qlora_finetune.md) | 模型QLoRA微调方案 |
| | [offline_dpo](pytorch/training/finetune/mcore/offline_dpo.md) | 离线DPO对齐方案 |
| | [layerwise_disaggregated_training](pytorch/training/finetune/mcore/layerwise_disaggregated_training.md) | 逐层分离训练微调方案 |
| | [pmcc_obfuscation](pytorch/training/finetune/mcore/pmcc_obfuscation.md) | PMCC混淆方案 |
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
| [quantization](pytorch/features/fsdp2/quantization.md) | FSDP2后端量化特性 |

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
| [data_process_sft_alpaca_style](pytorch/tools/data_process_sft_alpaca_style.md) | 指令微调Alpaca风格数据预处理 |
| [data_process_sft_sharegpt_style](pytorch/tools/data_process_sft_sharegpt_style.md) | 指令微调sharegpt风格数据预处理 |
| [data_process_dpo_pairwise](pytorch/tools/data_process_dpo_pairwise.md) | 偏好对齐pairwise数据处理 |
| [profiling](pytorch/tools/profiling.md) | 基于昇腾芯片采集profiling数据 |
| [deterministic_computation](pytorch/tools/deterministic_computation.md) | 基于昇腾芯片开启确定性计算 |

### 调优指南

| 内容 | 说明 |
|------|------|
| [performance_tuning](pytorch/tuning/performance_tuning.md) | 性能调优指南 |

### 精度定位

| 内容 | 说明 |
|------|------|
| [precision_issue](pytorch/develop/precision_issue.md) | 精度问题排查指南 |

### 其他

| 内容 | 说明 |
|------|------|
| [release_notes_llm](./release_notes_llm.md) | 版本发布说明 |
| [replace_ascend_path_guide](./replace_ascend_path_guide.md) | 昇腾路径替换指南 |
