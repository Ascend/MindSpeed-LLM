# MindSpeed LLM Documentation Guide

---

## Documentation Overview

The MindSpeed LLM documentation is organized by training framework and mainly includes the following core directories:

- **pytorch/**: Documentation based on the PyTorch training framework. It mainly supports the MCore and FSDP2 training backends and includes installation guides, model lists, feature descriptions, training solutions, and toolchains.
- **mindspore/**: Documentation based on the MindSpore training framework. It supports only the MCore training backend and provides usage guides and feature descriptions for the MindSpore framework.

### Documentation Directory Structure

The MindSpeed LLM documentation directory hierarchy is shown below:

``` shell
docs/zh/

├── introduction.md         # Project introduction
├── project_guide.md        # Project guide
├── docs_guide.md           # Documentation guide
├── appendixes.md           # Appendix documents
├── pytorch/                # Documentation related to the PyTorch training framework
│   ├── develop/            # Development guide
│   │   ├── mcore/          # MCore development guide
│   │   │   └── lora_finetune_adaptation.md # LoRA fine-tuning migration development
│   │   └── fsdp2/          # FSDP2 development guide
│   │       └── model_adaptation.md # FSDP2 model adaptation
│   ├── features/           # Feature documents
│   │   ├── mcore/          # MCore feature documents
│   │   └── fsdp2/          # FSDP2 feature documents
│   │       ├── arguments.md            # FSDP2 parameter descriptions
│   │       └── fsdp2_basic_features.md # FSDP2 feature descriptions
│   ├── figures/            # Images
│   ├── models/             # Models supported by the PyTorch framework
│   │   └── supported_models.md
│   ├── training/           # Training solution documents
│   │   ├── install_guide.md  # Installation guide
│   │   ├── quick_start.md    # Quick start guide
│   │   ├── evaluation/       # Model evaluation
│   │   │   ├── evaluation_guide.md
│   │   │   ├── models_evaluation.md
│   │   │   └── evaluation_datasets/  # Evaluation datasets
│   │   ├── finetune/       # Model fine-tuning
│   │   │   ├── mcore/      # MCore fine-tuning solutions
│   │   │   └── fsdp2/      # FSDP2 fine-tuning solutions
│   │   │       └── finetune_fsdp2.md
│   │   ├── inference/      # Model inference
│   │   │   ├── inference.md
│   │   │   └── chat.md
│   │   └── pretrain/       # Model pretraining
│   │       └── mcore/      # MCore pretraining solutions
│   │           ├── pretrain.md
│   │           ├── pretrain_eod.md
│   │           └── train_from_hf.md
│   └── tools/              # Tool documents
│       ├── data_process_sft_alpaca_style.md   # Alpaca-style data processing
│       ├── data_process_sft_sharegpt_style.md # ShareGPT-style data processing
│       ├── data_process_dpo_pairwise.md       # Pairwise data processing
│       ├── data_process_pretrain.md           # Pretraining data processing
│       ├── checkpoint_convert_hf_mcore.md     # Weight conversion
│       ├── checkpoint_convert_hf_mcore_large_params.md  # Weight conversion V2
│       ├── checkpoint_convert_hf_dcp.md       # Hugging Face-DCP weight conversion
│       ├── profiling.md                       # Performance analysis
│       └── deterministic_computation.md       # Deterministic computation
└── mindspore/              # Documentation related to the MindSpore training framework
    ├── readme.md           # MindSpore documentation overview
    ├── quick_start.md      # Quick start guide
    ├── install_guide.md    # Installation guide
    ├── features/           # MindSpore feature documents
    └── models/             # Models supported by the MindSpore framework
```

## Core Documentation Navigation

**Quick links:** [Getting Started](#getting-started) | [MCore Backend](#mcore-backend) | [FSDP2 Backend](#fsdp2-backend) | [Toolchain](#toolchain)

### Getting Started

| Content | Description |
|------|------|
| [install_guide_pytorch](./pytorch/training/install_guide.md) | Installation guidance for the PyTorch framework environment. |
| [quick_start_pytorch](./pytorch/training/quick_start.md) | Quick start guidance for the MCore backend, covering the full process from environment setup to model pretraining and fine-tuning on the PyTorch framework. |
| [install_guide_mindspore](./mindspore/install_guide.md) | Installation guidance for the MindSpore framework environment. |
| [quick_start_mindspore](./mindspore/quick_start.md) | Quick start guidance for the MCore backend, covering the full process from environment setup to model pretraining and fine-tuning on the MindSpore framework. |
| [finetune_fsdp2](pytorch/training/finetune/fsdp2/finetune_fsdp2.md) | Quick start guidance for the FSDP2 backend, covering the full process from environment setup to model training. |
| [supported_models](pytorch/models/supported_models.md) | Model support list. |

### MCore Backend

**Features**

| Content | Description |
|------|------|
| [features](pytorch/features/mcore) | A collection of performance optimization and memory optimization features supported by parts of the repository. |

**Development Guide**

| Content | Description |
|------|------|
| [lora_finetune_adaptation](pytorch/develop/mcore/lora_finetune_adaptation.md) | LoRA fine-tuning migration development guide. |

**Training Solutions**

| Category | Content | Description |
|------|------|------|
| Pretraining | [pretrain](pytorch/training/pretrain/mcore/pretrain.md) | Multi-sample pretraining method. |
| | [pretrain_eod](pytorch/training/pretrain/mcore/pretrain_eod.md) | Multi-sample pack pretraining method. |
| Fine-tuning | [instruction_finetune](pytorch/training/finetune/mcore/instruction_finetune.md) | Full-parameter model fine-tuning solution. |
| | [multi_sample_pack_finetune](pytorch/training/finetune/mcore/multi_sample_pack_finetune.md) | Multi-sample pack fine-tuning solution. |
| | [multi_turn_conversation](pytorch/training/finetune/mcore/multi_turn_conversation.md) | Multi-turn conversation fine-tuning solution. |
| | [lora_finetune](pytorch/training/finetune/mcore/lora_finetune.md) | LoRA model fine-tuning solution. |
| | [qlora_finetune](pytorch/training/finetune/mcore/qlora_finetune.md) | QLoRA model fine-tuning solution. |
| Inference | [inference](pytorch/training/inference/inference.md) | Model inference. |
| | [chat](pytorch/training/inference/chat.md) | Chat. |
| | [yarn](pytorch/features/mcore/yarn.md) | Uses the Yarn solution to extend context length and support long-sequence inference. |
| Evaluation | [evaluation_guide](pytorch/training/evaluation/evaluation_guide.md) | Model evaluation solution. |
| | [models_evaluation](pytorch/training/evaluation/models_evaluation.md) | Repository model evaluation list. |
| | [evaluation_datasets](pytorch/training/evaluation/evaluation_datasets) | Evaluation datasets supported by the repository. |

### FSDP2 Backend

**Features**

| Content | Description |
|------|------|
| [fsdp2_basic_features](pytorch/features/fsdp2/fsdp2_basic_features.md) | Introduction to FSDP2 backend features. |
| [arguments](pytorch/features/fsdp2/arguments.md) | Full parameter descriptions for the FSDP2 backend. |

**Development Guide**

| Content | Description |
|------|------|
| [model_adaptation](pytorch/develop/fsdp2/model_adaptation.md) | Model adaptation guide for the FSDP2 backend. |

**Training Solutions**

| Category | Content | Description |
|------|------|------|
| Fine-tuning | [finetune](pytorch/training/finetune/fsdp2/finetune_fsdp2.md) | Full-parameter fine-tuning method. |

### Toolchain

| Content | Description |
|------|------|
| [checkpoint_convert_hf_mcore](pytorch/tools/checkpoint_convert_hf_mcore.md) | Supports two-way weight conversion between Hugging Face and Megatron-core, and supports LoRA weight merging. |
| [checkpoint_convert_hf_mcore_large_params](pytorch/tools/checkpoint_convert_hf_mcore_large_params.md) | Supports weight conversion among different formats such as MCore and Hugging Face for large-parameter models. |
| [checkpoint_convert_hf_dcp](pytorch/tools/checkpoint_convert_hf_dcp.md) | Weight conversion tool between Hugging Face and DCP. |
| [data_process_pretrain](pytorch/tools/data_process_pretrain.md) | Data preprocessing for pretraining tasks. |
| [data_process_sft_alpaca_style](pytorch/tools/data_process_sft_alpaca_style.md) | Alpaca-style data preprocessing for instruction fine-tuning. |
| [data_process_sft_sharegpt_style](pytorch/tools/data_process_sft_sharegpt_style.md) | ShareGPT-style data preprocessing for instruction fine-tuning. |
| [data_process_dpo_pairwise](pytorch/tools/data_process_dpo_pairwise.md) | Pairwise data processing for preference alignment. |
| [profiling](pytorch/tools/profiling.md) | Profiling data collection based on Ascend chips. |
| [deterministic_computation](pytorch/tools/deterministic_computation.md) | Enables deterministic computation based on Ascend chips. |
