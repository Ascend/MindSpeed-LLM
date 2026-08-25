# MindSpeed LLM Documentation Guide

---

## Documentation Overview

The MindSpeed LLM documentation is organized by training framework and mainly includes the following core directories:

- **pytorch/**: Documentation based on the PyTorch training framework. It mainly supports the MCore and FSDP2 training backends and includes installation guides, model lists, feature descriptions, training solutions, and toolchains.

### Documentation Directory Structure

The MindSpeed LLM documentation directory hierarchy is shown below:

``` shell
docs/en/

├── introduction.md           # Project introduction
├── project_guide.md          # Project guide
├── docs_guide.md             # Documentation guide
├── appendixes.md             # Appendix documents
├── release_notes_llm.md      # Release notes
├── replace_ascend_path_guide.md  # Ascend path replacement guide
├── FAQ.md                    # Frequently asked questions
└── pytorch/                  # Documentation related to the PyTorch training framework
    ├── develop/              # Development guide
    │   ├── mcore/            # MCore development guide
    │   │   └── lora_finetune_adaptation.md # LoRA fine-tuning migration development
    │   ├── fsdp2/            # FSDP2 development guide
    │   │   └── model_adaptation.md # FSDP2 model adaptation
    │   └── precision_issue.md    # Precision issue guide
    ├── features/             # Feature documents
    │   ├── mcore/            # MCore feature documents
    │   │   ├── async_activation_offload.md       # Async activation offload
    │   │   ├── async_save_torch_dist.md          # Async save
    │   │   ├── cc_lora.md                        # CC-LoRA
    │   │   ├── checkpoint_resume.md              # Checkpoint resume
    │   │   ├── chunk_loss.md                     # Chunk loss
    │   │   ├── communication-over-computation.md  # Communication over computation
    │   │   ├── environment_variable.md            # Environment variables
    │   │   ├── fine-tuning-with-context-parallel.md # Fine-tuning with context parallelism
    │   │   ├── high_availability.md               # High availability
    │   │   ├── kvallgather-context-parallel.md    # KV all-gather context parallelism
    │   │   ├── layerwise_disaggregated_training.md # Layer-wise disaggregated training
    │   │   ├── mamba_context_parallel.md          # Mamba context parallelism
    │   │   ├── mc2.md                            # MC2 communication
    │   │   ├── multi-latent-attention.md          # Multi-latent attention
    │   │   ├── o2.md                             # O2 optimization
    │   │   ├── recompute_relative.md              # Recompute strategy
    │   │   ├── ring-attention-context-parallel.md # Ring attention context parallelism
    │   │   ├── tensor_parallel_2d.md              # 2D tensor parallelism
    │   │   ├── variable_length_flash_attention.md # Variable-length flash attention
    │   │   ├── virtual_pipeline_parallel.md       # Virtual pipeline parallelism
    │   │   └── yarn.md                           # YaRN context extension
    │   └── fsdp2/            # FSDP2 feature documents
    │       ├── arguments.md            # FSDP2 parameters
    │       ├── fsdp2_basic_features.md # FSDP2 feature descriptions
    │       └── quantization.md         # Quantization feature
    ├── figures/              # Images
    ├── models/               # Models supported by the PyTorch framework
    │   └── supported_models.md
    ├── training/             # Training solution documents
    │   ├── install_guide.md    # Installation guide
    │   ├── quick_start.md      # Quick start guide
    │   ├── evaluation/         # Model evaluation
    │   │   ├── evaluation_guide.md
    │   │   ├── models_evaluation.md
    │   │   └── evaluation_datasets/  # Evaluation datasets
    │   ├── finetune/         # Model fine-tuning
    │   │   ├── mcore/        # MCore fine-tuning solutions
    │   │   │   ├── instruction_finetune.md      # Full-parameter fine-tuning
    │   │   │   ├── lora_finetune.md             # LoRA fine-tuning
    │   │   │   ├── lu_lora_finetune.md          # LU-LoRA fine-tuning
    │   │   │   ├── multi_sample_pack_finetune.md # Multi-sample pack fine-tuning
    │   │   │   ├── single_sample_finetune.md    # Single-sample fine-tuning
    │   │   │   ├── multi_turn_conversation.md   # Multi-turn conversation fine-tuning
    │   │   │   ├── offline_dpo.md               # Offline DPO
    │   │   │   ├── layerwise_disaggregated_training.md # Layer-wise disaggregated training fine-tuning
    │   │   │   └── pmcc_obfuscation.md          # PMCC obfuscation
    │   │   └── fsdp2/        # FSDP2 fine-tuning solutions
    │   │       └── finetune_fsdp2.md  # FSDP2 fine-tuning guide
    │   ├── inference/        # Model inference
    │   │   ├── inference.md
    │   │   └── chat.md
    │   └── pretrain/         # Model pretraining
    │       └── mcore/        # MCore pretraining solutions
    │           ├── pretrain.md
    │           ├── pretrain_eod.md
    │           └── train_from_hf.md
    ├── tuning/               # Tuning documents
    │   └── fsdp2_backend_performance_optimization.md # FSDP2 backend model performance optimization
    └── tools/                # Tool documents
        ├── data_process_sft_alpaca_style.md   # Alpaca-style data processing
        ├── data_process_sft_sharegpt_style.md # ShareGPT-style data processing
        ├── data_process_dpo_pairwise.md       # Pairwise data processing
        ├── data_process_pretrain.md           # Pretraining data processing
        ├── checkpoint_convert_hf_mcore_large_params.md  # Weight conversion
        ├── checkpoint_convert_hf_dcp.md       # HF-DCP weight conversion
        ├── profiling.md                       # Performance analysis
        └── deterministic_computation.md       # Deterministic computation
```

## Core Documentation Navigation

**Quick links:** [Getting Started](#getting-started) | [MCore Backend](#mcore-backend) | [FSDP2 Backend](#fsdp2-backend) | [Toolchain](#toolchain) | [Tuning Guide](#tuning-guide) | [Others](#others)

### Getting Started

| Content | Description |
|------|------|
| [install_guide_pytorch](./pytorch/training/install_guide.md) |Installation guidance for the PyTorch framework environment|
| [quick_start_pytorch](./pytorch/training/quick_start.md) |Quick start guidance for the MCore backend, covering the full process from environment setup to model pretraining and fine-tuning on the PyTorch framework|
| [fsdp2_quick_start](./pytorch/training/fsdp2_quick_start.md) |Quick start guidance for the FSDP2 backend, covering the full process from environment setup to model pretraining and fine-tuning|
| [supported_models](pytorch/models/supported_models.md) |Model support list|

### MCore Backend

**Features**

| Content | Description |
|------|------|
| [features](pytorch/features/mcore) |A collection of performance optimization and memory optimization features supported by parts of the repository|

**Development Guide**

| Content | Description |
|------|------|
| [lora_finetune_adaptation](pytorch/develop/mcore/lora_finetune_adaptation.md) |LoRA fine-tuning migration development guide|

**Training Solutions**

| Category | Content | Description |
|------|------|------|
| Pretraining | [pretrain](pytorch/training/pretrain/mcore/pretrain.md) |Multi-sample pretraining method|
| | [pretrain_eod](pytorch/training/pretrain/mcore/pretrain_eod.md) |Multi-sample pack pretraining method|
| Fine-tuning | [instruction_finetune](pytorch/training/finetune/mcore/instruction_finetune.md) |Full-parameter model fine-tuning solution|
| | [single_sample_finetune](pytorch/training/finetune/mcore/single_sample_finetune.md) |Single-sample fine-tuning solution|
| | [multi_sample_pack_finetune](pytorch/training/finetune/mcore/multi_sample_pack_finetune.md) |Multi-sample pack fine-tuning solution|
| | [multi_turn_conversation](pytorch/training/finetune/mcore/multi_turn_conversation.md) |Multi-turn conversation fine-tuning solution|
| | [lora_finetune](pytorch/training/finetune/mcore/lora_finetune.md) |LoRA model fine-tuning solution|
| | [lu_lora_finetune](pytorch/training/finetune/mcore/lu_lora_finetune.md) |LU-LoRA model fine-tuning solution|
| | [offline_dpo](pytorch/training/finetune/mcore/offline_dpo.md) |Offline DPO alignment solution|
| | [layerwise_disaggregated_training](pytorch/training/finetune/mcore/layerwise_disaggregated_training.md) |Layer-wise disaggregated training fine-tuning solution|
| | [pmcc_obfuscation](pytorch/training/finetune/mcore/pmcc_obfuscation.md) |PMCC obfuscation solution|
| Inference | [inference](pytorch/training/inference/inference.md) |Model inference|
| | [chat](pytorch/training/inference/chat.md) |Chat|
| | [YaRN](pytorch/features/mcore/yarn.md) | Uses the YaRN solution to extend context length and support long-sequence inference. |
| Evaluation | [evaluation_guide](pytorch/training/evaluation/evaluation_guide.md) |Model evaluation solution|
| | [models_evaluation](pytorch/training/evaluation/models_evaluation.md) |Repository model evaluation list|
| | [evaluation_datasets](pytorch/training/evaluation/evaluation_datasets) |Evaluation datasets supported by the repository|

### FSDP2 Backend

**Features**

| Content | Description |
|------|------|
| [fsdp2_basic_features](pytorch/features/fsdp2/fsdp2_basic_features.md) |Introduction to FSDP2 backend features|
| [arguments](pytorch/features/fsdp2/arguments.md) |Full parameters for the FSDP2 backend|
| [quantization](pytorch/features/fsdp2/quantization.md) |Quantization feature for the FSDP2 backend|

**Development Guide**

| Content | Description |
|------|------|
| [model_adaptation](pytorch/develop/fsdp2/model_adaptation.md) |Model adaptation guide for the FSDP2 backend|

**Training Solutions**

| Category | Content | Description |
|------|------|------|
| Fine-tuning | [finetune_fsdp2](pytorch/training/finetune/fsdp2/finetune_fsdp2.md) |FSDP2 full-parameter fine-tuning guide (model, dataset, YAML configuration, and parameter description)|

### Toolchain

| Content | Description |
|------|------|
| [checkpoint_convert_hf_mcore_large_params](pytorch/tools/checkpoint_convert_hf_mcore_large_params.md) | Supports weight conversion among different formats such as MCore and Hugging Face for large-parameter models. |
| [checkpoint_convert_hf_dcp](pytorch/tools/checkpoint_convert_hf_dcp.md) |Weight conversion tool between Hugging Face and DCP|
| [data_process_pretrain](pytorch/tools/data_process_pretrain.md) |Data preprocessing for pretraining tasks|
| [data_process_sft_alpaca_style](pytorch/tools/data_process_sft_alpaca_style.md) |Alpaca-style data preprocessing for instruction fine-tuning|
| [data_process_sft_sharegpt_style](pytorch/tools/data_process_sft_sharegpt_style.md) |ShareGPT-style data preprocessing for instruction fine-tuning|
| [data_process_dpo_pairwise](pytorch/tools/data_process_dpo_pairwise.md) |Pairwise data processing for preference alignment|
| [profiling](pytorch/tools/profiling.md) |Profiling data collection based on Ascend chips|
| [deterministic_computation](pytorch/tools/deterministic_computation.md) | Enables deterministic computation based on Ascend chips. |

### Tuning Guide

| Content | Description |
|------|------|
| [fsdp2_backend_performance_optimization](pytorch/tuning/fsdp2_backend_performance_optimization.md) |FSDP2 backend model performance optimization guide|

### Precision Troubleshooting

| Content | Description |
|------|------|
| [precision_issue](pytorch/develop/precision_issue.md) |Precision issue troubleshooting guide|

### Others

| Content | Description |
|------|------|
| [release_notes_llm](./release_notes_llm.md) |Release notes|
| [replace_ascend_path_guide](./replace_ascend_path_guide.md) |Ascend path replacement guide|
