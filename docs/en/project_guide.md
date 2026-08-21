# MindSpeed LLM Project Guide

---

## Project Introduction

The MindSpeed LLM project code follows modular design principles and mainly includes the following core modules:

- `mindspeed_llm/`: The core code directory. It contains the core implementations for model training, feature management, weight conversion, online inference, and the evaluation toolchain.
- `docs/`: The project documentation directory. It provides the project introduction, quick start, installation guide, and feature descriptions.
- `configs/`: The configuration file directory. It provides configuration for weight conversion, evaluation, fine-tuning, FSDP2, and RLHF.
- `examples/`: The model example scripts. They cover multiple training backends and scenarios such as FSDP2, MCore, and RLHF.
- `docker/`: Docker image build configuration. It provides Dockerfiles and image build scripts.
- `pre-commit/`: Code quality hook configuration. It includes spell checking and code standards.
- `tests/`: The test case directory. It includes unit tests, system tests, and coverage tests.

## Code Directory Structure

The modules listed above are organized as directories and files in the project repository. The utility scripts in the root directory (such as weight conversion, model evaluation, and training processes) support the full process from data preprocessing to model training, evaluation, and online inference. The following is an overview of the project directory structure:

``` shell
MindSpeed-LLM/
 ├── ci/                        # CI test scripts
 ├── configs/                   # Configuration file directory
 ├── docker/                    # Docker build configuration
 ├── docs/                      # Project documentation directory
 ├── examples/                  # Model example scripts
 ├── mindspeed_llm/             # Core code directory
 │   ├── core/                  # Core functional modules
 │   ├── features_manager/      # Feature manager
 │   ├── legacy/                # Legacy code module
 │   ├── tasks/                 # Task module
 │   ├── training/              # Training module
 │   └── fsdp2/                 # FSDP2-related implementations
 ├── pre-commit/                # Pre-commit hook configuration
 ├── tests/                     # Test case directory
 ├── convert_ckpt.py            # Weight conversion tool
 ├── convert_ckpt_v2.py         # Weight conversion tool v2
 ├── evaluation.py              # Model evaluation tool
 ├── inference.py               # Model inference tool
 ├── inference_fsdp2.py         # FSDP2 inference tool
 ├── posttrain_gpt.py           # Post-training process
 ├── pretrain_deepseek4.py      # DeepSeek4 pretraining process
 ├── pretrain_gpt.py            # Pretraining process
 ├── pretrain_mamba.py          # Pretraining process
 ├── preprocess_data.py         # Data preprocessing tool
 ├── preprocess_prompt.py       # Prompt preprocessing tool
 ├── rlhf_gpt.py                # RLHF training process
 ├── train_fsdp2.py             # FSDP2 training process
 ├── setup.py                   # Installation configuration file
 ├── requirements.txt           # Python dependency file
 └── ...                        # Other configuration and documentation files
```

<details>
<summary>Click to view the complete directory structure (including subdirectory details)</summary>

``` shell
MindSpeed-LLM/
 ├── ci                        # CI test scripts
 ├── configs                   # Configuration file directory
 │   ├── checkpoint/           # Weight-related configuration
 │   ├── evaluate/             # Evaluation-related configuration
 │   ├── finetune/             # Fine-tuning-related configuration
 │   ├── fsdp2/                # FSDP2-related configuration
 │   └── rlhf/                 # RLHF-related configuration
 ├── docker                    # Docker build configuration
 │   ├── Dockerfile            # Docker image build file
 │   ├── image_build.sh        # Image build script
 │   ├── configure_apt_repo.sh # APT repository configuration script
 │   ├── configure_yum_repo.sh # YUM repository configuration script
 │   ├── OVERVIEW.md           # Docker overview documentation
 │   └── OVERVIEW.zh.md        # Docker overview documentation (Chinese)
 ├── docs                      # Project documentation directory
 ├── examples                  # Model example scripts
 │   ├── fsdp2/                # FSDP2 training backend examples
 │   ├── mcore/                # MCore training backend examples
 │   └── rlhf/                 # RLHF examples
 ├── mindspeed_llm             # Core code directory
 │   ├── core/                 # Core functional modules
 │   │   ├── context_parallel/         # Context parallelism
 │   │   ├── datasets/                 # Dataset processing
 │   │   ├── distributed/              # Distributed training
 │   │   ├── fusions/                  # Fusion operators
 │   │   ├── high_availability/        # High availability
 │   │   ├── layerwise_disaggregated_training/  # Layer-wise disaggregated training
 │   │   ├── models/                   # Model definitions
 │   │   │   ├── common/               # Common model components
 │   │   │   │   ├── embeddings/       # Embedding layers
 │   │   │   │   └── language_module/  # Language module
 │   │   │   └── gpt/                  # GPT model implementation
 │   │   ├── optimizer/           # Optimizers
 │   │   ├── pipeline_parallel/   # Pipeline parallelism
 │   │   │   └── dualpipe/        # DualPipe pipeline parallelism
 │   │   ├── ssm/                 # State space models
 │   │   ├── tensor_parallel/     # Tensor parallelism
 │   │   └── transformer/         # Transformer implementation
 │   │       ├── custom_layers/   # Custom layers
 │   │       └── moe/             # MoE implementation
 │   ├── features_manager/        # Feature manager
 │   │   ├── affinity/            # Affinity optimization
 │   │   ├── ai_framework/        # AI framework support
 │   │   ├── arguments/           # Argument management
 │   │   ├── common/              # Common functionality
 │   │   ├── context_parallel/    # Context parallelism features
 │   │   ├── convert_checkpoint/  # Weight conversion
 │   │   ├── dataset/             # Dataset features
 │   │   ├── dpo/                 # DPO training
 │   │   ├── evaluation/          # Evaluation features
 │   │   ├── finetune/            # Fine-tuning features
 │   │   ├── fsdp2/               # FSDP2 features
 │   │   ├── functional/          # Functional features
 │   │   ├── fusions/             # Fusion operator features
 │   │   ├── high_availability/   # High-availability features
 │   │   ├── inference/           # Inference features
 │   │   ├── layerwise_disaggregated_training/  # Layer-wise disaggregated training features
 │   │   ├── low_precision/       # Low-precision training
 │   │   ├── megatron_basic/      # Megatron basics
 │   │   ├── memory/              # Memory optimization
 │   │   ├── models/              # Model features
 │   │   ├── moe/                 # MoE features
 │   │   ├── optimizer/           # Optimizer features
 │   │   ├── pipeline_parallel/   # Pipeline parallelism
 │   │   ├── qat/                 # Quantization-aware training
 │   │   ├── tensor_parallel/     # Tensor parallelism
 │   │   ├── tokenizer/           # Tokenizer
 │   │   └── transformer/         # Transformer features
 │   │       ├── flash_attention/           # Flash Attention
 │   │       ├── multi_latent_attention/    # Multi-latent attention
 │   │       └── qwen3_next_attention/      # Qwen3 Next attention
 │   ├── legacy/                # Legacy code module
 │   │   └── data/              # Legacy data processing
 │   ├── tasks/                 # Task module
 │   │   ├── checkpoint/        # Checkpoint tasks
 │   │   ├── common/            # Common tasks
 │   │   ├── dataset/           # Dataset tasks
 │   │   ├── evaluation/        # Evaluation tasks
 │   │   ├── high_availability/ # High-availability tasks
 │   │   ├── inference/         # Inference tasks
 │   │   ├── megatron_basic/    # Megatron basic tasks
 │   │   ├── models/            # Model tasks
 │   │   ├── posttrain/         # Post-training tasks
 │   │   ├── preprocess/        # Preprocessing tasks
 │   │   └── utils/             # Task utilities
 │   ├── training/              # Training module
 │   │   └── tokenizer/         # Tokenizer module
 │   └── fsdp2/                # FSDP2-related implementations
 │       ├── checkpoint/        # Weight management
 │       ├── data/              # Data processing
 │       │   ├── megatron_data/   # Megatron datasets
 │       │   └── processor/       # Data processors
 │       ├── distributed/       # Distributed training
 │       │   ├── context_parallel/   # Context parallelism
 │       │   ├── expert_parallel/    # Expert parallelism
 │       │   └── fully_shard/        # Fully sharded
 │       ├── features/          # FSDP2 features
 │       ├── inference/         # Inference module
 │       ├── models/            # Model implementations
 │       │   ├── common/          # Common model components
 │       │   ├── gpt_oss/         # gpt-oss models
 │       │   ├── longcat_flash/   # LongCat Flash models
 │       │   ├── mamba3/          # Mamba3 models
 │       │   ├── minimax_m27/     # MiniMax M2 models
 │       │   ├── qwen3/           # Qwen3 models
 │       │   ├── qwen3_next/      # Qwen3 Next models
 │       │   └── step35/          # Step3.5 models
 │       ├── optim/             # Optimizers
 │       ├── train/             # Trainers
 │       └── utils/             # Utility functions
 ├── pre-commit                 # Pre-commit hook configuration
 │   ├── pyproject.toml         # Pre-commit project configuration
 │   └── typos.toml             # Spell checking configuration
 ├── tests                     # Test case directory
 ├── .clang-format             # Code formatting configuration
 ├── .gitignore                # Git ignore configuration
 ├── .pre-commit-config.yaml   # Pre-commit configuration
 ├── CONTRIBUTING.md            # Contribution guide
 ├── convert_ckpt.py           # Weight conversion tool
 ├── convert_ckpt_v2.py        # Weight conversion tool v2
 ├── evaluation.py             # Model evaluation tool
 ├── inference.py              # Model inference tool
 ├── inference_fsdp2.py        # FSDP2 inference tool
 ├── posttrain_gpt.py          # Post-training process
 ├── pretrain_deepseek4.py     # DeepSeek4 pretraining process
 ├── pretrain_gpt.py           # Pretraining process
 ├── pretrain_mamba.py         # Pretraining process
 ├── preprocess_data.py        # Data preprocessing tool
 ├── preprocess_prompt.py      # Prompt preprocessing tool
 ├── rlhf_gpt.py               # RLHF training process
 ├── setup.py                  # Installation configuration file
 ├── train_fsdp2.py            # FSDP2 training process
 ├── requirements.txt          # Python dependency file
 ├── LICENSE                   # License file
 ├── OWNERS                    # Maintainer list
 ├── README.md                 # Project documentation
 └── Third_Party_Open_Source_Software_Notice  # Third-party open-source software notice
```

</details>

## Core Submodules

The directory tree above shows the overall project structure, where `mindspeed_llm/` is the core code directory containing the majority of the project code. This directory contains five key submodules, each with its own responsibilities, working together to support the complete training process:

- `core/`: Low-level core functionality implementation, including basic capabilities such as parallel strategies (context parallelism, tensor parallelism, pipeline parallelism), model definitions, Transformer components, dataset processing, and high availability.
- `features_manager/`: Feature registration and management module. It injects various training features (such as Flash Attention, MoE routing, quantization-aware training, and so on) into the training process through a patching mechanism, enabling on-demand combination and activation of features.
- `tasks/`: Training task entry points and service logic. It contains implementations of specific tasks such as evaluation, inference, fine-tuning, DPO alignment, and weight conversion. It serves as the execution layer for your training process.
- `training/`: Training framework initialization and main loop. It is responsible for training lifecycle management including parameter parsing, distributed initialization, checkpoint management, and training loop control.
- `fsdp2/`: Independent FSDP2 backend implementation. It includes the complete training pipeline such as model definitions, data processing, distributed strategies, and inference engine, providing parallel support alongside the MCore backend.

In simple terms, `core/` provides basic capabilities, `features_manager/` handles feature injection, `training/` manages the training lifecycle, `tasks/` executes specific service logic, and `fsdp2/` provides an independent complete implementation for the FSDP2 backend. The collaboration relationship between these five submodules is as follows: `core/` → `features_manager/` → `training/` → `tasks/`, while `fsdp2/` exists in parallel as an independent backend.

## Model Run Examples

After understanding the project code structure, you can quickly experience how to use MindSpeed LLM through the example scripts in the `examples/` directory. This directory contains a rich set of model example scripts covering different training frameworks, model architectures, and training scenarios, helping you get started quickly. The example scripts in this directory are classified as follows:

``` shell
examples/
├── fsdp2/                # FSDP2 training backend
│   ├── gpt_oss/          # gpt-oss model examples
├── mcore/                # MCore training backend
│   ├── qwen3/            # Qwen3 model examples, including scripts for pretraining, fine-tuning, and evaluation
└── rlhf/                 # RLHF-related examples, including data preprocessing and training scripts
```

Each training framework provides complete training scripts for the core models. You can choose the corresponding scripts for training according to your needs:

1. gpt-oss model example for the FSDP2 training backend.

    - Fine-tuning: Run the `examples/fsdp2/gpt_oss/tune_gpt_oss_20b_a3b_4K_fsdp2_mindspeed.sh` script for model fine-tuning.
    - Detailed guide: Refer to [finetune_fsdp2.md](pytorch/training/finetune/fsdp2/finetune_fsdp2.md) for complete instructions.

2. Qwen3 8B pretraining example for the MCore training backend.

    - Data processing: Run the `data_convert_qwen3_pretrain.sh` script for pretraining data processing.
    - Pretraining: Run the `pretrain_qwen3_8b_4k.sh` script for model pretraining.
    - Detailed guide: Refer to [pretrain.md](pytorch/training/pretrain/mcore/pretrain.md) for complete instructions.

3. Qwen3 8B fine-tuning example for the MCore training backend.

    - Data processing: Run the `data_convert_qwen3_instruction.sh` script for fine-tuning data processing.
    - Weight conversion: Run the `ckpt_convert_qwen3_hf2mcore.sh` script for weight conversion.
    - Full-parameter fine-tuning: Run the `tune_qwen3_8b_4k_full.sh` script for full-parameter fine-tuning.
    - LoRA fine-tuning: Run the `tune_qwen3_8b_4k_lora.sh` script for LoRA fine-tuning.
    - Detailed guide: Refer to [single_sample_finetune.md](pytorch/training/finetune/mcore/single_sample_finetune.md) for complete instructions.

4. Qwen3 model toolchain example for the MCore training backend.

    - Online inference: Run the `generate_qwen3_8b_ptd.sh` script for model online inference.
    - Evaluation: Run the `evaluate_qwen3_8b.sh` script for model evaluation.

All example scripts provide complete CLI parameters and configuration examples. You can modify and extend them according to your needs. For more detailed training solutions and tool usage instructions, refer to the [MindSpeed LLM Documentation Guide](./docs_guide.md).

## Summary

MindSpeed LLM provides a complete LLM training solution with the following core features:

- **Multi-framework support**: Supports both the MCore and FSDP2 training backends based on PyTorch.
- **Modular design**: Organizes the code by functional module, which makes maintenance and extension easier.
- **Rich model support**: Covers many model architectures, including Dense, MoE, SSM, and Linear, and supports mainstream open-source LLMs.
- **Complete toolchain**: Provides an end-to-end toolchain from data preprocessing to model training, evaluation, and online inference.
- **Comprehensive documentation system**: Organizes documentation by training framework and functional module, which helps you get started quickly.

With a clear directory structure, detailed documentation, and a rich set of example scripts, the project helps you efficiently train, fine-tune, and deploy LLMs.
