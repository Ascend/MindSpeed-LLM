# MindSpeed LLM Project Guide

---

## Project Introduction

The MindSpeed LLM project code follows modular design principles and mainly includes the following core modules:

- **mindspeed_llm/**: The core code directory. It contains the core implementations for model training, feature management, weight conversion, online inference, and the evaluation toolchain.
- **docs/**: The project documentation directory. It provides the project introduction, quick start, installation guide, and feature descriptions.
- **configs/**: The configuration file directory. It provides configuration for weight conversion, evaluation, fine-tuning, FSDP2, and RLHF.
- **examples/**: The model example scripts. They cover multiple training backends and scenarios such as FSDP2, MCore, MindSpore, and RLHF.
- **tests/**: The test case directory. It includes unit tests, system tests, and coverage tests.

### Code Directory Structure

The project root contains multiple key utility scripts, such as weight conversion tools, model evaluation tools, and training processes. They support the full process from data preprocessing to model training, evaluation, and online inference. The project code directory hierarchy is described as follows:

``` shell
MindSpeed-LLM/
 ├── ci                        # CI test scripts.
 ├── configs                   # Configuration file directory.
 │   ├── checkpoint/           # Weight-related configuration.
 │   ├── evaluate/             # Evaluation-related configuration.
 │   ├── finetune/             # Fine-tuning-related configuration.
 │   ├── fsdp2/                # FSDP2-related configuration.
 │   └── rlhf/                 # RLHF-related configuration.
 ├── docs                      # Project documentation directory.
 ├── examples                  # Model example scripts.
 │   ├── fsdp2/                # FSDP2 training backend examples.
 │   ├── mcore/                # MCore training backend examples.
 │   ├── mindspore/            # MindSpore training backend examples.
 │   └── rlhf/                 # RLHF examples.
 ├── mindspeed_llm             # Core code directory.
 │   ├── __init__.py           # Core module package initialization.
 │   ├── core/                 # Core functional modules.
 │   │   ├── context_parallel/         # Context parallelism.
 │   │   ├── datasets/                 # Dataset processing.
 │   │   ├── distributed/              # Distributed training.
 │   │   ├── high_availability/        # High availability.
 │   │   ├── models/                   # Model definitions.
 │   │   │   ├── common/               # Common model components.
 │   │   │   │   ├── embeddings/       # Embedding layers.
 │   │   │   │   └── language_module/  # Language module.
 │   │   │   └── gpt/                  # GPT model implementation.
 │   │   ├── optimizer/           # Optimizers.
 │   │   ├── pipeline_parallel/   # Pipeline parallelism.
 │   │   ├── ssm/                 # State space models.
 │   │   ├── tensor_parallel/     # Tensor parallelism.
 │   │   └── transformer/         # Transformer implementation.
 │   │       ├── custom_layers/   # Custom layers.
 │   │       ├── moe/             # MoE implementation.
 │   │       ├── alibi_attention.py         # ALiBi attention.
 │   │       ├── attention.py               # Attention mechanism.
 │   │       ├── custom_dot_product_attention.py  # Custom dot-product attention.
 │   │       ├── mlp.py                     # Multi-layer perceptron.
 │   │       ├── multi_token_prediction.py  # Multi-token prediction.
 │   │       ├── transformer_block.py       # Transformer block.
 │   │       ├── transformer_config.py      # Transformer configuration.
 │   │       └── transformer_layer.py       # Transformer layer.
 │   │   ├── __init__.py          # Core module initialization.
 │   │   ├── fp8_utils.py         # FP8 utility functions.
 │   │   ├── optimizer_param_scheduler.py  # Optimizer parameter scheduler.
 │   │   ├── parallel_state.py    # Parallel state management.
 │   │   └── timers.py            # Timer utilities.
 │   ├── features_manager/        # Feature manager.
 │   │   ├── __init__.py          # Feature manager initialization.
 │   │   ├── affinity/            # Affinity optimization.
 │   │   ├── ai_framework/        # AI framework support.
 │   │   ├── arguments/           # Argument management.
 │   │   ├── common/              # Common functionality.
 │   │   ├── context_parallel/    # Context parallelism features.
 │   │   ├── convert_checkpoint/  # Weight conversion.
 │   │   ├── dataset/             # Dataset features.
 │   │   ├── dpo/                 # DPO training.
 │   │   ├── evaluation/          # Evaluation features.
 │   │   ├── finetune/            # Fine-tuning features.
 │   │   ├── fsdp2/               # FSDP2 features.
 │   │   ├── functional/          # Functional features.
 │   │   ├── high_availability/   # High-availability features.
 │   │   ├── inference/           # Inference features.
 │   │   ├── low_precision/       # Low-precision training.
 │   │   ├── megatron_basic/      # Megatron basics.
 │   │   ├── memory/              # Memory optimization.
 │   │   ├── models/              # Model features.
 │   │   ├── moe/                 # MoE features.
 │   │   ├── pipeline_parallel/   # Pipeline parallelism.
 │   │   ├── tensor_parallel/     # Tensor parallelism.
 │   │   ├── tokenizer/           # Tokenizer.
 │   │   └── transformer/         # Transformer features.
 │   │       ├── flash_attention/           # Flash Attention.
 │   │       ├── multi_latent_attention/    # Multi-latent attention.
 │   │       ├── qwen3_next_attention/      # Qwen3 Next attention.
 │   │       ├── mtp.py                     # Multi-token prediction.
 │   │       └── transformer_block.py       # Transformer block.
 │   ├── legacy/                # Legacy code module.
 │   │   ├── data/              # Legacy data processing.
 │   │   └── __init__.py        # Legacy module initialization.
 │   ├── mindspore/             # MindSpore backend module.
 │   │   ├── core/              # MindSpore core functionality.
 │   │   ├── tasks/             # MindSpore task module.
 │   │   ├── training/          # MindSpore training module.
 │   │   ├── convert_ckpt.py    # MindSpore weight conversion.
 │   │   ├── convert_ckpt_v2.py # MindSpore weight conversion v2.
 │   │   ├── mindspore_adaptor_v2.py  # MindSpore adapter.
 │   │   └── utils.py           # MindSpore utility functions.
 │   ├── tasks/                 # Task module.
 │   │   ├── checkpoint/        # Checkpoint tasks.
 │   │   ├── common/            # Common tasks.
 │   │   ├── dataset/           # Dataset tasks.
 │   │   ├── evaluation/        # Evaluation tasks.
 │   │   ├── high_availability/ # High-availability tasks.
 │   │   ├── inference/         # Inference tasks.
 │   │   ├── megatron_basic/    # Megatron basic tasks.
 │   │   ├── models/            # Model tasks.
 │   │   ├── posttrain/         # Post-training tasks.
 │   │   ├── preprocess/        # Preprocessing tasks.
 │   │   ├── utils/             # Task utilities.
 │   │   └── __init__.py        # Task module initialization.
 │   ├── training/              # Training module.
 │   │   ├── tokenizer/         # Tokenizer module.
 │   │   ├── __init__.py        # Training module initialization.
 │   │   ├── arguments.py       # Training arguments.
 │   │   ├── checkpointing.py   # Checkpoint management.
 │   │   ├── initialize.py      # Training initialization.
 │   │   ├── one_logger_utils.py  # Logging utilities.
 │   │   ├── training.py        # Main training logic.
 │   │   └── utils.py           # Training utility functions.
 │   └── fsdp2/                # FSDP2-related implementations.
 │       ├── __init__.py         # FSDP2 module initialization.
 │       ├── checkpoint/        # Weight management.
 │       ├── data/              # Data processing.
 │       │   ├── megatron_data/   # Megatron datasets.
 │       │   └── processor/       # Data processors.
 │       ├── distributed/       # Distributed training.
 │       │   ├── context_parallel/   # Context parallelism.
 │       │   ├── expert_parallel/    # Expert parallelism.
 │       │   └── fully_shard/        # Fully sharded.
 │       ├── features/          # FSDP2 features.
 │       ├── models/            # Model implementations.
 │       │   ├── common/          # Common model components.
 │       │   ├── gpt_oss/         # GPT OSS models.
 │       │   ├── qwen3/           # Qwen3 models.
 │       │   ├── qwen3_next/      # Qwen3 Next models.
 │       │   └── step35/          # Step3.5 models.
 │       ├── optim/             # Optimizers.
 │       ├── train/             # Trainers.
 │       └── utils/             # Utility functions.
 ├── tests                     # Test case directory.
 ├── convert_ckpt.py           # Weight conversion tool.
 ├── convert_ckpt_v2.py        # Weight conversion tool v2.
 ├── evaluation.py             # Model evaluation tool.
 ├── inference.py              # Model inference tool.
 ├── posttrain_gpt.py          # Post-training process.
 ├── pretrain_gpt.py          # Pretraining process.
 ├── pretrain_mamba.py         # Pretraining process.
 ├── preprocess_data.py        # Data preprocessing tool.
 ├── preprocess_prompt.py      # Prompt preprocessing tool.
 ├── rlhf_gpt.py               # RLHF training process.
 ├── setup.py                  # Installation configuration file.
 ├── train_fsdp2.py            # FSDP2 training process.
 ├── requirements.txt          # Python dependency file.
 ├── LICENSE                   # License file.
 ├── OWNERS                    # Maintainer list.
 ├── README.md                 # Project documentation.
 ├── SECURITYNOTE.md           # Security note.
 └── Third_Party_Open_Source_Software_Notice  # Third-party open-source software notice.
```

### Model Run Examples

The `examples/` directory contains a rich set of model example scripts. These scripts cover different training frameworks, model architectures, and training scenarios, and help you quickly get started with and use MindSpeed LLM. The example scripts in this directory are classified as follows:

``` shell
examples/
├── fsdp2/                # FSDP2 training backend.
│   ├── gpt_oss/          # GPT OSS model examples.
├── mcore/                # MCore training backend.
│   ├── qwen3/            # Qwen3 model examples, including scripts for pretraining, fine-tuning, and evaluation.
├── mindspore/            # MindSpore training framework.
│   ├── qwen25/           # Qwen2.5 MindSpore examples, including scripts for pretraining, fine-tuning, and evaluation.
└── rlhf/                 # RLHF-related examples, including data preprocessing and training scripts.
```

Each training framework provides complete training scripts for the core models. You can choose the corresponding scripts for training according to your needs:

1. GPT OSS model example for the FSDP2 training backend.

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

5. Qwen2.5 pretraining model example for the MindSpore training framework.

    - Data processing: Run the `data_convert_qwen25_pretrain.sh` script for pretraining data processing.
    - Pretraining: Run the `pretrain_qwen25_7b_32k_ms.sh` script for model pretraining.

All example scripts provide complete command-line parameters and configuration examples. You can modify and extend them according to your needs.

## Summary

MindSpeed LLM provides a complete LLM training solution with the following core features:

- **Multi-framework support**: Supports both the PyTorch training framework, including the MCore and FSDP2 training backends, and the MindSpore training framework.
- **Modular design**: Organizes the code by functional module, which makes maintenance and extension easier.
- **Rich model support**: Covers many model architectures, including Dense, MoE, SSM, and Linear, and supports mainstream open-source LLMs.
- **Complete toolchain**: Provides an end-to-end toolchain from data preprocessing to model training, evaluation, and online inference.
- **Comprehensive documentation system**: Organizes documentation by training framework and functional module, which helps you get started quickly.

With a clear directory structure, detailed documentation, and a rich set of example scripts, the project helps you efficiently train, fine-tune, and deploy LLMs.
