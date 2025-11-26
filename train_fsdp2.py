# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
from mindspeed_llm.fsdp2.auto_trainer import AutoTrainer

def launch():
    """
    Main entry point for the FSDP2 training workflow.

    This function utilizes the `AutoTrainer` factory pattern to dynamically
    select and instantiate the appropriate training strategy based on the
    provided command-line arguments.

    Supported Training Modes managed by AutoTrainer:
    ------------------------------------------------
    1. Pretraining (FSDP2PretrainTrainer):
       - Triggered by default.
       - Uses standard Megatron data loading for large-scale corpus pretraining.

    2. Finetuning (FSDP2SFTTrainer):
       - Triggered by the flag: `--stage sft`.
       - Specialized for Instruction Tuning (SFT).
       - Handles variable sequence lengths and specific instruction dataset formatting.

    3. Future Extensions (Extensible via AutoTrainer):
       - The structure supports easy integration of other paradigms such as
         DPO (Direct Preference Optimization) or RM (Reward Modeling)
         without modifying this entry script.
    """

    # Initialize the AutoTrainer.
    # Internally, this will:
    #   1. Call `initialize_megatron()` to parse arguments and setup distributed backends.
    #   2. Inspect arguments (e.g., `args.stage`).
    #   3. Instantiate the specific trainer class (Pretrain vs. Finetune).
    trainer = AutoTrainer()

    # Execute the training loop.
    trainer.train()


if __name__ == '__main__':
    launch()