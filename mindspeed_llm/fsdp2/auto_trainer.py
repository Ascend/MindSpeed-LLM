# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
from megatron.training import get_args, print_rank_0
from megatron.training.initialize import initialize_megatron

from mindspeed_llm.fsdp2.trainer.pretrain_trainer import FSDP2PretrainTrainer
from mindspeed_llm.fsdp2.trainer.sft_trainer import FSDP2SFTTrainer

# Dependency to be injected
from mindspeed_llm.fsdp2.model_factory import FSDP2ModelFactory


class AutoTrainer:
    """
    AutoTrainer: A factory class used to automatically manage environment
    initialization and Trainer selection.

    Act as the Composition Root for Dependency Injection.
    """

    def __init__(self):
        # 1. Centralize Megatron environment initialization here.
        initialize_megatron()

        # 2. Retrieve arguments for logic determination.
        self.args = get_args()

        # 3. Instantiate the specific Trainer.
        self.trainer = self._build_trainer()

    def train(self):
        """
        Proxy method that delegates the call to the specific Trainer's train function.
        """
        if self.trainer:
            self.trainer.train()
        else:
            raise RuntimeError("Failed to initialize a valid trainer.")

    def _build_trainer(self):
        """
        Dispatch to the specific Trainer implementation based on arguments.
        Injects the FSDP2ModelFactory.create method as the model_builder dependency.
        """
        # Define the dependency
        model_builder = FSDP2ModelFactory.create

        if self.args.stage == "sft":
            print_rank_0(">>> [AutoTrainer] Mode: Finetuning")
            # Inject dependency into SFT Trainer
            return FSDP2SFTTrainer(model_builder=model_builder)
        else:
            print_rank_0(">>> [AutoTrainer] Mode: Pretraining")
            # Inject dependency into Pretrain Trainer
            return FSDP2PretrainTrainer(model_builder=model_builder)