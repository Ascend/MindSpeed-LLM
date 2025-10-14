# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
import logging

from megatron.training import get_args
from megatron.training.initialize import initialize_megatron
from mindspeed_llm.tasks.posttrain.sft import SFTTrainer
from mindspeed_llm.tasks.posttrain.dpo import DPOTrainer

logger = logging.getLogger(__name__)


def get_trainer(stage):
    """
    Factory function to select the appropriate trainer based on the 'stage' argument.

    :param stage: A string representing the stage of the training.
    :return: An instance of the appropriate trainer class.
    """
    if stage == "sft":
        return SFTTrainer()
    elif stage == "dpo":
        return DPOTrainer()
    else:
        logger.info(f'Unknown Stage: {stage}')
        return None


class AutoTrainer:
    """
    AutoTrainer is an automatic trainer selector.
    It chooses the appropriate trainer (e.g., SFTTrainer, DPOTrainer, ORMTrainer...)
    based on the 'stage' argument.
    """

    def __init__(self):
        """
        Initializes the AutoTrainer.

        - Initializes the training system.
        - Retrieves the 'stage' argument.
        - Uses the 'stage' to select the correct trainer.
        """
        initialize_megatron()
        self.args = get_args()
        self.trainer = get_trainer(self.args.stage)

    def train(self):
        """
        Starts the training process by invoking the 'train()' method of the selected trainer.
        """
        self.trainer.train()

