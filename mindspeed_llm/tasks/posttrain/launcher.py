# Copyright 2024 Huawei Technologies Co., Ltd.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

from megatron.training import get_args
from megatron.training.initialize import initialize_megatron
from mindspeed_llm.tasks.posttrain.sft import SFTTrainer, DeepSeek4SFTTrainer
from mindspeed_llm.tasks.posttrain.dpo import DPOTrainer
from mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer import LDTSFTTrainer

logger = logging.getLogger(__name__)


def get_trainer(stage):
    """
    Factory function to select the appropriate trainer based on the 'stage' argument.

    :param stage: A string representing the stage of the training.
    :return: An instance of the appropriate trainer class.
    """
    if stage == "sft":
        if getattr(get_args(), 'layerwise_disaggregated_training', None):
            return LDTSFTTrainer()
        elif getattr(get_args(), 'prompt_type', None) == 'deepseek4':
            return DeepSeek4SFTTrainer()
        else:
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

