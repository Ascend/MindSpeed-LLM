# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
import os
import sys

if os.environ.get("MINDSPEED_LLM_VERSION", "012") == "018":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))
from mindspeed_llm import megatron_adaptor  # noqa: F401
from mindspeed_llm.tasks.posttrain.launcher import AutoTrainer
from mindspeed_llm.training.utils import auto_coverage


@auto_coverage
def launch():
    trainer = AutoTrainer()
    trainer.train()


if __name__ == '__main__':
    launch()
