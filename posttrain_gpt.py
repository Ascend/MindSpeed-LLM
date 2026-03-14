# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
from mindspeed_llm import megatron_adaptor
from mindspeed_llm.tasks.posttrain.launcher import AutoTrainer
from mindspeed_llm.training.utils import auto_coverage


@auto_coverage
def launch():
    trainer = AutoTrainer()
    trainer.train()


if __name__ == '__main__':
    launch()
