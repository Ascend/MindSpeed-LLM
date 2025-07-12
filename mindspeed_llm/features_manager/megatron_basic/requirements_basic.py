# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import sys
from argparse import ArgumentParser

import torch
from mindspeed.features_manager.megatron_basic.requirements_basic import RequirementsBasicFeature as MindspeedRequirementsBasicFeature


class RequirementsBasicFeature(MindspeedRequirementsBasicFeature):
    
    def register_patches(self, patch_manager, args):
        super().register_patches(patch_manager, args)
        
    def optimizer_selection(self, pm, args):
        from mindspeed.core.optimizer.adamw import FusedTorchAdamW, AdamW
        if args.o2_optimizer:
            # O2 optimizer
            from mindspeed_llm.tasks.models.common.adamw import O2AdamW
            pm.register_patch('apex.optimizers.FusedAdam', O2AdamW, create_dummy=True)
            
        else:
            if args.optimizer_selection == 'fused_torch_adamw':
                pm.register_patch('apex.optimizers.FusedAdam', FusedTorchAdamW, create_dummy=True)
            elif args.optimizer_selection == 'fused_adamw':
                pm.register_patch('apex.optimizers.FusedAdam', AdamW, create_dummy=True)
            pm.register_patch('apex.optimizers.FusedSGD', torch.optim.SGD, create_dummy=True)

