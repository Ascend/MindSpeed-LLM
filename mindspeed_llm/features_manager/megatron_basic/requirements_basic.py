# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import sys
from argparse import ArgumentParser

import torch
from mindspeed.features_manager.megatron_basic.requirements_basic import RequirementsBasicFeature as MindspeedRequirementsBasicFeature


class RequirementsBasicFeature(MindspeedRequirementsBasicFeature):

    def register_args(self, parser: ArgumentParser):
        super().register_args(parser)
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--o2-optimizer', action='store_true',
                            help='use bf16 exponential moving average to greatly save up memory.')
        group.add_argument('--o2-gradient', action='store_true',
                            help='use bf16 gradient accumulation to greatly save up memory.')
    
    def register_patches(self, patch_manager, args):
        super().register_patches(patch_manager, args)
        self.version_patch(patch_manager, args)

    def pre_register_patches(self, patch_manager, args):
        super().pre_register_patches(patch_manager, args)
        self.load_checkpoint_patch(patch_manager, args)
        
    def optimizer_selection(self, pm, args):
        from mindspeed.core.optimizer.adamw import FusedTorchAdamW, AdamW
        if args.o2_optimizer:
            # O2 optimizer
            from mindspeed_llm.tasks.models.common.adamw import O2AdamW
            pm.register_patch('apex.optimizers.FusedAdam', 
                               O2AdamW, create_dummy=True)
            
        else:
            if args.optimizer_selection == 'fused_torch_adamw':
                pm.register_patch('apex.optimizers.FusedAdam', 
                                   FusedTorchAdamW, create_dummy=True)
            elif args.optimizer_selection == 'fused_adamw':
                pm.register_patch('apex.optimizers.FusedAdam', 
                                   AdamW, create_dummy=True)
            pm.register_patch('apex.optimizers.FusedSGD', 
                               torch.optim.SGD, create_dummy=True)

    def version_patch(self, pm, args):
        from mindspeed_llm.tasks.megatron_basic.requirements_basic import version_wrapper
        pm.register_patch('importlib.metadata.version', version_wrapper)

    def load_checkpoint_patch(self, pm, args):
        if hasattr(args, 'lora_target_modules') and args.lora_target_modules:
            from mindspeed_llm.tasks.megatron_basic.requirements_basic import _load_from_state_dict_wrapper
            pm.register_patch('torch.nn.Module._load_from_state_dict', _load_from_state_dict_wrapper)

