import logging

import torch
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import Shard

from mindspeed_llm.fsdp2.distributed.parallel_engine_config import EPPlanConfig
from mindspeed.fsdp.utils.log import print_rank
from mindspeed.fsdp.utils.str_match import module_name_match

logger = logging.getLogger(__name__)


def expert_fully_shard_modules(model: torch.nn.Module, efsdp_mesh, plan: EPPlanConfig) -> torch.nn.Module:
    efsdp_modules = get_efsdp_modules(model, plan)
    config = {'mesh': efsdp_mesh,
              'mp_policy': MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
              'shard_placement_fn': lambda x: Shard(1)}

    for experts in efsdp_modules:
        if isinstance(experts, torch.nn.ModuleList):
            for expert in experts:
                fully_shard(expert, **config)
                set_gradient_divide_factor(expert, plan._gradient_divide_factor)
        else:
            fully_shard(experts, **config)
            set_gradient_divide_factor(experts, plan._gradient_divide_factor)

    return model


def get_efsdp_modules(modules: torch.nn.Module, plan: EPPlanConfig):
    efsdp_modules = []
    for plan_name in plan.apply_efsdp_modules:
        for name, module in modules.named_modules():
            if module_name_match(plan_name, name):
                print_rank(logger.debug, f'[Expert Fully Shard]: Apply efsdp to module <{name}>')
                efsdp_modules.append(module)
    if len(efsdp_modules) == 0:
        raise RuntimeError(f'[Expert Fully Shard] No module named {plan} or not be ModuleList')
    return efsdp_modules


def set_gradient_divide_factor(module, factor):
    if hasattr(module, 'set_gradient_divide_factor'):
        module.set_gradient_divide_factor(factor)
    else:
        module.set_reduce_scatter_divide_factor(factor)
