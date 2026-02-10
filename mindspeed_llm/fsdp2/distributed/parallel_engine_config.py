# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from dataclasses import dataclass, field
from typing import Any, Dict, List, Callable, Literal, Union, Optional

import torch


@dataclass
class FSDPPlanConfig:
    ignored_modules: List[str] = None
    apply_modules: Dict[str, Any] = None

    # mp_policy settings
    param_dtype: Optional[torch.dtype] = None
    reduce_dtype: Optional[torch.dtype] = None
    output_dtype: Optional[torch.dtype] = None
    cast_forward_inputs: bool = True

    # prefetch settings
    num_to_forward_prefetch: Optional[int] = 0
    num_to_backward_prefetch: Optional[int] = 0


@dataclass
class TPPlanConfig:
    colwise_parallel: List[str] = None
    rowwise_parallel: List[str] = None
    sequence_parallel: List[str] = None


@dataclass
class CPPlanConfig:
    context_parallel_type: str = None
    is_pack: bool = False


@dataclass
class EPPlanConfig:
    apply_modules: List[str] = None
    dispatcher: Union[Literal["eager", "fused", "mc2"], Callable] = None
    apply_efsdp_modules: List[str] = None
    _gradient_divide_factor: float = None


@dataclass
class QuantizeConfig:
    quant_format: Optional[str] = None
    quant_recipe: Optional[str] = None
    block_size: int = 32
    quant_apply_modules: List[str] = None
    quant_ignored_modules: List[str] = None
    converters: List[str] = None
    extra_args: Dict[str, Any] = field(default_factory=dict)  # for future extensibility


@dataclass
class ParallelEngineConfig:
    data_parallel_size: int = 1

    fully_shard_parallel_size: int = 1
    fsdp_plan: FSDPPlanConfig = None

    tensor_parallel_size: int = 1
    tp_plan: TPPlanConfig = None

    context_parallel_size: int = 1
    context_parallel_type: Literal["ulysses"] = "ulysses"
    cp_plan: CPPlanConfig = None

    expert_parallel_size: int = 1
    expert_fully_shard_parallel_size: int = 1
    expert_data_parallel_size: int = 1
    ep_plan: EPPlanConfig = None

    recompute: bool = False
    recompute_plan: List[str] = None

    quantization_plan: Optional[QuantizeConfig] = None

    def __post_init__(self):
        self.validate_tp_config()
        self.validate_ep_config()
        self.validate_cp_config()
        self.validate_recompute_config()
        self.validate_quantization_config()
        self.validate_fsdp_config()

    def validate_fsdp_config(self):
        ''' fully shard plan
        config = ParallelEngineConfig(
            fsdp_plan=FSDPPlanConfig(
                'ignored_modules':['*mlp.experts*'],
                'apply_modules': {
                    'model.layers.*': {reshard_after_forward=None, shard_placement_fn=None}
                }
            )
        )
        '''
        self.fsdp_plan = FSDPPlanConfig() if self.fsdp_plan is None else self.fsdp_plan
        if self.fully_shard_parallel_size > 1:
            if self.expert_parallel_size > 1:
                self.fsdp_plan.ignored_modules.extend(self.ep_plan.apply_modules)
            if self.tensor_parallel_size > 1:
                self.fsdp_plan.ignored_modules.extend(self.tp_plan.colwise_parallel)
                self.fsdp_plan.ignored_modules.extend(self.tp_plan.rowwise_parallel)
            self.fsdp_plan.ignored_modules = list(set(self.fsdp_plan.ignored_modules))  # remove duplicates

    def validate_tp_config(self):
        ''' tensor parallelize plan

        config = ParallelEngineConfig(
            tp_plan=TPPlanConfig(
                colwise_parallel=['*.q_proj', '*.k_proj', '*.v_proj'],
                rowwise_parallel=['*.o_proj']
            )
        )
        '''
        self.tp_plan = TPPlanConfig() if self.tp_plan is None else self.tp_plan
        self.tp_plan.colwise_parallel = [] if self.tp_plan.colwise_parallel is None else self.tp_plan.colwise_parallel
        self.tp_plan.rowwise_parallel = [] if self.tp_plan.rowwise_parallel is None else self.tp_plan.rowwise_parallel
        self.tp_plan.sequence_parallel = [] if self.tp_plan.sequence_parallel is None else self.tp_plan.sequence_parallel

    def validate_ep_config(self):
        ''' expert parallelize plan

        config = ParallelEngineConfig(
            ep_plan=EPPlanConfig(
                apply_modules: ['*mlp.experts*'],
                dispatcher: 'eager', 'fused', 'mc2'
            )
        )
        '''
        self.ep_plan = EPPlanConfig(apply_modules=[], dispatcher='eager') if self.ep_plan is None else self.ep_plan
        self.ep_plan._gradient_divide_factor = self.expert_parallel_size * self.expert_fully_shard_parallel_size * self.expert_data_parallel_size
        if self.ep_plan.apply_efsdp_modules is None:
            self.ep_plan.apply_efsdp_modules = []
            for ep_module in self.ep_plan.apply_modules:
                if ep_module.endswith('.experts'):
                    self.ep_plan.apply_efsdp_modules.append(ep_module.removesuffix('.experts'))

    def validate_recompute_config(self):
        self.recompute_plan = [] if self.recompute_plan is None else self.recompute_plan

    def validate_cp_config(self):
        if self.context_parallel_type not in ["ulysses"]:
            raise Exception("context parallel type must be `ulysses`.")

    def validate_quantization_config(self):
        self.quantization_plan = QuantizeConfig() if self.quantization_plan is None else self.quantization_plan


