# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from functools import wraps
from dataclasses import make_dataclass, field

import torch.nn.functional as F

from megatron.training import get_args


def transformer_config_post_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self):
        #Reset apply_rope_fusion to bypass Megatron core_r0.12.0 check.
        ori_apply_rope_fusion = self.apply_rope_fusion
        self.apply_rope_fusion = False
        
        _ori_moe_router_pre_softmax = None
        _ori_var_seq = None
        _ori_add_bias_linear = None
        _ori_gated_linear_unit = None
        
        if (
            self.moe_router_topk == 1
            and self.moe_router_score_function == 'softmax'
            and not self.moe_router_pre_softmax
            and self.moe_router_load_balancing_type != 'sinkhorn'
        ):
            _ori_moe_router_pre_softmax = False
            self.moe_router_pre_softmax = True
        if self.moe_token_dispatcher_type is not None and self.variable_seq_lengths:
            _ori_var_seq = self.variable_seq_lengths
            self.variable_seq_lengths = False
        if self.num_moe_experts is not None and self.add_bias_linear:
            _ori_add_bias_linear = self.add_bias_linear
            self.add_bias_linear = False
            
        if (
            self.activation_func == F.gelu
            and not self.gated_linear_unit
            and not self.add_bias_linear
        ):
            _ori_gated_linear_unit = getattr(self, 'gated_linear_unit', False)
            self.gated_linear_unit = True
            
        fn(self)
        
        if _ori_gated_linear_unit is not None:
            self.gated_linear_unit = _ori_gated_linear_unit
            
        if _ori_var_seq is not None:
            self.variable_seq_lengths = _ori_var_seq
            
        if _ori_add_bias_linear is not None:
            self.add_bias_linear = _ori_add_bias_linear
        if _ori_moe_router_pre_softmax is not None:
            self.moe_router_pre_softmax = _ori_moe_router_pre_softmax

        self.apply_rope_fusion = ori_apply_rope_fusion
        del ori_apply_rope_fusion

        args = get_args()
        fields = []
        for key, value in vars(args).items():
            field_name = str(key)
            field_type = type(value)
            if not hasattr(self, key):
                field_def = (field_name, field_type, field(init=False))
                fields.append(field_def)
        self.__class__ = make_dataclass(self.__class__.__name__, fields=fields, bases=(self.__class__,))

        for key, value in vars(args).items():
            if not hasattr(self, key):
                setattr(self, key, value)
    return wrapper