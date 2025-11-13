# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import types
from copy import deepcopy
from functools import wraps
import torch
import torch.nn.functional as F
from mindspeed.moe.utils import MoEAuxLossAutoScaler

from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer import build_module
from megatron.core.transformer.mlp import MLPSubmodules, MLP
from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP
from megatron.core.transformer.moe.moe_utils import save_to_aux_losses_tracker
from megatron.core.transformer.moe.legacy_a2a_token_dispatcher import MoEAlltoAllSEQTokenDispatcher
from megatron.training import get_args
from mindspeed.core.transformer.moe.moe_layer_overlap_all2all import MoELayerOverlapAll2All
from mindspeed.core.transformer.moe.moe_layer_overlap_allgather import MoELayerOverlapAllGather
from mindspeed.core.transformer.moe.moe_feature import MoELayer as MegatronMoELayer
from mindspeed_llm.tasks.posttrain.lora.utils import is_enable_lora




def parallel_transformer_layer_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        from mindspeed.core.transformer.moe.moe_feature.overlap.moe_layer import AllGatherOverlapMoeLayer, AlltoAllSeqOverlapMoeLayer
        if self.config.moe_alltoall_overlap_comm or self.config.moe_allgather_overlap_comm:
            if isinstance(self.mlp, (AllGatherOverlapMoeLayer, AlltoAllSeqOverlapMoeLayer)):
                self.mlp.experts.layer_number = self.layer_number
                if self.config.n_shared_experts:
                    self.mlp.shared_experts.layer_number = self.layer_number
            else:
                self.mlp.layer_number = self.layer_number
    return wrapper


def moe_layer_init_wrapper(init_func):
    @wraps(init_func)
    def moe_layer_init(*args, **kwargs):
        moe_config = deepcopy(kwargs["config"])
        global_args = get_args()
        if global_args.moe_intermediate_size:
            moe_config.ffn_hidden_size = global_args.moe_intermediate_size
        kwargs["config"] = moe_config

        init_func(*args, **kwargs)
        self = args[0]
        
        # In 0.10.0, the definition of shared_experts has conflict. Rename the MindSpeed version to 'with_shared_expert'.
        if self.use_shared_expert:
            self.shared_experts.with_shared_expert = True
            
        if moe_config.moe_grouped_gemm:
            if is_enable_lora():
                from peft import LoraConfig
                lora_config = LoraConfig(
                    r=global_args.lora_r,
                    lora_alpha=global_args.lora_alpha,
                    target_modules=global_args.lora_target_modules,
                    lora_dropout=0.0,
                    bias="none",
                    megatron_config=self.config,
                    megatron_core="megatron.core",
                )
                from mindspeed_llm.tasks.posttrain.lora.moe.experts import LoraParallelGroupedMLP
                self.experts = LoraParallelGroupedMLP(self.num_local_experts, moe_config, lora_config)
            else:
                self.experts = GroupedMLP(self.num_local_experts, moe_config)
                
        if global_args.n_shared_experts:
            shared_expert_config = deepcopy(moe_config)
            shared_expert_config.ffn_hidden_size = global_args.n_shared_experts * moe_config.ffn_hidden_size

            if global_args.moe_allgather_overlap_comm or global_args.moe_alltoall_overlap_comm:
                if is_enable_lora():
                    from mindspeed_llm.core.transformer.moe.layers import SEColumnParallelLinear, SERowParallelLinear
                    self.shared_experts = MLP(shared_expert_config, MLPSubmodules(linear_fc1=SEColumnParallelLinear,
                                                                                  linear_fc2=SERowParallelLinear),
                                              shared_expert=True)
                else:
                    from mindspeed.core.transformer.moe.layers import ColumnParallelLinear, RowParallelLinear
                    self.shared_experts = MLP(shared_expert_config, MLPSubmodules(linear_fc1=ColumnParallelLinear,
                                                                                  linear_fc2=RowParallelLinear),
                                              shared_expert=True)
            elif global_args.moe_fb_overlap:
                from mindspeed.core.pipeline_parallel.fb_overlap import ColumnParallelLinear, RowParallelLinear
                self.shared_experts = MLP(shared_expert_config, MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear), shared_expert=True)
            else:
                from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
                self.shared_experts = MLP(shared_expert_config, MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear))
                
            # For using layer_number when recompute activation function is enabled.
            self.shared_experts.layer_number = self.layer_number
            if global_args.shared_expert_gate:
                self.shared_expert_gate = build_module(
                    torch.nn.Linear,
                    shared_expert_config.hidden_size,
                    global_args.shared_expert_gate_output_dimension,
                    bias=False
                )
                
        self.moe_alltoall_overlap_comm = global_args.moe_alltoall_overlap_comm
        self.moe_allgather_overlap_comm = global_args.moe_allgather_overlap_comm
        
        self.moe_adaptive_recompute_activation = global_args.moe_adaptive_recompute_activation
        self.recompute_threshold = 0
        if hasattr(self.config, 'moe_token_dispatcher_type') and self.config.moe_token_dispatcher_type == 'allgather':
            self.moe_adaptive_recompute_activation_scale = global_args.moe_adaptive_recompute_activation_scale
            self.recompute_threshold = parallel_state.get_tensor_model_parallel_world_size() * parallel_state.get_data_parallel_world_size() * \
                                       self.config.moe_router_topk * self.moe_adaptive_recompute_activation_scale / self.config.num_moe_experts
            self.token_dispatcher.all_tokens_per_expert = None
            
    return moe_layer_init


def moe_layer_forward(self, hidden_states: torch.Tensor):
    if (
            self.training
            and self.config.tensor_model_parallel_size > 1
            and not self.config.sequence_parallel
    ):
        raise ValueError(
            "During training, performance may degrade if MoE and tensor parallelism"
            "are enabled without also enabling sequence parallelism."
        )

    # process MoE
    def custom_forward(hidden_states):
        probs, routing_map = self.router(hidden_states)
        (dispatched_input, tokens_per_expert, permuted_probs) = (
            self.token_dispatcher.token_permutation(hidden_states, probs, routing_map)
        )
        expert_output, mlp_bias = self.experts(
            dispatched_input, tokens_per_expert, permuted_probs
        )
        output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)

        args = get_args()
        if args.moe_router_load_balancing_type == "group_limited_greedy":
            # forward only need no loss track
            if hasattr(args, "do_train") and args.do_train:
                save_to_aux_losses_tracker(
                    "load_balancing_loss",
                    self.router.l_aux,
                    self.layer_number,
                    self.config.num_layers,
                )
                save_to_aux_losses_tracker(
                    "load_balancing_expert_level_loss",
                    self.router.l_expert_aux / args.moe_aux_loss_coeff,
                    self.layer_number,
                    self.config.num_layers,
                )
                if hasattr(self.router, 'l_device_aux'):
                    save_to_aux_losses_tracker(
                        "load_balancing_device_level_loss",
                        self.router.l_device_aux / args.moe_device_level_aux_loss_coeff,
                        self.layer_number,
                        self.config.num_layers,
                    )
                if hasattr(self.router, 'l_comm_aux'):
                    save_to_aux_losses_tracker(
                        "load_balancing_comm_level_loss",
                        self.router.l_comm_aux / args.moe_comm_aux_loss_coeff,
                        self.layer_number,
                        self.config.num_layers,
                    )
            output = MoEAuxLossAutoScaler.apply(output, self.router.l_aux)

        if self.use_shared_expert and not self.shared_expert_overlap:
            # if shared_expert_overlap is True, the expert calculation happens in
            # the token_dispatcher to overlap communications and computations
            output = output + self.shared_experts(hidden_states)
        return output, mlp_bias

    if self.moe_layer_recompute:
        output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
    else:
        output, mlp_bias = custom_forward(hidden_states)

    return output, mlp_bias




def lora_moe_layer_init(self, config, submodules=None, layer_number=None):
    """
    "moe-alltoall-overlap-comm" only supported "moe_grouped_gemm".
    """
    self.submodules = submodules
    self.config = config
    super(MegatronMoELayer, self).__init__(config)
    self.moe_layer_recompute = config.moe_layer_recompute
    global_args = get_args()

    if self.config.moe_tp_extend_ep:
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        if self.config.num_moe_experts % (self.expert_parallel_size * tp_size) != 0:
            raise ValueError("ep * tp must be divisible by the number of experts")
        # adjust the local expert split logic
        self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size // tp_size
        local_expert_indices_offset = (
                parallel_state.get_expert_model_parallel_rank() * self.num_local_experts * tp_size +
                parallel_state.get_tensor_model_parallel_rank() * self.num_local_experts
        )
    else:
        if self.config.num_moe_experts % self.expert_parallel_size != 0:
            raise ValueError("ep must be divisible by the number of experts")
        self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
        local_expert_indices_offset = (
                parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
        )

    self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None
    self.shared_expert_overlap = self.config.moe_shared_expert_overlap

    self.local_expert_indices = [
        local_expert_indices_offset + i for i in range(self.num_local_experts)
    ]
    if not all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices)):
        raise ValueError("local_expert_indices must all be <= experts")

    from mindspeed.core.transformer.moe.moe_feature import TopKRouter, build_module
    from mindspeed.core.transformer.moe.moe_feature.adaptor import MindSpeedMOEAlltoAllSeqOverLapDispatcherAdaptor
    
    # Initialize router
    self.router = TopKRouter(config=self.config)
    
    # Initialize experts with LoRA for MOE if grouped GEMM is enabled
    if self.config.moe_grouped_gemm:
        if is_enable_lora():
            from peft import LoraConfig
            lora_config = LoraConfig(
                r=global_args.lora_r,
                lora_alpha=global_args.lora_alpha,
                target_modules=global_args.lora_target_modules,
                lora_dropout=0.0,
                bias="none",
                megatron_config=self.config,
                megatron_core="megatron.core",
            )
            from mindspeed_llm.tasks.posttrain.lora.moe.experts import LoraParallelGroupedMLP
            self.experts = LoraParallelGroupedMLP(self.num_local_experts, self.config, lora_config)
    else:
        self.experts = build_module(self.submodules.experts, self.num_local_experts, self.config)

    # Initialize token dispatcher
    if hasattr(global_args, "moe_alltoall_overlap_comm") and global_args.moe_alltoall_overlap_comm:
        self.token_dispatcher = MindSpeedMOEAlltoAllSeqOverLapDispatcherAdaptor(
            self.num_local_experts, self.local_expert_indices, config=self.config
        )
    else:
        self.token_dispatcher = MoEAlltoAllSEQTokenDispatcher(
            self.num_local_experts, self.local_expert_indices, config=self.config
        )

    # Initialize shared experts if enabled
    if self.use_shared_expert:
        if global_args.moe_alltoall_overlap_comm:
            from mindspeed_llm.core.transformer.moe.layers import SEColumnParallelLinear, SERowParallelLinear
            self.config.with_shared_expert = True
            self.submodules.shared_experts.submodules.linear_fc1 = SEColumnParallelLinear
            self.submodules.shared_experts.submodules.linear_fc2 = SERowParallelLinear

        else:
            from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
            self.submodules.shared_experts.submodules.linear_fc1 = ColumnParallelLinear
            self.submodules.shared_experts.submodules.linear_fc2 = RowParallelLinear

        self.shared_experts = build_module(self.submodules.shared_experts, config=self.config)