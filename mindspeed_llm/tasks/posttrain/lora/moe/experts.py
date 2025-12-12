# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from einops import rearrange

from megatron.core import tensor_parallel
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer.moe.experts import GroupedMLP
from megatron.core.transformer.moe import grouped_gemm_util as gg
from megatron.training import get_args
from megatron.core.jit import jit_fuser
from mindspeed.core.fusions.fused_bias_swiglu import fused_swiglu
from mindspeed.ops.npu_groupmatmul_add import npu_groupmatmul_add_fp32
from mindspeed.core.transformer.moe.moe_feature.overlap.moe_layer_overlap_all2all import gmm_op
from mindspeed.model.transformer import should_recompute_activation
from mindspeed.core.transformer.moe.moe_feature.overlap.comm_utils import async_all_to_all
from mindspeed.core.transformer.moe.moe_feature.overlap.moe_common import (only_recompute_activation, 
                                                                           forward_func, backward_func,
                                                                           get_gemm_backward_need_tensors,
                                                                           set_all2all_experts_output,)
from mindspeed.core.transformer.moe.moe_feature import (permute,
                                                        sort_chunks_by_idxs,
                                                        parallel_state
                                                        )
from mindspeed_llm.tasks.posttrain.lora.cc_lora_forward import dequantize
try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None


class LoraParallelGroupedMlpWithCompAndCommOverlapAll2AllSEQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights1_a, weights1_b, weights2_a, weights2_b, args, moe_layer_ctx):
        weights1, weights2, original_weight1_a, original_weight1_b, \
        original_weight2_a, original_weight2_b, activation_func, \
        permuted_probs, group_list, layer_number, scaling, config = args
        ctx.config = config
        moe_zero_memory = config.moe_zero_memory
        ctx.layer_number = layer_number
        ctx.moe_zero_memory = moe_zero_memory
        ctx.activation_func = activation_func
        use_gmm = (inputs.nelement() != 0)
        ctx.use_gmm = use_gmm
        weights1_a_scaling = weights1_a * scaling
        weights2_a_scaling = weights2_a * scaling
        ctx.scaling = scaling

        ctx.use_gmm = use_gmm
        if use_gmm:
            mm1_out = gmm_op(inputs, weights1, [], group_list, 0)[0]
            mm1_a = gmm_op(inputs, weights1_a_scaling, [], group_list, 0)[0]
            mm1_b = gmm_op(mm1_a, weights1_b, [], group_list, 0)[0]
            mm1_out += mm1_b
        else:
            mm1_out = torch.matmul(inputs, weights1)
            mm1_a = torch.matmul(inputs, weights1_a_scaling)
            mm1_b = torch.matmul(mm1_a, weights1_b)
            mm1_out += mm1_b

        if moe_zero_memory != "disable":
            inputs.untyped_storage().resize_(0)
            mm1_a.untyped_storage().resize_(0)

        def activation_func_with_probs_detach(x, probs):
            dtype = x.dtype
            act_without_probs = activation_func(x)
            fin_res = act_without_probs * (probs.unsqueeze(-1))
            return fin_res.to(dtype), act_without_probs

        (act_out, act_without_probs), detached_act_inputs, permuted_probs_inputs_detach = forward_func(
                                                                                          activation_func_with_probs_detach,
                                                                                          (mm1_out,
                                                                                          permuted_probs))

        is_only_recompute_activation = only_recompute_activation(config, layer_number)
        if moe_zero_memory == "level1" and not is_only_recompute_activation:
            # In zm1, recompute mm1_out and permuted_probs.
            mm1_out.untyped_storage().resize_(0)
            permuted_probs.untyped_storage().resize_(0)
        if use_gmm:
            mm2_out = gmm_op(act_out, weights2, [], group_list, 0)[0]
            mm2_a = gmm_op(act_out, weights2_a_scaling, [], group_list, 0)[0]
            mm2_b = gmm_op(mm2_a, weights2_b, [], group_list, 0)[0]
            mm2_out += mm2_b
        else:
            mm2_out = torch.matmul(act_out, weights2)
            mm2_a = torch.matmul(act_out, weights2_a_scaling)
            mm2_b = torch.matmul(mm2_a, weights2_b)
            mm2_out += mm2_b

        if moe_zero_memory == "level1" and not is_only_recompute_activation:
            act_without_probs.untyped_storage().resize_(0)
            act_out.untyped_storage().resize_(0)
            moe_layer_ctx.recompute_tensors = (inputs, mm1_out, permuted_probs, act_out, act_without_probs)

        is_recompute_activation = moe_zero_memory == "level0" or should_recompute_activation(layer_number) or (
                moe_zero_memory == "level1" and is_only_recompute_activation)
        if is_recompute_activation:
            act_without_probs.untyped_storage().resize_(0)
            act_out.untyped_storage().resize_(0)

        if moe_zero_memory != "level0" and not (moe_zero_memory == "level1" and is_only_recompute_activation):
            ctx.save_for_backward(inputs, mm1_a, permuted_probs_inputs_detach, detached_act_inputs, act_out,
                                  act_without_probs, mm2_a, weights1, weights1_a_scaling, weights1_b, weights2,
                                  weights2_a_scaling, weights2_b, original_weight1_a, original_weight1_b,
                                  original_weight2_a, original_weight2_b, group_list)
        else:
            ctx.save_for_backward(mm1_a, permuted_probs_inputs_detach, detached_act_inputs, act_out, act_without_probs,
                                  mm2_a, weights1, weights1_a_scaling, weights1_b, weights2, weights2_a_scaling, weights2_b,
                                  original_weight1_a, original_weight1_b, original_weight2_a, original_weight2_b, group_list)

        return mm2_out, None

    @staticmethod
    def backward(ctx, *grad_outs):
        grad_outs = grad_outs[0]
        config = ctx.config
        layer_number = ctx.layer_number
        moe_zero_memory = ctx.moe_zero_memory
        is_only_recompute_activation = only_recompute_activation(config, layer_number)
        if moe_zero_memory != "level0" and not (moe_zero_memory == "level1" and is_only_recompute_activation):
            mm1_inputs, mm1_a, permuted_probs_inputs_detach, act_inputs, mm2_inputs, act_without_probs, mm2_a, \
            weights1, weights1_a, weights1_b, weights2, weights2_a, weights2_b, \
            original_weight1_a, original_weight1_b, original_weight2_a, original_weight2_b, \
            group_list = ctx.saved_tensors
        else:
            mm1_a, permuted_probs_inputs_detach, act_inputs, mm2_inputs, act_without_probs, mm2_a, \
            weights1, weights1_a, weights1_b, weights2, weights2_a, weights2_b, \
            original_weight1_a, original_weight1_b, original_weight2_a, original_weight2_b, \
            group_list = ctx.saved_tensors

        ((detach_input, probs, routing_map, num_global_tokens_per_local_expert_cpu, sort_input_by_local_experts),
         permute2_input_detach, permute2_graph,
         permute2_prob_detach, permute2_prob_graph,
         output_splits, input_splits, num_out_tokens) = get_gemm_backward_need_tensors()

        ep_group = parallel_state.get_expert_model_parallel_group()
        if config.moe_tp_extend_ep:
            ep_group = parallel_state.get_expert_tensor_and_model_parallel_group()

        # grad of mm2
        if ctx.use_gmm:
            weights2_tmp, _ = dequantize(weights2, grad_outs.dtype, grad_outs.device)
            weights2 = rearrange(weights2_tmp, 'n h f -> n f h')
            weights2_a = rearrange(weights2_a, 'n h f -> n f h')
            weights2_b = rearrange(weights2_b, 'n h f -> n f h')
            grad_mm2_inputs = gmm_op(grad_outs, weights2, [], group_list, 0)[0]
            grad_mm2_b_inputs = gmm_op(grad_outs, weights2_b, [], group_list, 0)[0]
            grad_mm2_inputs_a = gmm_op(grad_mm2_b_inputs, weights2_a, [], group_list, 0)[0]
        else:
            grad_mm2_inputs = torch.matmul(grad_outs, weights2.t())
            grad_mm2_b_inputs = torch.matmul(grad_outs, weights2_b.t())
            grad_mm2_inputs_a = torch.matmul(grad_mm2_b_inputs, weights2_a.t())

        grad_mm2_inputs += grad_mm2_inputs_a  # add
        act_graph = mm2_inputs
        is_recompute_activation = moe_zero_memory == "level0" or should_recompute_activation(layer_number) or (
                moe_zero_memory == "level1" and is_only_recompute_activation)

        if is_recompute_activation:
            activation_func = ctx.activation_func
            act_without_probs_ = activation_func(act_inputs)
            mm2_inputs = act_without_probs_ * permuted_probs_inputs_detach.unsqueeze(-1)
            act_without_probs_size = act_without_probs_.untyped_storage().size()
            act_without_probs.untyped_storage().resize_(act_without_probs_size)
            act_without_probs.untyped_storage().copy_(act_without_probs_.untyped_storage())
            act_without_probs = None
            act_without_probs_.untyped_storage().resize_(0)
            # add
            if ctx.use_gmm:
                mm2_a = gmm_op(mm2_inputs, weights2_a.transpose(-1, -2), [], group_list, 0)[0]
            else:
                mm2_a = torch.matmul(mm2_inputs, weights2_a)

        if ctx.use_gmm:
            if config.gemm_gradient_accumulation_fusion:
                npu_groupmatmul_add_fp32(mm2_inputs, grad_mm2_b_inputs * ctx.scaling, group_list,
                                         original_weight2_a.main_grad)
                npu_groupmatmul_add_fp32(mm2_a, grad_outs, group_list, original_weight2_b.main_grad)

                if hasattr(original_weight2_a, 'grad_added_to_main_grad'):
                    if getattr(original_weight2_a, 'zero_out_wgrad', False):
                        grad_weights2_a = torch.zeros(
                            weights2_a.transpose(-1, -2).shape,
                            dtype=weights2_a.dtype,
                            device=torch.cuda.current_device(),
                            requires_grad=False,
                        )
                        grad_weights2_b = torch.zeros(
                            weights2_b.transpose(-1, -2).shape,
                            dtype=weights2_a.dtype,
                            device=torch.cuda.current_device(),
                            requires_grad=False,
                        )
                    else:
                        grad_weights2_a = torch.empty(
                            weights2_a.transpose(-1, -2).shape,
                            dtype=weights2_a.dtype,
                            device=torch.cuda.current_device(),
                            requires_grad=False,
                        )
                        grad_weights2_b = torch.empty(
                            weights2_b.transpose(-1, -2).shape,
                            dtype=weights2_a.dtype,
                            device=torch.cuda.current_device(),
                            requires_grad=False,
                        )
                    original_weight2_a.grad_added_to_main_grad = True
                    original_weight2_b.grad_added_to_main_grad = True
                else:
                    grad_weights2_a = None
                    grad_weights2_b = None
            else:
                grad_weights2_a = gmm_op(mm2_inputs.t(), grad_mm2_b_inputs, [], group_list, 2)[0] * ctx.scaling
                grad_weights2_b = gmm_op(mm2_a.t(), grad_outs, [], group_list, 2)[0]
        else:
            grad_weights2_b = torch.matmul(mm2_a.t(), grad_outs)
            grad_weights2_a = torch.matmul(mm2_inputs.t(), grad_mm2_b_inputs)

        # grad of activation_func_with_probs.
        grad_outs.untyped_storage().resize_(0)
        mm2_inputs.untyped_storage().resize_(0)
        # Resize the storage of mm2_a to 0 to release its memory
        mm2_a.untyped_storage().resize_(0)
        act_graph.backward(grad_mm2_inputs)
        permuted_probs_inputs_detach.untyped_storage().resize_(0)
        grad_mm2_inputs.untyped_storage().resize_(0)
        # Resize the storage of grad_mm2_b_inputs to 0 to release its memory
        grad_mm2_b_inputs.untyped_storage().resize_(0)
        act_inputs.untyped_storage().resize_(0)

        if moe_zero_memory == "level0" or (moe_zero_memory == "level1" and is_only_recompute_activation):
            def alltoall_token_permutation1(hidden_states, routing_map, probs=None):
                hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
                permutated_local_input_tokens, permuted_probs_, _ = permute(
                    hidden_states, routing_map, probs, num_out_tokens=num_out_tokens, fused=ctx.config.moe_permute_fusion
                )
                return permutated_local_input_tokens, permuted_probs_

            permutated_local_input_tokens, permuted_probs_ = alltoall_token_permutation1(detach_input, routing_map, probs)

            probs.untyped_storage().resize_(0)

            _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
                permutated_local_input_tokens,
                output_splits,
                input_splits,
                ep_group,
            )
            # recompute global_probs.
            _, global_probs, permuted_probs_inputs_handle = async_all_to_all(
                permuted_probs_,
                output_splits,
                input_splits,
                ep_group
            )

        if not config.moe_permute_fusion:
            # Because the moe_permute_fusion fusion operator needs to save routing_map for backward
            routing_map.untyped_storage().resize_(0)

        if ctx.use_gmm:
            weights1_tmp, _ = dequantize(weights1, act_inputs.dtype, act_inputs.device)
            weights1 = rearrange(weights1_tmp, 'n h f -> n f h')
            weights1_a = rearrange(weights1_a, 'n h f -> n f h')
            weights1_b = rearrange(weights1_b, 'n h f -> n f h')
            mm1_inputs_grad = gmm_op(act_inputs.grad, weights1, [], group_list, 0)[0]
            mm1_b_inputs_grad = gmm_op(act_inputs.grad, weights1_b, [], group_list, 0)[0]
            mm1_inputs_grad += gmm_op(mm1_b_inputs_grad, weights1_a, [], group_list, 0)[0]
        else:
            mm1_inputs_grad = torch.matmul(act_inputs.grad, weights1.t())
            mm1_b_inputs_grad = torch.matmul(act_inputs.grad, weights1_b.t())
            mm1_inputs_grad += torch.matmul(mm1_b_inputs_grad, weights1_a.t())
        probs.untyped_storage().resize_(0)

        # backward for probs. 
        backward_func(permute2_prob_graph, permuted_probs_inputs_detach.grad) 

        _, permute1_prob_backward_input, bw_permute1_prob_all2all_handle = async_all_to_all(
            permute2_prob_detach.grad,
            input_splits,
            output_splits,
            ep_group,
        )

        # backward for expert.
        backward_func(permute2_graph, mm1_inputs_grad)
        mm1_inputs_grad.untyped_storage().resize_(0)

        if moe_zero_memory == "level0" or (moe_zero_memory == "level1" and is_only_recompute_activation):
            permute1_ep_all_to_all_handle.wait()
            permutated_local_input_tokens.untyped_storage().resize_(0)

        _, permute1_backward_input, bw_permute1_ep_all2all_handle = async_all_to_all(
            permute2_input_detach.grad,
            input_splits,
            output_splits,
            ep_group,
        )

        #prepare for permute1 backward. 
        set_all2all_experts_output((permute1_backward_input, bw_permute1_ep_all2all_handle, 
                                    permute1_prob_backward_input, bw_permute1_prob_all2all_handle))

        if moe_zero_memory == "level0" or (moe_zero_memory == "level1" and is_only_recompute_activation):
            permuted_probs_inputs_handle.wait()
            permuted_probs_.untyped_storage().resize_(0)
            mm1_inputs, permuted_probs_inputs_detach = sort_chunks_by_idxs(
                global_input_tokens,
                num_global_tokens_per_local_expert_cpu.ravel(),
                sort_input_by_local_experts,
                probs=global_probs
            )
            global_probs.untyped_storage().resize_(0)
            permuted_probs_inputs_detach.untyped_storage().resize_(0)
            global_input_tokens.untyped_storage().resize_(0)

        if ctx.use_gmm:
            if config.gemm_gradient_accumulation_fusion:
                npu_groupmatmul_add_fp32(mm1_a, act_inputs.grad, group_list, original_weight1_b.main_grad)
                npu_groupmatmul_add_fp32(mm1_inputs, mm1_b_inputs_grad * ctx.scaling, group_list,
                                         original_weight1_a.main_grad)
                if hasattr(original_weight1_b, 'grad_added_to_main_grad'):
                    if getattr(weights1, 'zero_out_wgrad', False):
                        grad_weights1_b = torch.zeros(
                            weights1_b.transpose(-1, -2).shape,
                            dtype=mm1_inputs.dtype,
                            device=torch.cuda.current_device(),
                            requires_grad=False,
                        )
                        grad_weights1_a = torch.zeros(
                            weights1_a.transpose(-1, -2).shape,
                            dtype=mm1_inputs.dtype,
                            device=torch.cuda.current_device(),
                            requires_grad=False,
                        )
                    else:
                        grad_weights1_b = torch.empty(
                            weights1_b.transpose(-1, -2).shape,
                            dtype=mm1_inputs.dtype,
                            device=torch.cuda.current_device(),
                            requires_grad=False,
                        )
                        grad_weights1_a = torch.empty(
                            weights1_a.transpose(-1, -2).shape,
                            dtype=mm1_inputs.dtype,
                            device=torch.cuda.current_device(),
                            requires_grad=False,
                        )
                    original_weight1_b.grad_added_to_main_grad = True
                    original_weight1_a.grad_added_to_main_grad = True
                else:
                    grad_weights1_b = None
                    grad_weights1_a = None
            else:
                grad_weights1_b = gmm_op(mm1_a.t(), act_inputs.grad, [], group_list, 2)[0]
                grad_weights1_a = gmm_op(mm1_inputs.t(), mm1_b_inputs_grad, [], group_list, 2)[0] * ctx.scaling
        else:
            grad_weights1_b = torch.matmul(mm1_a.t(), act_inputs.grad)
            grad_weights1_a = torch.matmul(mm1_inputs.t(), mm1_b_inputs_grad) * ctx.scaling

        act_inputs.grad.untyped_storage().resize_(0)
        # Resize the storage of mm1_b_inputs_grad to 0 to release its memory
        mm1_b_inputs_grad.untyped_storage().resize_(0)
        permuted_probs_inputs_detach.untyped_storage().resize_(0)
        return mm1_inputs_grad, grad_weights1_a, grad_weights1_b, grad_weights2_a, grad_weights2_b, None, None



def lora_parallel_grouped_mlp_with_comp_and_comm_overlap_all2all_seq(inputs, weights1_a, weights1_b,
                                                                     weights2_a, weight2_b, args, ctx):
    return LoraParallelGroupedMlpWithCompAndCommOverlapAll2AllSEQ.apply(inputs, weights1_a, weights1_b,
                                                                        weights2_a, weight2_b, args, ctx)


class LoraParallelGroupedMLP(GroupedMLP):
    def __init__(self, num_local_experts, config, lora_config):
        super().__init__(num_local_experts, config=config)
        self.lora_r = lora_config.r
        self.scaling = lora_config.lora_alpha / self.lora_r

        gg.assert_grouped_gemm_is_available()
        if config.add_bias_linear:
            raise ValueError("bias not supported in Grouped GEMM yet, please set '--disable-bias-linear' instead.")

        self.expert_parallel = config.expert_model_parallel_size > 1
        if self.config.gated_linear_unit:
            if self.config.activation_func not in (F.silu, F.gelu):
                raise ValueError("Activation function must be silu or gelu when using GroupedMLP.")

            self.activation_func = fused_swiglu
        else:
            self.activation_func = self.config.activation_func
        self.activation_recompute = (
            self.config.recompute_granularity == 'selective'
            and "moe_act" in self.config.recompute_modules
        )

        @jit_fuser
        def activation_func_with_probs(x, probs):
            dtype = x.dtype
            res = self.activation_func(x) * probs
            return res.to(dtype)

        self.activation_func_with_probs = activation_func_with_probs

        # How many feature each rank holds for fc1 and fc2, respectively.
        tp_size = parallel_state.get_expert_tensor_parallel_world_size()
        tp_rank = parallel_state.get_expert_tensor_parallel_rank()

        fc1_output_size = self.config.moe_ffn_hidden_size * self.num_local_experts
        if config.gated_linear_unit:
            # Project to 4h. If using swiglu double the output width,
            fc1_output_size *= 2
        fc1_output_size_per_partition = divide(fc1_output_size, tp_size)

        fc2_input_size = self.config.moe_ffn_hidden_size * self.num_local_experts
        fc2_input_size_per_partition = divide(fc2_input_size, tp_size)

        if config.use_cpu_initialization:
            self.weight1_lora_a = Parameter(
                torch.empty(
                    self.config.hidden_size,
                    self.lora_r * self.num_local_experts,
                    dtype=config.params_dtype,
                )
            )
            self.weight1_lora_b = Parameter(
                torch.empty(
                    self.lora_r,
                    fc1_output_size_per_partition,
                    dtype=config.params_dtype,
                )
            )
            self.weight2_lora_a = Parameter(
                torch.empty(
                    fc2_input_size_per_partition,
                    self.lora_r,
                    dtype=config.params_dtype,
                )
            )
            self.weight2_lora_b = Parameter(
                torch.empty(
                    self.lora_r * self.num_local_experts,
                    self.config.hidden_size,
                    dtype=config.params_dtype,
                )
            )
        else:
            self.weight1_lora_a = Parameter(
                torch.empty(
                    self.config.hidden_size,
                    self.lora_r * self.num_local_experts,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            self.weight1_lora_b = Parameter(
                torch.empty(
                    self.lora_r,
                    fc1_output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            self.weight2_lora_a = Parameter(
                torch.empty(
                    fc2_input_size_per_partition,
                    self.lora_r,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            self.weight2_lora_b = Parameter(
                torch.empty(
                    self.lora_r * self.num_local_experts,
                    self.config.hidden_size,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
        self.weight1.requires_grad = False
        self.weight2.requires_grad = False
        
        # initialize LoRA parameters
        # Initialize according to the shape of Linear weights (feature_out, feature_in)
        weight1_a = torch.zeros((self.num_local_experts, self.config.hidden_size, self.lora_r),
                                dtype=self.weight1_lora_a.dtype, device=self.weight1_lora_a.device)
        weight2_a = torch.zeros((self.num_local_experts, fc2_input_size_per_partition // self.num_local_experts,
                                 self.lora_r), dtype=self.weight2_lora_a.dtype, device=self.weight2_lora_a.device)
        for i in range(self.num_local_experts):
            torch.nn.init.kaiming_uniform_(weight1_a[i].t(), a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(weight2_a[i].t(), a=math.sqrt(5))

        self.weight1_lora_a.data = weight1_a.view(self.weight1_lora_a.shape)
        self.weight2_lora_a.data = weight2_a.view(self.weight2_lora_a.shape)

        torch.nn.init.zeros_(self.weight1_lora_b)
        torch.nn.init.zeros_(self.weight2_lora_b)
        # expert lora weight
        setattr(self.weight1_lora_b, 'allreduce', not self.expert_parallel)
        setattr(self.weight2_lora_a, 'allreduce', not self.expert_parallel)


    def forward(self,
                permuted_local_hidden_states: torch.Tensor,
                tokens_per_expert: torch.Tensor,
                permuted_probs: torch.Tensor,
                ctx=None):

        """Forward step of the GroupedMLP."""
        if self.activation_recompute:
            self.activation_checkpoint = tensor_parallel.CheckpointWithoutOutput()

        if self.config.moe_apply_probs_on_input:
            if self.config.moe_router_topk != 1:
                raise ValueError("`moe_apply_probs_on_input` only works with `moe_router_topk`=1.")
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = (
                permuted_probs.unsqueeze(-1) * permuted_local_hidden_states
            )
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            # Probs already applied, so reset to 1.
            permuted_probs = torch.ones_like(permuted_probs)

        if permuted_local_hidden_states.nelement() != 0:
            # Reshape the weights for the grouped GEMMs.
            # input is not empty
            w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
            w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)
            w1_a = self.weight1_lora_a.view(self.num_local_experts, -1, self.lora_r)
            w1_b = self.weight1_lora_b.view(self.num_local_experts, self.lora_r, -1)
            w2_a = self.weight2_lora_a.view(self.num_local_experts, -1, self.lora_r)
            w2_b = self.weight2_lora_b.view(self.num_local_experts, self.lora_r, -1)
            if hasattr(self.weight1, "quant_state"):
                self.weight1.quant_state.shape = (self.num_local_experts, self.config.hidden_size, w1.shape[-1] * 2)
                self.weight2.quant_state.shape = (self.num_local_experts, w2.shape[1] * 2, self.config.hidden_size)
        else:
            # No token is allocated for local experts.
            if torch.count_nonzero(tokens_per_expert) != 0:
                raise ValueError("There are non-zero tokens per expert, which is not expected.")

            # Make sure params of experts still have gradients even given zero tokens.
            # input is not empty
            w1 = self.weight1.view(self.config.hidden_size, -1)
            w2 = self.weight2.view(-1, self.config.hidden_size)
            w1_a = self.weight1_lora_a.view(self.num_local_experts, -1, self.lora_r)[0]
            w1_b = self.weight1_lora_b.view(self.lora_r, -1)
            w2_a = self.weight2_lora_a.view(-1, self.lora_r)
            w2_b = self.weight2_lora_b.view(self.num_local_experts, self.lora_r, -1)[0]
            if hasattr(self.weight1, "quant_state"):
                self.weight1.quant_state.shape = (self.config.hidden_size, w1.shape[-1] * 2)
                self.weight2.quant_state.shape = (w2.shape[0] * 2, self.config.hidden_size)
        if hasattr(self.weight1, "quant_state"):
            w1, w2 = self.weight1, self.weight2

        args = get_args()
        # alltoall-overlap-comm
        if args.moe_alltoall_overlap_comm:
            group_list = torch.cumsum(tokens_per_expert, dim=0)
            return lora_parallel_grouped_mlp_with_comp_and_comm_overlap_all2all_seq(permuted_local_hidden_states,
                                                                                    w1_a, w1_b,
                                                                                    w2_a,w2_b,
                                                                                    (w1, w2,
                                                                                    self.weight1_lora_a,
                                                                                    self.weight1_lora_b,
                                                                                    self.weight2_lora_a,
                                                                                    self.weight2_lora_b,
                                                                                    self.activation_func,
                                                                                    permuted_probs,
                                                                                    group_list,
                                                                                    self.layer_number,
                                                                                    self.scaling,
                                                                                    self.config),
                                                                                    ctx=ctx)
        # origin gemm
        else:
            if permuted_local_hidden_states.nelement() != 0:
                fc1_output = gg.ops.gmm(permuted_local_hidden_states, w1, tokens_per_expert, trans_b=False)
                mm1_a = gg.ops.gmm(permuted_local_hidden_states, w1_a, tokens_per_expert, trans_b=False)
                mm1_b = gg.ops.gmm(mm1_a, w1_b, tokens_per_expert, trans_b=False) * self.scaling

                if self.activation_recompute:
                    intermediate_parallel = self.activation_checkpoint.checkpoint(
                        self.activation_func_with_probs, fc1_output + mm1_b, permuted_probs.unsqueeze(-1)
                    )
                    mm2_a = gg.ops.gmm(intermediate_parallel, w2_a, tokens_per_expert, trans_b=False)
                    mm2_b = gg.ops.gmm(mm2_a, w2_b, tokens_per_expert, trans_b=False) * self.scaling
                    fc2_output = gg.ops.gmm(intermediate_parallel, w2, tokens_per_expert, trans_b=False)
                    self.activation_checkpoint.discard_output_and_register_recompute(fc2_output)
                    fc2_output += mm2_b
                else:
                    intermediate_parallel = self.activation_func_with_probs(
                        fc1_output + mm1_b, permuted_probs.unsqueeze(-1))

                    mm2_a = gg.ops.gmm(intermediate_parallel, w2_a, tokens_per_expert, trans_b=False)
                    mm2_b = gg.ops.gmm(mm2_a, w2_b, tokens_per_expert, trans_b=False) * self.scaling
                    fc2_output = gg.ops.gmm(intermediate_parallel, w2, tokens_per_expert, trans_b=False)
                    fc2_output += mm2_b
                
            else:
                h = torch.matmul(permuted_local_hidden_states, w1)
                mm1_a = torch.matmul(permuted_local_hidden_states, w1_a)
                mm1_b = torch.matmul(mm1_a, w1_b) * self.scaling
                if self.activation_recompute:
                    h = self.activation_checkpoint.checkpoint(
                        self.activation_func_with_probs, h + mm1_b, permuted_probs.unsqueeze(-1)
                    )
                    mm2_a = torch.matmul(h, w2_a)
                    mm2_b = torch.matmul(mm2_a, w2_b) * self.scaling
                    fc2_output = torch.matmul(h, w2)
                    self.activation_checkpoint.discard_output_and_register_recompute(fc2_output)
                    fc2_output = h + mm2_b
                else:
                    h = self.activation_func_with_probs(h + mm1_b, permuted_probs.unsqueeze(-1))
                    mm2_a = torch.matmul(h, w2_a)
                    mm2_b = torch.matmul(mm2_a, w2_b) * self.scaling
                    fc2_output = torch.matmul(h, w2)
                    fc2_output = h + mm2_b

            return fc2_output, None