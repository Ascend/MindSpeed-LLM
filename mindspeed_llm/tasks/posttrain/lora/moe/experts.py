# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import math
import torch

from torch.nn.parameter import Parameter
from einops import rearrange

from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
)
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer.moe.moe_utils import permute
from megatron.core.transformer.moe.experts import GroupedMLP
from megatron.core.transformer.moe import grouped_gemm_util as gg
from megatron.training import get_args
from megatron.core.parallel_state import (
    get_expert_model_parallel_group,
    get_tensor_and_expert_parallel_group,
    get_tensor_and_expert_parallel_world_size,
    get_tensor_model_parallel_world_size
)

from mindspeed.ops.gmm import GMMFunction
from mindspeed.core.transformer.moe.moe_layer_overlap_all2all import gmm_op
from mindspeed.ops.npu_groupmatmul_add import npu_groupmatmul_add_fp32
from mindspeed.model.transformer import should_recompute_activation
from mindspeed.core.transformer.moe.moe_layer_overlap_all2all import forward_func, backward_func
from mindspeed.core.transformer.moe.comm_utils import async_all_to_all
from mindspeed.core.transformer.moe.moe_utils import (
    only_recompute_activation,
    get_gemm_backward_need_tensors,
    set_all2all_experts_output
)
from mindspeed_llm.tasks.posttrain.lora.cc_lora_forward import dequantize


class LoraParallelGroupedMlpWithCompAndCommOverlapAll2All(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights1_a, weights1_b, weights2_a, weights2_b, args, moe_layer_ctx):
        weights1, weights2, original_weight1_a, original_weight1_b, original_weight2_a, original_weight2_b, \
            activation_func, group_list, layer_number, scaling = args
        global_args = get_args()

        weights1_a_scaling = weights1_a * scaling
        weights2_a_scaling = weights2_a * scaling

        ctx.scaling = scaling
        moe_zero_memory = global_args.moe_zero_memory
        ctx.is_only_recompute_activation = only_recompute_activation(layer_number)

        ctx.is_recompute_activation = moe_zero_memory == "level0" or should_recompute_activation(layer_number) or (
                moe_zero_memory == "level1" and ctx.is_only_recompute_activation)

        ctx.save_inputs = moe_zero_memory != "level0" and not (
                moe_zero_memory == "level1" and ctx.is_only_recompute_activation)

        ctx.recompute_level_1_total = moe_zero_memory == "level1" and not ctx.is_only_recompute_activation

        ctx.layer_number = layer_number
        use_gmm = (inputs.nelement() != 0)
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

        act_out, detached_act_inputs = forward_func(activation_func, mm1_out)

        if ctx.recompute_level_1_total:
            mm1_out.untyped_storage().resize_(0)

        if use_gmm:
            mm2_out = gmm_op(act_out, weights2, [], group_list, 0)[0]
            mm2_a = gmm_op(act_out, weights2_a_scaling, [], group_list, 0)[0]
            mm2_b = gmm_op(mm2_a, weights2_b, [], group_list, 0)[0]
            mm2_out += mm2_b
        else:
            mm2_a = torch.matmul(act_out, weights2_a_scaling)
            mm2_out = torch.matmul(act_out, weights2)
            mm2_b = torch.matmul(mm2_a, weights2_b)
            mm2_out += mm2_b

        if ctx.recompute_level_1_total:
            act_out.untyped_storage().resize_(0)
            moe_layer_ctx.recompute_tensors = (inputs, mm1_out, act_out)

        if ctx.is_recompute_activation:
            act_out.untyped_storage().resize_(0)
            ctx.activation_func = activation_func

        if not ctx.save_inputs:
            inputs, mm1_a = None, None
        ctx.save_for_backward(inputs, mm1_a,
                              detached_act_inputs,
                              act_out, mm2_a,
                              weights1, weights1_a_scaling, weights1_b,
                              weights2, weights2_a_scaling, weights2_b,
                              original_weight1_a, original_weight1_b, original_weight2_a, original_weight2_b,
                              group_list)
        return mm2_out, None

    @staticmethod
    def backward(ctx, *grad_outs):
        grad_outs = grad_outs[0]
        global_args = get_args()
        use_gemm_add_fusion = get_args().gemm_gradient_accumulation_fusion

        inputs, mm1_a, act_inputs, mm2_inputs, mm2_a, \
            weights1, weights1_a, weights1_b, weights2, weights2_a, weights2_b, \
            original_weight1_a, original_weight1_b, original_weight2_a, original_weight2_b, group_list = ctx.saved_tensors

        ((detach_input, indices, scores_ep, router_topk, global_input_tokens_local_experts_indices),
         permute2_input_detach, permute2_graph, output_splits, input_splits,
         input_splits_tp_ep) = get_gemm_backward_need_tensors()

        ep_group = get_expert_model_parallel_group()
        if global_args.moe_tp_extend_ep:
            ep_group = get_tensor_and_expert_parallel_group()

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

        grad_mm2_inputs += grad_mm2_inputs_a
        act_graph = mm2_inputs
        if ctx.is_recompute_activation:
            activation_func = ctx.activation_func
            mm2_inputs = activation_func(act_inputs)
            if ctx.use_gmm:
                mm2_a = gmm_op(mm2_inputs, weights2_a.transpose(-1, -2), [], group_list, 0)[0]
            else:
                mm2_a = torch.matmul(mm2_inputs, weights2_a)
        if ctx.use_gmm:
            if use_gemm_add_fusion:
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

        # grad of activation_func
        grad_outs.untyped_storage().resize_(0)
        mm2_inputs.untyped_storage().resize_(0)
        mm2_a.untyped_storage().resize_(0)

        act_graph.backward(grad_mm2_inputs)
        grad_mm2_inputs.untyped_storage().resize_(0)
        grad_mm2_b_inputs.untyped_storage().resize_(0)
        act_inputs.untyped_storage().resize_(0)

        if not ctx.save_inputs:
            def alltoall_token_permutation1(hidden_states, indices, router_topk):
                hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
                permutated_local_input_tokens, _ = permute(
                    hidden_states, indices
                )
                return permutated_local_input_tokens

            permutated_local_input_tokens = alltoall_token_permutation1(detach_input, indices, router_topk)

            _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
                permutated_local_input_tokens,
                output_splits,
                input_splits,
                ep_group,
            )

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

        backward_func(permute2_graph, mm1_inputs_grad)
        mm1_inputs_grad.untyped_storage().resize_(0)

        if not ctx.save_inputs:
            permute1_ep_all_to_all_handle.wait()
            permutated_local_input_tokens.untyped_storage().resize_(0)

        _, permute1_backward_input, bw_permute1_ep_all2all_handle = async_all_to_all(
            permute2_input_detach.grad,
            input_splits,
            output_splits,
            ep_group,
        )

        if not ctx.save_inputs:
            inputs, _ = permute(
                global_input_tokens, global_input_tokens_local_experts_indices
            )
            if ctx.use_gmm:
                mm1_a = gmm_op(inputs, weights1_a.transpose(-1, -2), [], group_list, 0)[0]
            else:
                mm1_a = inputs @ weights1_a
            global_input_tokens.untyped_storage().resize_(0)
        set_all2all_experts_output((permute1_backward_input, bw_permute1_ep_all2all_handle))

        if ctx.use_gmm:
            if use_gemm_add_fusion:
                npu_groupmatmul_add_fp32(mm1_a, act_inputs.grad, group_list, original_weight1_b.main_grad)
                npu_groupmatmul_add_fp32(inputs, mm1_b_inputs_grad * ctx.scaling, group_list,
                                         original_weight1_a.main_grad)
                if hasattr(original_weight1_b, 'grad_added_to_main_grad'):
                    if getattr(weights1, 'zero_out_wgrad', False):
                        grad_weights1_b = torch.zeros(
                            weights1_b.transpose(-1, -2).shape,
                            dtype=inputs.dtype,
                            device=torch.cuda.current_device(),
                            requires_grad=False,
                        )
                        grad_weights1_a = torch.zeros(
                            weights1_a.transpose(-1, -2).shape,
                            dtype=inputs.dtype,
                            device=torch.cuda.current_device(),
                            requires_grad=False,
                        )
                    else:
                        grad_weights1_b = torch.empty(
                            weights1_b.transpose(-1, -2).shape,
                            dtype=inputs.dtype,
                            device=torch.cuda.current_device(),
                            requires_grad=False,
                        )
                        grad_weights1_a = torch.empty(
                            weights1_a.transpose(-1, -2).shape,
                            dtype=inputs.dtype,
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
                grad_weights1_a = gmm_op(inputs.t(), mm1_b_inputs_grad, [], group_list, 2)[0] * ctx.scaling
        else:
            grad_weights1_b = torch.matmul(mm1_a.t(), act_inputs.grad)
            grad_weights1_a = torch.matmul(inputs.t(), mm1_b_inputs_grad) * ctx.scaling

        act_inputs.grad.untyped_storage().resize_(0)
        mm1_b_inputs_grad.untyped_storage().resize_(0)
        return mm1_inputs_grad, grad_weights1_a, grad_weights1_b, grad_weights2_a, grad_weights2_b, None, None


def lora_parallel_grouped_mlp_with_comp_and_comm_overlap_all2all(inputs, weights1_a, weights1_b, weights2_a, weight2_b,
                                                                 args, ctx):
    return LoraParallelGroupedMlpWithCompAndCommOverlapAll2All.apply(inputs, weights1_a, weights1_b, weights2_a,
                                                                     weight2_b, args, ctx)


class LoraParallelGroupedMLP(GroupedMLP):
    def __init__(self, num_local_experts, config, lora_config):
        super().__init__(num_local_experts, config=config)
        self.lora_r = lora_config.r
        self.scaling = lora_config.lora_alpha / self.lora_r

        if config.moe_extended_tp:
            tp_size = get_tensor_and_expert_parallel_world_size()
        else:
            tp_size = get_tensor_model_parallel_world_size()

        fc1_output_size = self.config.ffn_hidden_size * self.num_local_experts
        if config.gated_linear_unit:
            fc1_output_size *= 2
        fc1_output_size_per_partition = divide(fc1_output_size, tp_size)

        fc2_input_size = self.config.ffn_hidden_size * self.num_local_experts
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
        
        # init lora parameters
        # 按照Linear的权重形状feature_out, feature_in格式初始化，保持初始化相同

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

        if get_args().moe_hierarchical_alltoallv or get_args().moe_experts_pipeline_degree:
            raise AssertionError("Currently GMM LoRA Finetune not support moe_hierarchical_alltoallv")

    def forward(self, permuted_local_hidden_states, tokens_per_expert, ctx=None):
        args = get_args()

        if permuted_local_hidden_states.nelement() != 0:
            # input is empty
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

        if args.moe_alltoall_overlap_comm:
            # alltoall-overlap-comm
            group_list = torch.cumsum(tokens_per_expert, dim=0)
            return lora_parallel_grouped_mlp_with_comp_and_comm_overlap_all2all(permuted_local_hidden_states,
                                                                                w1_a, w1_b,
                                                                                w2_a, w2_b,
                                                                                (w1, w2,
                                                                                 self.weight1_lora_a,
                                                                                 self.weight1_lora_b,
                                                                                 self.weight2_lora_a,
                                                                                 self.weight2_lora_b,
                                                                                 self.activation_func,
                                                                                 group_list, self.layer_number,
                                                                                 self.scaling), ctx=ctx)
        else:
            # origin gemm
            if permuted_local_hidden_states.nelement() != 0:
                # Reshape the weights for the grouped GEMMs.
                fc1_output = gg.ops.gmm(permuted_local_hidden_states, w1, tokens_per_expert, trans_b=False)
                mm1_a = gg.ops.gmm(permuted_local_hidden_states, w1_a, tokens_per_expert, trans_b=False)
                mm1_b = gg.ops.gmm(mm1_a, w1_b, tokens_per_expert, trans_b=False) * self.scaling

                intermediate_parallel = self.activation_func(fc1_output + mm1_b)

                mm2_a = gg.ops.gmm(intermediate_parallel, w2_a, tokens_per_expert, trans_b=False)
                mm2_b = gg.ops.gmm(mm2_a, w2_b, tokens_per_expert, trans_b=False) * self.scaling
                fc2_output = gg.ops.gmm(intermediate_parallel, w2, tokens_per_expert, trans_b=False)

                fc2_output += mm2_b
            else:
                h = torch.matmul(permuted_local_hidden_states, w1)
                mm1_a = torch.matmul(permuted_local_hidden_states, w1_a)
                mm1_b = torch.matmul(mm1_a, w1_b) * self.scaling

                h = self.activation_func(h + mm1_b)

                mm2_a = torch.matmul(h, w2_a)
                mm2_b = torch.matmul(mm2_a, w2_b) * self.scaling
                h = torch.matmul(h, w2)

                fc2_output = h + mm2_b

            return fc2_output, None