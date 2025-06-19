# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
import torch
import torch_npu
from torch.nn.parameter import Parameter

from megatron.training import get_args
from megatron.core import mpu
from megatron.core.utils import get_model_config
from megatron.core.enums import ModelType
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.parallel_state import get_tensor_model_parallel_group
from mindspeed.ops.npu_groupmatmul_add import npu_groupmatmul_add_fp32
from mindspeed.ops.gmm import GMMFunction
from mindspeed_llm.training.training import model_provider_func_wrapper
from mindspeed_llm.tasks.posttrain.lora.utils import is_enable_qlora

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None


def parallel_linear_init_wrapper(fn):
    def wrapper(self, input_size, output_size, **kwargs):
        fn(self, input_size, output_size, **kwargs)
        if is_enable_qlora():
            self.weight.data = self.weight.data.to("cpu")
    return wrapper


def linear_with_frozen_weight_forward(
        ctx, input_, weight, bias, allreduce_dgrad
    ):
    ctx.save_for_backward(weight)
    ctx.allreduce_dgrad = allreduce_dgrad
    if hasattr(weight, "quant_state"):
        weight_tmp = bnb.functional.dequantize_4bit(weight.data, weight.quant_state).to(input_.dtype)
    else:
        weight_tmp = weight
    output = torch.matmul(input_, weight_tmp.t())
    if bias is not None:
        output = output + bias
    return output


def linear_with_frozen_weight_backward(ctx, grad_output):
    (weight,) = ctx.saved_tensors
    if hasattr(weight, "quant_state"):
        weight_tmp = bnb.functional.dequantize_4bit(weight.data, weight.quant_state).to(grad_output.dtype)
    else:
        weight_tmp = weight
    grad_input = grad_output.matmul(weight_tmp)
    if ctx.allreduce_dgrad:
        # All-reduce. Note: here async and sync are effectively the same.
        torch.distributed.all_reduce(grad_input, group=get_tensor_model_parallel_group())

    return grad_input, None, None, None


def parallel_linear_save_to_state_dict_wrapper(fn):
    def wrapper(self, destination, prefix, keep_vars):
        """
        save weight and bias,
        then fill state_dict with components of quant_state
        """
        args = get_args()
        fn(self, destination, prefix, keep_vars)

        if args.qlora_save_dequantize and getattr(self.weight, "quant_state", None) is not None:
            device = self.weight.device
            dequantized_weight = Parameter(bnb.functional.dequantize_4bit(self.weight.data.to(device), self.weight.quant_state)).cpu()
            destination[prefix + "weight"] = dequantized_weight.detach()

        if getattr(self.weight, "quant_state", None) is not None and not args.qlora_save_dequantize:
            for k, v in self.weight.quant_state.as_dict(packed=True).items():
                destination[prefix + "weight." + k] = v if keep_vars else v.detach()

    return wrapper


def parallel_linear_load_from_state_dict_wrapper(fn):
    def wrapper(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if any(['bitsandbytes' in i for i in state_dict.keys()]):  # is quantized linear
            qs_dict = {}
            for k, v in state_dict.items():
                key = k.replace(prefix, "")
                if key != '_extra_state':
                    qs_dict[key] = v

            self.weight = bnb.nn.Params4bit.from_prequantized(
                data=qs_dict.get('weight'),
                quantized_stats={key.replace('weight.', ''): qs_dict[key] for key in qs_dict if key != 'weight' and key != 'bias'},
                requires_grad=False,
                device='npu')
        fn(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        self.weight.data = self.weight.data.to("npu")
    return wrapper


def get_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
    """Build the model."""
    from megatron.core import tensor_parallel
    from megatron.core.transformer.module import Float16Module
    from megatron.core.distributed import DistributedDataParallelConfig
    
    tpl = tensor_parallel.layers
    model_provider_func = model_provider_func_wrapper(model_provider_func)
    args = get_args()
    args.model_type = model_type

    # Build model.
    if mpu.get_pipeline_model_parallel_world_size() > 1 and \
       args.virtual_pipeline_model_parallel_size is not None:
        if model_type == ModelType.encoder_and_decoder:
            raise ValueError("Interleaved schedule not supported for model with both encoder and decoder")
        model = []
        for i in range(args.virtual_pipeline_model_parallel_size):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            this_model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process
            )
            this_model.model_type = model_type
            model.append(this_model)
    else:
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        add_encoder = True
        add_decoder = True
        if model_type == ModelType.encoder_and_decoder:
            if mpu.get_pipeline_model_parallel_world_size() > 1:
                if args.pipeline_model_parallel_split_rank is None:
                    raise ValueError("Split rank needs to be specified for model with both encoder and decoder")
                rank = mpu.get_pipeline_model_parallel_rank()
                split_rank = args.pipeline_model_parallel_split_rank
                world_size = mpu.get_pipeline_model_parallel_world_size()
                pre_process = rank == 0 or rank == split_rank
                post_process = (rank == (split_rank - 1)) or (
                        rank == (world_size - 1))
                add_encoder = mpu.is_pipeline_stage_before_split()
                add_decoder = mpu.is_pipeline_stage_after_split()
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process,
                add_encoder=add_encoder,
                add_decoder=add_decoder)
        else:
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process
            )
        model.model_type = model_type

    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on (tensor, pipeline) '
              'model parallel rank ({}, {}): {}'.format(
            mpu.get_tensor_model_parallel_rank(),
            mpu.get_pipeline_model_parallel_rank(),
            sum([sum([p.nelement() for p in model_module.parameters()])
                 for model_module in model])), flush=True)

    # start of megatron_adaptation,
    # here we keep the main model's linear layers on CPU to avoid OOM in QLoRA.
    # GPU allocation.
    for model_module in model:
        if is_enable_qlora():
            for name, module in model_module.base_model.named_modules():
                if not hasattr(module, "weight") or hasattr(module, "base_layer"):
                    continue

                is_lora_adapter = any(substring in name for substring in ["lora_A", "lora_B"])
                is_target_linear = (
                    isinstance(module, (tpl.ColumnParallelLinear, tpl.RowParallelLinear))
                    and "layers" in name
                )
                if not (is_target_linear and not is_lora_adapter):
                    module.weight.data = module.weight.data.to(torch.cuda.current_device())
                if hasattr(module, "expert_bias") and module.expert_bias is not None:
                    module.expert_bias = module.expert_bias.to(torch.cuda.current_device())
                if hasattr(module, "local_tokens_per_expert") and module.local_tokens_per_expert is not None:
                    module.local_tokens_per_expert = module.local_tokens_per_expert.to(torch.cuda.current_device())
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias.data = module.bias.data.to(torch.cuda.current_device())
        else:
            model_module.cuda(torch.cuda.current_device())
    # end of megatron_adaptation

    # Fp16 conversion.
    if args.fp16 or args.bf16:
        config = get_model_config(model[0])
        model = [Float16Module(config, model_module) for model_module in model]

    if wrap_with_ddp:
        config = get_model_config(model[0])
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=args.accumulate_allreduce_grads_in_fp32,
            overlap_grad_reduce=args.overlap_grad_reduce,
            use_distributed_optimizer=args.use_distributed_optimizer,
            check_for_nan_in_grad=args.check_for_nan_in_loss_and_grad,
            bucket_size=args.ddp_bucket_size)
        model = [DDP(config,
                     ddp_config,
                     model_chunk,
                     # Turn off bucketing for model_chunk 2 onwards, since communication for these
                     # model chunks is overlapped with compute anyway.
                     disable_bucketing=(model_chunk_idx > 0))
                 for (model_chunk_idx, model_chunk) in enumerate(model)]

        # Broadcast params from data parallel src rank to other data parallel ranks.
        if args.data_parallel_random_init:
            for model_module in model:
                model_module.broadcast_params()

    return model


def _load_bnb_nf4_weight(state_dict, prefix, weight_name):
    prefix_name = prefix + weight_name
    quantized_weight = state_dict.get(prefix_name)
    if quantized_weight is None:
        raise ValueError(f"quantized_weight is None, expected a tensor of type torch.uint8.")
    elif not isinstance(quantized_weight, torch.Tensor) or quantized_weight.dtype != torch.uint8:
        raise TypeError(f"Expected quantized_weight to be of type torch.uint8, but got {quantized_weight.dtype}")
    
    qs_dict = {}
    for k, v in state_dict.items():
        if prefix_name not in k or prefix_name == k:
            continue
        key = k.replace(prefix_name, "")[1:]
        if "lora_a" == key or "lora_b" == key:
            continue
        qs_dict[key] = v

    return bnb.nn.Params4bit.from_prequantized(
        data=quantized_weight,
        quantized_stats=qs_dict,
        requires_grad=False,
        device='npu')


def groupedmlp_load_from_state_dict_wrapper(fn):
    def wrapper(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self.weight1 = _load_bnb_nf4_weight(state_dict, prefix, "weight1")
        self.weight2 = _load_bnb_nf4_weight(state_dict, prefix, "weight2")
        fn(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
    return wrapper


def infer_dequant_shape(hidden_size, weight_shape):
    num_local_experts = weight_shape[0]
    out_shape = [num_local_experts]
    if weight_shape[1] == hidden_size:
        out_shape.extend([hidden_size, 2 * weight_shape[2]])
    else:
        out_shape.extend([2 * weight_shape[1], hidden_size])
    return out_shape


class WeightNf4QuantGMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, original_weight, x, weight, bias, group_args):
        group_list, group_type, gemm_fusion, group_list_type, group_list_data_type = group_args
        weight_tmp = bnb.functional.dequantize_4bit(
            original_weight.data,
            original_weight.quant_state
        ).to(x.dtype)
        hidden_size = list(set(weight_tmp.shape) & set(weight.shape))[0]
        weight_tmp_shape = infer_dequant_shape(hidden_size, weight.shape)
        weight_tmp = weight_tmp.reshape(*weight_tmp_shape)
        if bias is not None and bias.requires_grad:
            raise ValueError("Bias is not supported to compute gradient!")
        if (x.requires_grad or weight.requires_grad) and group_type != 0:
            raise ValueError("group_type must be zero to compute gradients of x and weight!")
        bias = [] if bias is None else [bias]
        if group_list_type == 0:
            outputs = GMMFunction.builder.load().npu_gmm([x], [weight_tmp], bias, group_list, group_type, group_list_type)
        elif group_list_type == 1:
            outputs = GMMFunction.builder2.load().npu_gmm([x], [weight_tmp], bias, group_list, group_type, group_list_type)
        if group_list_data_type == 0:
            ctx.save_for_backward(x, weight, original_weight)
            ctx.group_list = group_list
        else:
            ctx.save_for_backward(x, weight, group_list, original_weight)
        ctx.weight_tmp_shape = weight_tmp_shape
        ctx.gemm_fusion = gemm_fusion
        ctx.group_list_type = group_list_type
        ctx.group_list_data_type = group_list_data_type

        return outputs[0]

    @staticmethod
    def backward(ctx, grad_outputs):
        if ctx.group_list_data_type == 0:
            x, weight, original_weight = ctx.saved_tensors
            group_list = ctx.group_list
        else:
            x, weight, group_list, original_weight = ctx.saved_tensors
        weight_tmp = bnb.functional.dequantize_4bit(
            original_weight.data,
            original_weight.quant_state
        ).to(grad_outputs.dtype).reshape(ctx.weight_tmp_shape)

        if ctx.gemm_fusion:
            if ctx.group_list_type == 0:
                dx, _, dbias = GMMFunction.builder.load().npu_gmm_backward_fusion([grad_outputs], [weight_tmp], group_list,
                                                                    ctx.group_list_type)
                npu_groupmatmul_add_fp32(x, grad_outputs, group_list, original_weight.main_grad)
                
            elif ctx.group_list_type == 1:
                dx, _, dbias = GMMFunction.builder2.load().npu_gmm_backward_fusion([grad_outputs], [weight_tmp], group_list,
                                                                    ctx.group_list_type)
                group_list_v2 = torch.cumsum(group_list, dim=0)                                           
                npu_groupmatmul_add_fp32(x, grad_outputs, group_list_v2, original_weight.main_grad)

            dbias = None if len(dbias) == 0 else dbias[0]
  
            if hasattr(original_weight, 'grad_added_to_main_grad'):
                if getattr(weight, 'zero_out_wgrad', False):
                    grad_weight = torch.zeros(
                        weight.shape,
                        dtype=x.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                else:
                    grad_weight = torch.empty(
                        weight.shape,
                        dtype=x.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                original_weight.grad_added_to_main_grad = True
            else:
                grad_weight = None

            return None, dx[0], grad_weight, dbias, None
        else:
            if ctx.group_list_type == 0:
                dx, dw, dbias = GMMFunction.builder.load().npu_gmm_backward([grad_outputs], [x], [weight_tmp], group_list,
                                                                    ctx.group_list_type)
            elif ctx.group_list_type == 1:
                dx, dw, dbias = GMMFunction.builder2.load().npu_gmm_backward([grad_outputs], [x], [weight_tmp], group_list,
                                                                    ctx.group_list_type)
            dbias = None if len(dbias) == 0 else dbias[0]

            return None, dx[0], dw[0], dbias, None


def grouped_gemm_util_ops_gmm(a, b, batch_sizes, trans_b=False, gemm_fusion=False, original_weight=None):
    if trans_b:
        b = b.t()
    group_list = torch.cumsum(batch_sizes, dim=0).to('npu')
    if isinstance(group_list, (torch.Tensor, type(None))):
        group_list_data_type = 1
    else:
        group_list_data_type = 0
    group_args = (group_list, 0, gemm_fusion, 0, group_list_data_type)
    return WeightNf4QuantGMMFunction.apply(original_weight, a, b, bias=None, group_args=group_args)


def moe_layer_overlap_all2all_gmm_op_wrapper(fn):
    def wrapper(x, weight, bias, group_list, group_type):
        if hasattr(weight, "quant_state"):
            weight_tmp = bnb.functional.dequantize_4bit(weight.data, weight.quant_state).to(x.dtype)
        else:
            weight_tmp = weight
        return fn(x, weight_tmp, bias, group_list, group_type)
    return wrapper
