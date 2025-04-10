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
        if args.qlora_save_dequantize and getattr(self.weight, "quant_state", None) is not None:
            self.weight = Parameter(bnb.functional.dequantize_4bit(self.weight.data, self.weight.quant_state))
        fn(self, destination, prefix, keep_vars)
        if getattr(self.weight, "quant_state", None) is not None:
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
    from megatron.legacy.model import Float16Module
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
        model = [Float16Module(model_module, args) for model_module in model]

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
