from functools import wraps
from typing import Optional, Callable

import torch
from torch.nn.parameter import Parameter

from megatron.core import parallel_state, ModelParallelConfig
from megatron.core.tensor_parallel import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region
)
from megatron.core.tensor_parallel.utils import VocabUtility
from megatron.core.tensor_parallel.layers import (
    linear_with_frozen_weight,
    linear_with_grad_accumulation_and_async_allreduce,
    ColumnParallelLinear,
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
    VocabParallelEmbedding,
)
from megatron.legacy.model.fused_layer_norm import MixedFusedLayerNorm
from megatron.training import get_args

import torch.nn.functional as F
from megatron.core.tensor_parallel.mappings import (
    reduce_scatter_to_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
)
from mindspeed.utils import get_actual_seq_len, set_actual_seq_len


def vocab_embedding_init_func(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        init_method: Callable,
        config: ModelParallelConfig,
        reduce_scatter_embeddings: bool = False,
        skip_weight_param_allocation: bool = False,
):
    """Patch for legacy norm."""
    super(VocabParallelEmbedding, self).__init__()
    # Keep the input dimensions.
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.reduce_scatter_embeddings = reduce_scatter_embeddings
    self.tensor_model_parallel_size = parallel_state.get_tensor_model_parallel_world_size()
    # Divide the weight matrix along the vocaburaly dimension.
    (
        self.vocab_start_index,
        self.vocab_end_index,
    ) = VocabUtility.vocab_range_from_global_vocab_size(
        self.num_embeddings, parallel_state.get_tensor_model_parallel_rank(), self.tensor_model_parallel_size
    )
    self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index
    self.deterministic_mode = config.deterministic_mode

    # Allocate weights and initialize.
    if not skip_weight_param_allocation:
        if config.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition, self.embedding_dim, dtype=config.params_dtype
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight,
                    self.num_embeddings,
                    self.embedding_dim,
                    self.num_embeddings_per_partition,
                    0,
                    init_method,
                    params_dtype=config.params_dtype,
                )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition,
                    self.embedding_dim,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(self.weight, init_method, partition_dim=0, stride=1)
    else:
        self.weight = None

    args = get_args()
    if parallel_state.is_pipeline_first_stage() and args.embed_layernorm:
        norm = MixedFusedLayerNorm(args.hidden_size)
        self.norm = norm


def vocab_parallel_embedding_forward(self, input_, weight=None):
    if weight is None:
        if self.weight is None:
            raise RuntimeError(
                "weight was not supplied to VocabParallelEmbedding forward pass "
                "and skip_weight_param_allocation is True."
            )
        weight = self.weight

    if self.tensor_model_parallel_size > 1:
        # Build the mask.
        input_mask = (input_ < self.vocab_start_index) | \
                     (input_ >= self.vocab_end_index)
        # Mask the input.
        masked_input = input_.clone() - self.vocab_start_index
        masked_input *= ~input_mask
    else:
        masked_input = input_
        # Get the embeddings.

    # For higher accumulation accuracy for bf16 on NPU.
    output_parallel = F.embedding(masked_input, weight)

    # Mask the output embedding.
    if self.tensor_model_parallel_size > 1:
        output_parallel *= ~input_mask[..., None]
    if self.reduce_scatter_embeddings:
        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        output_parallel = output_parallel.transpose(0, 1).contiguous()
        output = reduce_scatter_to_sequence_parallel_region(output_parallel)
    else:
        # Reduce across all the model parallel GPUs.
        output = reduce_from_tensor_model_parallel_region(output_parallel)
    args_ = get_args()
    if hasattr(self, 'norm'):
        output = self.norm(output)
    return output * args_.embedding_multiplier_scale if args_.embedding_multiplier_scale else output


class SegmentedColumnParallelLinear(ColumnParallelLinear):
    def __int__(self):
        super(ColumnParallelLinear, self).__init__()

    def forward(self, input_: torch.Tensor, weight: Optional[torch.Tensor] = None):
        """Forward of ColumnParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

            weight (optional): weight tensor to use, compulsory when
                skip_weight_param_allocation is True.

        Returns:
            - output
            - bias

        """
        args_ = get_args()
        if weight is None:
            if self.weight is None:
                raise RuntimeError(
                    "weight was not supplied to ColumnParallelLinear forward pass "
                    "and skip_weight_param_allocation is True."
                )
            weight = self.weight
        else:
            # Check the weight passed in is the correct shape
            expected_shape = (self.output_size_per_partition, self.input_size)
            if weight.shape != expected_shape:
                raise RuntimeError(
                    f"supplied weight's shape is {tuple(weight.shape)}, "
                    f"not {expected_shape} as expected"
                )

        if self.config._cpu_offloading_context is not None:
            if self.config._cpu_offloading_context.inside_context:
                if self.config.cpu_offloading:
                    raise ValueError("CPU Offloading cannot be enabled while using non-TE modules")

        bias = self.bias if not self.skip_bias_add else None

        if (
                self.allreduce_dgrad
                or self.sequence_parallel
                or self.explicit_expert_comm
                or self.disable_grad_reduce
        ):
            input_parallel = input_
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input_)

        if self.config.defer_embedding_wgrad_compute:
            self.embedding_activation_buffer.append(input_parallel)

        # Matrix multiply.
        if not weight.requires_grad:
            self._forward_impl = linear_with_frozen_weight
        else:
            self._forward_impl = linear_with_grad_accumulation_and_async_allreduce


        allreduce_dgrad = False if self.explicit_expert_comm else self.allreduce_dgrad

        weight = torch.split(weight, weight.shape[0] // args_.output_layer_slice_num, dim=0)

        output_parallel = []
        for i in range(args_.output_layer_slice_num):
            output_parallel.append(self._forward_impl(
                input=input_parallel,
                weight=weight[i],
                bias=bias,
                gradient_accumulation_fusion=self.gradient_accumulation_fusion,
                async_grad_allreduce=allreduce_dgrad,
                sequence_parallel=False if self.explicit_expert_comm else self.sequence_parallel,
                grad_output_buffer=self.grad_output_buffer
                if self.config.defer_embedding_wgrad_compute
                else None,
                allreduce_dgrad=allreduce_dgrad,
            ))
        output_parallel = torch.cat(output_parallel, dim=2)

        if self.gather_output:
            # All-gather across the partitions.
            assert not self.sequence_parallel
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


def checkpoint_forward_wrapper(fn):
    def wrapper(ctx, run_function, distribute_saved_activations, *args):
        ctx.actual_seq_len = get_actual_seq_len()
        return fn(ctx, run_function, distribute_saved_activations, *args)

    return wrapper


def checkpoint_backward_wrapper(fn):
    def wrapper(ctx, *args):
        set_actual_seq_len(ctx.actual_seq_len)
        return fn(ctx, *args)

    return wrapper
