# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, inputs, output_split_sizes, input_split_sizes):
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        world_size = torch.distributed.get_world_size(group=group)
        if world_size == 1:
            return inputs

        inputs = inputs.contiguous()
        if output_split_sizes is None:
            output = torch.empty_like(inputs)  # Equal split (all2all)
        else:
            # Unequal split (all2all-v)
            output = inputs.new_empty(size=[sum(output_split_sizes)] + list(inputs.size()[1:]),
                                      dtype=inputs.dtype, device=torch.accelerator.current_device_index())
        torch.distributed.all_to_all_single(output, inputs, output_split_sizes=output_split_sizes,
                                            input_split_sizes=input_split_sizes, group=group)
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        """Backward function."""
        return None, _AllToAll.apply(ctx.group, *grad_output, ctx.input_split_sizes, ctx.output_split_sizes), None, None


def all_to_all(group, inputs, output_split_sizes=None, input_split_sizes=None):
    """Wrapper for autograd function"""
    return _AllToAll.apply(group, inputs, output_split_sizes, input_split_sizes)


def gather_along_first_dim_expert_parallel(input_, group, async_op=False):
    """Gather tensors and concatenate along the first dimension."""
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return input_, None

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.accelerator.current_device_index())
    handle = torch.distributed.all_gather_into_tensor(output, input_.contiguous(), group=group, async_op=async_op)

    return output, handle
