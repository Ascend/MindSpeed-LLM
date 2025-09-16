# Copyright (c) 2025; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

import torch


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, send_count_matrix, mode):
        """Forward function."""
        ctx.group = group
        ctx.send_count_matrix = send_count_matrix
        ctx.mode = mode

        world_size = torch.distributed.get_world_size(group=group)
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input

        input = input.contiguous()
        if output_split_sizes is None:
            # Equal split (all2all)
            output = torch.empty_like(input)
        else:
            # Unequal split (all2all-v)
            output = input.new_empty(
                size=[sum(output_split_sizes)] + list(input.size()[1:]),
                dtype=input.dtype,
                device=torch.cuda.current_device(),
            )
        torch.distributed.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        """Backward function."""
        return (
            None,
            _AllToAll.apply(ctx.group, *grad_output, ctx.send_count_matrix_T, ctx.mode),
            None,
            None,
        )


def all_to_all(group, input_, send_count_matrix=None, mode=None):
    """Wrapper for autograd function"""
    return _AllToAll.apply(group, input_, send_count_matrix, mode)