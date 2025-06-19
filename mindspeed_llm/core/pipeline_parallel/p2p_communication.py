# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional
import torch
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank,
)


def _batched_p2p_ops(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    group: torch.distributed.ProcessGroup,
    prev_pipeline_rank: int,
    next_pipeline_rank: int,
):
    ops = []
    if tensor_send_prev is not None:
        send_prev_op = torch.distributed.P2POp(
            torch.distributed.isend,
            tensor_send_prev,
            prev_pipeline_rank,
            group,
        )
        ops.append(send_prev_op)
    if tensor_recv_prev is not None:
        recv_prev_op = torch.distributed.P2POp(
            torch.distributed.irecv,
            tensor_recv_prev,
            prev_pipeline_rank,
            group,
        )
        ops.append(recv_prev_op)
    if tensor_recv_next is not None:
        recv_next_op = torch.distributed.P2POp(
            torch.distributed.irecv,
            tensor_recv_next,
            next_pipeline_rank,
            group,
        )
        ops.append(recv_next_op)
    if tensor_send_next is not None:
        send_next_op = torch.distributed.P2POp(
            torch.distributed.isend,
            tensor_send_next,
            next_pipeline_rank,
            group,
        )
        ops.append(send_next_op)
    if len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
    else:
        reqs = []
    return reqs



def _communicate_shapes(tensor_send_next, tensor_send_prev, recv_prev, recv_next, config):
    """
    Add group=get_pipeline_model_parallel_group() in P2POp shape communications to avoid error.
    Currently only enable in ray scenarios.
    """

    recv_prev_shape_tensor = None
    recv_next_shape_tensor = None
    send_prev_shape_tensor = None
    send_next_shape_tensor = None
    if recv_prev:
        recv_prev_shape_tensor = torch.empty(
            (3), device=torch.cuda.current_device(), dtype=torch.int64
        )
    if recv_next:
        recv_next_shape_tensor = torch.empty(
            (3), device=torch.cuda.current_device(), dtype=torch.int64
        )
    if tensor_send_prev is not None:
        send_prev_shape_tensor = torch.tensor(
            tensor_send_prev.size(), device=torch.cuda.current_device(), dtype=torch.int64
        )
    if tensor_send_next is not None:
        send_next_shape_tensor = torch.tensor(
            tensor_send_next.size(), device=torch.cuda.current_device(), dtype=torch.int64
        )

    if config.use_ring_exchange_p2p:
        torch.distributed.ring_exchange(
            tensor_send_prev=send_prev_shape_tensor,
            tensor_recv_prev=recv_prev_shape_tensor,
            tensor_send_next=send_next_shape_tensor,
            tensor_recv_next=recv_next_shape_tensor,
            group=get_pipeline_model_parallel_group(),
        )
    else:
        ops = []
        if send_prev_shape_tensor is not None:
            send_prev_op = torch.distributed.P2POp(
                torch.distributed.isend,
                send_prev_shape_tensor,
                get_pipeline_model_parallel_prev_rank(),
                group=get_pipeline_model_parallel_group(),
            )
            ops.append(send_prev_op)
        if recv_prev_shape_tensor is not None:
            recv_prev_op = torch.distributed.P2POp(
                torch.distributed.irecv,
                recv_prev_shape_tensor,
                get_pipeline_model_parallel_prev_rank(),
                group=get_pipeline_model_parallel_group(),
            )
            ops.append(recv_prev_op)
        if send_next_shape_tensor is not None:
            send_next_op = torch.distributed.P2POp(
                torch.distributed.isend,
                send_next_shape_tensor,
                get_pipeline_model_parallel_next_rank(),
                group=get_pipeline_model_parallel_group(),
            )
            ops.append(send_next_op)
        if recv_next_shape_tensor is not None:
            recv_next_op = torch.distributed.P2POp(
                torch.distributed.irecv,
                recv_next_shape_tensor,
                get_pipeline_model_parallel_next_rank(),
                group=get_pipeline_model_parallel_group(),
            )
            ops.append(recv_next_op)
        if len(ops) > 0:
            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        # To protect against race condition when using batch_isend_irecv().
        # should take this out once the bug with batch_isend_irecv is resolved.
        torch.cuda.synchronize()

    recv_prev_shape = [0, 0, 0]
    if recv_prev_shape_tensor is not None:
        recv_prev_shape = recv_prev_shape_tensor.tolist()

    recv_next_shape = [0, 0, 0]
    if recv_next_shape_tensor is not None:
        recv_next_shape = recv_next_shape_tensor.tolist()

    return recv_prev_shape, recv_next_shape
