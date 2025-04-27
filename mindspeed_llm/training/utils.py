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

"""General utilities."""
import os
import stat
import random
import warnings
from functools import wraps
from typing import Optional, Union, List

from itertools import takewhile
import torch
import torch_npu
from torch import distributed as dist
import numpy as np
import megatron
from megatron.training import get_args
from megatron.core import mpu
from tqdm import tqdm

from mindspeed.utils import (set_actual_seq_len, set_position_ids,
                             _get_batch_on_this_cp_rank_in_megatron_cp,
                             _get_batch_on_this_cp_rank_in_ulysses_cp,
                             _get_batch_on_this_cp_rank_in_hybrid_cp_general,
                             _get_batch_on_this_cp_rank_in_hybrid_cp,
                             _get_batch_on_this_cp_rank_in_adaptive_cp,
                             _get_batch_on_this_cp_rank_in_hybrid_adaptive_cp)
from mindspeed.core.tensor_parallel_y_union_cp import TensorParallelYUnionCP
from mindspeed.model.transformer import set_attention_mask
from mindspeed.utils import _get_batch_on_this_tp_y_cp_rank_in_megatron_cp

WRITE_FILE_DEFAULT_FLAGS = os.O_WRONLY | os.O_CREAT
WRITE_FILE_DEFAULT_MODES = stat.S_IWUSR | stat.S_IRUSR


def compute_actual_seq_len(seq, stride=0):
    """
    compute actual seq len.
    MTP layer bring extra n tokens, which should be cut.
    """
    res = list()
    _args = get_args()
    seq_length = _args.seq_length
    batch_size = seq.shape[0]
    for batch_idx in range(batch_size):
        batch_seq = seq[batch_idx]
        if batch_idx == 0:
            zero_pos = (batch_seq == 0).nonzero()[1:].squeeze(dim=1)
        else:
            zero_pos = (batch_seq == 0).nonzero().squeeze(dim=1)
        res.extend((zero_pos + (batch_idx * seq_length - stride)).tolist())
        batch_len = len(batch_seq) + batch_idx * seq_length - stride
        if batch_len > seq_length * (batch_idx + 1):
            batch_len = seq_length * (batch_idx + 1)
        if batch_idx == batch_size - 1:
            res.append(batch_len)
    return res


def generate_actual_seq_len(batch):
    position_ids = batch.get('position_ids').transpose(0, 1).contiguous()
    set_position_ids(position_ids)
    position_ids = batch.get('position_ids')
    actual_seq_len = compute_actual_seq_len(position_ids)
    set_actual_seq_len(actual_seq_len)


def parse_args():
    return megatron.training.arguments.parse_args()


def is_rank_0():
    """Check whether it is rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0 or (
                torch.distributed.get_rank() % torch.cuda.device_count() == 0
        ):
            return True
        else:
            return False
    else:
        return True


def print_rank0_by_args(args, message):
    """Before initialization of distributed, we only print on rank 0."""
    if args.rank == 0:
        print(message, flush=True)


def get_tune_attention_mask(attention_mask_1d):
    args = get_args()
    micro_batch_size, seq_length = attention_mask_1d.size()
    if args.reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1

    if args.tokenizer_padding_side == "left":
        attention_mask = torch.tril(
            torch.ones(seq_length, seq_length, device=attention_mask_1d.device, dtype=torch.bool)).view(1, 1,
                                                                                                        seq_length,
                                                                                                        seq_length)
        attention_mask_tran = attention_mask_1d.view(seq_length, 1, -1)
        attention_mask = attention_mask.masked_fill((attention_mask_tran < 0.5).view(-1, 1, 1, seq_length), value=0)
    else:
        attention_mask = torch.tril(torch.ones(
            (att_mask_batch, seq_length, seq_length), device=attention_mask_1d.device)).view(
            att_mask_batch, 1, seq_length, seq_length)
    attention_mask = attention_mask.masked_fill((attention_mask_1d < 0.5).view(-1, 1, 1, seq_length), value=0)
    attention_mask = (attention_mask < 0.5)
    return attention_mask


def print_args_wrapper(fn):
    """
    Add switch for controlling when to print arguments.
    """

    @wraps(fn)
    def wrapper(title, args, after_validate=False):
        if after_validate:
            fn(title, args)

    return wrapper


def print_args(title, args):
    """
    Provide a public func for printing arguments.
    """
    # here global process group has not been initialized, that's why we use args.rank
    if args.rank == 0:
        print(f'------------------------ {title} ------------------------', flush=True)
        str_list = []
        for arg in vars(args):
            dots = '.' * (48 - len(arg))
            str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print(f'-------------------- end of {title} ---------------------',
              flush=True)


def seed_all(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    torch_npu.npu.manual_seed_all(seed)
    torch_npu.npu.manual_seed(seed)


def emit(self, record):
    try:
        rank = dist.get_rank()
    except Exception:
        rank = -1  # 如果获取rank失败，则设置为一个不合法的rank

    if rank == 0 or rank == -1:
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def get_device_wrapper(fn):
    @wraps(fn)
    def wrapper(local_rank=None, *arg, **kwargs):
        backend = torch.distributed.get_backend()
        if backend == 'hccl':
            if local_rank is None:
                device = torch.device('npu')
            else:
                device = torch.device(f'npu:{local_rank}')
        else:
            device = fn(local_rank)
        return device

    return wrapper


def unwrap_model_wrapper(fn):
    @wraps(fn)
    def wrapper(model, module_instances=None):
        if not module_instances:
            module_instances = megatron.training.utils.ALL_MODULE_WRAPPER_CLASSNAMES
        return fn(model, module_instances)

    return wrapper


def get_finetune_data_on_this_tp_rank(data_iterator):
    try:
        ds = next(data_iterator)
    except StopIteration as e:
        warnings.warn(f"An exception occurred in dataloader: {e}")
        data_iterator = iter(data_iterator)
        ds = next(data_iterator)
    tokens = ds.get('input_ids').long().cuda(non_blocking=True)
    tokens_shape = tokens.shape
    micro_batch_size = tokens_shape[0]

    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(item, mpu.get_tensor_model_parallel_src_rank(),
                                        group=mpu.get_tensor_model_parallel_group())

    if mpu.get_tensor_model_parallel_rank() == 0:
        via_length = torch.LongTensor([tokens_shape[1]]).cuda(non_blocking=True)
        _broadcast(via_length)
        _broadcast(tokens)
        attention_mask_1d = ds.get('attention_mask').long().cuda(non_blocking=True)
        _broadcast(attention_mask_1d)
        attention_mask = get_tune_attention_mask(attention_mask_1d)
    else:
        via_length = torch.empty((1), dtype=torch.int64, device=torch.cuda.current_device())
        _broadcast(via_length)
        tokens = torch.empty((micro_batch_size, via_length), dtype=torch.int64, device=torch.cuda.current_device())
        _broadcast(tokens)
        attention_mask_1d = torch.empty((micro_batch_size, via_length), dtype=torch.int64,
                                        device=torch.cuda.current_device())
        _broadcast(attention_mask_1d)
        attention_mask = get_tune_attention_mask(attention_mask_1d)

    return tokens, attention_mask


def get_batch_on_this_tp_rank(data_iterator):
    args = get_args()

    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(item, mpu.get_tensor_model_parallel_src_rank(),
                                        group=mpu.get_tensor_model_parallel_group())

    if mpu.get_tensor_model_parallel_rank() == 0:
        if data_iterator is not None:
            data = next(data_iterator)
        else:
            data = None

        if args.return_document_ids and mpu.get_context_parallel_rank() == 0 and mpu.get_pipeline_model_parallel_rank() == 0:
            document_ids = [
                [x.item() for x in takewhile(lambda y: y.item() != -100, row)]
                for row in data['document_ids']
            ]
            data_idx = [
                [x.item() for x in takewhile(lambda y: y.item() != -100, row)]
                for row in data['idx']
            ]

            data.pop("document_ids", None)
            data.pop("idx", None)

            batch = {
                'tokens': data["tokens"].cuda(non_blocking=True),
                'labels': data["labels"].cuda(non_blocking=True),
                'loss_mask': data["loss_mask"].cuda(non_blocking=True),
                'attention_mask': None if "attention_mask" not in data else data["attention_mask"].cuda(non_blocking=True),
                'position_ids': data["position_ids"].cuda(non_blocking=True),
                'document_ids': document_ids,
                'idx': data_idx
            }
        else:
            batch = {
                'tokens': data["tokens"].cuda(non_blocking=True),
                'labels': data["labels"].cuda(non_blocking=True),
                'loss_mask': data["loss_mask"].cuda(non_blocking=True),
                'attention_mask': None if "attention_mask" not in data else data["attention_mask"].cuda(non_blocking=True),
                'position_ids': data["position_ids"].cuda(non_blocking=True)
            }
        if args.pipeline_model_parallel_size == 1:
            _broadcast(batch['tokens'])
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])

        elif mpu.is_pipeline_first_stage():
            _broadcast(batch['tokens'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])
            if args.schedules_method == 'dualpipev':
                _broadcast(batch['loss_mask'])
                _broadcast(batch['labels'])

        elif mpu.is_pipeline_last_stage():
            if args.num_nextn_predict_layers or args.schedules_method == 'dualpipev':
                _broadcast(batch['tokens'])
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
            if args.reset_position_ids or args.num_nextn_predict_layers or args.schedules_method == 'dualpipev':
                _broadcast(batch['position_ids'])
        else:
            _broadcast(batch['attention_mask'])
            if args.reset_position_ids:
                _broadcast(batch['position_ids'])

    else:

        tokens = torch.empty((args.micro_batch_size, args.seq_length + args.num_nextn_predict_layers),
                             dtype=torch.int64,
                             device=torch.cuda.current_device())
        labels = torch.empty((args.micro_batch_size, args.seq_length + args.num_nextn_predict_layers),
                             dtype=torch.int64,
                             device=torch.cuda.current_device())
        loss_mask = torch.empty((args.micro_batch_size, args.seq_length + args.num_nextn_predict_layers),
                                dtype=torch.float32,
                                device=torch.cuda.current_device())
        if args.create_attention_mask_in_dataloader:
            attention_mask = torch.empty(
                (args.micro_batch_size, 1, args.seq_length + args.num_nextn_predict_layers,
                 args.seq_length + args.num_nextn_predict_layers), dtype=torch.bool,
                device=torch.cuda.current_device()
            )
        else:
            attention_mask = None
        position_ids = torch.empty((args.micro_batch_size, args.seq_length + args.num_nextn_predict_layers),
                                   dtype=torch.int64,
                                   device=torch.cuda.current_device())

        if args.pipeline_model_parallel_size == 1:
            _broadcast(tokens)
            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            _broadcast(position_ids)

        elif mpu.is_pipeline_first_stage():
            _broadcast(tokens)
            _broadcast(attention_mask)
            _broadcast(position_ids)
            if args.schedules_method == 'dualpipev':
                _broadcast(loss_mask)
                _broadcast(labels)
            else:
                labels = None
                loss_mask = None

        elif mpu.is_pipeline_last_stage():
            if args.num_nextn_predict_layers or args.schedules_method == 'dualpipev':
                _broadcast(tokens)
            else:
                tokens = None
            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            if args.reset_position_ids or args.num_nextn_predict_layers or args.schedules_method == 'dualpipev':
                _broadcast(position_ids)
            else:
                position_ids = None

        else:
            tokens = None
            labels = None
            loss_mask = None
            _broadcast(attention_mask)
            if args.reset_position_ids:
                _broadcast(position_ids)
            else:
                position_ids = None

        batch = {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids
        }

    return batch


def get_batch_on_this_cp_rank(batch):
    """ Slice batch input along sequence dimension into multiple chunks,
        which are parallelized across GPUs in a context parallel group.
    """

    # With causal masking, each token only attends to its prior tokens. Simply split
    # sequence into CP chunks can result in severe load imbalance. That's to say, chunks
    # at the end of sequence have bigger workload than others. To address this issue,
    # we split sequence into 2*CP ranks. Assuming CP=2, we then get 4 chunks, chunk_0
    # and chunk_3 are assigned to GPU0, chunk_1 and chunk_2 are assigned to GPU1, so
    # that we can get balanced workload among GPUs in a context parallel group.
    args = get_args()
    tp_y_cp_size = TensorParallelYUnionCP().get_parallel_group_world_size() if args.tp_2d else args.context_parallel_size
    if not tp_y_cp_size > 1:
        return batch

    if args.cp_attention_mask_type == 'general' and batch.get("attention_mask", None) is not None:
        set_attention_mask(batch['attention_mask'].squeeze())

    cp_expanded_by_2d_tp = args.tp_y > 1
    if args.context_parallel_algo == 'megatron_cp_algo':
        if args.cp_attention_mask_type == 'general':
            batch = _get_batch_on_this_cp_rank_in_megatron_cp_general(batch)
        elif cp_expanded_by_2d_tp:
            batch = _get_batch_on_this_tp_y_cp_rank_in_megatron_cp(batch)
        else:
            batch = _get_batch_on_this_cp_rank_in_megatron_cp(batch)
    elif args.context_parallel_algo == 'ulysses_cp_algo':
        batch = _get_batch_on_this_cp_rank_in_ulysses_cp(batch)
    elif args.context_parallel_algo == 'hybrid_cp_algo':
        if args.cp_attention_mask_type == 'general':
            batch = _get_batch_on_this_cp_rank_in_hybrid_cp_general(batch)
        else:
            batch = _get_batch_on_this_cp_rank_in_hybrid_cp(batch)
    elif args.context_parallel_algo == 'adaptive_cp_algo':
        batch = _get_batch_on_this_cp_rank_in_adaptive_cp(batch)
    elif args.context_parallel_algo == 'hybrid_adaptive_cp_algo':
        batch = _get_batch_on_this_cp_rank_in_hybrid_adaptive_cp(batch)
    return batch


def generate_adaptive_cp_grid_mask_by_user(cp_size):
    from mindspeed.utils import get_actual_seq_len
    from mindspeed.core.context_parallel.utils import set_adaptive_cp_grid_mask_by_user
    args = get_args()
    actual_seq_len = get_actual_seq_len()
    seq_length = args.seq_length
    grid_mask = torch.zeros(cp_size, cp_size)
    sub_seq_length = seq_length // cp_size

    grid_actual_seq_len_dict = {}
    for seq_len in actual_seq_len:
        grid_actual_seq_len_dict[seq_len // sub_seq_length + 1] = seq_len % sub_seq_length == 0
    grid_actual_seq_len = list(grid_actual_seq_len_dict.items())
    start_index = 0
    for i, _ in enumerate(grid_actual_seq_len):
        end_index = grid_actual_seq_len[i][0]
        grid_mask[start_index:end_index, start_index:end_index] = 1

        if i != 0:
            if grid_actual_seq_len[i - 1][1]:
                start_index = grid_actual_seq_len[i - 1][0] - 1
            else:
                start_index = grid_actual_seq_len[i - 1][0]
    grid_mask = torch.tril(grid_mask)
    set_adaptive_cp_grid_mask_by_user(grid_mask)


def generate_adaptive_cp_mask_list_by_user(opt_seq, opt_scheduling, cp_rank, cp_size):
    from mindspeed.utils import get_actual_seq_len
    from mindspeed.core.context_parallel.utils import set_adaptive_cp_mask_list_by_user
    actual_seq_len = get_actual_seq_len()
    round_num = len(opt_scheduling)
    grid_size = (opt_seq[-1] + 1) // cp_size
    mask_list = []
    for rnd_idx in range(round_num):
        task_id = opt_scheduling[rnd_idx][cp_rank]
        if task_id == -1:
            mask_list.append(None)
            continue
        rank_x = task_id // cp_size
        rank_y = task_id % cp_size
        if rank_x == rank_y:
            mask = torch.tril(torch.ones((grid_size, grid_size), device=torch.npu.current_device()))
            for i in actual_seq_len:
                if i - 1 < grid_size * rank_y:
                    continue
                elif i - 1 >= grid_size * (rank_y + 1):
                    break
                else:
                    mask[(i - 1 - grid_size * cp_rank + 1):, : (i - 1 - grid_size * cp_rank + 1)] = 0
        elif cp_rank > rank_y:
            mask = torch.zeros((grid_size, grid_size), device=torch.npu.current_device())
            start_index = 0
            end_index = grid_size
            for i in actual_seq_len:
                if i - 1 < grid_size * rank_y:
                    start_index = i - 1
                    continue
                elif i - 1 >= grid_size * (rank_y + 1):
                    end_index = i - 1
                    break
                else:
                    start_index = i - 1
            start_index -= rank_y * grid_size
            if start_index < 0:
                start_index = 0
            elif start_index > grid_size:
                start_index = grid_size
            end_index -= cp_rank * grid_size
            if end_index < 0:
                end_index = 0
            elif end_index > grid_size:
                end_index = grid_size
            mask[: (end_index + 1), (start_index + 1):] = 1
        else:
            mask = torch.zeros((grid_size, grid_size), device=torch.npu.current_device())
        if mask is not None:
            # Convert attention mask to binary:
            mask = mask < 0.5
        mask_list.append(mask)
    set_adaptive_cp_mask_list_by_user(mask_list)


def _get_batch_on_this_cp_rank_in_megatron_cp_general(batch):
    cp_rank = mpu.get_context_parallel_rank()
    cp_size = mpu.get_context_parallel_world_size()
    for key, val in batch.items():
        if key == 'attention_mask' and val is not None:
            seq_dim = 2 if len(val.shape) == 4 else 0
            mask_row = val.chunk(cp_size, dim=seq_dim)[cp_rank].contiguous()
            mask_list = [m.contiguous() for m in mask_row.chunk(cp_size, dim=seq_dim + 1)]
            batch[key] = mask_list
            continue
        if val is not None:
            seq_dim = 1 
            val = val.chunk(cp_size, dim=seq_dim)[cp_rank].contiguous()
            batch[key] = val 
    
    return batch


def tensor_slide(
        tensor: Optional[torch.Tensor],
        slice_num: int,
        dims: Union[int, List[int]] = -1,
        step: int = 1,
        return_first=False,
) -> List[Union[torch.Tensor, None]]:
    """通用滑动窗口函数，支持任意维度"""
    if tensor is None:
        # return `List[None]` to avoid NoneType Error
        return [None] * (slice_num + 1)
    if slice_num == 0:
        return [tensor]
    window_size = tensor.shape[-1] - slice_num
    dims = [dims] if isinstance(dims, int) else sorted(dims, reverse=True)

    # 连续多维度滑动
    slices = []
    for i in range(0, tensor.size(dims[-1]) - window_size + 1, step):
        slice_obj = [slice(None)] * tensor.dim()
        for dim in dims:
            slice_obj[dim] = slice(i, i + window_size)
        slices.append(tensor[tuple(slice_obj)])
        if return_first:
            return slices
    return slices
