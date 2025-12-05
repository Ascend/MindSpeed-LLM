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
import time
import stat
import random
import warnings
from functools import wraps
from logging import getLogger
from typing import Optional, Union, List
from itertools import takewhile
from packaging.version import Version as PkgVersion

import acl
import torch
import torch_npu
import socket
from torch import distributed as dist
import numpy as np
import megatron
from megatron.training import get_args
from megatron.core import mpu
from megatron.core.parallel_state import is_pipeline_last_stage
from tqdm import tqdm

from mindspeed.core.context_parallel.utils import pad_data
from mindspeed.core.context_parallel.get_batch_utils import  set_actual_seq_len
from mindspeed.utils import ( set_position_ids,
                             _get_batch_on_this_cp_rank_in_megatron_cp,
                             _get_batch_on_this_cp_rank_in_hybrid_cp_general,
                             _get_batch_on_this_cp_rank_in_hybrid_cp,
                             broadcast_dynamic, _broadcast, get_ring_degree)
from mindspeed.core.tensor_parallel_y_union_cp import TensorParallelYUnionCP
from mindspeed.model.transformer import set_attention_mask
from mindspeed.utils import _get_batch_on_this_tp_y_cp_rank_in_megatron_cp
from mindspeed_llm.tasks.dataset.shared_memory_manager import SharedMemoryManager

try:
    from mindspeed.core.pipeline_parallel.dualpipev.dualpipev_schedules import get_post_process_flag
except Exception:
    pass

try:
    _torch_version = PkgVersion(torch.__version__)
except Exception:
    # This is a WAR for building docs, where torch is not actually imported
    _torch_version = PkgVersion("0.0.0")


logger = getLogger(__name__)


WRITE_FILE_DEFAULT_FLAGS = os.O_WRONLY | os.O_CREAT
WRITE_FILE_DEFAULT_MODES = stat.S_IWUSR | stat.S_IRUSR

_MTP_POSITION_ID = None
_MTP_BATCH_LIST = None

_ACTUAL_SEQ_LEN_LIST = None
_ACTUAL_ATTN_RATIO_LIST = []
_ACTUAL_COUNT = 0


def get_attn_ratio(actual_seq_len, seq_length):
    first_seq_list = np.array(actual_seq_len[:-1])
    last_seq_list = np.array(actual_seq_len[1:])

    seq_list_without_first = (last_seq_list - first_seq_list).tolist()
    seq_length_list = np.array([actual_seq_len[0], ] + seq_list_without_first)
    ratio = 0.5 * sum(seq_length_list * seq_length_list) / (seq_length * seq_length)

    return ratio


def clear_actual_attn_ratio():
    global _ACTUAL_ATTN_RATIO_LIST, _ACTUAL_COUNT
    _ACTUAL_ATTN_RATIO_LIST = []
    _ACTUAL_COUNT = 0


def set_actual_seq_len_list(actual_seq_len):
    global _ACTUAL_SEQ_LEN_LIST, _ACTUAL_ATTN_RATIO_LIST, _ACTUAL_COUNT
    _ACTUAL_SEQ_LEN_LIST = actual_seq_len

    args = get_args()
    if actual_seq_len is not None and args.log_throughput and is_pipeline_last_stage():
        actual_attn_ratio = get_attn_ratio(actual_seq_len, args.seq_length)
        _ACTUAL_ATTN_RATIO_LIST.append(actual_attn_ratio)
        _ACTUAL_COUNT += 1


def get_actual_seq_len_list():
    global _ACTUAL_SEQ_LEN_LIST
    return _ACTUAL_SEQ_LEN_LIST


def get_actual_attn_ratio():
    global _ACTUAL_ATTN_RATIO_LIST, _ACTUAL_COUNT
    return _ACTUAL_ATTN_RATIO_LIST, _ACTUAL_COUNT


def set_mtp_batch_list(mtp_batch_list):
    global _MTP_BATCH_LIST
    _MTP_BATCH_LIST = mtp_batch_list


def get_mtp_batch_list():
    """Get mtp_batch_list"""
    global _MTP_BATCH_LIST
    return _MTP_BATCH_LIST


def set_mtp_position_ids(position_ids_mtp):
    """set_postprocess_chunk for mtp position id"""
    global _MTP_POSITION_ID
    _MTP_POSITION_ID = position_ids_mtp


def get_torch_version():
    """Get torch version from __version__."""

    global _torch_version
    return _torch_version


def get_mtp_position_ids():
    global _MTP_POSITION_ID
    if _MTP_POSITION_ID is not None:
        return _MTP_POSITION_ID
    else:
        raise AssertionError("_MTP_POSITION_ID is None")


def _compute_actual_seq_len(origin_seq):
    seq = origin_seq.view(-1)
    zero_pos = (seq == 0).nonzero()[1:].squeeze(dim=1)
    res = zero_pos.tolist()
    res.append(len(seq))
    return res


def recompute_valid_actual_seq_len(actual_seq_len, micro_batch_size):
    if len(actual_seq_len) <= 1:
        return actual_seq_len
    s = torch.tensor(actual_seq_len)
    diffs = s[1:] - s[:-1]
    indices = (diffs == 1).nonzero()
    if len(indices) < micro_batch_size:
        return s
    first_continuous = indices[micro_batch_size - 1].item()
    return torch.cat([s[:first_continuous + 1], s[-1:]]).tolist()


def compute_actual_seq_len(origin_seq):
    args = get_args()
    actual_seq_len = _compute_actual_seq_len(origin_seq)
    if args.mtp_num_layers:
        seq_len = origin_seq.shape[1]
        mtp_res = [actual_seq_len]
        for i in range(1, args.mtp_num_layers + 1):
            next_actual_seq_len = []
            for j in actual_seq_len:
                if j % seq_len == 0:
                    next_actual_seq_len.append(j)
                else:
                    next_actual_seq_len.append(j - i)
            mtp_res.append(next_actual_seq_len)
        return mtp_res
    return actual_seq_len





def regenerate_position_ids(tensor, offset):
    if tensor is None:
        return None
    tensor = tensor.clone()
    for i in range(tensor.size(0)):
        row = tensor[i]
        zero_mask = (row == 0)
        if zero_mask.any():
            first_zero_idx = torch.argmax(zero_mask.int()).item()
            tensor[i, :first_zero_idx] = torch.arange(first_zero_idx)
        else:
            tensor = tensor - offset
    return tensor


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

    if args.stage in ['dpo']:
        micro_batch_size = attention_mask_1d.shape[0] // 2
        attention_mask_1d = attention_mask_1d[:micro_batch_size]

    attention_mask = torch.ones((micro_batch_size, seq_length, seq_length),
                                 device=attention_mask_1d.device,
                                 dtype=torch.bool).tril_().view(micro_batch_size, 1, seq_length, seq_length)

    if args.tokenizer_padding_side == "left":
        attention_mask_1d = attention_mask_1d.view(seq_length, 1, -1)

    attention_mask = attention_mask.masked_fill_(attention_mask_1d.bool().bitwise_not_().view(-1, 1, 1, seq_length), value=0)
    attention_mask.bitwise_not_()

    return attention_mask


def get_batch_on_this_cp_rank_wrapper(fn):
    @wraps(fn)
    def wrapper(batch):
        batch = fn(batch)
        args = get_args()
        if 'position_ids' in batch:
            if args.reset_position_ids:
                set_position_ids(batch['position_ids'].transpose(0, 1).contiguous())
            else:
                set_position_ids(batch['position_ids'])
        return batch

    return wrapper

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
    args = get_args()
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


_GLOBAL_SHM_MANAGER = None  # Shared Memory Manager Instance
_SHM_SKIP_FLAG = False  # Whether to not use shared memory
BASE_SHM_NAME = "g_shm"


def reset_sharedmem_mgr():
    """
    Reset the shared memory manager and status flags.
    """
    global _GLOBAL_SHM_MANAGER, _SHM_SKIP_FLAG

    if _GLOBAL_SHM_MANAGER is not None:
        try:
            _GLOBAL_SHM_MANAGER.close()
        except Exception as e:
            print(f"[SharedMemoryManager] [WARN] Error during SharedMemoryManager shutdown: {e}")

    _GLOBAL_SHM_MANAGER = None
    _SHM_SKIP_FLAG = False


def get_sharedmem_mgr(base_shm_name="g_shm", buffer_length=4096):
    """
    Retrieve the global shared memory manager for data transfer through shared memory.
    :param base_shm_name: Base name of the shared memory
    :param buffer_length: Size of the shared memory buffer, default: 4K
    :return: `SharedMemoryManager` instance
    """
    global _GLOBAL_SHM_MANAGER, _SHM_SKIP_FLAG

    if _SHM_SKIP_FLAG:
        return None

    if _GLOBAL_SHM_MANAGER is not None:
        return _GLOBAL_SHM_MANAGER

    rank = mpu.get_tensor_model_parallel_rank()
    global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1

    if not torch.distributed.is_initialized():
        print(
            f"[SharedMemoryManager][Rank {rank}][global_rank {global_rank}]"
            f"[Func: get_sharedmem_mgr] <ERROR> "
            f"torch.distributed not initialized, skipping..."
        )
        return None

    args = get_args()
    reset_position_ids = args.reset_position_ids
    enable_shm = args.enable_share_memory
    tp_size = mpu.get_tensor_model_parallel_world_size()
    device_count = torch.cuda.device_count()

    if not (reset_position_ids and enable_shm and tp_size > 1 and tp_size <= device_count):
        print(
            f"[SharedMemoryManager][Rank {rank}][global_rank {global_rank}]"
            f"[Func: get_sharedmem_mgr] <INFO> Skip creation. "
            f"reset_position_ids={reset_position_ids}, enable_shm={enable_shm}, "
            f"tp_size={tp_size}, device_count={device_count}"
        )
        _SHM_SKIP_FLAG = True
        return None

    if rank == 0:
        pid = os.getpid()
        _GLOBAL_SHM_MANAGER = SharedMemoryManager(
            base_shm_name, rank0_pid=pid, buffer_length=buffer_length, tp_size=tp_size
        )
        print(
            f"[SharedMemoryManager][Rank {rank}][global_rank {global_rank}] <INFO> Created: "
            f"{_GLOBAL_SHM_MANAGER.shm_name}, TP_size: {tp_size}, TP_Group: {_GLOBAL_SHM_MANAGER.tp_group_id}"
        )

    try:
        torch.distributed.barrier(group=mpu.get_tensor_model_parallel_group())
    except RuntimeError as e:
        print(
            f"[SharedMemoryManager][Rank {rank}][global_rank {global_rank}]"
            f"[Func: get_sharedmem_mgr] <ERROR> Barrier timeout: {e}"
        )

    if rank == 0:
        pid = os.getpid()
        pid_tensor = torch.tensor([pid], dtype=torch.int32, device="cuda")
        torch.distributed.broadcast(pid_tensor, mpu.get_tensor_model_parallel_src_rank(),
                                    group=mpu.get_tensor_model_parallel_group())
    else:
        pid_tensor = torch.zeros(1, dtype=torch.int32, device="cuda")
        torch.distributed.broadcast(pid_tensor, mpu.get_tensor_model_parallel_src_rank(),
                                    group=mpu.get_tensor_model_parallel_group())
        pid = pid_tensor.item()
        _GLOBAL_SHM_MANAGER = SharedMemoryManager(
            base_shm_name, rank0_pid=pid, buffer_length=buffer_length, tp_size=tp_size, existing=True
        )
        print(
            f"[SharedMemoryManager][Rank {rank}][global_rank {global_rank}] <INFO> Connected to: "
            f"{_GLOBAL_SHM_MANAGER.shm_name}, TP_size: {tp_size}, TP_Group: {_GLOBAL_SHM_MANAGER.tp_group_id}"
        )

    torch.distributed.barrier(group=mpu.get_tensor_model_parallel_group())
    return _GLOBAL_SHM_MANAGER


def get_batch_on_this_tp_rank(data_iterator):
    args = get_args()

    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(item, mpu.get_tensor_model_parallel_src_rank(),
                                        group=mpu.get_tensor_model_parallel_group())

    shm_manager = None
    actual_seq_len = None
    if args.enable_share_memory:
        shm_manager = get_sharedmem_mgr(BASE_SHM_NAME, args.micro_batch_size * args.seq_length)

    if mpu.get_tensor_model_parallel_rank() == 0:
        if data_iterator is not None:
            data = next(data_iterator)
        else:
            data = None

        if args.enable_share_memory and shm_manager is not None:
            position_ids = data["position_ids"]
            actual_seq_len = compute_actual_seq_len(position_ids)
            shm_manager.write(actual_seq_len)

            if '910B' not in acl.get_soc_name() and args.mtp_num_layers and get_post_process_flag():
                from mindspeed_llm.core.transformer.multi_token_prediction import roll_tensor
                position_ids_mtp = []
                cur_position_id = data["position_ids"]
                for _ in range(args.mtp_num_layers):
                    cur_position_id, _ = roll_tensor(cur_position_id, shifts=-1, dims=-1)
                    cur_position_id = regenerate_position_ids(cur_position_id, 1)
                    position_ids_mtp.append(cur_position_id)
                set_mtp_position_ids((position_ids_mtp, shm_manager))

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
            # Multi-Token Prediction (MTP) layers need tokens and position_ids to calculate embedding.
            # Currently the Multi-Token Prediction (MTP) layers is fixed on the last stage, so we need
            # to broadcast tokens and position_ids to all of the tensor parallel ranks on the last stage.
            if args.mtp_num_layers or args.schedules_method == 'dualpipev':
                _broadcast(batch['tokens'])
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
            if args.reset_attention_mask  or args.mtp_num_layers or args.schedules_method == 'dualpipev':
                _broadcast(batch['position_ids'])
        elif args.reset_attention_mask:
            _broadcast(batch['position_ids'])
        else:
            _broadcast(batch['attention_mask'])
        if args.reset_attention_mask:
            actual_seq_len = broadcast_dynamic(data['actual_seq_len'])
            if args.attention_mask_type == 'causal' \
              and args.context_parallel_size > 1 \
              and args.context_parallel_algo == 'megatron_cp_algo':
                actual_seq_len = pad_data(actual_seq_len, batch, args.context_parallel_size, args.tensor_model_parallel_size)
                actual_seq_len /= get_ring_degree()
            set_actual_seq_len(actual_seq_len)

    else:
        if args.enable_share_memory and shm_manager is not None:
            actual_seq_len = shm_manager.read()
            if '910B' not in acl.get_soc_name() and args.mtp_num_layers and get_post_process_flag():
                set_mtp_position_ids((None, shm_manager))

        tokens = torch.empty((args.micro_batch_size, args.seq_length),
                             dtype=torch.int64,
                             device=torch.cuda.current_device())
        labels = torch.empty((args.micro_batch_size, args.seq_length),
                             dtype=torch.int64,
                             device=torch.cuda.current_device())
        loss_mask = torch.empty((args.micro_batch_size, args.seq_length),
                                dtype=torch.float32,
                                device=torch.cuda.current_device())
        if getattr(args, 'create_attention_mask_in_dataloader', False):
            attention_mask = torch.empty(
                (args.micro_batch_size, 1, args.seq_length, args.seq_length), dtype=torch.bool, device=torch.cuda.current_device()
            )
        else:
            attention_mask = None
        position_ids = torch.empty((args.micro_batch_size, args.seq_length),
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
            if args.mtp_num_layers or args.schedules_method == 'dualpipev':
                _broadcast(tokens)
            else:
                tokens = None
            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            if args.reset_attention_mask or args.mtp_num_layers or args.schedules_method == 'dualpipev':
                _broadcast(position_ids)
            else:
                position_ids = None

        else:
            tokens = None
            labels = None
            loss_mask = None
            _broadcast(attention_mask)
            if args.reset_attention_mask:
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
        if args.reset_attention_mask:
            actual_seq_len = broadcast_dynamic(None)
            if args.attention_mask_type == 'causal' \
                    and args.context_parallel_size > 1 \
                    and args.context_parallel_algo == 'megatron_cp_algo':
                actual_seq_len = pad_data(actual_seq_len, batch, args.context_parallel_size,
                                          args.tensor_model_parallel_size)
                actual_seq_len /= get_ring_degree()
            set_actual_seq_len(actual_seq_len)
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

    if args.attention_mask_type == 'general' and batch.get("attention_mask", None) is not None:
        set_attention_mask(batch['attention_mask'].squeeze())

    cp_expanded_by_2d_tp = args.tp_y > 1
    if args.context_parallel_algo == 'megatron_cp_algo':
        if args.attention_mask_type == 'general':
            batch = _get_batch_on_this_cp_rank_in_megatron_cp_general(batch)
        elif cp_expanded_by_2d_tp:
            batch = _get_batch_on_this_tp_y_cp_rank_in_megatron_cp(batch)
        else:
            batch = _get_batch_on_this_cp_rank_in_megatron_cp(batch)
    elif args.context_parallel_algo == 'ulysses_cp_algo' or args.context_parallel_algo == 'mamba_cp_algo':
        batch = _get_batch_on_this_cp_rank_in_ulysses_cp(batch)
    elif args.context_parallel_algo == 'hybrid_cp_algo':
        if args.attention_mask_type == 'general':
            batch = _get_batch_on_this_cp_rank_in_hybrid_cp_general(batch)
        else:
            batch = _get_batch_on_this_cp_rank_in_hybrid_cp(batch)
    return batch


def _get_batch_on_this_cp_rank_in_megatron_cp_general(batch):
    cp_rank = mpu.get_context_parallel_rank()
    cp_size = mpu.get_context_parallel_world_size()
    for key, val in batch.items():
        if key == 'attention_mask' and val is not None:
            seq_dim = 2 if len(val.shape) == 4 else 0
            mask_row = val.chunk(cp_size, dim=seq_dim)[cp_rank].contiguous()
            mask_tensor = torch.stack([m.contiguous() for m in mask_row.chunk(cp_size, dim=seq_dim + 1)])
            batch[key] = mask_tensor
            continue
        if val is not None:
            seq_dim = 1
            val = val.chunk(cp_size, dim=seq_dim)[cp_rank].contiguous()
            batch[key] = val

    return batch


def _get_batch_on_this_cp_rank_in_ulysses_cp(batch):
    cp_rank = mpu.get_context_parallel_rank()
    cp_size = mpu.get_context_parallel_world_size()
    for key, val in batch.items():
        if key == 'attention_mask':
            continue
        if val is not None:
            seq_dim = 1 if key != 'attention_mask' else 2
            val = val.chunk(cp_size, dim=seq_dim)[cp_rank].contiguous()
            batch[key] = val

    return batch


def is_last_rank_wrapper(fn):
    @wraps(fn)
    def wrapper():
        """
        In the context of scale-in training scenarios, use the scale-in world group to determine
        if it is the last rank.
        """
        from mindspeed_llm.core.high_availability import elastic_training_common
        if not elastic_training_common.zit_scale_in_running_state():
            return fn()
        else:
            return torch.distributed.get_rank() == torch.distributed.get_process_group_ranks(
                group=elastic_training_common.zit_get_scale_in_world_group())[-1]
    return wrapper


def print_rank_last_wrapper(fn):
    @wraps(fn)
    def wrapper(message):
        """
        In the context of scale-in training scenarios, use the get_args().global_batch_size to
        replace the batch_size.
        """
        from mindspeed_llm.core.high_availability import elastic_training_common
        if elastic_training_common.zit_scale_in_running_state():
            args = get_args()
            from megatron.core.num_microbatches_calculator import get_num_microbatches
            batch_size = args.micro_batch_size * args.data_parallel_size * \
                         get_num_microbatches()
            src_str = f' global batch size: {batch_size:5d} |'
            batch_size = get_args().global_batch_size
            dest_str = f' global batch size: {batch_size:5d} |'
            message = message.replace(src_str, dest_str)
        return fn(message)
    return wrapper


def is_shared_path(path: str, retry: int = 3, wait: float = 0.5) -> bool:

    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        return True

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    hostname = socket.gethostname()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    os.makedirs(path, exist_ok=True)
    marker_file = os.path.join(path, f".share_test_{hostname}")

    try:
        if local_rank == 0:
            with open(marker_file, "w") as f:
                f.write(f"marker from {hostname}")
        torch.distributed.barrier()

        visible_files = set()
        for _ in range(retry):
            visible_files = {f for f in os.listdir(path) if f.startswith(".share_test_")}
            if len(visible_files) > 1 or world_size == 1:
                break
            time.sleep(wait)

        visible_count = len(visible_files)
        visible_tensor = torch.tensor(
            [visible_count],
            dtype=torch.int,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        torch.distributed.all_reduce(visible_tensor, op=torch.distributed.ReduceOp.MAX)
        total_visible = visible_tensor.item()

        if rank == 0:
            if total_visible > 1:
                logger.info(f"[is_shared_path] Detection result: Shared storage ({path}), detected {total_visible} node marker files.")
                shared = True
            elif total_visible == 1:
                logger.info(f"[is_shared_path] Detection result: Non-shared storage ({path}), only local node can access its own marker.")
                shared = False
            else:
                raise RuntimeError(f"[is_shared_path] Detection failed: No visible marker files, please check mount configuration.")
        else:
            shared = None

        shared = torch.tensor([1 if shared else 0], dtype=torch.int, device="cuda" if torch.cuda.is_available() else "cpu")
        torch.distributed.broadcast(shared, src=0)

        torch.distributed.barrier()
        if local_rank == 0 and os.path.exists(marker_file):
            os.remove(marker_file)
        torch.distributed.barrier()

        return bool(shared.item())

    except Exception as e:
        if rank == 0:
            logger.info(f"[is_shared_path] Exception during shared path check: {e}")
        raise