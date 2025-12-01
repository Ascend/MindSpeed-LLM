# Copyright (c) 2025; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

"""General utilities."""
import logging
from itertools import takewhile
import torch
import numpy as np
from megatron.training import get_args
from megatron.core import mpu
import acl
from mindspeed_llm.training.utils import (get_sharedmem_mgr, BASE_SHM_NAME, compute_actual_seq_len,
                                          set_mtp_position_ids, regenerate_position_ids)
from mindspeed.core.context_parallel.utils import pad_data
from mindspeed.core.context_parallel.get_batch_utils import  set_actual_seq_len
from mindspeed.utils import broadcast_dynamic, get_ring_degree

try:
    from mindspeed.core.pipeline_parallel.dualpipev.dualpipev_schedules import get_post_process_flag
except ImportError as e:
    logging.warning(f"Import failed: {e}")


def _compute_actual_seq_len(origin_seq):
    origin_seq_np = origin_seq.numpy()
    seq = origin_seq_np.reshape(-1)
    tmp = (seq == 0).nonzero()
    tmp_stack = np.stack(tmp, axis=1)
    tmp_squeeze = tmp_stack[1:].squeeze(axis=1)
    res = tmp_squeeze.tolist()

    res.append(len(seq))
    return res


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
                'tokens': data["tokens"],
                'labels': data["labels"],
                'loss_mask': data["loss_mask"],
                'attention_mask': None if "attention_mask" not in data else data["attention_mask"],
                'position_ids': data["position_ids"],
                'document_ids': document_ids,
                'idx': data_idx
            }
        else:
            batch = {
                'tokens': data["tokens"],
                'labels': data["labels"],
                'loss_mask': data["loss_mask"],
                'attention_mask': None if "attention_mask" not in data else data["attention_mask"],
                'position_ids': data["position_ids"]
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