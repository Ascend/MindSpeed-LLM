# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
import hashlib

import torch
import torch.distributed as dist
from megatron.training import get_args
from megatron.core import parallel_state as mpu

import mindspeed_llm.tasks.posttrain.rlxf.training.parallel_state as ps


RECEIVE_PARAM_NUMS = None
MODEL_SEND_GROUP = None
MODEL_RECEIVE_GROUPS = []


def init_comm_groups():
    """
    Initialize model auto-mapping communication groups
    Scenario example:
    Training: tp 2 pp 2
    Inference: tp 2 dp 2
    Principle: Same tp rank weight data is sequentially broadcast to different dp on the inference side,
    and aggregated across different dp for training pp dimension.
    Number of communication groups: tp * pp
    groups = [[0,4,6],[2,4,6],[1,5,7],[3,5,7]]
    """
    args = get_args()
    pipeline_parallel_groups = []
    data_parallel_groups = []
    for i in range(args.tensor_model_parallel_size):
        ranks = list(range(i, args.num_gpus_for_train, args.tensor_model_parallel_size))
        pipeline_parallel_groups.append(ranks)
        ranks = list(range(args.num_gpus_for_train + i, args.num_gpus_for_train + args.num_gpus_for_infer,
                           args.tensor_model_parallel_size))
        data_parallel_groups.append(ranks)
    comm_groups = []
    for data_parallel_group, pipeline_parallel_group in zip(data_parallel_groups, pipeline_parallel_groups):
        for rank in pipeline_parallel_group:
            tmp_group = [rank] + list(data_parallel_group)
            comm_groups.append(tmp_group)
    print(comm_groups)
    return comm_groups


def init_parameter_mapping_distributed():
    """
    Automatically mapping communication groups for model initialization
    Scenario example:
    Training: tp 2 pp 2
    Inference: tp 2 dp 2
    Given a world size of 8 as above
    rank 0 acts as the sender, the communication group is [0, 4, 6]
    rank 2 acts as the sender, the communication group is [2, 4, 6]
    rank 4 acts as the receiver, the communication group is [[0, 4, 6], [2, 4, 6]]
    """
    global MODEL_SEND_GROUP, MODEL_RECEIVE_GROUPS
    groups = init_comm_groups()
    rank = dist.get_rank()
    for group in groups:
        tmp_group = dist.new_group(group)
        if not ps.in_mg2_inference_group() and rank in group:
            MODEL_SEND_GROUP = tmp_group
        if ps.in_mg2_inference_group() and rank in group:
            MODEL_RECEIVE_GROUPS.append((group[0], tmp_group))
    print("init_distributed_sucess")


def get_model_send_group():
    return MODEL_SEND_GROUP


def get_model_receive_groups():
    return MODEL_RECEIVE_GROUPS


def get_receive_param_nums():
    return RECEIVE_PARAM_NUMS


def param_nums_is_initialized():
    return get_receive_param_nums() is not None


def sync_param_nums(moudle: torch.nn.Module):
    """
    Synchronize the number of parameters for sending and receiving, ensuring that
    the number of broadcast communications aligns and the model parameters align.
    """
    args = get_args()
    if not ps.in_mg2_inference_group():
        args_need_broadcast = torch.tensor([args.iteration, args.consumed_train_samples], dtype=torch.int64,
                                           device=torch.cuda.current_device())
        dist.broadcast(args_need_broadcast, group=get_model_send_group(), src=dist.get_rank())
        num_parameters = torch.tensor([sum(1 for _ in moudle.named_parameters())], dtype=torch.int64,
                                      device=torch.cuda.current_device())
        dist.broadcast(num_parameters, group=get_model_send_group(), src=dist.get_rank())
    else:
        global RECEIVE_PARAM_NUMS

        recv_param_nums = []
        for group in get_model_receive_groups():
            args_need_broadcast = torch.empty(2, dtype=torch.int64, device=torch.cuda.current_device())
            dist.broadcast(args_need_broadcast, group=group[1], src=group[0], async_op=True)
            args.iteration, args.consumed_train_samples = args_need_broadcast

            tmp_num_parameters = torch.empty(1, dtype=torch.int64, device=torch.cuda.current_device())
            dist.broadcast(tmp_num_parameters, group=group[1], src=group[0], async_op=True)
            recv_param_nums.append(tmp_num_parameters)
        RECEIVE_PARAM_NUMS = recv_param_nums


def compute_model_hash(model, hash_func):
    hash_value = hash_func()
    for param in model.parameters():
        param_bytes = param.data.cpu().numpy().tobytes()
        hash_value.update(param_bytes)
    md5_tensor = torch.tensor([int(h, 16) for h in hash_value.hexdigest()])
    return md5_tensor


def send_model_to_infer_model(moudle: torch.nn.Module):
    """
    Decompose model information and transfer it directly from the model.
    """
    args = get_args()
    model_send_group = get_model_send_group()
    if args.md5_validate:
        hash_value = hashlib.md5()

    is_reuse_output_weights = (not args.untie_embeddings_and_output_weights and
                               args.pipeline_model_parallel_size >= 2 and
                               mpu.is_pipeline_last_stage(ignore_virtual=True))
    for name, param in moudle.named_parameters():
        if is_reuse_output_weights and 'output_layer.weight' in name:
            continue
        param_info_data = param.data
        dist.broadcast(param_info_data, group=model_send_group, src=dist.get_rank(), async_op=True)
        if args.md5_validate:
            param_bytes = param_info_data.to(torch.float32).cpu().numpy().tobytes()
            hash_value.update(param_bytes)
    if args.md5_validate:
        md5_tensor = torch.tensor([int(h, 16) for h in hash_value.hexdigest()], dtype=torch.int64,
                                  device=torch.cuda.current_device())
        dist.broadcast(md5_tensor, group=model_send_group, src=dist.get_rank(), async_op=True)


def recv_model_from_train_model(moudle: torch.nn.Module):
    """
    Decompose model parameters and directly loaded them into the model.
    """
    args = get_args()
    model_receive_groups = get_model_receive_groups()
    recv_param_nums = get_receive_param_nums()
    flag = True
    idx = 0
    if args.md5_validate:
        hash_value = hashlib.md5()
    for _, param in moudle.named_parameters():
        if flag:
            cur_num = int(recv_param_nums[idx])
            cur_group = model_receive_groups[idx]
            flag = False

        param_info_data = param.data
        torch.distributed.broadcast(param_info_data, group=cur_group[1], src=cur_group[0], async_op=False)
        if args.md5_validate:
            param_bytes = param_info_data.to(torch.float32).cpu().numpy().tobytes()
            hash_value.update(param_bytes)

        cur_num -= 1
        if cur_num == 0:
            if args.md5_validate:
                md5_tensor = torch.tensor([int(h, 16) for h in hash_value.hexdigest()], dtype=torch.int64,
                                          device=torch.cuda.current_device())
                md5_tensor_src = torch.zeros_like(md5_tensor, dtype=torch.int64, device=torch.cuda.current_device())
                dist.broadcast(md5_tensor_src, group=cur_group[1], src=cur_group[0], async_op=False)
                if torch.equal(md5_tensor_src, md5_tensor):
                    print("MD5 Hash: The weights of the two models match.")
                else:
                    print("MD5 Hash: The weights of the two models do not match.")
                hash_value = hashlib.md5()
            flag = True
            idx += 1

    if cur_num != 0:
        if args.md5_validate:
            md5_tensor = torch.tensor([int(h, 16) for h in hash_value.hexdigest()], dtype=torch.int64,
                                      device=torch.cuda.current_device())
            md5_tensor_src = torch.zeros_like(md5_tensor, dtype=torch.int64, device=torch.cuda.current_device())
            dist.broadcast(md5_tensor_src, group=cur_group[1], src=cur_group[0], async_op=False)
            if torch.equal(md5_tensor_src, md5_tensor):
                print("MD5 Hash: The weights of the two models match.")
            else:
                print("MD5 Hash: The weights of the two models do not match.")


def run_auto_mapping(model):
    if ps.in_mg2_inference_group():
        recv_model_from_train_model(model)
    else:
        send_model_to_infer_model(model)
