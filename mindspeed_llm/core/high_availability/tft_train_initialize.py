# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import os
from megatron.training import get_args
from .tft_replica_group import ttp_get_replica_dp_num, ttp_get_dp_cp_ranks, ttp_get_dp_ep_ranks, \
    build_local_embedding_group, ttp_get_dp_ranks, reinit_node_group
from .tft_arf_group_repair import arf_rebuild_process_group_callback, update_arf_reboot_flag, tft_is_arf_reboot_node
from .tft_dump import tft_save_callback, tft_rename_callback
from .tft_stop_clean import stop_callback, clean_callback, torch_sync
from .tft_optimizer_data_repair import repair_callback, set_build_data_args
from .tft_rollback import rollback_callback
from logging import getLogger

from mindio_ttp.framework_ttp import (tft_exception_handler, tft_init_controller, tft_start_controller, tft_init_processor,
                             tft_start_processor, tft_register_rename_handler, set_mindio_export_version,
                             tft_register_save_ckpt_handler, tft_set_optimizer_replica, tft_set_dp_group_info,
                             tft_register_stop_handler, tft_register_clean_handler, tft_register_repair_handler,
                             tft_register_rollback_handler, tft_register_rebuild_group_handler, tft_is_reboot_node,
                             tft_register_stream_sync_handler)

ttp_logger = getLogger(__name__)
REPLICA_OFFSET = 0


def tft_init_controller_processor():
    args = get_args()
    default_ip = '127.0.0.1'
    ttp_ip = os.getenv('TTP_ADDR', default_ip)
    controller_ip = os.getenv('CONTROLLER_ADDR', default_ip)
    if controller_ip == default_ip:
        controller_ip = ttp_ip
    processor_ip = os.getenv('PROCESSOR_ADDR', default_ip)
    if processor_ip == default_ip:
        processor_ip = ttp_ip
    port = 8000

    cur_rank = args.rank
    world_size = args.world_size

    enable_worker_reboot = args.enable_worker_reboot if hasattr(args, 'enable_worker_reboot') else False
    enable_hbmfault_repair = args.enable_hbmfault_repair if hasattr(args, 'enable_hbmfault_repair') else False
    enable_elastic_training = args.enable_elastic_training if hasattr(args, 'enable_elastic_training') else False

    enable_mindx = os.getenv('MINDX_TASK_ID')
    if cur_rank == 0 and enable_mindx is None:
        tft_init_controller(cur_rank, world_size, False, enable_worker_reboot, enable_elastic_training)
        tft_start_controller(controller_ip, port, False, '')
    tft_init_processor(cur_rank, world_size, False, False, '',
                       enable_hbmfault_repair, enable_worker_reboot, enable_elastic_training)
    tft_start_processor(processor_ip, port)
    if tft_is_reboot_node():
        update_arf_reboot_flag(True)


def tft_register_processor(train_valid_test_dataset_provider, model_provider, model_type):
    args = get_args()
    replica_info = []
    dp_cp_ranks = ttp_get_dp_cp_ranks()
    dp_ranks = ttp_get_dp_ranks()
    dense_replica_cnt = ttp_get_replica_dp_num() if args.use_distributed_optimizer else len(dp_cp_ranks)
    replica_offset = REPLICA_OFFSET
    moe_flag = args.expert_model_parallel_size > 1
    cur_rank = args.rank

    replica_dict = {
        "rank_list": dp_cp_ranks,
        "replica_cnt": dense_replica_cnt,
        "replica_shift": replica_offset
    }
    replica_info.append(replica_dict)

    if moe_flag:
        dp_ep_ranks = ttp_get_dp_ep_ranks()
        moe_replica_cnt = ttp_get_replica_dp_num() if args.use_distributed_optimizer else len(dp_ep_ranks)
        replica_dict = {
            "rank_list": dp_ep_ranks,
            "replica_cnt": moe_replica_cnt,
            "replica_shift": replica_offset
        }
        replica_info.append(replica_dict)

    build_local_embedding_group()
    set_build_data_args(model_type, model_provider, train_valid_test_dataset_provider)
    tft_set_optimizer_replica(cur_rank, replica_info)
    tft_set_dp_group_info(cur_rank, dp_ranks)
    tft_register_save_ckpt_handler(tft_save_callback)
    tft_register_rename_handler(tft_rename_callback)
    tft_register_stop_handler(stop_callback)
    tft_register_clean_handler(clean_callback)
    tft_register_repair_handler(repair_callback)
    tft_register_rollback_handler(rollback_callback)
    tft_register_rebuild_group_handler(arf_rebuild_process_group_callback)
    tft_register_stream_sync_handler(torch_sync)

    reinit_node_group()


@tft_exception_handler
def tft_train(args: list, test_data_iterator: list):
    try:
        from mindspeed_llm.training.training import train
        # gitee MindSpeed-LLM/1.0.0
        set_mindio_export_version("MindSpeed-LLM")

    except ModuleNotFoundError:
        ttp_logger.error(f'Mindio only support MindSpeed-LLM!')
        raise

    if tft_is_arf_reboot_node():
        raise RuntimeError("ARF FINISH")

    return train(*args)
