#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import unittest
from datetime import timedelta
from unittest import mock
from unittest.mock import patch, MagicMock


class TestElasticTrainingScaleOutRebuild(unittest.TestCase):
    @patch('mindspeed_llm.core.high_availability.tft_arf_group_repair.get_args')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_out_rebuild.get_args')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_out_rebuild.get_timers')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_out_rebuild.rebuild_process_group')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_out_rebuild.update_model_and_optim_related_group')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_out_rebuild.torch.distributed.get_rank')
    def test_scale_out_rebuild_process_group_callback(self, *mocks):
        mock_get_rank, \
        mock_update_model_and_optim_related_group, mock_rebuild_process_group, \
        mock_get_timers, mock_get_args, mock_repair_get_args = mocks
        from mindspeed_llm.core.high_availability import (elastic_training_scale_out_rebuild, tft_arf_group_repair,
                                                          elastic_training_common)
        fault_ranks = [1, 2]
        mock_model = mock.MagicMock()
        mock_optimizer = mock.MagicMock()
        train_args = {0: {0: mock_model, 1: mock_optimizer}}
        params = '{"scale-out-strategy": "DP"}'
        mock_get_rank.return_value = 0
        mock_args = mock.MagicMock()
        mock_args.distributed_timeout_minutes = 30
        mock_args.nccl_communicator_config_path = None
        mock_get_args.return_value = mock_args
        mock_timer = mock.MagicMock()
        mock_timers_instance = mock.MagicMock()
        mock_timers_instance.return_value = mock_timer
        mock_get_timers.return_value = mock_timers_instance
        elastic_training_common.ORIGIN_DP_SIZE = 2
        # test args error
        try:
            elastic_training_scale_out_rebuild.scale_out_rebuild_process_group_callback(
                fault_ranks, train_args, params)
        except Exception as e:
            self.assertIn('train_args error:', str(e))

        # test normal
        tft_arf_group_repair.ARF_REBOOT_FLAG = False
        elastic_training_common.update_scale_in_flag(False)
        train_args = {0: {0: MagicMock(), 1: mock_model, 2: mock_optimizer}}
        elastic_training_scale_out_rebuild.scale_out_rebuild_process_group_callback(
            fault_ranks, train_args, params)
        self.assertEqual(mock_args.data_parallel_size, 2)
        self.assertFalse(tft_arf_group_repair.tft_is_arf_reboot_node())
        self.assertFalse(elastic_training_common.zit_scale_in_running_state())

    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_out_rebuild.init_data_parallel_with_cp_group')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_out_rebuild.destroy_sub_process_group')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_out_rebuild.destroy_all_process_group')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_out_rebuild.ttp_initialize_replica_dp_group')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_out_rebuild.init_data_parallel_group')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_out_rebuild.init_model_parallel_group')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_out_rebuild.init_tensor_parallel_group')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_out_rebuild.init_pipeline_parallel_group')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_out_rebuild.init_context_parallel_group')
    @patch('torch.distributed.get_rank')
    @patch('torch.distributed.init_process_group')
    def test_rebuild_process_group(self, *mocks):
        from mindspeed_llm.core.high_availability import elastic_training_scale_out_rebuild, elastic_training_common
        mock_init_process_group, mock_get_rank, \
        mock_init_context_parallel_group, mock_init_pipeline_parallel_group, mock_init_tensor_parallel_group, \
        mock_init_model_parallel_group, mock_init_data_parallel_group, mock_ttp_initialize_replica_dp_group, \
        mock_destroy_all_process_group, mock_destroy_sub_process_group, mock_init_data_parallel_with_cp_group = mocks
        mock_args = mock.MagicMock()
        mock_args.data_parallel_size = 2
        mock_args.tensor_model_parallel_size = 2
        mock_args.pipeline_model_parallel_size = 1
        mock_args.context_parallel_size = 1
        mock_args.world_size = 8
        mock_args.rank = 0
        mock_args.distributed_timeout_minutes = 1
        timeout = timedelta(minutes=30)
        mock_nccl_cfgs = {}
        mock_get_rank.return_value = 0
        elastic_training_common.SCALE_IN_DP_CP_REPLICA_GROUP = 1
        elastic_training_common.SCALE_IN_DP_CP_REPLICA_GROUP_GLOO = 1

        elastic_training_scale_out_rebuild.rebuild_process_group(mock_args, timeout, mock_nccl_cfgs)

        mock_init_process_group.assert_called_once()
        mock_destroy_all_process_group.assert_called_once()
        mock_ttp_initialize_replica_dp_group.assert_called_once()
        mock_init_data_parallel_group.assert_called_once()
        mock_init_model_parallel_group.assert_called_once()
        mock_init_tensor_parallel_group.assert_called_once()
        mock_init_pipeline_parallel_group.assert_called_once()
        mock_init_context_parallel_group.assert_called_once()
        mock_init_data_parallel_with_cp_group.assert_called_once()
        self.assertEqual(elastic_training_common.SCALE_IN_DP_CP_REPLICA_GROUP, None)
        self.assertEqual(elastic_training_common.SCALE_IN_DP_CP_REPLICA_GROUP_GLOO, None)
    
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_out_rebuild.torch.distributed.get_rank')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_out_rebuild.torch.distributed.get_world_size')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_out_rebuild.get_args')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_out_rebuild.torch.distributed.new_group')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_out_rebuild.destroy_sub_process_group')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_out_rebuild.build_dp_cp_replica_group')
    def test_ttp_initialize_replica_dp_group(self, *mocks):
        from mindspeed_llm.core.high_availability import elastic_training_scale_out_rebuild
        mock_build_dp_cp_replica_group, mock_destroy_sub_process_group, mock_new_group, mock_get_args, mock_get_world_size, mock_get_rank = mocks
        mock_get_rank.return_value = 0
        mock_get_world_size.return_value = 8
        mock_args_obj = mock.MagicMock()
        mock_args_obj.use_distributed_optimizer = True
        mock_get_args.return_value = mock_args_obj
        # test with correct parameters
        elastic_training_scale_out_rebuild.ttp_initialize_replica_dp_group(
            pipeline_model_parallel_size=1,
            tensor_model_parallel_size=2,
            context_parallel_size=1,
            world_size=8
        )
        # Verify build_dp_cp_replica_group is called with the correct dp_cp_ranks
        # In this case, with tensor_parallel_size=2, world_size=8, pipeline_parallel_size=1
        # The dp_cp_ranks for rank 0 should be [0, 2, 4, 6]
        from mindspeed_llm.core.high_availability import tft_replica_group
        self.assertEqual(tft_replica_group.DP_CP_ORIGIN_RANKS, [0, 2, 4, 6])
        mock_build_dp_cp_replica_group.assert_called_with([0, 2, 4, 6], 0)
    
    @patch('torch.distributed.get_rank')
    @patch('torch.distributed.new_group')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_out_rebuild.destroy_sub_process_group')
    @patch('mindspeed_llm.core.high_availability.tft_replica_group')
    def test_build_dp_cp_replica_group(self, *mocks):
        from mindspeed_llm.core.high_availability import elastic_training_scale_out_rebuild
        mock_tft_replica_group, mock_destroy_sub_process_group, mock_new_group, mock_get_rank = mocks
        mock_new_group.return_value = 1
        mock_get_rank.return_value = 0
        dp_cp_ranks = [0, 1, 2, 3]
        cur_rank = 0

        elastic_training_scale_out_rebuild.build_dp_cp_replica_group(dp_cp_ranks, cur_rank)

        self.assertEqual(mock_new_group.call_count, 2)
        self.assertEqual(mock_destroy_sub_process_group.call_count, 2)
        mock_tft_replica_group.DP_CP_REPLICA_GROUP_GLOO = 1
        mock_tft_replica_group.DP_CP_REPLICA_GROUP = 1

    @patch('torch.distributed.get_rank')
    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.new_group')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_out_rebuild.get_nccl_options')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_out_rebuild.destroy_sub_process_group')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_out_rebuild.init_data_parallel_with_cp_group')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_out_rebuild.elastic_training_common')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_out_rebuild.get_args')
    @patch('megatron.core')
    def test_init_data_parallel_group(self, *mocks):
        # 第一个参数现在是mock_megatron_core，我们需要从中获取mpu
        mock_megatron_core, mock_get_args, mock_elastic_common, mock_init_data_parallel_with_cp_group, \
        mock_destroy_sub_process_group, mock_get_nccl_options, mock_new_group, mock_get_world_size, mock_get_rank = mocks
        mock_mpu = mock_megatron_core.mpu
        from mindspeed_llm.core.high_availability import elastic_training_scale_out_rebuild
        mock_args = mock.MagicMock()
        mock_timeout = 300
        mock_nccl_comm_cfgs = {}
        mock_get_rank.return_value = 0
        mock_get_world_size.return_value = 8
        mock_args.tensor_model_parallel_size = 1
        mock_args.pipeline_model_parallel_size = 1
        mock_args.context_parallel_size = 1
        mock_get_args.return_value = mock_args
        mock_mpu._DATA_PARALLEL_GROUP = mock.MagicMock()
        mock_mpu._DATA_PARALLEL_GROUP_GLOO = mock.MagicMock()
        mock_group = mock.MagicMock()
        mock_group_gloo = mock.MagicMock()
        mock_new_group.side_effect = [mock_group, mock_group_gloo]
        elastic_training_scale_out_rebuild.init_data_parallel_group(mock_args, mock_timeout, mock_nccl_comm_cfgs)
        self.assertEqual(mock_new_group.call_count, 2)
        self.assertEqual(mock_destroy_sub_process_group.call_count, 2)
    
    @patch('torch.distributed.get_rank')
    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.new_group')
    def test_init_model_parallel_group(self, *mocks):
        mock_new_group, mock_get_world_size, mock_get_rank = mocks
        from mindspeed_llm.core.high_availability import elastic_training_scale_out_rebuild
        mock_args = mock.MagicMock()
        mock_timeout = 300
        mock_nccl_comm_cfgs = {}
        mock_all_dp_ranks_with_cp = [[0, 2], [1, 3]]
        mock_get_rank.return_value = 0
        mock_get_world_size.return_value = 4
        mock_args.tensor_model_parallel_size = 2
        mock_args.pipeline_model_parallel_size = 1
        mock_args.context_parallel_size = 1
        mock_new_group.return_value = 1
        elastic_training_scale_out_rebuild.init_model_parallel_group(mock_args, mock_timeout, mock_nccl_comm_cfgs, mock_all_dp_ranks_with_cp)
        from megatron.core import mpu
        self.assertEqual(mpu._MODEL_PARALLEL_GROUP, 1)

    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_out_rebuild.mpu')
    @patch('torch.distributed.new_group')
    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.get_rank')
    def test_init_tensor_parallel_group(self, *mocks):
        (mock_get_rank, mock_get_world_size, mock_new_group, mock_mpu) = mocks
        from mindspeed_llm.core.high_availability import elastic_training_scale_out_rebuild
        mock_args = mock.MagicMock()
        mock_args.tensor_model_parallel_size = 2
        mock_args.pipeline_model_parallel_size = 1
        mock_args.context_parallel_size = 1
        mock_get_rank.return_value = 0
        mock_get_world_size.return_value = 8
        mock_mpu._TENSOR_MODEL_PARALLEL_GROUP = None
        mock_timeout = timedelta(minutes=30)
        mock_nccl_comm_cfgs = {}
        mock_new_group.return_value = 1
        elastic_training_scale_out_rebuild.init_tensor_parallel_group(mock_args, mock_timeout, mock_nccl_comm_cfgs)
        mock_new_group.assert_called_once_with(range(0, 2), timeout=mock_timeout,
                                              pg_options=None,
                                              use_local_synchronization=True)
        assert mock_mpu._TENSOR_MODEL_PARALLEL_GROUP == 1
    
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_out_rebuild.destroy_sub_process_group')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_out_rebuild.mpu')
    @patch('torch.distributed.new_group')
    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.get_rank')
    def test_init_pipeline_parallel_group(self, *mocks):
        (mock_get_rank, mock_get_world_size, mock_new_group, mock_mpu, mock_destroy_sub_process_group) = mocks
        from mindspeed_llm.core.high_availability import elastic_training_scale_out_rebuild
        mock_args = mock.MagicMock()
        mock_args.tensor_model_parallel_size = 2
        mock_args.pipeline_model_parallel_size = 4
        mock_args.context_parallel_size = 1
        mock_new_group.return_value = 1
        mock_get_rank.return_value = 0
        mock_get_world_size.return_value = 8
        mock_mpu._PIPELINE_MODEL_PARALLEL_GROUP = None
        mock_timeout = timedelta(minutes=30)
        mock_nccl_comm_cfgs = {}
        elastic_training_scale_out_rebuild.init_pipeline_parallel_group(mock_args, mock_timeout, mock_nccl_comm_cfgs)
        pipeline_expected_ranks = range(0, 8, 2)
        embedding_expected_ranks = [0, 6]
        position_expected_ranks = [0]
        mock_new_group.assert_called()
        mock_destroy_sub_process_group.assert_called()
        assert mock_mpu._PIPELINE_GLOBAL_RANKS == pipeline_expected_ranks
        assert mock_mpu._EMBEDDING_GLOBAL_RANKS == embedding_expected_ranks
        assert mock_mpu._POSITION_EMBEDDING_GLOBAL_RANKS == position_expected_ranks

    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_out_rebuild.torch.distributed.get_rank')
    @patch('torch.distributed.new_group')
    @patch('torch.distributed.get_world_size')
    def test_init_context_parallel_group(self, *mocks):
        mock_get_world_size, mock_new_group, mock_get_rank = mocks
        mock_get_world_size.return_value = 8
        mock_get_rank.return_value = 0
        mock_new_group.return_value = 1
        from mindspeed_llm.core.high_availability import elastic_training_scale_out_rebuild
        mock_args = MagicMock()
        mock_args.tensor_model_parallel_size = 2
        mock_args.pipeline_model_parallel_size = 2
        mock_args.context_parallel_size = 1
        timeout = timedelta(minutes=30)
        nccl_comm_cfgs = {}
        elastic_training_scale_out_rebuild.init_context_parallel_group(mock_args, timeout, nccl_comm_cfgs)
        mock_new_group.assert_called_once()
        from megatron.core import mpu
        self.assertEqual(mpu._CONTEXT_PARALLEL_GROUP, 1)

    @patch('torch.distributed')
    def test_get_nccl_options_with_default_values(self, mock_dist):
        mock_nccl = MagicMock()
        mock_options = MagicMock()
        mock_nccl.Options = MagicMock()
        mock_nccl.Options.return_value = mock_options
        mock_dist.ProcessGroupNCCL = mock_nccl
        pg_name = 'test_pg'
        nccl_comm_cfgs = {
            pg_name: {
                'cga_cluster_size': 16
            }
        }
        from mindspeed_llm.core.high_availability import elastic_training_scale_out_rebuild
        result = elastic_training_scale_out_rebuild.get_nccl_options(pg_name, nccl_comm_cfgs)

        self.assertEqual(result, mock_options)
        self.assertEqual(mock_options.config.cga_cluster_size, 16)
        self.assertEqual(mock_options.config.max_ctas, 32)
        self.assertEqual(mock_options.config.min_ctas, 1)

    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_out_rebuild.mpu')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_out_rebuild.destroy_sub_process_group')
    @patch('torch.distributed.new_group')
    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.get_rank')
    def test_init_data_parallel_with_cp_group_rank_in_group(self, *mocks):
        (mock_get_rank, mock_get_world_size, mock_new_group, mock_destroy_sub_process_group, mock_mpu) = mocks
        mock_get_rank.return_value = 1
        mock_get_world_size.return_value = 8
        mock_group = MagicMock()
        mock_new_group.return_value = mock_group

        args = MagicMock()
        args.tensor_model_parallel_size = 2
        args.pipeline_model_parallel_size = 2
        timeout = MagicMock()
        nccl_comm_cfgs = {}
        from mindspeed_llm.core.high_availability import elastic_training_scale_out_rebuild
        result = elastic_training_scale_out_rebuild.init_data_parallel_with_cp_group(args, timeout,
                                                                                     nccl_comm_cfgs)
        self.assertEqual(mock_new_group.call_count, 2)
        expected_ranks = range(1, 4, 2)
        self.assertEqual(mock_destroy_sub_process_group.call_count, 2)
        self.assertEqual(mock_mpu._DATA_PARALLEL_GROUP_WITH_CP, mock_group)
        self.assertEqual(mock_mpu._DATA_PARALLEL_GROUP_WITH_CP_GLOO, mock_group)
        self.assertEqual(mock_mpu._DATA_PARALLEL_GLOBAL_RANKS_WITH_CP, expected_ranks)
        expected_result = [
            [0, 2],
            [1, 3],
            [4, 6],
            [5, 7]
        ]
        self.assertEqual(result, expected_result)