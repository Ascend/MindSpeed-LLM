#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import unittest
from unittest import mock
from unittest.mock import patch


class TestElasticTrainingScaleInRebuild(unittest.TestCase):
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.build_dataset')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.rebuild_not_changed_group')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.build_scale_in_dp_cp_replica_group')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.get_fault_msgs')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.get_changed_old_dp_ranks')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.tft_replica_group.ttp_get_dp_cp_replica_group')
    @patch('megatron.core.mpu.get_data_parallel_group')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.get_timers')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.change_num_micro_batches')
    @patch('megatron.core.num_microbatches_calculator')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.get_num_microbatches')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.get_args')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.torch.distributed.new_group')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.torch.distributed.get_rank')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.update_model_and_optim_related_group')
    @patch('torch.distributed.get_process_group_ranks')
    def test_scale_in_rebuild_callback(self, *mocks):
        (mock_get_process_group_ranks_mock, mock_update_model_and_optim_related_group, mock_get_rank, mock_new_group,
         mock_get_args,
         mock_get_num_microbatches,
         mock_num_microbatches_calculator, mock_change_num_micro_batches, mock_get_timers,
         mock_get_data_parallel_group, mock_ttp_get_dp_cp_replica_group,
         mock_get_changed_old_dp_ranks, mock_get_fault_msgs, mock_build_scale_in_replica_group,
         mock_rebuild_not_changed_group, mock_build_dataset) = mocks
        mock_get_changed_old_dp_ranks.return_value = (False, [1, 2])
        mock_get_fault_msgs.return_value = ([0], [0], True)
        mock_get_rank.return_value = 0
        mock_args = mock.MagicMock()
        mock_args.expert_model_parallel_size = 1
        mock_args.context_parallel_size = 1
        mock_args.distributed_backend = 'nccl'
        mock_args.rampup_batch_size = None
        mock_get_args.return_value = mock_args
        mock_get_num_microbatches.return_value = 4
        mock_calculator = mock.MagicMock()
        mock_num_microbatches_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR = mock_calculator
        mock_timer = mock.MagicMock()
        mock_timers = mock.MagicMock()
        mock_timers._timers = {'timer1': mock_timer, 'timer2': mock_timer}
        mock_get_timers.return_value = mock_timers
        mock_get_data_parallel_group.return_value = mock.MagicMock()
        mock_ttp_get_dp_cp_replica_group.return_value = mock.MagicMock()
        mock_get_process_group_ranks_mock.return_value = [0, 1, 2, 3]
        from mindspeed_llm.core.high_availability import elastic_training_scale_in_rebuild
        new_dp_ranks = [0, 2]
        new_world_ranks = [0, 2]
        args = [[mock.MagicMock(), mock.MagicMock(), mock.MagicMock()]]
        params = '{"scale-in-strategy": "DP"}'

        elastic_training_scale_in_rebuild.scale_in_rebuild_callback(new_dp_ranks, new_world_ranks, args, params)

        mock_get_changed_old_dp_ranks.assert_called_once()
        mock_get_fault_msgs.assert_called_once()
        mock_build_scale_in_replica_group.assert_called_once()
        mock_rebuild_not_changed_group.assert_called_once()
        mock_update_model_and_optim_related_group.assert_called_once()
        mock_change_num_micro_batches.assert_called_once()
        from mindspeed_llm.core.high_availability import elastic_training_common
        self.assertTrue(elastic_training_common.SCALE_IN_RUNNING_STATE)
        mock_get_timers.assert_called_once()
        self.assertEqual(mock_timer.set_barrier_group.call_count, 2)  # There are 2 timers
        self.assertEqual(mock_timer.reset.call_count, 2)  # There are 2 timers
        mock_build_dataset.assert_called_once_with(args)

    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.torch.distributed.get_process_group_ranks')
    def test_get_changed_old_dp_ranks(self, mock_get_process_group_ranks):
        mock_get_process_group_ranks.return_value = [0, 1]
        from mindspeed_llm.core.high_availability import elastic_training_scale_in_rebuild
        cur_rank = 0
        old_dp_ranks = [0, 1, 2, 3]
        new_dp_ranks = [0, 1, 3]  # Only rank 2 is faulty, it's in the right replica group
        dp_cp_replica_ranks = [0, 1]
        both_replica_group_fault, changed_old_dp_ranks = elastic_training_scale_in_rebuild.get_changed_old_dp_ranks(
            cur_rank, old_dp_ranks, new_dp_ranks, dp_cp_replica_ranks
        )
        self.assertFalse(both_replica_group_fault)
        self.assertEqual(changed_old_dp_ranks, old_dp_ranks)

    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.build_new_dp_cp_group')
    def test_get_fault_msgs(self, mock_build_new_dp_cp_group):
        from mindspeed_llm.core.high_availability import elastic_training_scale_in_rebuild
        cur_rank = 0
        old_dp_ranks = [0, 1, 2, 3]
        changed_old_dp_ranks = [0, 1, 2, 3]
        new_dp_ranks = [0, 2]
        dp_cp_replica_ranks = [0, 1]

        fault_idxs, fault_local_idxs, fault_first_group = elastic_training_scale_in_rebuild.get_fault_msgs(
            cur_rank, old_dp_ranks, changed_old_dp_ranks, new_dp_ranks, dp_cp_replica_ranks
        )

        self.assertEqual(fault_idxs, [1, 3])
        self.assertEqual(fault_local_idxs, [1, 1])
        mock_build_new_dp_cp_group.assert_called_once()

    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.get_num_microbatches')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.num_microbatches_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.torch.distributed.get_rank')
    def test_change_num_micro_batches(self, *mocks):
        from mindspeed_llm.core.high_availability import elastic_training_scale_in_rebuild
        mock_get_rank, mock_calculator, mock_get_num_microbatches = mocks
        mock_get_num_microbatches.return_value = 4
        mock_get_rank.return_value = 0
        old_dp_ranks = [0, 1, 2, 3]
        new_dp_ranks = [0, 2]
        arguments = mock.MagicMock()
        arguments.rampup_batch_size = None

        elastic_training_scale_in_rebuild.change_num_micro_batches(old_dp_ranks, new_dp_ranks, arguments)

        self.assertEqual(mock_calculator.num_micro_batches, 8)

    def test_delete_ranks_from_src_by_ids(self):
        from mindspeed_llm.core.high_availability import elastic_training_scale_in_rebuild
        src_ranks = [0, 1, 2, 3]
        reversed_idxs = [3, 1]  # Delete indexes 1 and 3 from the end

        result = elastic_training_scale_in_rebuild.delete_ranks_from_src_by_ids(src_ranks, reversed_idxs)

        self.assertEqual(result, [0, 2])

    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.torch.distributed.'
          'reinit_process_group')
    def test_init_context_parallel_group(self, mock_reinit):
        from mindspeed_llm.core.high_availability import elastic_training_scale_in_rebuild
        from megatron.core import mpu

        elastic_training_scale_in_rebuild.init_context_parallel_group()
        mock_reinit.assert_not_called()

        mpu._CONTEXT_PARALLEL_GROUP = 1
        elastic_training_scale_in_rebuild.init_context_parallel_group()
        mock_reinit.assert_called_once_with(1, rebuild_link=True)

    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.torch.distributed'
           '.reinit_process_group')
    def test_init_model_parallel_group(self, mock_reinit):
        from mindspeed_llm.core.high_availability import elastic_training_scale_in_rebuild
        from megatron.core import mpu

        elastic_training_scale_in_rebuild.init_model_parallel_group()
        mock_reinit.assert_not_called()

        mpu._MODEL_PARALLEL_GROUP = 1
        elastic_training_scale_in_rebuild.init_model_parallel_group()
        mock_reinit.assert_called_once_with(1, rebuild_link=True)

    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.torch.distributed.'
           'reinit_process_group')
    def test_init_tensor_parallel_group(self, mock_reinit):
        from mindspeed_llm.core.high_availability import elastic_training_scale_in_rebuild
        from megatron.core import mpu

        elastic_training_scale_in_rebuild.init_tensor_parallel_group()
        mock_reinit.assert_not_called()

        mpu._TENSOR_MODEL_PARALLEL_GROUP = 1
        elastic_training_scale_in_rebuild.init_tensor_parallel_group()
        mock_reinit.assert_called_once_with(1, rebuild_link=True)

    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.torch.distributed.'
           'reinit_process_group')
    @patch('mindspeed_llm.core.high_availability.elastic_training_common.destroy_sub_process_group')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.torch.distributed.new_group')
    def test_init_pipeline_parallel_group(self, *mocks):
        (mock_new_group, mock_destroy_sub_process_group, mock_reinit) = mocks
        rank = 0
        mock_mpu = mock.MagicMock(
            _PIPELINE_GLOBAL_RANKS=[0, 1, 2, 3],
            _PIPELINE_MODEL_PARALLEL_GROUP=mock.MagicMock(),
            _EMBEDDING_GLOBAL_RANKS=[0, 1],
            _EMBEDDING_GROUP=mock.MagicMock(),
            _POSITION_EMBEDDING_GLOBAL_RANKS=[0, 1],
            _POSITION_EMBEDDING_GROUP=mock.MagicMock()
        )
        from mindspeed_llm.core.high_availability import elastic_training_scale_in_rebuild
        with patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.mpu', mock_mpu):
            elastic_training_scale_in_rebuild.init_pipeline_parallel_group(rank)
            self.assertEqual(mock_reinit.call_count, 3)

    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.torch.distributed.get_rank')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.torch.distributed.get_world_size')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.delete_ranks_from_src_by_ids')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.destroy_sub_process_group')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.torch.distributed.new_group')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.get_args')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.mpu')
    def test_build_new_dp_cp_group(self, *mocks):
        (mock_mpu, mock_get_args, mock_new_group, mock_destroy_sub_process_group,
                                   mock_delete_ranks, mock_get_world_size, mock_get_rank) = mocks
        from mindspeed_llm.core.high_availability import elastic_training_scale_in_rebuild
        mock_mpu.get_pipeline_model_parallel_world_size.return_value = 1  # Set to 1 to ensure condition is met
        mock_mpu.get_tensor_model_parallel_world_size.return_value = 1
        mock_mpu.get_context_parallel_world_size.return_value = 1
        mock_mpu._DATA_PARALLEL_GROUP = mock.MagicMock()
        mock_mpu._DATA_PARALLEL_GROUP_GLOO = mock.MagicMock()
        mock_mpu._DATA_PARALLEL_GROUP_WITH_CP = mock.MagicMock()
        mock_mpu._DATA_PARALLEL_GROUP_WITH_CP_GLOO = mock.MagicMock()
        mock_get_args.return_value = mock.MagicMock(data_parallel_size=4)
        mock_get_world_size.return_value = 4
        mock_get_rank.return_value = 0
        mock_delete_ranks.return_value = [0]
        mock_new_group.return_value = mock.MagicMock()
        elastic_training_scale_in_rebuild.build_new_dp_cp_group([1, 2, 3])  # Modify fault_idxs to ensure condition is met

        mock_delete_ranks.assert_called()
        mock_new_group.assert_called()
        self.assertTrue(mock_destroy_sub_process_group.called, "destroy_sub_process_group should have been called")

    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.create_new_replica_group_for_changed_left')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.create_scale_in_replica_group')
    @patch('mindspeed_llm.core.high_availability.tft_replica_group.ttp_get_replica_dp_num')
    def test_build_scale_in_dp_cp_replica_group(self, mock_get_replica_dp_num, mock_create_scale_in_replica_group, mock_create_new_replica_group):
        from mindspeed_llm.core.high_availability import elastic_training_scale_in_rebuild
        mock_get_replica_dp_num.return_value = 2
        fault_local_idxs = [1]
        fault_first_group = False
        both_replica_group_fault = False
        changed_old_dp_ranks = [0, 1, 2, 3]

        elastic_training_scale_in_rebuild.build_scale_in_dp_cp_replica_group(
            fault_local_idxs, fault_first_group, both_replica_group_fault, changed_old_dp_ranks
        )

        mock_create_scale_in_replica_group.assert_called_once()
        mock_create_new_replica_group.assert_not_called()

    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.destroy_sub_process_group')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.torch.distributed.new_group')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.torch.distributed.get_rank')
    def test_create_scale_in_replica_group(self, mock_get_rank, mock_new_group, mock_destroy_sub_process_group):
        from mindspeed_llm.core.high_availability import (elastic_training_scale_in_rebuild,
                                                          elastic_training_common, tft_replica_group)
        mock_get_rank.return_value = 0
        mock_new_group.return_value = mock.MagicMock()
        # test is fault replica rank
        fault_first_group = True
        ranks_left = [0, 1]
        ranks_right = [2, 3]
        elastic_training_common.IS_FAULT_REPLICA_RANK = False
        elastic_training_scale_in_rebuild.create_scale_in_replica_group(fault_first_group, ranks_left, ranks_right)
        mock_new_group.assert_called()
        mock_destroy_sub_process_group.assert_called()
        self.assertIsNotNone(tft_replica_group.DP_CP_REPLICA_GROUP)
        self.assertIsNotNone(tft_replica_group.DP_CP_REPLICA_GROUP_GLOO)

        # test is not fault replica rank
        fault_first_group = False
        ranks_right = [0, 1]
        ranks_left = [2, 3]
        elastic_training_common.IS_FAULT_REPLICA_RANK = True
        tft_replica_group.DP_CP_REPLICA_GROUP = None
        tft_replica_group.DP_CP_REPLICA_GROUP_GLOO = None
        elastic_training_scale_in_rebuild.create_scale_in_replica_group(fault_first_group, ranks_left, ranks_right)
        self.assertIsNone(tft_replica_group.DP_CP_REPLICA_GROUP)
        self.assertIsNone(tft_replica_group.DP_CP_REPLICA_GROUP_GLOO)

    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.destroy_sub_process_group')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.torch.distributed.new_group')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.tft_replica_group')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.torch.distributed.get_rank')
    def test_create_new_replica_group_for_changed_left(self, mock_get_rank, mock_tft_replica_group, mock_new_group, mock_destroy_sub_process_group):
        from mindspeed_llm.core.high_availability import elastic_training_scale_in_rebuild, elastic_training_common
        mock_get_rank.return_value = 0
        mock_new_group.return_value = mock.MagicMock()
        mock_tft_replica_group.DP_CP_REPLICA_GROUP = mock.MagicMock()
        mock_tft_replica_group.DP_CP_REPLICA_GROUP_GLOO = mock.MagicMock()

        elastic_training_common.IS_FAULT_REPLICA_RANK = True
        elastic_training_common.FAULT_RANK_IN_DP_CP_REPLICA_GROUP = True
        left_ranks = [0, 2]
        elastic_training_scale_in_rebuild.create_new_replica_group_for_changed_left(left_ranks)
        self.assertFalse(elastic_training_common.FAULT_RANK_IN_DP_CP_REPLICA_GROUP)
        mock_new_group.assert_called()
        mock_destroy_sub_process_group.assert_called()

    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_out_rebuild.ttp_initialize_replica_dp_group')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.init_pipeline_parallel_group')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.init_tensor_parallel_group')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.init_model_parallel_group')
    @patch('mindspeed_llm.core.high_availability.elastic_training_scale_in_rebuild.init_context_parallel_group')
    def test_rebuild_not_changed_group(self, mock_init_context, mock_init_model, mock_init_tensor, mock_init_pipeline, mock_ttp_initialize):
        from mindspeed_llm.core.high_availability import elastic_training_scale_in_rebuild, elastic_training_common
        cur_rank = 0
        both_replica_group_fault = False
        args = mock.MagicMock()
        args.pipeline_model_parallel_size = 1
        args.tensor_model_parallel_size = 1
        args.context_parallel_size = 1
        args.world_size = 4
        elastic_training_common.FAULT_RANK_IN_DP_CP_REPLICA_GROUP = False
        
        elastic_training_scale_in_rebuild.rebuild_not_changed_group(cur_rank, both_replica_group_fault, args)
        mock_init_context.assert_called_once()
        mock_init_model.assert_called_once()
        mock_init_tensor.assert_called_once()
        mock_init_pipeline.assert_called_once_with(cur_rank)
        mock_ttp_initialize.assert_called_once_with(
            args.pipeline_model_parallel_size,
            args.tensor_model_parallel_size,
            args.context_parallel_size,
            args.world_size
        )

    def test_get_ranks_after_change_left(self):
        from mindspeed_llm.core.high_availability import elastic_training_scale_in_rebuild, elastic_training_common
        dp_cp_replica_ranks_length = 2
        fault_idxs = [0, 2]
        changed_old_dp_ranks = [0, 1, 2, 3]
        cur_rank = 2

        result = elastic_training_scale_in_rebuild.get_ranks_after_change_left(
            dp_cp_replica_ranks_length, fault_idxs, changed_old_dp_ranks, cur_rank
        )
        self.assertEqual(result, [2, 1, 2, 3])
        self.assertTrue(elastic_training_common.IS_FAULT_REPLICA_RANK)

