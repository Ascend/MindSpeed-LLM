#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import unittest
from unittest import mock
from unittest.mock import patch
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

class mockOptimizerParamScheduler(OptimizerParamScheduler):
    def __init__(self, num_steps):
        self.num_steps = num_steps
        self.global_batch_size = None

    def step(self, global_batch_size):
        self.global_batch_size = global_batch_size


class TestRollbackCallback(unittest.TestCase):
    @patch('mindspeed_llm.core.high_availability.elastic_training_rollback.build_dataset')
    @patch('mindspeed_llm.core.high_availability.elastic_training_rollback.rebuild_global_vars')
    @patch('mindspeed_llm.core.high_availability.elastic_training_rollback.training_log_repair')
    @patch('mindspeed_llm.core.high_availability.elastic_training_rollback.feature_rollback')
    @patch('mindspeed_llm.core.high_availability.elastic_training_rollback.gather_model_params_from_optimizer')
    @patch('mindspeed_llm.core.high_availability.elastic_training_rollback.get_args')
    @patch('torch.distributed.get_rank')
    @patch('torch.npu.set_device')
    @patch('mindio_ttp.framework_ttp.ttp_decorator.get_device')
    def test_rollback_callback(self, *mocks):
        (mock_get_device, mock_set_device, mock_get_rank,
         mock_get_args, mock_gather_model_params, mock_feature_rollback,
         mock_training_log_repair, mock_rebuild_global_vars, mock_build_dataset) = mocks
        mock_get_args.return_value = mock.MagicMock()
        from megatron.core import num_microbatches_calculator
        ORIGIN_GLOBAL_NUM_MICROBATCHES_CALCULATOR = num_microbatches_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR
        num_microbatches_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR = mock.MagicMock()
        from mindspeed_llm.core.high_availability import elastic_training_common, elastic_training_rollback
        train_args = [[1, 1, 1, mockOptimizerParamScheduler(num_steps=1)]]
        from mindspeed_llm.core.high_availability.tft_optimizer_data_repair import set_load_ckpt, get_load_ckpt
        set_load_ckpt(True)
        from mindspeed_llm.core.high_availability.elastic_training_rollback import get_args
        args = get_args()
        args.rampup_batch_size = [1, 2, 3]
        args.global_batch_size = 8
        args.iteration = 1
        args.train_samples = None
        params = "{\"scale-out-strategy\": \"DP\"}"
        elastic_training_common.ORIGIN_DP_SIZE = 4
        elastic_training_common.ORIGIN_NUM_MICRO_BATCHES = 1
        # test load_ckpt is True and train_samples is None
        elastic_training_rollback.rollback_callback(1, train_args, params)
        mock_get_rank.assert_called()
        mock_rebuild_global_vars.assert_called()
        self.assertFalse(get_load_ckpt())
        from mindspeed_llm.core.high_availability.utils import ha_constant
        self.assertEqual(train_args[ha_constant.TRAIN_PARAM][ha_constant.SCHEDULER_INDEX].num_steps, 8)
        self.assertEqual(args.consumed_train_samples, 8)

        # test load_ckpt is True and train_samples is not None
        set_load_ckpt(True)
        args.consumed_train_samples = 2
        train_args[ha_constant.TRAIN_PARAM][ha_constant.SCHEDULER_INDEX].num_steps = 1
        args.train_samples = mock.MagicMock()
        elastic_training_rollback.rollback_callback(1, train_args, params)
        mock_get_rank.assert_called()
        mock_rebuild_global_vars.assert_called()
        self.assertFalse(get_load_ckpt())
        from mindspeed_llm.core.high_availability.utils import ha_constant
        self.assertEqual(train_args[ha_constant.TRAIN_PARAM][ha_constant.SCHEDULER_INDEX].num_steps, 1)
        self.assertEqual(args.consumed_train_samples, 2)
        self.assertEqual(train_args[ha_constant.TRAIN_PARAM][ha_constant.SCHEDULER_INDEX].global_batch_size, 8)
        num_microbatches_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR = ORIGIN_GLOBAL_NUM_MICROBATCHES_CALCULATOR