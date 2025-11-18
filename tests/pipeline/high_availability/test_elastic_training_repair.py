#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import unittest
from unittest import mock
from unittest.mock import patch, MagicMock


class TestRepairCallback(unittest.TestCase):
    @patch('torch.distributed.get_rank')
    @patch('mindspeed_llm.core.high_availability.tft_optimizer_data_repair.send_rank_repair')
    @patch('mindspeed_llm.core.high_availability.tft_optimizer_data_repair.recv_rank_repair')
    @patch('mindspeed_llm.tasks.megatron_adaptor_v2.FeatureAdaptor.execute')
    @patch('torch.npu.set_device')
    @patch('mindio_ttp.framework_ttp.ttp_decorator.get_device')
    def test_repair_callback(self, *mocks):
        (mock_get_device, mock_set_device,
        mock_execute, mock_recv_rank_repair,
        mock_send_rank_repair, mock_get_rank) = mocks
        from mindspeed_llm.core.high_availability import elastic_training_repair
        from mindio_ttp.framework_ttp import RepairType
        mock_get_rank.return_value = 0
        """Test repair_callback with invalid step"""
        step = 0
        need_rebuild = False
        error_ranks = []
        repair_type_str = 'repair_type'
        repair_info = {repair_type_str: RepairType.RT_SEND.value}
        train_args = {0: {}}
        params = "{\"scale-out-strategy\": \"DP\"}"
        try:
            elastic_training_repair.repair_callback(step, need_rebuild, error_ranks, repair_info, train_args, params)
        except Exception as e:
            self.assertIn("repair step 0 is not valid", str(e))

        # test send repair type
        step = 1
        elastic_training_repair.repair_callback(step, need_rebuild, error_ranks, repair_info, train_args, params)
        self.assertEqual(mock_send_rank_repair.call_count, 1)

        # test recv repair type
        repair_info = {repair_type_str: RepairType.RT_RECV_REPAIR.value}
        elastic_training_repair.repair_callback(step, need_rebuild, error_ranks, repair_info, train_args, params)
        self.assertEqual(mock_recv_rank_repair.call_count, 1)

        # test invalid repair type
        repair_info = {repair_type_str: RepairType.RT_UCE_HIGHLEVEL.value}
        try:
            elastic_training_repair.repair_callback(step, need_rebuild, error_ranks, repair_info, train_args, params)
        except Exception as e:
            self.assertIn("rank:0 repair type 1 not supported", str(e))