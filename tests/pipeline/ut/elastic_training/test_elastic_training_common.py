#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import unittest
from unittest.mock import patch, MagicMock
import json
import os
import importlib.util


def load_module_by_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
target_module_path = os.path.join(project_root, "mindspeed_llm", "core", "high_availability", "elastic_training_common.py")
common_module = load_module_by_path("mindspeed_llm.core.high_availability.elastic_training_common", target_module_path)


class TestCommon(unittest.TestCase):

    def setUp(self):
        # Reset global variables before each test
        common_module.ORIGIN_DP_SIZE = None
        common_module.ORIGIN_NUM_MICRO_BATCHES = None
        common_module.SCALE_IN_WORLD_GROUP = None
        common_module.SCALE_IN_DP_CP_REPLICA_GROUP = None
        common_module.SCALE_IN_DP_CP_REPLICA_GROUP_GLOO = None
        common_module.FAULT_RANK_IN_DP_CP_REPLICA_GROUP = False
        common_module.IS_FAULT_REPLICA_RANK = False
        common_module.FAULT_REPLICA_RANK = None
        common_module.SCALE_IN_RUNNING_STATE = False
        common_module.HAS_DATA = None

    def test_update_scale_in_flag(self):
        """Test updating scale in flag"""
        common_module.update_scale_in_flag(True)
        self.assertTrue(common_module.SCALE_IN_RUNNING_STATE)

        common_module.update_scale_in_flag(False)
        self.assertFalse(common_module.SCALE_IN_RUNNING_STATE)

    def test_zit_get_has_data_index(self):
        """Test getting has data index"""
        common_module.HAS_DATA = 5
        self.assertEqual(common_module.zit_get_has_data_index(), 5)

        common_module.HAS_DATA = None
        self.assertIsNone(common_module.zit_get_has_data_index())

    def test_zit_get_scale_in_world_group(self):
        """Test getting scale in world group"""
        common_module.SCALE_IN_WORLD_GROUP = "test_group"
        self.assertEqual(common_module.zit_get_scale_in_world_group(), "test_group")

        common_module.SCALE_IN_WORLD_GROUP = None
        self.assertIsNone(common_module.zit_get_scale_in_world_group())

    def test_zit_is_fault_replica_rank(self):
        """Test checking if is fault replica rank"""
        common_module.IS_FAULT_REPLICA_RANK = True
        self.assertTrue(common_module.zit_is_fault_replica_rank())

        common_module.IS_FAULT_REPLICA_RANK = False
        self.assertFalse(common_module.zit_is_fault_replica_rank())

    def test_zit_fault_rank_in_dp_cp_replica_group(self):
        """Test checking if fault rank in dp cp replica group"""
        common_module.FAULT_RANK_IN_DP_CP_REPLICA_GROUP = True
        self.assertTrue(common_module.zit_fault_rank_in_dp_cp_replica_group())

        common_module.FAULT_RANK_IN_DP_CP_REPLICA_GROUP = False
        self.assertFalse(common_module.zit_fault_rank_in_dp_cp_replica_group())

    def test_zit_scale_in_running_state(self):
        """Test checking scale in running state"""
        common_module.SCALE_IN_RUNNING_STATE = True
        self.assertTrue(common_module.zit_scale_in_running_state())

        common_module.SCALE_IN_RUNNING_STATE = False
        self.assertFalse(common_module.zit_scale_in_running_state())

    def test_zit_get_scale_in_dp_cp_replica_group(self):
        """Test getting scale in dp cp replica group"""
        common_module.SCALE_IN_DP_CP_REPLICA_GROUP = "test_dp_cp_group"
        self.assertEqual(common_module.zit_get_scale_in_dp_cp_replica_group(), "test_dp_cp_group")

        common_module.SCALE_IN_DP_CP_REPLICA_GROUP = None
        self.assertIsNone(common_module.zit_get_scale_in_dp_cp_replica_group())

    def test_zit_get_scale_in_dp_cp_replica_group_gloo(self):
        """Test getting scale in dp cp replica group gloo"""
        common_module.SCALE_IN_DP_CP_REPLICA_GROUP_GLOO = "test_gloo_group"
        self.assertEqual(common_module.zit_get_scale_in_dp_cp_replica_group_gloo(), "test_gloo_group")

        common_module.SCALE_IN_DP_CP_REPLICA_GROUP_GLOO = None
        self.assertIsNone(common_module.zit_get_scale_in_dp_cp_replica_group_gloo())

    def test_zit_get_fault_replica_rank(self):
        """Test getting fault replica rank"""
        common_module.FAULT_REPLICA_RANK = 3
        self.assertEqual(common_module.zit_get_fault_replica_rank(), 3)

        common_module.FAULT_REPLICA_RANK = None
        self.assertIsNone(common_module.zit_get_fault_replica_rank())

    @patch.object(common_module, 'ttp_logger')
    def test_check_scale_out_params(self, mock_logger):
        params = '{"scale-out-strategy": "DP"}'
        # Should not raise exception
        try:
            common_module.check_scale_out_params(params)
        except Exception as e:
            self.fail(f"check_scale_out_params raised exception: {e}")
        mock_logger.info.assert_called_once()

        params = '{"scale-out-strategy": "INVALID"}'
        with self.assertRaises(Exception) as context:
            common_module.check_scale_out_params(params)
        self.assertIn("Only support DP strategy", str(context.exception))
        self.assertEqual(mock_logger.info.call_count, 2)
        self.assertEqual(mock_logger.error.call_count, 1)

        params = '{}'
        with self.assertRaises(Exception) as context:
            common_module.check_scale_out_params(params)
        self.assertIn("Only support DP strategy", str(context.exception))
        self.assertEqual(mock_logger.info.call_count, 3)
        self.assertEqual(mock_logger.error.call_count, 2)

        params = 'invalid json'
        with self.assertRaises(json.JSONDecodeError):
            common_module.check_scale_out_params(params)
        self.assertEqual(mock_logger.info.call_count, 3)
        self.assertEqual(mock_logger.error.call_count, 2)


    @patch.object(common_module, 'ttp_logger')
    def test_check_scale_in_params(self, mock_logger):
        params = '{"scale-in-strategy": "DP"}'
        # Should not raise exception
        try:
            common_module.check_scale_in_params(params)
        except Exception as e:
            self.fail(f"check_scale_in_params raised exception: {e}")
        mock_logger.info.assert_called_once()

        params = '{"scale-in-strategy": "INVALID"}'
        with self.assertRaises(Exception) as context:
            common_module.check_scale_in_params(params)
        self.assertIn("Only support DP strategy", str(context.exception))
        self.assertEqual(mock_logger.info.call_count, 2)
        self.assertEqual(mock_logger.error.call_count, 1)

        params = '{}'
        with self.assertRaises(Exception) as context:
            common_module.check_scale_in_params(params)
        self.assertIn("Only support DP strategy", str(context.exception))
        self.assertEqual(mock_logger.info.call_count, 3)
        self.assertEqual(mock_logger.error.call_count, 2)

        params = 'invalid json'
        with self.assertRaises(json.JSONDecodeError):
            common_module.check_scale_in_params(params)
        self.assertEqual(mock_logger.info.call_count, 3)
        self.assertEqual(mock_logger.error.call_count, 2)

    @patch.object(common_module, 'ttp_logger')
    @patch('torch.distributed')
    def test_destroy_sub_process_group(self, mock_dist, mock_logger):
        mock_dist.get_rank.return_value = 0
        mock_dist.reinit_process_group = MagicMock()
        mock_dist.destroy_process_group = MagicMock()
        common_module.destroy_sub_process_group(None)
        self.assertEqual(mock_dist.reinit_process_group.call_count, 0)
        self.assertEqual(mock_logger.debug.call_count, 1)

        common_module.destroy_sub_process_group("")
        self.assertEqual(mock_dist.reinit_process_group.call_count, 1)
        self.assertEqual(mock_dist.destroy_process_group.call_count, 1)
        self.assertEqual(mock_logger.debug.call_count, 2)


if __name__ == '__main__':
    unittest.main()