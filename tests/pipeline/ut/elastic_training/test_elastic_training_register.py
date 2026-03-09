#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import unittest
from unittest.mock import patch


class TestElasticTrainingRegister(unittest.TestCase):
    @patch('mindio_ttp.framework_ttp.tft_register_zit_downgrade_rebuild_handler')
    @patch('mindio_ttp.framework_ttp.tft_register_zit_upgrade_rollback_handler')
    @patch('mindio_ttp.framework_ttp.tft_register_zit_upgrade_repair_handler')
    @patch('mindio_ttp.framework_ttp.tft_register_zit_upgrade_rebuild_handler')
    def test_scale_out_rebuild_process_group_callback(self, *mocks):
        (mock_upgrade_rebuild_handler, mock_upgrade_repair_handler, mock_upgrade_rollback_handler,
         mock_downgrade_rebuild_handler) = mocks
        from mindspeed_llm.core.high_availability import elastic_training_register
        elastic_training_register.register_callbacks()
        mock_upgrade_rebuild_handler.assert_called_once()
        mock_upgrade_repair_handler.assert_called_once()
        mock_upgrade_rollback_handler.assert_called_once()
        mock_downgrade_rebuild_handler.assert_called_once()