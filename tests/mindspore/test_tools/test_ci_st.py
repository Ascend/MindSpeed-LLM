# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
import pytest
from mindspeed_llm import megatron_adaptor
from tests.mindspore.test_tools.acquire_json import transfer_logs_as_json, read_json

LOSS = "lm loss"


class TestMargin:
    _MARGIN_NAME = " margin"
    loss = 0.02


class TestCIST:


    def _get_baseline(self, baseline_json):
        # acquire expected results
        self.expected = read_json(baseline_json)

    def _get_actual(self, generate_log, generate_json):
        # acquire actual results
        transfer_logs_as_json(generate_log, generate_json)
        self.actual = read_json(generate_json)
    
    def _test_helper(self, test_obj):
        """
        Core test function

        Args:
            test_obj: the object we want to test compare.
            test_type: deterministic or approximate, default is None.

        Here we temperally test `lm loss`
        """
        comparison_selection = {
            LOSS: self._compare_lm_loss
        }
        
        if test_obj in comparison_selection:
            expected_list = self.expected[test_obj]
            if not expected_list:
                return
            print(f"===================== Begin comparing {test_obj} ===================")
            actual_list = self.actual[test_obj]
            print(f"The list of expected values: {expected_list}")
            print(f"The list of actual values: {actual_list}")
            # Check if lists exist and are non-empty
            if not actual_list:
                raise ValueError(f"Actual list for {test_obj} is empty or not found. Maybe program has failed! Check it.")

            # Check if lists have the same length
            if len(expected_list) != len(actual_list):
                raise ValueError(f"Actual lengths of the lists for {test_obj} do not match. Maybe program has failed! Check it.")

            compare_func = comparison_selection[test_obj]
            compare_func(expected_list, actual_list)
        else:
            raise ValueError(f"Unsupported test object: {test_obj}")
            
    def _compare_lm_loss(self, expected_list, actual_list):
        for step, (expected_val, actual_val) in enumerate(zip(expected_list, actual_list)):
            print(f"Checking step {step + 1} for lm loss")
            assert actual_val == pytest.approx(expected=expected_val, rel=TestMargin.loss), \
                f"The loss at step {step} should be approximate to {expected_val} but it is {actual_val}."

    def test_lm_loss(self, baseline_json, generate_log, generate_json):
        # expected training loss curve at different global steps.
        self._get_baseline(baseline_json)
        self._get_actual(generate_log, generate_json)
        self._test_helper("lm loss")