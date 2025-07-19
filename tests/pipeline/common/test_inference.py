# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests of Inference"""

import sys
import os
from pathlib import Path
import re
import logging
from torch import distributed as dist
import pytest
from inference import main
from tests.test_tools.dist_test import DistributedTest
from tests.test_tools.utils import build_args, create_testconfig, setup_logger


PATTERN = r"MindSpeed-LLM:\n(.*)"


def acquire_context(log_capture):
    # Acquire the final score for evaluation tasks, still universal.
    context_str = log_capture[0]
    context_pattern = r"MindSpeed-LLM:\s*([\s\S]*)"
    match = re.search(context_pattern, context_str)
    if match:
        context = match.group(1)
    else:
        raise ValueError("No matching context found in the provided log.")
    return context


class TestInferenceWorldSize2(DistributedTest):
    world_size = 2
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("params", test_config["test_chatglm3_mcore_greedy_search"])
    def test_chatglm3_mcore_greedy_search(self, build_args, params):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        os.environ["CLOSE_MATMUL_K_SHIFT"] = "1"
        if dist.get_rank() == 0:
            handler, log_capture = setup_logger(PATTERN)

        main()
        if dist.get_rank() == 0:
            print("=============== chatglm3 mcore greedy search =============")
            print(log_capture)
            context = acquire_context(log_capture)
            assert [context] == [
                "I'm fine, thanks.\nI'm fine, thanks.\nI'm fine, thanks.\nI'm fine,"
            ], "forward pass has been changed, check it!"


class TestInference(DistributedTest):
    world_size = 8
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("params", test_config["test_baichuan2_mcore_greedy_search"])
    def test_baichuan2_mcore_greedy_search(self, build_args, params):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        os.environ["CLOSE_MATMUL_K_SHIFT"] = "1"
        if dist.get_rank() == 0:
            handler, log_capture = setup_logger(PATTERN)

        main()
        if dist.get_rank() == 0:
            print("=============== baichuan2 mcore greedy search =============")
            print(log_capture) 
            context = acquire_context(log_capture)
            assert [context] == [
                "” “I’m fine.” “I’m glad to hear it.” “I’m glad to hear it too.” “"
            ], "forward pass has been changed, check it!"