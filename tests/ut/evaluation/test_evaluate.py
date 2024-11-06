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
"""Tests of Evaluation"""

import sys
import os
from pathlib import Path
import logging
import re
import math
import pytest
import torch.distributed as dist
from evaluation import main
from tests.test_tools.dist_test import DistributedTest
from tests.test_tools.utils import build_args, create_testconfig, setup_logger


PATTERN = r"acc = (.*)"


def acquire_score(log_capture):
    # Acquire the final score for evaluation tasks, still universal.
    score_str = log_capture[0]
    score_pattern = r"(\d+\.\d+(?:e[+-]?\d+)?)"
    match = re.search(score_pattern, score_str)
    if match:
        score = float(match.group(1).strip())
    else:
        raise ValueError("No matching context found in the provided log.")
    return score


class TestEvaluate(DistributedTest):
    world_size = 8
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("params", test_config["test_llama2_mmlu_evaluate"])
    def test_llama2_mmlu_evaluate(self, build_args, params):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        if dist.get_rank() == 0:
            handler, log_capture = setup_logger(PATTERN)
        
        main()
        if dist.get_rank() == 0:
            print("=================== llama2 MMLU score ===============")
            print(log_capture)

            expected_score = acquire_score(log_capture)
            assert math.isclose(expected_score, 0.4970, abs_tol=1e-2), f"score {expected_score}, forward pass has been changed, check it!" 

    @pytest.mark.parametrize("params", test_config["test_qwen_prompt_mmlu_evaluate"])
    def test_qwen_prompt_mmlu_evaluate(self, build_args, params):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        if dist.get_rank() == 0:
            handler, log_capture = setup_logger(PATTERN)
        
        main()
        if dist.get_rank() == 0:
            print("=================== Qwen MMLU score ===============")
            print(log_capture)

            expected_score = acquire_score(log_capture)
            assert math.isclose(expected_score,  0.5526, abs_tol=1e-2), f"score {expected_score}, forward pass has been changed, check it!"

    @pytest.mark.parametrize("params", test_config["test_qwen_prompt_boolq_evaluate"])
    def test_qwen_prompt_boolq_evaluate(self, build_args, params):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        if dist.get_rank() == 0:
            handler, log_capture = setup_logger(PATTERN)
        
        main()
        if dist.get_rank() == 0:
            print("=================== Qwen boolq score ===============")
            print(log_capture)

            expected_score = acquire_score(log_capture)
            assert math.isclose(expected_score, 0.5666, abs_tol=1e-2), f"score {expected_score}, forward pass has been changed, check it!"

    @pytest.mark.parametrize("params", test_config["test_qwen_prompt_ceval_evaluate"])
    def test_qwen_prompt_ceval_evaluate(self, build_args, params):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        if dist.get_rank() == 0:
            handler, log_capture = setup_logger(PATTERN)
        
        main()
        if dist.get_rank() == 0:
            print("=================== Qwen ceval score ===============")
            print(log_capture)

            expected_score = acquire_score(log_capture)
            assert math.isclose(expected_score, 0.6154, abs_tol=1e-2), f"score {expected_score}, forward pass has been changed, check it!"

    @pytest.mark.parametrize("params", test_config["test_lora_mmlu_evaluate"])
    def test_lora_mmlu_evaluate(self, build_args, params):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        if dist.get_rank() == 0:
            handler, log_capture = setup_logger(PATTERN)
        
        main()
        if dist.get_rank() == 0:
            print("=================== lora mmlu evaluate score ===============")
            print(log_capture)

            expected_score = acquire_score(log_capture)
            assert math.isclose(expected_score, 0.5087, abs_tol=2e-2), f"score {expected_score}, forward pass has been changed, check it!"

    @pytest.mark.parametrize("params", test_config["test_llama2_agieval_evaluate"])
    def test_llama2_agieval_evaluate(self, build_args, params):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        if dist.get_rank() == 0:
            handler, log_capture = setup_logger(PATTERN)

        main()
        if dist.get_rank() == 0:
            print("=================== llama2 AGIEVAL score ===============")
            print(log_capture)

            expected_score = acquire_score(log_capture)
            assert math.isclose(expected_score, 0.192771,
                                abs_tol=1e-2), f"score {expected_score}, forward pass has been changed, check it!"

    @pytest.mark.parametrize("params", test_config["test_llama2_bbh_evaluate"])
    def test_llama2_bbh_evaluate(self, build_args, params):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        if dist.get_rank() == 0:
            handler, log_capture = setup_logger(PATTERN)

        main()
        if dist.get_rank() == 0:
            print("=================== llama2 BBH score ===============")
            print(log_capture)

            expected_score = acquire_score(log_capture)
            assert math.isclose(expected_score, 0.7441860465116279,
                                abs_tol=1e-2), f"score {expected_score}, forward pass has been changed, check it!"

    @pytest.mark.parametrize("params", test_config["test_llama2_humaneval_evaluate"])
    def test_llama2_humaneval_evaluate(self, build_args, params):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        if dist.get_rank() == 0:
            handler, log_capture = setup_logger(PATTERN)

        main()
        if dist.get_rank() == 0:
            print("=================== llama2 HUMANEVAL score ===============")
            print(log_capture)

            expected_score = acquire_score(log_capture)
            assert math.isclose(expected_score, 0.3333333333333333,
                                abs_tol=1e-2), f"score {expected_score}, forward pass has been changed, check it!"


class TestEvaluateWorldSize1(DistributedTest):
    world_size = 1
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("params", test_config["test_qwen2_mmlu_evaluate"])
    def test_qwen2_mmlu_evaluate(self, build_args, params):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        if dist.get_rank() == 0:
            handler, log_capture = setup_logger(PATTERN)

        main()
        if dist.get_rank() == 0:
            print("=================== Qwen2 MMLU score ===============")
            print(log_capture)

            expected_score = acquire_score(log_capture)
            assert math.isclose(expected_score, 0.547, abs_tol=2e-2), f"score {expected_score}, forward pass has been changed, check it!" 

    
